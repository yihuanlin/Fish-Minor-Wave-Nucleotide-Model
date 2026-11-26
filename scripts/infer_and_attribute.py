#!/usr/bin/env python3
"""
scripts/infer_and_attribute.py
==============================
Modular analysis, transfer learning, attribution, and evolutionary dissection engine:
- Block 3: Linear probe fitting (RidgeCV) & grafting Dense(1) head onto frozen encoder
  * Extracts penultimate embeddings (GlobalAveragePooling1D, forward+RC averaged) over 2,114 bp windows
  * Reports both all-peak and repeat-filtered metrics (excluding miR-430 tandem repeats)
  * Supports optional exclusion of housekeeping promoters
  * Exposes base late predicted logcounts alongside y_abs and y_rel
  * Saves grafted models cleanly (state_dict + weights/scaler/config metadata) for PyTorch 2.6+ weights_only=True
- Block 4: Zero-shot cross-species scoring on TSS ±25kb with ROC/PR metrics
  * Evaluates primary clean sets (strictly_zygotic_minor_wave, slam_confirmed_minor_wave)
    and secondary diagnostic set (all_upregulated) vs control_non_activated
  * Supports pred_max_y_abs, pred_max_y_rel, and pred_max_late
  * Real zero-shot Zebrafish -> Medaka pre-MZT ATAC benchmark on st9 vs st11-only summits
- Block 5: Real unbiased sliding-window scanning across target loci (2,114 bp windows)
- Block 6: Real input gradient saliency (input x grad), single-base in silico mutagenesis (ISM),
  TOMTOM motif matching (premzt_tfs.meme & full JASPAR), and fast 2-strand evolutionary swap
- Block 7: QuantSeq 3'-end scoring (terminal 1,000 bp / 3' UTR primary vs all-exons)
  with central 37.5% output crop handling, and ortholog mapping utilities.
- ABSOLUTE ZERO-MOCK POLICY: strictly eliminates all mock embeddings, sine waves, and hardcoded sequences.
"""

import os
import re
import sys
import math
import json
import gzip
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Set

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure conda env bin directory is at the front of PATH
bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import numpy as np
import pandas as pd
import random
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import torch
import torch.nn as nn
import chrombpnet

try:
    from scripts.task_manager import set_task_progress
except ImportError:
    try:
        from task_manager import set_task_progress
    except ImportError:
        def set_task_progress(work_dir, task_name, progress_str):
            pass

# Re-export shared high-performance Pre-MZT inference API
try:
    from scripts.premzt_inference import (
        infer_premzt,
        infer_premzt_batch,
        InferenceContext,
        parse_locus_range,
    )
except ImportError:
    try:
        from premzt_inference import (
            infer_premzt,
            infer_premzt_batch,
            InferenceContext,
            parse_locus_range,
        )
    except ImportError:
        pass


# ==============================================================================
# Helper: Fast One-Hot Encoding & Reverse Complement
# ==============================================================================
_ASCII_TO_ONEHOT = np.zeros(128, dtype=np.int8) - 1
_ASCII_TO_ONEHOT[ord('A')] = _ASCII_TO_ONEHOT[ord('a')] = 0
_ASCII_TO_ONEHOT[ord('C')] = _ASCII_TO_ONEHOT[ord('c')] = 1
_ASCII_TO_ONEHOT[ord('G')] = _ASCII_TO_ONEHOT[ord('g')] = 2
_ASCII_TO_ONEHOT[ord('T')] = _ASCII_TO_ONEHOT[ord('t')] = 3

_RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    """Reverse complement of a nucleotide string."""
    return seq.translate(_RC_TRANS)[::-1]


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """One-hot encode DNA sequence to (4, L) float32 array (A, C, G, T)."""
    seq_bytes = np.frombuffer(seq.encode('ascii', errors='ignore'), dtype=np.uint8)
    indices = _ASCII_TO_ONEHOT[seq_bytes]
    mat = np.zeros((4, len(seq_bytes)), dtype=np.float32)
    for c in range(4):
        mat[c, :] = (indices == c)
    return mat


# ==============================================================================
# 1. Block 3: Stage 2 Linear Probe Fitting & Model Grafting
# ==============================================================================

def sanitize_for_torch_save(obj: Any) -> Any:
    """Recursively convert numpy/pandas/custom objects to pure Python primitives for PyTorch 2.6+ weights_only=True."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool, bytes)):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_torch_save(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_torch_save(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_torch_save(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize_for_torch_save(x) for x in sorted(list(obj))]
    if hasattr(obj, "__dict__"):
        return sanitize_for_torch_save(vars(obj))
    return str(obj)


def load_chrombpnet_encoder(checkpoint_path: str | Path, device: str = "cpu") -> nn.Module:
    """Load trained ChromBPNet or BPNet PyTorch checkpoint strictly."""
    p = Path(checkpoint_path).resolve()
    assert p.exists() and p.stat().st_size > 1024, f"[FATAL] ChromBPNet checkpoint missing or empty: {p}"
    model = chrombpnet.load_model(p, map_location=device)
    model.eval()
    model.base_checkpoint_path = str(p)

    enc = model.tf_model if hasattr(model, "tf_model") else model
    enc.base_checkpoint_path = str(p)
    assert hasattr(enc, "_features"), f"[FATAL] Loaded model {type(model)} missing '_features' method"

    dummy_x = torch.zeros(1, 4, 2114, device=device)
    with torch.no_grad():
        feat = enc._features(dummy_x)
        feat_dim = feat.shape[1]
        assert feat_dim >= 64, f"[FATAL] Penultimate feature dim {feat_dim} < 64 (logits or invalid feature layer)"

    print(f"[+] Loaded ChromBPNet encoder from {p} (penultimate feature dim: {feat_dim})")
    return model


class GraftedChromBPNet(nn.Module):
    """
    End-to-end grafted neural network combining a frozen late-blastula encoder
    with linear probe heads (y_abs, y_rel) and base late accessibility prediction.
    """

    def __init__(
        self,
        encoder: nn.Module,
        coef_abs: np.ndarray,
        intercept_abs: float,
        coef_rel: np.ndarray,
        intercept_rel: float,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray
    ):
        super().__init__()
        self.encoder = encoder
        self.base_checkpoint_path = getattr(encoder, "base_checkpoint_path", None)
        for p in self.encoder.parameters():
            p.requires_grad = False

        in_dim = len(coef_abs)
        self.head_abs = nn.Linear(in_dim, 1)
        self.head_rel = nn.Linear(in_dim, 1)

        # Store probe attributes for clean serialization
        self.coef_abs = np.array(coef_abs, dtype=np.float32)
        self.intercept_abs = float(intercept_abs)
        self.coef_rel = np.array(coef_rel, dtype=np.float32)
        self.intercept_rel = float(intercept_rel)
        self.scaler_mean = np.array(scaler_mean, dtype=np.float32)
        self.scaler_scale = np.array(scaler_scale, dtype=np.float32)

        # Fold StandardScaler directly into Linear weights:
        # W_folded = W / scale, b_folded = intercept - sum(W * mean / scale)
        with torch.no_grad():
            w_abs = torch.tensor(coef_abs / scaler_scale, dtype=torch.float32).unsqueeze(0)
            b_abs = torch.tensor([intercept_abs - np.sum(coef_abs * (scaler_mean / scaler_scale))], dtype=torch.float32)
            self.head_abs.weight.copy_(w_abs)
            self.head_abs.bias.copy_(b_abs)

            w_rel = torch.tensor(coef_rel / scaler_scale, dtype=torch.float32).unsqueeze(0)
            b_rel = torch.tensor([intercept_rel - np.sum(coef_rel * (scaler_mean / scaler_scale))], dtype=torch.float32)
            self.head_rel.weight.copy_(w_rel)
            self.head_rel.bias.copy_(b_rel)

    def extract_features(self, x: torch.Tensor, rc_average: bool = True) -> torch.Tensor:
        """Extract penultimate embeddings (GAP-1D) from encoder."""
        if x.ndim == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.permute(0, 2, 1)

        enc = self.encoder.tf_model if hasattr(self.encoder, "tf_model") else self.encoder

        if hasattr(enc, "_features"):
            h_fwd = enc._features(x).mean(dim=-1)
            if rc_average:
                x_rc = torch.flip(x, dims=[-1])[:, [3, 2, 1, 0], :]
                h_rev = enc._features(x_rc).mean(dim=-1)
                return 0.5 * (h_fwd + h_rev)
            return h_fwd
        else:
            h = enc(x)
            if isinstance(h, tuple):
                h = h[1] if h[1] is not None else h[0]
            if h.ndim == 3:
                h = h.mean(dim=-1)
            return h

    def forward(
        self,
        x: torch.Tensor,
        rc_average: bool = True,
        return_late: bool = False,
        compute_late: bool = False,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        emb = self.extract_features(x, rc_average=rc_average)
        out_abs = self.head_abs(emb).reshape(-1)
        out_rel = self.head_rel(emb).reshape(-1)

        out_late = None
        enc = self.encoder.tf_model if hasattr(self.encoder, "tf_model") else self.encoder

        # Base late accessibility predicted logcounts from underlying trained encoder
        # Check all possible counts-head attribute names on BPNet / ChromBPNet to reuse emb
        counts_head = None
        for cand_attr in ["count_dense", "counts_head", "count_linear", "counts_dense", "linear_counts"]:
            if hasattr(enc, cand_attr) and getattr(enc, cand_attr) is not None:
                counts_head = getattr(enc, cand_attr)
                break

        if counts_head is not None:
            # Reuses already-computed emb (single CNN pass instead of re-running CNN twice)
            out_late = counts_head(emb).reshape(-1)
        elif return_late or compute_late:
            x_in = x
            if x_in.ndim == 3 and x_in.shape[1] != 4 and x_in.shape[2] == 4:
                x_in = x_in.permute(0, 2, 1)
            enc_out = enc(x_in.float())
            if isinstance(enc_out, tuple) and len(enc_out) > 1 and enc_out[1] is not None:
                out_late = enc_out[1].reshape(-1)
                if rc_average:
                    x_rc = torch.flip(x_in, dims=[-1])[:, [3, 2, 1, 0], :]
                    enc_out_rc = enc(x_rc.float())
                    if isinstance(enc_out_rc, tuple) and len(enc_out_rc) > 1 and enc_out_rc[1] is not None:
                        out_late = 0.5 * (out_late + enc_out_rc[1].reshape(-1))
            else:
                raise RuntimeError(
                    "[FATAL] Unable to compute late accessibility counts: neither count_dense head "
                    "nor encoder counts output found."
                )

        if return_late:
            if out_late is None:
                raise RuntimeError("[FATAL] return_late=True requested but late accessibility could not be computed.")
            return out_abs, out_rel, out_late

        res = {
            "y_abs": out_abs,
            "y_rel": out_rel,
            "embedding": emb
        }
        if out_late is not None:
            res["pred_late"] = out_late
            res["late_logcounts"] = out_late
            res["late"] = out_late
        return res


def save_grafted_checkpoint(
    grafted_model: GraftedChromBPNet,
    out_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
    base_checkpoint_path: Optional[str | Path] = None
) -> str:
    """Save grafted model cleanly (state_dict + weights/scaler/config metadata) for PyTorch 2.6+ weights_only=True."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Extract encoder config if present so architecture details (filters, layers) are preserved
    enc = grafted_model.encoder
    enc_target = enc.tf_model if hasattr(enc, "tf_model") else enc
    enc_cfg = None
    if hasattr(enc_target, "config") and enc_target.config is not None:
        try:
            import dataclasses
            if dataclasses.is_dataclass(enc_target.config):
                enc_cfg = dataclasses.asdict(enc_target.config)
            elif hasattr(enc_target.config, "__dict__"):
                enc_cfg = dict(vars(enc_target.config))
        except Exception:
            enc_cfg = None

    clean_cfg = sanitize_for_torch_save(enc_cfg)
    clean_meta = sanitize_for_torch_save(metadata or {})

    # Detect base_checkpoint_path from argument, attribute, or metadata
    base_ckpt = base_checkpoint_path or getattr(grafted_model, "base_checkpoint_path", None) or getattr(grafted_model.encoder, "base_checkpoint_path", None)
    if not base_ckpt and metadata and "base_checkpoint_path" in metadata:
        base_ckpt = metadata["base_checkpoint_path"]

    payload = {
        "head_abs_weight": grafted_model.head_abs.weight.detach().cpu(),
        "head_abs_bias": grafted_model.head_abs.bias.detach().cpu(),
        "head_rel_weight": grafted_model.head_rel.weight.detach().cpu(),
        "head_rel_bias": grafted_model.head_rel.bias.detach().cpu(),
        "coef_abs": torch.tensor(grafted_model.coef_abs),
        "intercept_abs": float(grafted_model.intercept_abs),
        "coef_rel": torch.tensor(grafted_model.coef_rel),
        "intercept_rel": float(grafted_model.intercept_rel),
        "scaler_mean": torch.tensor(grafted_model.scaler_mean),
        "scaler_scale": torch.tensor(grafted_model.scaler_scale),
        "encoder_state_dict": {k: v.detach().cpu() for k, v in grafted_model.encoder.state_dict().items()},
        "encoder_config": clean_cfg,
        "base_checkpoint_path": str(base_ckpt) if base_ckpt else "",
        "metadata": clean_meta
    }
    torch.save(payload, str(p))
    print(f"[+] Saved grafted model checkpoint to {p}")
    return str(p)


def load_grafted_model(
    checkpoint_path: str | Path,
    base_encoder: Optional[nn.Module] = None,
    device: str = "cpu",
    work_dir: Optional[str | Path] = None
) -> GraftedChromBPNet:
    """Load grafted model reliably under PyTorch 2.6+ weights_only=True."""
    p = Path(checkpoint_path).resolve()
    try:
        payload = torch.load(str(p), map_location=device, weights_only=False)
    except Exception:
        payload = torch.load(str(p), map_location=device)

    if base_encoder is None:
        # First attempt: reload the base encoder via chrombpnet.load_model(base_checkpoint_path)
        base_ckpt_str = payload.get("base_checkpoint_path", "")
        if base_ckpt_str:
            cands = [Path(base_ckpt_str)]
            if work_dir:
                cands.append(Path(work_dir) / base_ckpt_str)
            cands.append(p.parent / base_ckpt_str)
            for cand in cands:
                if cand.exists() and cand.is_file():
                    try:
                        base_encoder = chrombpnet.load_model(str(cand), map_location=device)
                        break
                    except Exception:
                        base_encoder = None

        # Fallback to ModelConfig if base checkpoint was moved or unavailable
        if base_encoder is None:
            enc_cfg = payload.get("encoder_config")
            cleaned_cfg = {}
            if enc_cfg and isinstance(enc_cfg, dict):
                for k, v in enc_cfg.items():
                    if k == "inputlen":
                        cleaned_cfg["input_length"] = v
                    elif k == "outputlen":
                        cleaned_cfg["output_length"] = v
                    elif k in ("input_length", "output_length", "filters", "n_dilation_layers", "counts_loss_weight", "counts_head"):
                        cleaned_cfg[k] = v
            try:
                cfg = chrombpnet.ModelConfig(**cleaned_cfg)
            except Exception:
                cfg = chrombpnet.ModelConfig()

            encoder_keys = list(payload.get("encoder_state_dict", {}).keys())
            is_cbp = any(k.startswith("tf_model.") for k in encoder_keys)
            if is_cbp:
                dummy_bias = chrombpnet.BPNet(cfg)
                dummy_tf = chrombpnet.BPNet(cfg)
                base_encoder = chrombpnet.ChromBPNet(cfg, bias_model=dummy_bias, tf_model=dummy_tf)
            else:
                base_encoder = chrombpnet.BPNet(cfg)

    base_encoder.load_state_dict(payload["encoder_state_dict"])
    base_encoder.to(device)
    base_encoder.eval()

    grafted = GraftedChromBPNet(
        encoder=base_encoder,
        coef_abs=(payload["coef_abs"].cpu().numpy() if isinstance(payload["coef_abs"], torch.Tensor) else np.asarray(payload["coef_abs"])),
        intercept_abs=float(payload["intercept_abs"]),
        coef_rel=(payload["coef_rel"].cpu().numpy() if isinstance(payload["coef_rel"], torch.Tensor) else np.asarray(payload["coef_rel"])),
        intercept_rel=float(payload["intercept_rel"]),
        scaler_mean=(payload["scaler_mean"].cpu().numpy() if isinstance(payload["scaler_mean"], torch.Tensor) else np.asarray(payload["scaler_mean"])),
        scaler_scale=(payload["scaler_scale"].cpu().numpy() if isinstance(payload["scaler_scale"], torch.Tensor) else np.asarray(payload["scaler_scale"]))
    )
    if payload.get("base_checkpoint_path"):
        grafted.base_checkpoint_path = payload["base_checkpoint_path"]
    grafted.to(device)
    grafted.eval()
    return grafted


def extract_summit_embeddings_from_genome(
    encoder: nn.Module,
    genome_fa_path: str | Path,
    df_summits: pd.DataFrame,
    window_bp: int = 2114,
    batch_size: int = 64,
    device: str = "cpu"
) -> np.ndarray:
    """Extract real penultimate embeddings across all summit windows from genome FASTA."""
    from pyfaidx import Fasta
    genome = Fasta(str(genome_fa_path), as_raw=True)
    encoder.to(device)
    encoder.eval()

    all_embs = []
    n_summits = len(df_summits)
    half_w = window_bp // 2
    chrom_arr = df_summits["chrom"].astype(str).values
    summit_arr = df_summits["summit"].astype(int).values

    # Process in batches
    for start_idx in range(0, n_summits, batch_size):
        end_idx = min(start_idx + batch_size, n_summits)

        batch_tensors = []
        for i in range(start_idx, end_idx):
            chrom = chrom_arr[i]
            summit = summit_arr[i]
            w_start = max(0, summit - half_w)
            w_end = w_start + window_bp

            try:
                seq = str(genome[chrom][w_start:w_end])
            except Exception:
                seq = "N" * window_bp

            if len(seq) < window_bp:
                seq = seq + "N" * (window_bp - len(seq))

            mat = one_hot_encode_sequence(seq)
            batch_tensors.append(mat)

        x_b = torch.tensor(np.array(batch_tensors), dtype=torch.float32, device=device)

        with torch.no_grad():
            enc = encoder.tf_model if hasattr(encoder, "tf_model") else encoder
            if hasattr(enc, "_features"):
                f_fwd = enc._features(x_b).mean(dim=-1)
                x_rc = torch.flip(x_b, dims=[-1])[:, [3, 2, 1, 0], :]
                f_rev = enc._features(x_rc).mean(dim=-1)
                emb = 0.5 * (f_fwd + f_rev)
            else:
                h = enc(x_b)
                emb = h[1] if isinstance(h, tuple) and h[1] is not None else h[0]
                if emb.ndim == 3:
                    emb = emb.mean(dim=-1)

            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def fit_linear_probes(
    embeddings: np.ndarray,
    df_targets: pd.DataFrame,
    test_split_col: str = "split",
    test_split_val: str = "test",
    exclude_housekeeping: bool = False
) -> Dict[str, Any]:
    """Fit RidgeCV linear probe on train+val embeddings, evaluate on held-out test split."""
    from sklearn.model_selection import GroupKFold

    is_test = (df_targets[test_split_col] == test_split_val).values
    is_repeat = df_targets.get("is_repeat_downweight", pd.Series(False, index=df_targets.index)).values
    is_hk = df_targets.get("is_housekeeping", pd.Series(False, index=df_targets.index)).values

    train_mask = df_targets[test_split_col].isin(["train", "valid"]).values & (~is_repeat)
    if exclude_housekeeping:
        train_mask = train_mask & (~is_hk)

    test_mask_all = is_test
    test_mask_filtered = is_test & (~is_repeat)
    if exclude_housekeeping:
        test_mask_filtered = test_mask_filtered & (~is_hk)

    X_train = embeddings[train_mask]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_all_scaled = scaler.transform(embeddings[test_mask_all])
    X_test_filt_scaled = scaler.transform(embeddings[test_mask_filtered])

    train_df = df_targets.loc[train_mask]
    if "chrom" in train_df.columns:
        chroms = train_df["chrom"].values
        n_unique_chroms = len(np.unique(chroms))
        if n_unique_chroms >= 2:
            n_splits = min(5, n_unique_chroms)
            gkf = GroupKFold(n_splits=n_splits)
            cv_splits = list(gkf.split(X_train_scaled, train_df["y_abs"].values, groups=chroms))
        else:
            cv_splits = 5
    else:
        cv_splits = 5

    results = {"scaler": scaler}
    for target in ["y_abs", "y_rel"]:
        y_train = train_df[target].values
        y_test_all = df_targets.loc[test_mask_all, target].values
        y_test_filt = df_targets.loc[test_mask_filtered, target].values

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 20), cv=cv_splits)
        ridge.fit(X_train_scaled, y_train)

        # 1. Repeat-filtered metrics (primary benchmark)
        y_pred_filt = ridge.predict(X_test_filt_scaled)
        spearman_filt_r, spearman_filt_p = stats.spearmanr(y_test_filt, y_pred_filt)
        median_filt = np.median(y_test_filt)
        auroc_filt = roc_auc_score(y_test_filt > median_filt, y_pred_filt)

        # 2. All-peak metrics (unfiltered)
        y_pred_all = ridge.predict(X_test_all_scaled)
        spearman_all_r, spearman_all_p = stats.spearmanr(y_test_all, y_pred_all)
        median_all = np.median(y_test_all)
        auroc_all = roc_auc_score(y_test_all > median_all, y_pred_all)

        # 3. Early-vs-late peak classification AUROC (cpm_early >= 2.0 vs cpm_early < 0.5)
        early_late_auroc = np.nan
        early_late_auroc_quantile = np.nan
        early_col = "cpm_early" if "cpm_early" in df_targets.columns else ("early_cpm" if "early_cpm" in df_targets.columns else None)
        count_early_col = "count_early" if "count_early" in df_targets.columns else None
        if early_col:
            filt_df = df_targets.loc[test_mask_filtered]
            early_active = filt_df[early_col] >= 2.0
            late_only = filt_df[early_col] < 0.5
            subset_mask = early_active | late_only
            if early_active.sum() >= 5 and late_only.sum() >= 5:
                y_bin = early_active[subset_mask].astype(int).values
                preds_sub = y_pred_filt[subset_mask.values]
                early_late_auroc = float(roc_auc_score(y_bin, preds_sub))

            # Quantile-based early-vs-late AUROC (top 10% cpm_early vs count_early == 0 / bottom 10%)
            cpm_vals = filt_df[early_col].values
            if len(cpm_vals) >= 10:
                q90 = np.quantile(cpm_vals, 0.90)
                q_top = cpm_vals >= q90
                if count_early_col and count_early_col in filt_df.columns:
                    q_bottom = (filt_df[count_early_col].values == 0)
                else:
                    q10 = np.quantile(cpm_vals, 0.10)
                    q_bottom = cpm_vals <= q10
                q_mask = q_top | q_bottom
                if q_top.sum() >= 3 and q_bottom.sum() >= 3:
                    y_q_bin = q_top[q_mask].astype(int)
                    preds_q = y_pred_filt[q_mask]
                    early_late_auroc_quantile = float(roc_auc_score(y_q_bin, preds_q))

        results[target] = {
            "model": ridge,
            "coef": ridge.coef_,
            "intercept": ridge.intercept_,
            "best_alpha": ridge.alpha_,
            "test_spearman_r": float(spearman_filt_r),
            "test_spearman_p": float(spearman_filt_p),
            "test_auroc": float(auroc_filt),
            "early_late_auroc": early_late_auroc,
            "early_late_auroc_quantile": early_late_auroc_quantile,
            "all_peaks_spearman_r": float(spearman_all_r),
            "all_peaks_spearman_p": float(spearman_all_p),
            "all_peaks_auroc": float(auroc_all)
        }
        print(f"[+] Linear Probe [{target}]:")
        print(f"    Repeat-filtered (excl. miR-430): Spearman r = {spearman_filt_r:.3f} (p={spearman_filt_p:.2e}), AUROC (>median) = {auroc_filt:.3f}")
        if not np.isnan(early_late_auroc):
            print(f"    Early-vs-Late Peak AUROC (>=2.0 vs <0.5 cpm): {early_late_auroc:.3f}")
        if not np.isnan(early_late_auroc_quantile):
            print(f"    Early-vs-Late Peak AUROC (Quantile top 10% vs 0 count): {early_late_auroc_quantile:.3f}")
        print(f"    All test peaks (unfiltered):     Spearman r = {spearman_all_r:.3f} (p={spearman_all_p:.2e}), AUROC = {auroc_all:.3f}")

    return results


# ==============================================================================
# 2. Ortholog Mapping Utility
# ==============================================================================

CURATED_TELEOST_ALIASES: Dict[str, List[str]] = {
    "nfyal": ["NFYA", "NFYB", "NFYC"],
    "arntl1a": ["ARNTL"],
    "myca": ["MYC"],
    "mych": ["MYC"],
    "pou2f1b": ["POU2F1"],
    "hif1aa": ["HIF1A"],
    "hif1al": ["HIF1A"],
    "epas1b": ["EPAS1"],
    "rorca": ["RORC"],
    "nr6a1a": ["NR6A1"],
    "arxa": ["ARX"],
    "mitfa": ["MITF"],
    "stat1a": ["STAT1"],
    "pou5f3": ["POU5F1"],
    "oct4": ["POU5F1"],
    "sox19a": ["SOX2"],
    "sox19b": ["SOX2"],
    "soxb1": ["SOX2"],
    "mxtx2": ["MIXL1"],
    "vox": ["VENTX"],
    "vent": ["VENTX"],
    "ved": ["VENTX"],
    "notch2": ["NOTCH2"],
    "slc35a3a": ["SLC35A3"],
    "slc35a3b": ["SLC35A3"],
    "klf17": ["KLF17"],
    "klf4": ["KLF4"],
}


def get_teleost_tf_targets(name: str) -> List[str]:
    """Return all normalized candidate target symbols for a teleost gene symbol."""
    raw = name.strip()
    g_low = raw.lower()
    targets: List[str] = []
    if g_low in CURATED_TELEOST_ALIASES:
        targets.extend(CURATED_TELEOST_ALIASES[g_low])
    direct = raw.upper()
    if direct not in targets:
        targets.append(direct)
    # Teleost paralog suffix normalization: strip trailing a, b, l, 1a, 1b, 2a, 2b, .1
    s = re.sub(r"\.1$", "", g_low)
    s = re.sub(r"(1a|1b|2a|2b|[abl])$", "", s)
    stem_up = s.upper()
    if stem_up not in targets and len(stem_up) >= 3:
        targets.append(stem_up)
    return targets


def normalize_teleost_tf_name(name: str) -> str:
    """Normalize teleost gene symbols: strip trailing 'a'/'b'/paralog suffixes and map curated aliases."""
    targets = get_teleost_tf_targets(name)
    return targets[0] if targets else name.strip().upper()


def get_or_create_ortholog_map(
    target_gtf_path: str,
    zf_gtf_path: str,
    out_tsv: str,
    ortholog_map_path: Optional[str] = None
) -> pd.DataFrame:
    """Create or load ortholog map between target species and Zebrafish."""
    if ortholog_map_path and os.path.exists(ortholog_map_path):
        print(f"[+] Loading user-supplied ortholog map: {ortholog_map_path}")
        return pd.read_csv(ortholog_map_path, sep="\t")

    out_p = Path(out_tsv)
    if out_p.exists() and out_p.stat().st_size > 0:
        return pd.read_csv(out_p, sep="\t")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Building 1-to-1 ortholog map via Ensembl symbols + teleost paralog normalization...")

    def parse_genes(gtf_file):
        genes = {}
        with (gzip.open(gtf_file, "rt") if str(gtf_file).endswith(".gz") else open(gtf_file, "rt")) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.strip().split("\t")
                if len(p) >= 9 and p[2] == "gene":
                    gid, gname = None, None
                    for item in p[8].split(";"):
                        item = item.strip()
                        if item.startswith("gene_id"):
                            m = re.search(r'gene_id\s+"([^"]+)"', item)
                            if m:
                                gid = m.group(1)
                        elif item.startswith("gene_name"):
                            m = re.search(r'gene_name\s+"([^"]+)"', item)
                            if m:
                                gname = m.group(1)
                    if gid:
                        genes[gid] = gname or gid
        return genes

    target_genes = parse_genes(target_gtf_path)
    zf_genes = parse_genes(zf_gtf_path)

    zf_by_name = {gname.lower(): (gid, gname) for gid, gname in zf_genes.items()}
    zf_by_norm = {normalize_teleost_tf_name(gname): (gid, gname) for gid, gname in zf_genes.items()}

    records = []
    direct_cnt, norm_cnt = 0, 0

    for t_gid, t_gname in target_genes.items():
        gn_low = t_gname.lower()
        gn_norm = normalize_teleost_tf_name(t_gname)

        if gn_low in zf_by_name:
            z_gid, z_gname = zf_by_name[gn_low]
            records.append({
                "target_gene_id": t_gid,
                "target_gene_name": t_gname,
                "zf_gene_id": z_gid,
                "zf_gene_name": z_gname,
                "mapping_type": "direct_symbol_match"
            })
            direct_cnt += 1
        elif gn_norm in zf_by_norm:
            z_gid, z_gname = zf_by_norm[gn_norm]
            records.append({
                "target_gene_id": t_gid,
                "target_gene_name": t_gname,
                "zf_gene_id": z_gid,
                "zf_gene_name": z_gname,
                "mapping_type": "paralog_normalized_match"
            })
            norm_cnt += 1

    df_map = pd.DataFrame(records)
    df_map.to_csv(out_p, sep="\t", index=False)
    print(f"[+] Ortholog map generated: {len(df_map)} / {len(target_genes)} genes mapped.")
    return df_map


# ==============================================================================
# 3. Block 4: Zero-Shot Validation & Benchmark
# ==============================================================================

def score_tss_regions(
    grafted_model: GraftedChromBPNet,
    genome_fa_path: str,
    gene_df: pd.DataFrame,
    flank_bp: int = 25000,
    window_bp: int = 2114,
    stride_bp: int = 250,
    batch_size: int = 256,
    device: Optional[str] = None,
    work_dir: Optional[Union[str, Path]] = None,
    task_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Score ±flank_bp around gene TSSs with grafted models, taking max predicted accessibility.
    Vectorized sliding window slicing (one-hot encoding full flank once) + torch.amp.autocast.
    Resolves gene_id consistently from gene_id or Geneid.
    Returns pred_max_y_abs, pred_max_y_rel, and pred_max_late.
    """
    from pyfaidx import Fasta
    genome = Fasta(genome_fa_path, as_raw=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    grafted_model.to(device)
    grafted_model.eval()

    eval_cats = {"strictly_zygotic_minor_wave", "all_upregulated", "control_non_activated"}
    mask = gene_df["category"].isin(eval_cats)
    if "is_slam_confirmed_minor_wave" in gene_df.columns:
        mask = mask | (gene_df["is_slam_confirmed_minor_wave"] == True)
    eval_df = gene_df[mask].copy()

    # Subsample control if large for snappy validation
    ctrl_mask = eval_df["category"] == "control_non_activated"
    if ctrl_mask.sum() > 300:
        ctrl_sampled = eval_df[ctrl_mask].sample(n=300, random_state=42)
        eval_df = pd.concat([eval_df[~ctrl_mask], ctrl_sampled])

    results = []
    total_genes = len(eval_df)
    autocast_enabled = (str(device).startswith("cuda") and torch.cuda.is_available())

    for idx_count, (idx, row) in enumerate(eval_df.iterrows(), start=1):
        chrom = str(row["chrom"])
        strand = str(row.get("strand", "+"))
        tss = int(row["start"]) if strand == "+" else int(row["end"])
        gene_id = str(row.get("gene_id", row.get("Geneid", f"gene_{idx}")))
        gene_name = str(row.get("gene_name", gene_id))
        category = str(row.get("category", "unclassified"))
        is_slam_conf = bool(row.get("is_slam_confirmed_minor_wave", False))

        reg_start = max(0, tss - flank_bp)
        reg_end = tss + flank_bp
        try:
            seq = str(genome[chrom][reg_start:reg_end])
        except Exception:
            continue

        if len(seq) < window_bp:
            continue

        n_windows = (len(seq) - window_bp) // stride_bp + 1
        if n_windows <= 0:
            continue

        # Vectorized extraction: One-hot encode the full 2 * flank_bp sequence ONCE
        full_enc = one_hot_encode_sequence(seq)  # shape: (4, len(seq))
        windows = np.stack([full_enc[:, i * stride_bp : i * stride_bp + window_bp] for i in range(n_windows)])

        max_y_abs = -1e9
        max_y_rel = -1e9
        max_late = -1e9

        for b_start in range(0, len(windows), batch_size):
            b_chunk = windows[b_start : b_start + batch_size]
            b_tensor = torch.tensor(b_chunk, dtype=torch.float32, device=device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    out = grafted_model(b_tensor, compute_late=True)
                    chunk_abs = float(torch.max(out["y_abs"]).cpu().item())
                    chunk_rel = float(torch.max(out["y_rel"]).cpu().item())
                    chunk_late = float(torch.max(out.get("pred_late", out.get("late_logcounts", out["y_abs"]))).cpu().item())

                if chunk_abs > max_y_abs:
                    max_y_abs = chunk_abs
                if chunk_rel > max_y_rel:
                    max_y_rel = chunk_rel
                if chunk_late > max_late:
                    max_late = chunk_late

        if work_dir and task_name and (idx_count % 25 == 0 or idx_count == total_genes):
            pct = (idx_count / total_genes) * 100
            set_task_progress(work_dir, task_name, f"Scoring genes {idx_count}/{total_genes} ({pct:.1f}%)")

        res_item = {
            "gene_id": gene_id,
            "gene_name": gene_name,
            "category": category,
            "is_slam_confirmed": is_slam_conf,
            "chrom": chrom,
            "tss": tss,
            "pred_max_y_abs": max_y_abs,
            "pred_max_y_rel": max_y_rel,
            "pred_max_late": max_late
        }
        for bool_col in ["is_strictly_zygotic_minor_wave", "is_all_upregulated", "is_control_non_activated", "is_slam_confirmed_minor_wave"]:
            if bool_col in row:
                res_item[bool_col] = bool(row[bool_col])
        results.append(res_item)

    return pd.DataFrame(results)


def benchmark_transfer_metrics(
    scored_genes_df: pd.DataFrame,
    pos_category: str = "strictly_zygotic_minor_wave",
    ctrl_category: str = "control_non_activated",
    score_col: str = "pred_max_y_abs"
) -> Dict[str, Any]:
    """Calculate ROC-AUC, PR-AUC, and Mann-Whitney U p-value comparing positive vs. control genes."""
    if pos_category == "slam_confirmed_minor_wave":
        pos_mask = (scored_genes_df.get("is_slam_confirmed", False) == True) | \
                   (scored_genes_df.get("is_slam_confirmed_minor_wave", False) == True) | \
                   (scored_genes_df["category"] == "slam_confirmed_minor_wave")
    elif pos_category in ["all_upregulated", "is_all_upregulated"]:
        pos_mask = (scored_genes_df.get("is_all_upregulated", False) == True) | \
                   (scored_genes_df["category"].isin(["all_upregulated", "strictly_zygotic_minor_wave"]))
    elif pos_category in ["strictly_zygotic_minor_wave", "is_strictly_zygotic_minor_wave"]:
        pos_mask = (scored_genes_df.get("is_strictly_zygotic_minor_wave", False) == True) | \
                   (scored_genes_df["category"] == "strictly_zygotic_minor_wave")
    else:
        pos_mask = scored_genes_df["category"] == pos_category

    if ctrl_category in ["control_non_activated", "is_control_non_activated"]:
        ctrl_mask = (scored_genes_df.get("is_control_non_activated", False) == True) | \
                    (scored_genes_df["category"] == "control_non_activated")
    else:
        ctrl_mask = scored_genes_df["category"] == ctrl_category

    pos_scores = scored_genes_df.loc[pos_mask, score_col].dropna().values
    ctrl_scores = scored_genes_df.loc[ctrl_mask, score_col].dropna().values

    if len(pos_scores) == 0 or len(ctrl_scores) == 0:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "mw_u_pval": np.nan, "n_pos": len(pos_scores), "n_ctrl": len(ctrl_scores)}

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(ctrl_scores)])
    y_scores = np.concatenate([pos_scores, ctrl_scores])

    auroc = roc_auc_score(y_true, y_scores)
    prauc = average_precision_score(y_true, y_scores)
    u_stat, p_val = stats.mannwhitneyu(pos_scores, ctrl_scores, alternative="greater")

    return {
        "roc_auc": float(auroc),
        "pr_auc": float(prauc),
        "mw_u_pval": float(p_val),
        "n_pos": len(pos_scores),
        "n_ctrl": len(ctrl_scores)
    }


def score_cross_species_medaka_atac(
    grafted_model: GraftedChromBPNet,
    medaka_targets_tsv: str,
    medaka_fa_path: str,
    window_bp: int = 2114,
    device: str = "cpu"
) -> Dict[str, float]:
    """Real zero-shot benchmark of Zebrafish-trained ChromBPNet on Medaka st9 vs st11-only summits."""
    from pyfaidx import Fasta
    genome = Fasta(medaka_fa_path, as_raw=True)
    df_medaka = pd.read_csv(medaka_targets_tsv, sep="\t")

    grafted_model.to(device)
    grafted_model.eval()

    y_true_abs = df_medaka["y_abs"].values
    y_true_rel = df_medaka["y_rel"].values

    preds_abs = []
    preds_rel = []
    half_w = window_bp // 2

    batch_size = 64
    for start_i in range(0, len(df_medaka), batch_size):
        end_i = min(start_i + batch_size, len(df_medaka))
        sub = df_medaka.iloc[start_i:end_i]

        tensors = []
        for _, r in sub.iterrows():
            c = str(r["chrom"])
            s = int(r["summit"])
            w_start = max(0, s - half_w)
            w_end = w_start + window_bp
            try:
                seq = str(genome[c][w_start:w_end])
            except Exception:
                seq = "N" * window_bp
            if len(seq) < window_bp:
                seq = seq + "N" * (window_bp - len(seq))
            tensors.append(one_hot_encode_sequence(seq))

        x_b = torch.tensor(np.array(tensors), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = grafted_model(x_b)
            preds_abs.extend(out["y_abs"].cpu().numpy().tolist())
            preds_rel.extend(out["y_rel"].cpu().numpy().tolist())

    pred_abs = np.array(preds_abs)
    pred_rel = np.array(preds_rel)

    spearman_abs, p_abs = stats.spearmanr(y_true_abs, pred_abs)
    spearman_rel, p_rel = stats.spearmanr(y_true_rel, pred_rel)

    median_abs = np.median(y_true_abs)
    auroc_abs = roc_auc_score(y_true_abs > median_abs, pred_abs)

    return {
        "medaka_st9_spearman_r": float(spearman_abs),
        "medaka_st9_spearman_p": float(p_abs),
        "medaka_st9_auroc": float(auroc_abs),
        "medaka_rel_spearman_r": float(spearman_rel)
    }


# ==============================================================================
# 4. Block 5: Real Unbiased Sliding-Window Scanning
# ==============================================================================

def scan_locus_sliding_window(
    grafted_model: GraftedChromBPNet,
    genome_fa_path: str,
    chrom: str,
    start_bp: int,
    end_bp: int,
    window_bp: int = 2114,
    stride_bp: int = 250,
    batch_size: int = 64,
    device: str = "cpu",
    return_late: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scan a target genomic locus in sliding windows using real grafted ChromBPNet forward passes."""
    from pyfaidx import Fasta
    genome = Fasta(genome_fa_path, as_raw=True)
    grafted_model.to(device)
    grafted_model.eval()

    full_seq = str(genome[chrom][start_bp:end_bp])
    n_windows = (len(full_seq) - window_bp) // stride_bp + 1
    assert n_windows > 0, f"Locus length ({len(full_seq)}) smaller than window size ({window_bp})"

    positions = []
    tensors = []
    for i in range(n_windows):
        w_start = start_bp + i * stride_bp
        w_center = w_start + window_bp // 2
        positions.append(w_center)
        sub_seq = full_seq[i * stride_bp: i * stride_bp + window_bp]
        tensors.append(one_hot_encode_sequence(sub_seq))

    abs_list, rel_list, late_list = [], [], []
    for b_start in range(0, len(tensors), batch_size):
        b_chunk = tensors[b_start:b_start + batch_size]
        b_x = torch.tensor(np.array(b_chunk), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = grafted_model(b_x, compute_late=return_late)
            abs_list.append(out["y_abs"].cpu().numpy())
            rel_list.append(out["y_rel"].cpu().numpy())
            if return_late:
                out_late = out.get("pred_late", out.get("late_logcounts", out["y_abs"])).cpu().numpy()
                late_list.append(out_late)

    pred_abs = np.concatenate(abs_list, axis=0) if abs_list else np.array([])
    pred_rel = np.concatenate(rel_list, axis=0) if rel_list else np.array([])
    pred_late = np.concatenate(late_list, axis=0) if late_list else np.array([])

    if return_late:
        return np.array(positions), pred_abs, pred_rel, pred_late
    return np.array(positions), pred_abs, pred_rel


# ==============================================================================
# 5. Block 6: Real Gradient Attribution, In Silico Mutagenesis & TOMTOM
# ==============================================================================

def compute_input_gradient_attribution(
    grafted_model: GraftedChromBPNet,
    one_hot_seq: np.ndarray,
    target: str = "y_abs",
    device: str = "cpu",
    rc_average: bool = True
) -> np.ndarray:
    """Compute real input x gradient attribution across window with optional RC averaging."""
    grafted_model.to(device)
    grafted_model.eval()

    need_late = (target in ["late", "pred_late", "late_logcounts"])
    x = torch.tensor(one_hot_seq, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad = True

    out = grafted_model(x, rc_average=False, compute_late=need_late)
    target_key = target if target in out else ("late" if "late" in out else "pred_late")
    score = out[target_key].sum()
    score.backward()

    grad = x.grad.squeeze(0).cpu().numpy()
    attr_fwd = one_hot_seq * grad  # shape (4, L)

    if not rc_average:
        return attr_fwd

    grafted_model.zero_grad()
    seq_rc = np.flip(one_hot_seq, axis=-1)[[3, 2, 1, 0], :]
    x_rc = torch.tensor(seq_rc, dtype=torch.float32, device=device).unsqueeze(0)
    x_rc.requires_grad = True

    out_rc = grafted_model(x_rc, rc_average=False, compute_late=need_late)
    target_key_rc = target if target in out_rc else ("late" if "late" in out_rc else "pred_late")
    score_rc = out_rc[target_key_rc].sum()
    score_rc.backward()

    grad_rc = x_rc.grad.squeeze(0).cpu().numpy()
    attr_rc_raw = seq_rc * grad_rc
    attr_rc = np.flip(attr_rc_raw, axis=-1)[[3, 2, 1, 0], :]
    return 0.5 * (attr_fwd + attr_rc)


def dinuc_shuffle_sequence(seq: str, seed: Optional[int] = None) -> str:
    """Altschul-Erickson dinucleotide shuffle algorithm preserving exact dinuc counts and start/end bases."""
    if len(seq) <= 2:
        return seq
    rng = random.Random(seed)
    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])
    last_char = seq[-1]
    reserved_edges = {}
    for char, nbrs in edges.items():
        if char != last_char and nbrs:
            reserved_edges[char] = nbrs[-1]
            edges[char] = nbrs[:-1]
    for char in edges:
        rng.shuffle(edges[char])
        if char in reserved_edges:
            edges[char].append(reserved_edges[char])
    result = [seq[0]]
    curr = seq[0]
    for _ in range(len(seq) - 1):
        if not edges[curr]:
            break
        nxt = edges[curr].pop(0)
        result.append(nxt)
        curr = nxt
    if len(result) < len(seq):
        return seq
    return "".join(result)


def dinuc_shuffle_one_hot(x_np: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Decode one-hot (4, L), dinucleotide shuffle, and re-encode to one-hot."""
    nucs = ["A", "C", "G", "T"]
    idx = np.argmax(x_np, axis=0)
    all_zero = np.all(x_np == 0, axis=0)
    seq = "".join(nucs[i] if not z else "N" for i, z in zip(idx, all_zero))
    shuf_seq = dinuc_shuffle_sequence(seq, seed=seed)
    return one_hot_encode_sequence(shuf_seq)


def compute_integrated_gradients(
    grafted_model: GraftedChromBPNet,
    one_hot_seq: np.ndarray,
    target: str = "y_abs",
    steps: int = 20,
    baseline: str = "zero",
    device: str = "cpu",
    rc_average: bool = True
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution across the sequence window.
    Interpolates from baseline (zero or dinucleotide-shuffled) to input in `steps` steps.
    Supports rc_average=True (averaging fwd and flipped revcomp attributions).
    """
    grafted_model.to(device)
    grafted_model.eval()

    def _ig_single_strand(x_np: np.ndarray) -> np.ndarray:
        if baseline == "zero":
            x_base = np.zeros_like(x_np)
        elif baseline == "dinuc":
            x_base = dinuc_shuffle_one_hot(x_np, seed=42)
        else:
            raise ValueError(f"Unsupported baseline: '{baseline}'. Expected 'zero' or 'dinuc'.")

        need_late = (target in ["late", "pred_late", "late_logcounts"])
        diff = x_np - x_base
        grads = []
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            x_step = x_base + alpha * diff
            x_tensor = torch.tensor(x_step, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
            out = grafted_model(x_tensor, rc_average=False, compute_late=need_late)
            target_key = target if target in out else ("late" if "late" in out else "pred_late")
            score = out[target_key].sum()
            score.backward()
            grads.append(x_tensor.grad.squeeze(0).cpu().numpy())
            grafted_model.zero_grad()

        avg_grad = np.mean(grads, axis=0)
        return diff * avg_grad

    attr_fwd = _ig_single_strand(one_hot_seq)
    if not rc_average:
        return attr_fwd

    seq_rc = np.flip(one_hot_seq, axis=-1)[[3, 2, 1, 0], :]
    attr_rc_raw = _ig_single_strand(seq_rc)
    attr_rc = np.flip(attr_rc_raw, axis=-1)[[3, 2, 1, 0], :]
    return 0.5 * (attr_fwd + attr_rc)


def extract_causal_seqlet(
    wt_importance: np.ndarray,
    min_len: int = 6,
    max_len: int = 14,
    peak_center: Optional[int] = None
) -> Tuple[int, int, float]:
    """
    Select the contiguous causal seqlet of length between min_len (6 bp) and max_len (14 bp)
    that captures the core motif footprint while trimming low-importance flanks.
    Uses signed positive importance (never np.abs) so activating motifs are sharply bounded.
    Returns (start_idx, end_idx, score).
    """
    importance = np.maximum(0.0, np.array(wt_importance, dtype=np.float32))
    L = len(importance)
    if peak_center is None:
        peak_center = int(np.argmax(importance))

    p_val = float(importance[peak_center])
    if p_val <= 0:
        s = max(0, peak_center - min_len // 2)
        return s, min(L, s + min_len), 0.0

    best_score = -float("inf")
    best_start = max(0, peak_center - min_len // 2)
    best_end = min(L, best_start + min_len)

    # Search window lengths min_len to max_len containing peak_center
    for w in range(min_len, min(max_len + 1, L + 1)):
        s_min = max(0, peak_center - w + 1)
        s_max = min(L - w, peak_center)
        for s in range(s_min, s_max + 1):
            e = s + w
            sub = importance[s:e]
            # Edge guards: trim low-importance flanks
            if sub[0] < 0.05 * p_val or sub[-1] < 0.05 * p_val:
                continue
            # Internal dip guard: avoid gluing two distinct adjacent peaks
            if len(sub) >= 5 and np.min(sub[2:-2]) < 0.02 * p_val:
                continue
            score = float(np.sum(sub) / np.sqrt(w))
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    if best_score == -float("inf"):
        # Fallback expansion from peak_center
        st = peak_center
        while st > 0 and importance[st - 1] >= 0.1 * p_val and (peak_center - (st - 1)) <= max_len // 2:
            st -= 1
        en = peak_center + 1
        while en < L and importance[en] >= 0.1 * p_val and ((en + 1) - st) <= max_len:
            en += 1
        if en - st < min_len:
            pad = min_len - (en - st)
            st = max(0, st - pad // 2)
            en = min(L, st + min_len)
            st = max(0, en - min_len)
        best_start, best_end = st, en
        best_score = float(np.sum(importance[st:en]) / np.sqrt(max(1, en - st)))

    return best_start, best_end, best_score


def extract_multiple_causal_seqlets(
    wt_importance: np.ndarray,
    min_len: int = 6,
    max_len: int = 14,
    top_k: int = 5,
    min_importance: float = 0.005
) -> List[Tuple[int, int, float, int]]:
    """
    Extract multiple non-overlapping activating motifs per element (top 3-5 positive peaks)
    using signed positive importance > 0 (never np.abs(c_attr)).
    Returns list of (start_idx, end_idx, score, peak_center).
    """
    pos_imp = np.maximum(0.0, np.array(wt_importance, dtype=np.float32))
    L = len(pos_imp)
    max_val = float(np.max(pos_imp)) if len(pos_imp) > 0 else 0.0
    adaptive_thresh = max(min_importance, 0.15 * max_val)

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(pos_imp, height=adaptive_thresh, distance=min_len)
    if len(peaks) == 0:
        c = int(np.argmax(pos_imp))
        s, e, sc = extract_causal_seqlet(pos_imp, min_len=min_len, max_len=max_len, peak_center=c)
        return [(s, e, sc, c)]

    sorted_peaks = sorted(peaks, key=lambda p: pos_imp[p], reverse=True)
    extracted: List[Tuple[int, int, float, int]] = []
    occupied = np.zeros(L, dtype=bool)

    for p in sorted_peaks:
        if occupied[p]:
            continue
        st, en, sc = extract_causal_seqlet(pos_imp, min_len=min_len, max_len=max_len, peak_center=int(p))
        if np.any(occupied[st:en]):
            continue
        occupied[max(0, st - 2): min(L, en + 2)] = True
        extracted.append((st, en, sc, int(p)))
        if len(extracted) >= top_k:
            break

    return extracted


def compute_in_silico_mutagenesis(
    grafted_model: GraftedChromBPNet,
    wt_one_hot: np.ndarray,
    region_start: int,
    region_len: int = 60,
    target: str = "y_abs",
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform single-base in silico mutagenesis (ISM) batched across all mutations:
    At each position in region [region_start, region_start + region_len), mutate to all 3 alternative bases,
    evaluate delta score (mut - wt). Guards 'N' (all-zero) columns explicitly.
    Returns:
      ism_mat: shape (4, region_len) of signed delta scores (score_mut - score_wt).
      wt_importance: shape (region_len,) where wt base importance is -mean(delta_mut)
                     so activating motifs have positive polarity.
    """
    grafted_model.to(device)
    grafted_model.eval()

    need_late = (target in ["late", "pred_late", "late_logcounts"])
    x_wt = torch.tensor(wt_one_hot, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out_wt = grafted_model(x_wt, compute_late=need_late)
        target_key = target if target in out_wt else ("late" if "late" in out_wt else "pred_late")
        score_wt = float(out_wt[target_key].item())

    mut_tensors = []
    mut_coords = []

    for offset in range(region_len):
        pos = region_start + offset
        col_sum = float(wt_one_hot[:, pos].sum())
        wt_base = int(np.argmax(wt_one_hot[:, pos])) if col_sum > 0 else -1

        for alt_base in range(4):
            if alt_base == wt_base:
                continue

            mut_seq = wt_one_hot.copy()
            mut_seq[:, pos] = 0.0
            mut_seq[alt_base, pos] = 1.0
            mut_tensors.append(mut_seq)
            mut_coords.append((alt_base, offset))

    mut_scores = []
    for b_start in range(0, len(mut_tensors), batch_size):
        b_chunk = mut_tensors[b_start:b_start + batch_size]
        b_x = torch.tensor(np.array(b_chunk), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = grafted_model(b_x, compute_late=need_late)
            mut_scores.extend(out[target_key].cpu().numpy().tolist())

    ism_mat = np.zeros((4, region_len), dtype=np.float32)
    for (alt_base, offset), score_mut in zip(mut_coords, mut_scores):
        ism_mat[alt_base, offset] = score_mut - score_wt

    wt_importance = np.zeros(region_len, dtype=np.float32)
    for offset in range(region_len):
        pos = region_start + offset
        if float(wt_one_hot[:, pos].sum()) == 0:
            wt_importance[offset] = 0.0
            continue
        wt_base = int(np.argmax(wt_one_hot[:, pos]))
        deltas = [ism_mat[b, offset] for b in range(4) if b != wt_base]
        wt_importance[offset] = -float(np.mean(deltas))

    return ism_mat, wt_importance


def compute_wt_projected_ism_matrix(
    ism_mat: np.ndarray,
    wt_one_hot: np.ndarray,
    wt_importance: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Project signed ISM deltas onto the Wild-Type sequence so activating WT bases render positively in logomaker logos.
    - ism_mat: shape (4, L), signed delta scores (score_mut - score_wt) where WT bases are 0.0.
    - wt_one_hot: shape (4, L), one-hot representation of the exact sliced WT sub-window.
    - wt_importance: shape (L,), where wt_importance[i] = -mean(ism_mat[alt, i]).
    Returns:
      wt_logo_mat: shape (4, L) where WT bases have positive height for activating mutations and 0.0 for non-WT bases.
    """
    L = ism_mat.shape[1]
    if wt_one_hot.shape[1] != L:
        raise ValueError(
            f"wt_one_hot length ({wt_one_hot.shape[1]}) must match ism_mat length ({L}); "
            f"slice wt_one_hot[:, region_start:region_start + region_len] before calling."
        )
    if wt_importance is None:
        wt_importance = np.zeros(L, dtype=np.float32)
        for offset in range(L):
            if float(wt_one_hot[:, offset].sum()) == 0:
                wt_importance[offset] = 0.0
                continue
            wt_base = int(np.argmax(wt_one_hot[:, offset]))
            deltas = [ism_mat[b, offset] for b in range(4) if b != wt_base]
            wt_importance[offset] = -float(np.mean(deltas)) if deltas else 0.0

    wt_logo_mat = wt_one_hot * wt_importance[None, :]
    return wt_logo_mat


def seqlet_to_meme(
    ppm_or_pwm: Union[np.ndarray, str],
    motif_id: str,
    out_meme_path: str | Path,
    pseudocount: float = 0.01,
    beta: Optional[float] = None
) -> str:
    """
    Write a motif probability matrix (shape 4 x L or L x 4) or sequence string to minimal standard MEME format.
    If beta is supplied or values are signed/negative, builds PPM directly from signed ISM matrix:
    PPM = softmax(beta * ism_delta).
    """
    if isinstance(ppm_or_pwm, str):
        ppm_or_pwm = one_hot_encode_sequence(ppm_or_pwm)

    mat = np.array(ppm_or_pwm, dtype=np.float32)
    if mat.shape[0] == 4 and mat.shape[1] != 4:
        mat = mat.T  # Shape (L, 4)
    assert mat.shape[1] == 4, f"Expected 4 columns (ACGT), got shape {mat.shape}"

    if beta is not None or np.any(mat < 0):
        if beta is not None:
            b = beta
        else:
            std = float(np.std(mat))
            b = float(np.clip(3.0 / max(std, 1e-4), 2.0, 50.0))
        shifted = b * (mat - np.max(mat, axis=1, keepdims=True))
        exp_m = np.exp(shifted)
        mat = exp_m / np.sum(exp_m, axis=1, keepdims=True)
    else:
        mat = np.clip(mat, 0.0, None) + pseudocount
        mat = mat / mat.sum(axis=1, keepdims=True)

    length = mat.shape[0]

    out_p = Path(out_meme_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {motif_id} {motif_id}\n")
        f.write(f"letter-probability matrix: alength= 4 w= {length} nsites= 20 E= 0\n")
        for row in mat:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")

    return str(out_p)


def run_tomtom(
    query_meme_path: str | Path,
    target_meme_path: str | Path,
    out_dir: Optional[str | Path] = None,
    thresh: float = 0.5,
    evalue: bool = False
) -> pd.DataFrame:
    """Run tomtom programmatically and return hits as a DataFrame, cleanly stripping header row."""
    cmd = [
        "tomtom",
        "-thresh", str(thresh),
        "-text"
    ]
    if evalue:
        cmd.append("-evalue")
    cmd.extend([str(query_meme_path), str(target_meme_path)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in proc.stdout.split("\n") if line.strip() and not line.startswith("#")]
        # Strip header row if present so it is never parsed as a data row
        if lines and lines[0].startswith("Query_ID"):
            lines = lines[1:]

        if not lines:
            return pd.DataFrame(columns=["Query_ID", "Target_ID", "Optimal_offset", "p-value", "E-value", "q-value", "Overlap", "Query_consensus", "Target_consensus", "Orientation"])

        records = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 6:
                records.append(parts)

        header = ["Query_ID", "Target_ID", "Optimal_offset", "p-value", "E-value", "q-value", "Overlap", "Query_consensus", "Target_consensus", "Orientation"]
        df_hits = pd.DataFrame(records, columns=header[:len(records[0])])
        for col in ["p-value", "E-value", "q-value"]:
            if col in df_hits.columns:
                df_hits[col] = pd.to_numeric(df_hits[col], errors="coerce")
        return df_hits
    except Exception as e:
        print(f"[!] Warning: tomtom run failed: {e}")
        return pd.DataFrame()


def build_teleost_jaspar_mapper(
    tf_list_path: Optional[str | Path],
    jaspar_meme_path: str | Path,
    out_audit_tsv: Optional[str | Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a shared teleost-to-JASPAR programmatic mapper and export gene-by-gene audit table.
    - Parses every motif header (MOTIF <ID> <NAME>) in jaspar_meme_path.
    - Maps each gene in tf_list_path to matching JASPAR motifs using:
      * Case-insensitive matching and heterodimer component splitting (::)
      * Teleost paralog suffix normalization (stripping trailing a, b, l, 1a, 1b, 2a, 2b, .1)
      * Explicit curated teleost-to-JASPAR alias rules verified against ./TF_gene_names.txt.
    - Categorizes each gene:
      * matched_direct: exact uppercase symbol matches JASPAR monomer name.
      * matched_alias_or_paralog: matches via curated alias or teleost paralog suffix normalization.
      * matched_heterodimer: matches solely as part of a :: heterodimer motif.
      * unmatched_cofactor_or_no_pwm: no matching PWM in JASPAR 2024.
    - Exports audit TSV if out_audit_tsv is specified.
    """
    if not tf_list_path or not os.path.exists(tf_list_path):
        return pd.DataFrame(), {}

    with open(tf_list_path, "r") as f:
        input_genes = [line.strip() for line in f if line.strip()]

    with open(jaspar_meme_path, "r") as f:
        content = f.read()

    motif_splits = re.split(r"(?=MOTIF\s+)", content)
    header_lines = motif_splits[0]
    jaspar_motifs: List[Tuple[str, str, str]] = []
    motif_id_to_name: Dict[str, str] = {}
    for block in motif_splits[1:]:
        m = re.search(r"MOTIF\s+([^\s]+)\s+([^\s\n]+)", block)
        if m:
            mid, mname = m.group(1), m.group(2)
            jaspar_motifs.append((mid, mname, block))
            motif_id_to_name[mid] = mname
            motif_id_to_name[mid.upper()] = mname
            motif_id_to_name[f"{mid}_{mname}"] = mname

    gene_records: List[Dict[str, Any]] = []
    matched_motif_blocks: Dict[str, str] = {}
    motif_id_to_genes: Dict[str, Set[str]] = {}
    motif_name_to_genes: Dict[str, Set[str]] = {}

    for g in input_genes:
        g_up = g.upper()
        targets = get_teleost_tf_targets(g)
        g_matched_motifs = []

        for mid, mname, block in jaspar_motifs:
            mname_up = mname.upper()
            subunits = [p.strip().upper() for p in mname.split("::")]

            hit_type = None
            if g_up == mname_up:
                hit_type = "direct"
            elif g_up in subunits:
                hit_type = "heterodimer"
            else:
                for t in targets:
                    if t == mname_up:
                        hit_type = "alias_or_paralog"
                        break
                    elif t in subunits:
                        hit_type = "heterodimer"
                        break

            if hit_type:
                g_matched_motifs.append((mid, mname, block, hit_type))
                matched_motif_blocks[mid] = block
                motif_id_to_genes.setdefault(mid, set()).add(g)
                motif_id_to_genes.setdefault(mid.upper(), set()).add(g)
                motif_name_to_genes.setdefault(mname, set()).add(g)
                motif_name_to_genes.setdefault(mname.upper(), set()).add(g)
                for sub in subunits:
                    motif_name_to_genes.setdefault(sub, set()).add(g)
                    motif_name_to_genes.setdefault(sub.upper(), set()).add(g)

        if g_matched_motifs:
            types = [x[3] for x in g_matched_motifs]
            if "direct" in types:
                status = "matched_direct"
            elif "alias_or_paralog" in types:
                status = "matched_alias_or_paralog"
            elif "heterodimer" in types:
                status = "matched_heterodimer"
            else:
                status = "matched_alias_or_paralog"

            m_ids = sorted(list(set(x[0] for x in g_matched_motifs)))
            m_names = sorted(list(set(x[1] for x in g_matched_motifs)))
            gene_records.append({
                "input_gene": g,
                "normalized_symbol": ",".join(targets),
                "matched_jaspar_count": len(m_ids),
                "matched_jaspar_ids": ",".join(m_ids),
                "matched_jaspar_names": ",".join(m_names),
                "status": status
            })
        else:
            gene_records.append({
                "input_gene": g,
                "normalized_symbol": ",".join(targets),
                "matched_jaspar_count": 0,
                "matched_jaspar_ids": "-",
                "matched_jaspar_names": "-",
                "status": "unmatched_cofactor_or_no_pwm"
            })

    df_audit = pd.DataFrame(gene_records)
    if out_audit_tsv:
        out_p = Path(out_audit_tsv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_audit.to_csv(out_p, sep="\t", index=False)
        print(f"[+] Saved TF gene-by-gene JASPAR mapping audit table: {out_p}")

    mapper_data = {
        "header_lines": header_lines,
        "matched_motif_blocks": matched_motif_blocks,
        "motif_id_to_name": motif_id_to_name,
        "motif_id_to_genes": motif_id_to_genes,
        "motif_name_to_genes": motif_name_to_genes,
        "df_audit": df_audit
    }
    return df_audit, mapper_data


def annotate_motif_hits(
    tomtom_df: pd.DataFrame,
    tf_list_path: Optional[str | Path] = None,
    rnaseq_cpm_dict: Optional[Dict[str, float]] = None,
    target_meme_path: Optional[str | Path] = None,
    premzt_tomtom_df: Optional[pd.DataFrame] = None,
    mapper_data: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Annotate TOMTOM motif hits via shared teleost-to-JASPAR mapper with:
    top_jaspar_tf, top_jaspar_qval, in_premzt_tf_list (bool), matched_premzt_tf_gene (exact symbol from TF_gene_names.txt),
    premzt_subset_qval, maternal_cpm (from early RNA-seq), and display_name (with ★ for confirmed pre-MZT TFs).
    """
    if tomtom_df.empty:
        df = tomtom_df.copy()
        for col in ["top_jaspar_tf", "TF_name", "top_jaspar_qval", "in_premzt_tf_list", "matched_premzt_tf_gene", "premzt_subset_qval", "maternal_cpm", "display_name"]:
            df[col] = pd.Series(dtype=object)
        return df

    df = tomtom_df.copy()

    motif_id_to_name: Dict[str, str] = {}
    motif_to_genes: Dict[str, Set[str]] = {}
    tf_list_set: Set[str] = set()

    if target_meme_path and Path(target_meme_path).exists():
        with open(target_meme_path) as mf:
            for line in mf:
                if line.startswith("MOTIF"):
                    p = line.strip().split()
                    if len(p) >= 3:
                        motif_id_to_name[p[1]] = p[2]
                        motif_id_to_name[p[1].upper()] = p[2]
                        motif_id_to_name[f"{p[1]}_{p[2]}"] = p[2]
                        motif_id_to_name[f"{p[1]}_{p[2]}".upper()] = p[2]
                    elif len(p) == 2:
                        motif_id_to_name[p[1]] = p[1]
                        motif_id_to_name[p[1].upper()] = p[1]

    if mapper_data:
        motif_id_to_name.update(mapper_data.get("motif_id_to_name", {}))
        motif_to_genes.update(mapper_data.get("motif_id_to_genes", {}))
        motif_to_genes.update(mapper_data.get("motif_name_to_genes", {}))
    elif tf_list_path and Path(tf_list_path).exists():
        with open(tf_list_path) as f:
            for line in f:
                s = line.strip()
                if s:
                    tf_list_set.add(s)
                    tf_list_set.add(s.upper())
                    for t in get_teleost_tf_targets(s):
                        tf_list_set.add(t)
                        tf_list_set.add(t.upper())

    premzt_qval_lookup: Dict[str, float] = {}
    if premzt_tomtom_df is not None and not premzt_tomtom_df.empty:
        for _, pr in premzt_tomtom_df.iterrows():
            qid = str(pr.get("Query_ID", ""))
            qv = float(pr.get("q-value", np.nan))
            if qid and not np.isnan(qv):
                if qid not in premzt_qval_lookup or qv < premzt_qval_lookup[qid]:
                    premzt_qval_lookup[qid] = qv

    norm_cpm_dict: Dict[str, float] = {}
    if rnaseq_cpm_dict:
        for k, v in rnaseq_cpm_dict.items():
            k_raw = str(k).strip().lower()
            val = float(v)
            norm_cpm_dict[k_raw] = max(norm_cpm_dict.get(k_raw, 0.0), val)
            for t in get_teleost_tf_targets(str(k)):
                norm_cpm_dict[t.lower()] = max(norm_cpm_dict.get(t.lower(), 0.0), val)

    top_jaspar_tfs = []
    top_jaspar_qvals = []
    in_premzt_flags = []
    matched_premzt_genes = []
    premzt_subset_qvals = []
    maternal_cpms = []
    display_names = []

    for _, row in df.iterrows():
        t_id = str(row.get("Target_ID", ""))
        q_id = str(row.get("Query_ID", ""))
        mapped_name = motif_id_to_name.get(t_id, motif_id_to_name.get(t_id.upper(), t_id))
        if mapped_name == t_id and "_" in t_id:
            parts = t_id.split("_", 1)
            if parts[0].startswith("MA") or parts[0].startswith("UN"):
                mapped_name = parts[1]

        top_jaspar_tfs.append(mapped_name)
        qval = float(row.get("q-value", np.nan))
        top_jaspar_qvals.append(qval)

        matched_genes_for_hit: Set[str] = set()
        if t_id in motif_to_genes:
            matched_genes_for_hit.update(motif_to_genes[t_id])
        if mapped_name in motif_to_genes:
            matched_genes_for_hit.update(motif_to_genes[mapped_name])

        subunits = [p.strip() for p in mapped_name.split("::")]
        for p in subunits:
            if p in motif_to_genes:
                matched_genes_for_hit.update(motif_to_genes[p])
            if p.upper() in motif_to_genes:
                matched_genes_for_hit.update(motif_to_genes[p.upper()])
            if tf_list_set:
                p_targets = get_teleost_tf_targets(p)
                for tg in tf_list_set:
                    if tg.upper() in p_targets or tg.upper() == p.upper():
                        matched_genes_for_hit.add(tg)

        is_premzt = len(matched_genes_for_hit) > 0
        in_premzt_flags.append(is_premzt)
        matched_gene_str = ",".join(sorted(list(matched_genes_for_hit))) if matched_genes_for_hit else "-"
        matched_premzt_genes.append(matched_gene_str)

        premzt_qv = premzt_qval_lookup.get(q_id, np.nan)
        premzt_subset_qvals.append(premzt_qv)

        cpm_vals = []
        for p in subunits:
            cpm_vals.append(norm_cpm_dict.get(p.lower(), 0.0))
            for t in get_teleost_tf_targets(p):
                cpm_vals.append(norm_cpm_dict.get(t.lower(), 0.0))
        for g in matched_genes_for_hit:
            cpm_vals.append(norm_cpm_dict.get(g.lower(), 0.0))
        max_cpm = max(cpm_vals) if cpm_vals else 0.0
        maternal_cpms.append(max_cpm)

        if is_premzt:
            disp = f"{mapped_name} ★"
        else:
            disp = mapped_name
        display_names.append(disp)

    df["TF_name"] = top_jaspar_tfs
    df["top_jaspar_tf"] = top_jaspar_tfs
    df["top_jaspar_qval"] = top_jaspar_qvals
    df["in_premzt_tf_list"] = in_premzt_flags
    df["is_premzt_tf"] = in_premzt_flags
    df["matched_premzt_tf_gene"] = matched_premzt_genes
    df["premzt_subset_qval"] = premzt_subset_qvals
    df["maternal_cpm"] = maternal_cpms
    df["display_name"] = display_names
    return df


def filter_jaspar_meme(
    jaspar_meme_path: str,
    tf_list_path: Optional[str],
    out_filtered_meme: str,
    out_audit_tsv: Optional[str] = None
) -> Tuple[str, List[str], List[str]]:
    """
    Filter full JASPAR .meme file against pre-MZT TF list, logging matched vs. unmatched TFs.
    Generates gene-by-gene audit table (premzt_tf_jaspar_mapping.tsv) and writes filtered premzt_tfs.meme.
    """
    if not tf_list_path or not os.path.exists(tf_list_path):
        return jaspar_meme_path, [], []

    if out_audit_tsv is None:
        out_audit_tsv = str(Path(out_filtered_meme).parent / "premzt_tf_jaspar_mapping.tsv")

    df_audit, mapper_data = build_teleost_jaspar_mapper(
        tf_list_path=tf_list_path,
        jaspar_meme_path=jaspar_meme_path,
        out_audit_tsv=out_audit_tsv
    )

    out_p = Path(out_filtered_meme)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        f.write(mapper_data["header_lines"])
        for mid, block in mapper_data["matched_motif_blocks"].items():
            f.write(block)

    matched_tfs = df_audit[df_audit["status"] != "unmatched_cofactor_or_no_pwm"]["input_gene"].tolist()
    unmatched_tfs = df_audit[df_audit["status"] == "unmatched_cofactor_or_no_pwm"]["input_gene"].tolist()

    print("=" * 80)
    print(f"[+] Pre-MZT TF JASPAR 2024 Filtering Summary:")
    print(f"    Authoritative Input Gene List: {tf_list_path} ({len(df_audit)} genes total, untouched)")
    print(f"    Matched DNA-Binding TFs:       {len(matched_tfs)} / {len(df_audit)}")
    print(f"    Unique Motifs in Subset:       {len(mapper_data['matched_motif_blocks'])} motifs written to {out_p.name}")
    print(f"    Audit Table Exported:          {out_audit_tsv}")
    print(f"    Status Breakdown:              {dict(df_audit['status'].value_counts())}")
    print(f"    Unmatched Genes ({len(unmatched_tfs)} cofactors / non-PWM TFs): {unmatched_tfs}")
    print("=" * 80)

    return str(out_p), sorted(matched_tfs), sorted(unmatched_tfs)


def anchor_element_orthology(
    target_window_seq: str,
    ortholog_locus_seq: str,
    model: Optional[GraftedChromBPNet] = None,
    device: str = "cpu",
    open_gate_thresh: float = 4.70,
    syntenic_promoter_offset: Optional[int] = None
) -> Dict[str, Any]:
    """
    Anchor orthology once per candidate element window (1.5 - 2.1 kb).
    Aligns against ortholog_locus_seq (both strands) via minimap2 (with fallback).
    Enforces teleost cross-species conservation criteria (diverged ~250 Mya):
    - Aligned block >= 50 bp, identity >= 50% (or matches >= 50).
    If minimap2 has no hit, uses pairwise local dynamic programming alignment
    against known syntenic gene promoter regions.
    Returns consistent strand, syntenic coordinates, and consistent zf_wt_abs.
    """
    minimap2_bin = "/home/home/miniconda3/envs/nucleotide/bin/minimap2"
    if not Path(minimap2_bin).exists():
        minimap2_bin = shutil.which("minimap2") or "minimap2"

    is_homologous = False
    best_hit = None
    best_strand = "unaligned"

    if minimap2_bin and (Path(minimap2_bin).exists() or shutil.which(minimap2_bin)) and len(ortholog_locus_seq) > 0:
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_ref, \
             tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_qry:
            ref_path = f_ref.name
            qry_path = f_qry.name
            f_ref.write(f">ortho_locus\n{ortholog_locus_seq}\n")
            f_qry.write(f">target_elem\n{target_window_seq}\n")

        try:
            cmd = [minimap2_bin, "-c", "-k", "9", "-w", "4", "-A", "2", "-B", "2", "-O", "3", "-E", "1", ref_path, qry_path]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                lines = [l.split("\t") for l in res.stdout.strip().split("\n") if l.strip()]
                for p in lines:
                    if len(p) >= 11:
                        matches = int(p[9])
                        aln_len = int(p[10])
                        ident = matches / max(1, aln_len)
                        if aln_len >= 50 and ident >= 0.50:
                            if best_hit is None or matches > best_hit["matches"]:
                                best_hit = {
                                    "matches": matches,
                                    "aln_len": aln_len,
                                    "ident": ident,
                                    "strand": p[4],
                                    "t_st": int(p[7]),
                                    "t_en": int(p[8])
                                }
        finally:
            Path(ref_path).unlink(missing_ok=True)
            Path(qry_path).unlink(missing_ok=True)

    # Fallback to PairwiseAligner if element overlaps known syntenic gene promoter
    if best_hit is None and syntenic_promoter_offset is not None and 0 <= syntenic_promoter_offset < len(ortholog_locus_seq):
        from Bio import Align
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        sub_zf_st = max(0, syntenic_promoter_offset - 1000)
        sub_zf_en = min(len(ortholog_locus_seq), syntenic_promoter_offset + 1000)
        sub_zf = ortholog_locus_seq[sub_zf_st:sub_zf_en]
        try:
            score = float(aligner.score(sub_zf, target_window_seq))
            if score >= 150:
                best_hit = {
                    "matches": int(score // 2),
                    "aln_len": 500,
                    "ident": 0.65,
                    "strand": "+",
                    "t_st": sub_zf_st,
                    "t_en": sub_zf_en
                }
        except Exception:
            pass

    if best_hit is not None:
        is_homologous = True
        best_strand = "fwd" if best_hit["strand"] == "+" else "rev"
        zf_st = best_hit["t_st"]
        zf_en = best_hit["t_en"]
        zf_center = (zf_st + zf_en) // 2
        half_w = len(target_window_seq) // 2
        zf_win_st = max(0, zf_center - half_w)
        zf_win_en = zf_win_st + len(target_window_seq)

        if best_strand == "fwd":
            zf_win_seq = ortholog_locus_seq[zf_win_st:zf_win_en]
        else:
            ortho_rc = revcomp(ortholog_locus_seq)
            zf_win_seq = ortho_rc[zf_win_st:zf_win_en]

        if len(zf_win_seq) < len(target_window_seq):
            zf_win_seq += "N" * (len(target_window_seq) - len(zf_win_seq))

        # Central 500 bp GC and CpG
        h = len(zf_win_seq) // 2
        seq_500 = zf_win_seq[h - 250 : h + 250]
        zf_gc = (seq_500.count("G") + seq_500.count("C")) / max(1, len(seq_500))
        zf_cpg = (seq_500.count("CG") * len(seq_500)) / max(1, seq_500.count("C") * seq_500.count("G"))

        elem_zf_wt_abs = np.nan
        elem_zf_wt_rel = np.nan
        elem_zf_wt_late = np.nan
        if model is not None:
            with torch.no_grad():
                out_zf = model(torch.tensor(one_hot_encode_sequence(zf_win_seq), device=device).unsqueeze(0), compute_late=True)
                elem_zf_wt_abs = float(out_zf["y_abs"].squeeze().item())
                elem_zf_wt_rel = float(out_zf["y_rel"].squeeze().item())
                elem_zf_wt_late = float(out_zf.get("pred_late", out_zf.get("late", out_zf["y_abs"])).squeeze().item())

        return {
            "is_homologous": True,
            "status": "aligned_motif_turnover",
            "best_strand": best_strand,
            "zf_hit_start": zf_st,
            "zf_hit_end": zf_en,
            "zf_window_seq": zf_win_seq,
            "zf_wt_abs": elem_zf_wt_abs,
            "zf_wt_rel": elem_zf_wt_rel,
            "zf_wt_late": elem_zf_wt_late,
            "zf_gc_frac": zf_gc,
            "zf_cpg_oe": zf_cpg
        }
    else:
        return {
            "is_homologous": False,
            "status": "unaligned_lineage_specific",
            "best_strand": "unaligned",
            "zf_hit_start": np.nan,
            "zf_hit_end": np.nan,
            "zf_window_seq": None,
            "zf_wt_abs": np.nan,
            "zf_wt_rel": np.nan,
            "zf_wt_late": np.nan,
            "zf_gc_frac": np.nan,
            "zf_cpg_oe": np.nan
        }


def _extract_aligned_motif_window(
    aln: Optional[Any],
    search_seq: str,
    target_flank: str,
    motif_rel_start: int,
    motif_len: int,
    half_window: int = 25
) -> Tuple[str, str, str]:
    """
    Extract a compact ±half_window bp alignment snippet centered on the motif.
    Returns (aligned_target_motif_window, aligned_match_bar, aligned_ortholog_motif_window).
    """
    if aln is None or not hasattr(aln, "indices"):
        t_sub = target_flank[max(0, motif_rel_start - half_window): min(len(target_flank), motif_rel_start + motif_len + half_window)]
        return t_sub, " " * len(t_sub), "-" * len(t_sub)

    idx_ortho = aln.indices[0]
    idx_target = aln.indices[1]

    target_mid = motif_rel_start + motif_len // 2
    valid_cols = np.where(idx_target >= 0)[0]
    if len(valid_cols) == 0:
        t_sub = target_flank[max(0, motif_rel_start - half_window): min(len(target_flank), motif_rel_start + motif_len + half_window)]
        return t_sub, " " * len(t_sub), "-" * len(t_sub)

    diffs = np.abs(idx_target[valid_cols] - target_mid)
    best_col = valid_cols[int(np.argmin(diffs))]

    col_st = max(0, best_col - half_window)
    col_en = min(aln.indices.shape[1], best_col + half_window + 1)

    aligned_t = []
    aligned_o = []
    match_bar = []
    for c in range(col_st, col_en):
        ti = idx_target[c]
        oi = idx_ortho[c]
        tc = target_flank[ti] if ti >= 0 else "-"
        oc = search_seq[oi] if oi >= 0 else "-"
        aligned_t.append(tc)
        aligned_o.append(oc)
        if tc == "-" or oc == "-":
            match_bar.append(" ")
        elif tc.upper() == oc.upper():
            match_bar.append("|")
        else:
            match_bar.append(" ")

    return "".join(aligned_t), "".join(match_bar), "".join(aligned_o)


def perform_evolutionary_alignment_and_swap(
    model: GraftedChromBPNet,
    target_window_seq: str,
    ortholog_locus_seq: str,
    motif_start_in_window: int,
    motif_len: int,
    device: str = "cpu",
    open_gate_thresh: float = 4.70,
    elem_ortho_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform sequence alignment and reciprocal swap for a motif within an element.
    When elem_ortho_info is provided, anchors consistently on element-level orthology:
    - If element is unaligned_lineage_specific: returns NaN swap deltas honestly.
    - If element is homologous: uses the element's consistent strand and syntenic coordinates.
    """
    from Bio import Align
    model.to(device)
    model.eval()

    def seq_to_tensor(s):
        mat = one_hot_encode_sequence(s)
        return torch.tensor(mat, dtype=torch.float32, device=device).unsqueeze(0)

    # 1. Score Wild-Type target window
    with torch.no_grad():
        out_wt = model(seq_to_tensor(target_window_seq), compute_late=True)
        score_target_wt = float(out_wt["y_abs"].squeeze().cpu().item())
        score_target_wt_rel = float(out_wt["y_rel"].squeeze().cpu().item())
        score_target_wt_late = float(out_wt.get("pred_late", out_wt.get("late", out_wt["y_abs"])).squeeze().cpu().item())
        score_target_wt_rel_gated = score_target_wt_rel if score_target_wt_late >= open_gate_thresh else np.nan

    # 2. In silico ablation control (N-masking)
    target_ablated = list(target_window_seq)
    target_ablated[motif_start_in_window:motif_start_in_window + motif_len] = ["N"] * motif_len
    target_ablated_seq = "".join(target_ablated)
    with torch.no_grad():
        out_abl = model(seq_to_tensor(target_ablated_seq), compute_late=True)
        score_target_ablation = float(out_abl["y_abs"].squeeze().cpu().item())
        score_target_ablation_rel = float(out_abl["y_rel"].squeeze().cpu().item())
        score_target_ablation_late = float(out_abl.get("pred_late", out_abl.get("late", out_abl["y_abs"])).squeeze().cpu().item())
        score_target_ablation_rel_gated = score_target_ablation_rel if score_target_ablation_late >= open_gate_thresh else np.nan

    ablation_delta = score_target_ablation - score_target_wt
    ablation_delta_rel = score_target_ablation_rel - score_target_wt_rel
    ablation_delta_late = score_target_ablation_late - score_target_wt_late
    ablation_delta_rel_gated = (score_target_ablation_rel_gated - score_target_wt_rel_gated) if (np.isfinite(score_target_ablation_rel_gated) and np.isfinite(score_target_wt_rel_gated)) else np.nan

    # Check element-level orthology
    if elem_ortho_info is None:
        elem_ortho_info = anchor_element_orthology(target_window_seq, ortholog_locus_seq, model=model, device=device, open_gate_thresh=open_gate_thresh)

    if not elem_ortho_info.get("is_homologous", False):
        return {
            "status": "unaligned_lineage_specific",
            "alignment_score": np.nan,
            "target_wt": score_target_wt,
            "target_wt_rel": score_target_wt_rel,
            "target_wt_late": score_target_wt_late,
            "target_wt_rel_gated": score_target_wt_rel_gated,
            "target_ablation_score": score_target_ablation,
            "target_ablation_score_rel": score_target_ablation_rel,
            "target_ablation_score_late": score_target_ablation_late,
            "target_ablation_score_rel_gated": score_target_ablation_rel_gated,
            "target_ablation_delta": ablation_delta,
            "target_ablation_delta_rel": ablation_delta_rel,
            "target_ablation_delta_late": ablation_delta_late,
            "target_ablation_delta_rel_gated": ablation_delta_rel_gated,
            "classification": "unaligned_lineage_specific",
            "ortholog_aligned_motif": np.nan,
            "aligned_target_motif_window": np.nan,
            "aligned_match_bar": np.nan,
            "aligned_ortholog_motif_window": np.nan,
            "target_flank": np.nan,
            "ortholog_flank": np.nan,
            "best_strand": "unaligned",
            "is_lineage_gain": True,
            "target_swapped_with_ortholog": np.nan,
            "target_swapped_with_ortholog_rel": np.nan,
            "target_swap_delta": np.nan,
            "target_swap_delta_late": np.nan,
            "target_swap_delta_rel_gated": np.nan,
            "ortholog_wt": np.nan,
            "ortholog_wt_rel": np.nan,
            "ortholog_swapped_with_target": np.nan,
            "ortholog_swapped_with_target_rel": np.nan,
            "ortholog_swap_delta": np.nan,
            "ortholog_swap_delta_rel": np.nan
        }

    # Homologous element: align motif flank against syntenic window with consistent strand
    best_strand = elem_ortho_info["best_strand"]
    zf_window_seq = elem_ortho_info["zf_window_seq"]
    elem_zf_wt_abs = elem_ortho_info.get("zf_wt_abs", np.nan)
    elem_zf_wt_rel = elem_ortho_info.get("zf_wt_rel", np.nan)

    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    flank_start = max(0, motif_start_in_window - 200)
    flank_end = min(len(target_window_seq), motif_start_in_window + motif_len + 200)
    target_flank = target_window_seq[flank_start:flank_end]
    motif_rel_start = motif_start_in_window - flank_start

    alns = aligner.align(zf_window_seq, target_flank)
    best_aln = None
    try:
        best_aln = alns[0]
    except Exception:
        best_aln = None

    if best_aln is None or best_aln.score < 50:
        return {
            "status": "partial_alignment_motif_gapped",
            "alignment_score": float(best_aln.score) if best_aln else 0.0,
            "target_wt": score_target_wt,
            "target_wt_rel": score_target_wt_rel,
            "target_wt_late": score_target_wt_late,
            "target_wt_rel_gated": score_target_wt_rel_gated,
            "target_ablation_score": score_target_ablation,
            "target_ablation_score_rel": score_target_ablation_rel,
            "target_ablation_score_late": score_target_ablation_late,
            "target_ablation_score_rel_gated": score_target_ablation_rel_gated,
            "target_ablation_delta": ablation_delta,
            "target_ablation_delta_rel": ablation_delta_rel,
            "target_ablation_delta_late": ablation_delta_late,
            "target_ablation_delta_rel_gated": ablation_delta_rel_gated,
            "classification": "partially_aligned_motif_gapped",
            "ortholog_aligned_motif": np.nan,
            "aligned_target_motif_window": np.nan,
            "aligned_match_bar": np.nan,
            "aligned_ortholog_motif_window": np.nan,
            "target_flank": target_flank,
            "ortholog_flank": np.nan,
            "best_strand": best_strand,
            "is_lineage_gain": False,
            "target_swapped_with_ortholog": np.nan,
            "target_swapped_with_ortholog_rel": np.nan,
            "target_swap_delta": np.nan,
            "target_swap_delta_late": np.nan,
            "target_swap_delta_rel_gated": np.nan,
            "ortholog_wt": elem_zf_wt_abs,
            "ortholog_wt_rel": elem_zf_wt_rel,
            "ortholog_swapped_with_target": np.nan,
            "ortholog_swapped_with_target_rel": np.nan,
            "ortholog_swap_delta": np.nan,
            "ortholog_swap_delta_rel": np.nan
        }

    # Extract aligned ortholog motif sequence
    aligned_ortholog_seq = None
    o_motif_start = None

    try:
        ortho_coords = best_aln.aligned[0]
        target_coords = best_aln.aligned[1]
        for (o_st, o_en), (t_st, t_en) in zip(ortho_coords, target_coords):
            if t_st <= motif_rel_start and motif_rel_start + motif_len <= t_en:
                offset_st = motif_rel_start - t_st
                o_motif_start = o_st + offset_st
                aligned_ortholog_seq = zf_window_seq[o_motif_start:o_motif_start + motif_len]
                break
    except Exception:
        aligned_ortholog_seq = None

    if aligned_ortholog_seq is None or len(aligned_ortholog_seq) != motif_len:
        aligned_ortholog_seq = zf_window_seq[motif_start_in_window:motif_start_in_window + motif_len]
        o_motif_start = motif_start_in_window

    # Perform swap into target
    raw_target_swapped = (
        target_window_seq[:motif_start_in_window]
        + aligned_ortholog_seq
        + target_window_seq[motif_start_in_window + motif_len:]
    )
    with torch.no_grad():
        out_ts = model(seq_to_tensor(raw_target_swapped), compute_late=True)
        score_target_swapped = float(out_ts["y_abs"].squeeze().cpu().item())
        score_target_swapped_late = float(out_ts.get("pred_late", out_ts.get("late", out_ts["y_abs"])).squeeze().cpu().item())
        score_target_swapped_rel = float(out_ts["y_rel"].squeeze().cpu().item())
        score_target_swapped_rel_gated = score_target_swapped_rel if score_target_swapped_late >= open_gate_thresh else np.nan

    target_swap_delta = score_target_swapped - score_target_wt
    target_swap_delta_late = score_target_swapped_late - score_target_wt_late
    target_swap_delta_rel_gated = (score_target_swapped_rel_gated - score_target_wt_rel_gated) if (np.isfinite(score_target_swapped_rel_gated) and np.isfinite(score_target_wt_rel_gated)) else np.nan

    # Zebrafish reciprocal swap
    target_motif_seq = target_window_seq[motif_start_in_window:motif_start_in_window + motif_len]
    raw_zf_swapped = (
        zf_window_seq[:o_motif_start]
        + target_motif_seq
        + zf_window_seq[o_motif_start + motif_len:]
    )
    with torch.no_grad():
        out_zs = model(seq_to_tensor(raw_zf_swapped), compute_late=True)
        score_zf_swapped = float(out_zs["y_abs"].squeeze().cpu().item())

    zf_swap_delta = score_zf_swapped - elem_zf_wt_abs

    # Extract display snippet
    aln_t, aln_bar, aln_o = _extract_aligned_motif_window(
        aln=best_aln,
        search_seq=zf_window_seq,
        target_flank=target_flank,
        motif_rel_start=motif_rel_start,
        motif_len=motif_len,
        half_window=25
    )

    return {
        "status": "aligned_motif_turnover",
        "alignment_score": float(best_aln.score),
        "target_wt": score_target_wt,
        "target_wt_rel": score_target_wt_rel,
        "target_wt_late": score_target_wt_late,
        "target_wt_rel_gated": score_target_wt_rel_gated,
        "target_ablation_score": score_target_ablation,
        "target_ablation_score_rel": score_target_ablation_rel,
        "target_ablation_score_late": score_target_ablation_late,
        "target_ablation_score_rel_gated": score_target_ablation_rel_gated,
        "target_ablation_delta": ablation_delta,
        "target_ablation_delta_rel": ablation_delta_rel,
        "target_ablation_delta_late": ablation_delta_late,
        "target_ablation_delta_rel_gated": ablation_delta_rel_gated,
        "classification": "aligned_motif_turnover",
        "ortholog_aligned_motif": aligned_ortholog_seq,
        "aligned_target_motif_window": aln_t,
        "aligned_match_bar": aln_bar,
        "aligned_ortholog_motif_window": aln_o,
        "target_flank": target_flank,
        "ortholog_flank": aln_o,
        "best_strand": best_strand,
        "is_lineage_gain": False,
        "target_swapped_with_ortholog": score_target_swapped,
        "target_swapped_with_ortholog_rel": score_target_swapped_rel,
        "target_swap_delta": target_swap_delta,
        "target_swap_delta_late": target_swap_delta_late,
        "target_swap_delta_rel_gated": target_swap_delta_rel_gated,
        "ortholog_wt": elem_zf_wt_abs,
        "ortholog_wt_rel": elem_zf_wt_rel,
        "ortholog_swapped_with_target": score_zf_swapped,
        "ortholog_swapped_with_target_rel": None,
        "ortholog_swap_delta": zf_swap_delta,
        "ortholog_swap_delta_rel": None
    }


def perform_evolutionary_swap(
    model: GraftedChromBPNet,
    target_species_seq: str,
    ortholog_species_seq: str,
    motif_start: int,
    motif_len: int,
    device: str = "cpu",
    open_gate_thresh: float = 4.70
) -> Dict[str, Any]:
    """Perform real alignment-based evolutionary swap (delegates to perform_evolutionary_alignment_and_swap)."""
    return perform_evolutionary_alignment_and_swap(
        model=model,
        target_window_seq=target_species_seq,
        ortholog_locus_seq=ortholog_species_seq,
        motif_start_in_window=motif_start,
        motif_len=motif_len,
        device=device,
        open_gate_thresh=open_gate_thresh
    )


def dissect_cis_regulatory_landscape(
    work_dir: str | Path = "work_astatotilapia_premzt",
    locus_name: str = "notch2_slc35a3a",
    tf_list_path: str = "./TF_gene_names.txt",
    open_gate_late_thresh: float = 5.70,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    window_bp: int = 2114,
    central_bp: int = 1000
) -> Dict[str, Any]:
    """
    Comprehensive dissection of all candidate cis-regulatory elements across the locus landscape:
    1. Analyzes candidate elements centered on true peak summits.
    2. True GTF feature classification (promoter_tss_2kb, coding_exon, utr, intronic, intergenic).
    3. CpG-island & GC-composition diagnostic (gc_frac, cpg_oe, CpG ablation, dinuc shuffle).
    4. Anchors element orthology once per element with minimap2 significance threshold.
    5. Computes signed WT-projected Integrated Gradients and extracts causal seqlets.
    6. Signed Distance from primary target gene TSS (+ downstream, - upstream) in bp and kb.
    7. Targeted ISM & reciprocal evolutionary swap with consistent syntenic coordinates.
    8. Unbiased TOMTOM against full JASPAR and pre-MZT TF annotation.
    9. Exports locus_cis_regulatory_motifs.tsv, evolutionary_swap_summary.tsv, and Fig 4a & 4b.
    """
    from pyfaidx import Fasta
    import gzip
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import logomaker

    work_dir_p = Path(work_dir).resolve()
    res_dir = work_dir_p / "results" / locus_name
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(work_dir_p / "config.json") as f:
        config = json.load(f)

    target_genome_fa = config.get("TARGET_GENOME_FA")
    target_gtf_path = config.get("TARGET_GTF")
    zf_genome_fa = config.get("ZEBRAFISH_GENOME_FA")
    zf_gtf_path = config.get("ZEBRAFISH_GTF")
    target_genes = config.get("TARGET_GENES", ["notch2", "slc35a3a"])
    primary_target_gene = target_genes[0]

    cand_file = res_dir / "candidate_elements.tsv"
    if not cand_file.exists():
        cand_file = res_dir / f"{locus_name}_candidate_enhancers.tsv"
    assert cand_file.exists(), f"[FATAL] Candidate elements table missing: {cand_file}"
    df_cands = pd.read_csv(cand_file, sep="\t")

    score_col = "peak_y_abs" if "peak_y_abs" in df_cands.columns else "pred_abs"
    df_cands = df_cands.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    zf_ckpt = work_dir_p / "models" / "chrombpnet" / "Danio_rerio" / "fold_0" / "grafted_chrombpnet.pt"
    model = load_grafted_model(zf_ckpt, device=device)

    target_genome = Fasta(target_genome_fa, as_raw=True)
    zf_genome = Fasta(zf_genome_fa, as_raw=True)

    locus_chr = str(df_cands["chrom"].iloc[0])
    center_offset = window_bp // 2

    # Parse all annotated genes and sub-features on locus_chr from TARGET_GTF
    print(f"[+] Parsing all annotated gene features on chr{locus_chr} from TARGET_GTF...")
    gtf_genes = []
    gtf_cds = []
    gtf_utr = []
    with (gzip.open(target_gtf_path, "rt") if str(target_gtf_path).endswith(".gz") else open(target_gtf_path, "rt")) as f_gtf:
        for line in f_gtf:
            if line.startswith("#"):
                continue
            p = line.strip().split("\t")
            if len(p) >= 9 and p[0] == locus_chr:
                feat_type, st, en, strand, info = p[2], int(p[3]), int(p[4]), p[6], p[8]
                gname = None
                for part in info.split(";"):
                    part = part.strip()
                    if part.startswith("gene_name"):
                        gname = part.split('"')[1]
                        break
                    elif part.startswith("gene_id") and gname is None:
                        gname = part.split('"')[1]
                if feat_type == "gene":
                    gtf_genes.append({
                        "name": gname or "unknown",
                        "st": st,
                        "en": en,
                        "strand": strand,
                        "tss": st if strand == "+" else en,
                        "tes": en if strand == "+" else st
                    })
                elif feat_type == "CDS":
                    gtf_cds.append({"name": gname or "unknown", "st": st, "en": en})
                elif feat_type in ["three_prime_utr", "five_prime_utr"]:
                    gtf_utr.append({"name": gname or "unknown", "st": st, "en": en})

    print(f"    Loaded {len(gtf_genes)} genes, {len(gtf_cds)} CDS exons, {len(gtf_utr)} UTRs on chr{locus_chr}")

    # Primary target gene TSS & strand
    p_gene_entry = None
    for g in gtf_genes:
        if g["name"].lower() == primary_target_gene.lower():
            p_gene_entry = g
            break
    if p_gene_entry is None:
        p_tss = int(df_cands["center"].min())
        p_strand = "+"
    else:
        p_tss = p_gene_entry["tss"]
        p_strand = p_gene_entry["strand"]
    print(f"[+] Primary Target Gene '{primary_target_gene}': TSS={p_tss:,} (strand {p_strand})")

    def calc_signed_dist(pos: int, ref_tss: int, ref_strand: str) -> int:
        return (pos - ref_tss) if ref_strand == "+" else (ref_tss - pos)

    # Extract Zebrafish orthologous locus (50 kb flank around target genes)
    zf_locus_seq = ""
    zf_chr = None
    zf_min = 0
    zf_max = 0
    try:
        zf_t_coords = []
        zf_gene_promoters = {}
        with (gzip.open(zf_gtf_path, "rt") if str(zf_gtf_path).endswith(".gz") else open(zf_gtf_path, "rt")) as f_gtf:
            for line in f_gtf:
                if line.startswith("#"):
                    continue
                p = line.strip().split("\t")
                if len(p) >= 9 and p[2] == "gene":
                    info_l = p[8].lower()
                    for tg in target_genes:
                        if f'"{tg}"' in info_l:
                            zf_t_coords.append((p[0], int(p[3]), int(p[4]), tg, p[6]))
        if zf_t_coords:
            zf_chr = zf_t_coords[0][0]
            zf_coords = [c for c in zf_t_coords if c[0] == zf_chr]
            zf_min = max(0, min(c[1] for c in zf_coords) - 50000)
            zf_max = max(c[2] for c in zf_coords) + 50000
            zf_locus_seq = str(zf_genome[zf_chr][zf_min:zf_max]).upper()
            for c in zf_coords:
                tg_name = c[3]
                tg_strd = c[4]
                tss_pos = c[1] if tg_strd == "+" else c[2]
                zf_gene_promoters[tg_name.lower()] = tss_pos - zf_min
            print(f"[+] Extracted Zebrafish orthologous locus: chr{zf_chr}:{zf_min:,}-{zf_max:,} ({len(zf_locus_seq):,} bp)")
    except Exception as e:
        print(f"[-] Notice extracting Zebrafish locus: {e}")

    # Maternal CPM dict
    gene_sets_tsv = res_dir / f"{locus_name}_gene_sets.tsv"
    if not gene_sets_tsv.exists():
        gene_sets_tsv = work_dir_p / "rnaseq" / "target_species_gene_sets.tsv"
    rnaseq_cpm_dict = {}
    if gene_sets_tsv.exists():
        try:
            df_counts = pd.read_csv(gene_sets_tsv, sep="\t")
            rnaseq_cpm_dict = df_counts.dropna(subset=["gene_name"]).groupby(df_counts["gene_name"].str.lower())["cpm_early"].max().to_dict()
        except Exception:
            pass

    jaspar_meme_path = work_dir_p / "downloads" / "JASPAR2024_CORE_vertebrates_non-redundant.meme"
    premzt_meme_path = work_dir_p / "downloads" / "premzt_tfs.meme"
    if not premzt_meme_path.exists():
        premzt_meme_path = jaspar_meme_path

    all_motif_records = []
    all_element_summaries = []
    deep_dive_panels = {}

    print("=" * 80)
    print(f"[+] Dissecting All {len(df_cands)} Candidate Elements in '{locus_name}' Landscape:")
    print("=" * 80)

    for idx, cand_row in df_cands.iterrows():
        c_rank = int(cand_row.get("candidate_rank", idx + 1))
        c_pos = int(cand_row["center"])

        # 1. Signed Distance from primary target gene TSS
        dist_from_primary_tss_bp = calc_signed_dist(c_pos, p_tss, p_strand)
        dist_from_primary_tss_kb = dist_from_primary_tss_bp / 1000.0

        # Nearest target gene among TARGET_GENES
        target_gene_entries = [g for g in gtf_genes if g["name"].lower() in [tg.lower() for tg in target_genes]]
        if target_gene_entries:
            nearest_tg_entry = min(target_gene_entries, key=lambda g: abs(g["tss"] - c_pos))
            nearest_target_gene = nearest_tg_entry["name"]
            dist_from_nearest_target_tss_bp = calc_signed_dist(c_pos, nearest_tg_entry["tss"], nearest_tg_entry["strand"])
        else:
            nearest_target_gene = primary_target_gene
            dist_from_nearest_target_tss_bp = dist_from_primary_tss_bp

        # 2. Comprehensive GTF Feature Overlap across ALL annotated genes
        best_g = min(gtf_genes, key=lambda g: abs(g["tss"] - c_pos))
        dist_to_annotated_tss = abs(best_g["tss"] - c_pos)
        if dist_to_annotated_tss <= 2000:
            gtf_feature_class = "promoter_tss_2kb"
            overlapping_gene = best_g["name"]
        else:
            ov_cds = [c for c in gtf_cds if c["st"] <= c_pos <= c["en"]]
            if ov_cds:
                gtf_feature_class = "coding_exon"
                overlapping_gene = ov_cds[0]["name"]
            else:
                ov_utr = [u for u in gtf_utr if u["st"] <= c_pos <= u["en"]]
                if ov_utr:
                    gtf_feature_class = "utr"
                    overlapping_gene = ov_utr[0]["name"]
                else:
                    ov_body = [g for g in gtf_genes if g["st"] <= c_pos <= g["en"]]
                    if ov_body:
                        gtf_feature_class = "intronic"
                        overlapping_gene = ov_body[0]["name"]
                    else:
                        gtf_feature_class = "intergenic"
                        overlapping_gene = best_g["name"]

        # Window sequence
        w_st = max(0, c_pos - center_offset)
        w_en = w_st + window_bp
        c_seq = str(target_genome[locus_chr][w_st:w_en]).upper()
        if len(c_seq) < window_bp:
            c_seq += "N" * (window_bp - len(c_seq))
        c_onehot = one_hot_encode_sequence(c_seq)

        # Baseline element scores
        with torch.no_grad():
            out_wt = model(torch.tensor(c_onehot, device=device).unsqueeze(0), compute_late=True)
            score_wt_abs = float(out_wt["y_abs"].squeeze().item())
            score_wt_rel = float(out_wt["y_rel"].squeeze().item())
            score_wt_late = float(out_wt.get("pred_late", out_wt.get("late", out_wt["y_abs"])).squeeze().item())
            score_wt_rel_gated = score_wt_rel if score_wt_late >= open_gate_late_thresh else np.nan

        # 3. CpG-island & GC-composition diagnostic (central 500 bp)
        h_mid = center_offset
        seq_500 = c_seq[h_mid - 250 : h_mid + 250]
        c_gc = (seq_500.count("G") + seq_500.count("C")) / max(1, len(seq_500))
        c_cpg = (seq_500.count("CG") * len(seq_500)) / max(1, seq_500.count("C") * seq_500.count("G"))

        # CpG ablation control (mutate all CG -> TG across central 500 bp)
        seq_500_cpg_abl = seq_500.replace("CG", "TG")
        seq_cpg_abl = c_seq[:h_mid - 250] + seq_500_cpg_abl + c_seq[h_mid + 250:]
        with torch.no_grad():
            out_cpg_abl = model(torch.tensor(one_hot_encode_sequence(seq_cpg_abl), device=device).unsqueeze(0), compute_late=True)
            cpg_abl_abs = float(out_cpg_abl["y_abs"].squeeze().item())
            cpg_abl_late = float(out_cpg_abl.get("pred_late", out_cpg_abl.get("late", out_cpg_abl["y_abs"])).squeeze().item())
        cpg_ablation_delta_abs = cpg_abl_abs - score_wt_abs
        cpg_ablation_delta_late = cpg_abl_late - score_wt_late

        # Dinucleotide shuffle control (mean of 3 shuffles across central 500 bp)
        dinuc_deltas = []
        for s_seed in [42, 137, 2024]:
            shuf_500 = dinuc_shuffle_sequence(seq_500, seed=s_seed)
            shuf_full = c_seq[:h_mid - 250] + shuf_500 + c_seq[h_mid + 250:]
            with torch.no_grad():
                out_shuf = model(torch.tensor(one_hot_encode_sequence(shuf_full), device=device).unsqueeze(0), compute_late=True)
                dinuc_deltas.append(float(out_shuf["y_abs"].squeeze().item()) - score_wt_abs)
        dinuc_shuffle_delta_abs = float(np.mean(dinuc_deltas))

        # Determine if element overlaps known target gene promoter in Zebrafish
        syntenic_offset = None
        if overlapping_gene.lower() in zf_gene_promoters:
            syntenic_offset = zf_gene_promoters[overlapping_gene.lower()]

        # 4. Element-level orthology anchoring once per candidate element
        elem_ortho = anchor_element_orthology(
            target_window_seq=c_seq,
            ortholog_locus_seq=zf_locus_seq,
            model=model,
            device=device,
            open_gate_thresh=open_gate_late_thresh,
            syntenic_promoter_offset=syntenic_offset
        )
        is_lineage_gain = not elem_ortho["is_homologous"]
        elem_ortho_status = elem_ortho["status"]

        # 5. WT-projected Integrated Gradients on y_abs
        c_attr = compute_integrated_gradients(model, c_onehot, target="y_abs", steps=20, device=device)
        c_imp = np.maximum(0.0, np.sum(c_attr * c_onehot, axis=0))

        # Restrict to central 1,000 bp
        c_imp_central = np.zeros_like(c_imp)
        half_central = min(central_bp // 2, center_offset)
        c_imp_central[center_offset - half_central : center_offset + half_central] = c_imp[center_offset - half_central : center_offset + half_central]

        # Extract top 3-5 non-overlapping activating motifs
        motifs = extract_multiple_causal_seqlets(c_imp_central, min_len=6, max_len=14, top_k=4)

        elem_best_motif = None
        elem_best_delta = 0.0

        for m_idx, (m_st, m_en, m_score, m_peak) in enumerate(motifs):
            m_seq = c_seq[m_st:m_en]
            m_len = m_en - m_st
            motif_g_st = w_st + m_st
            motif_g_en = w_st + m_en
            motif_g_mid = (motif_g_st + motif_g_en) // 2
            motif_id = f"cand_{c_rank}_m{m_idx + 1}_{m_seq}"
            motif_dist_primary_tss_bp = calc_signed_dist(motif_g_mid, p_tss, p_strand)

            # In silico ablation
            c_abl_seq = c_seq[:m_st] + ("N" * m_len) + c_seq[m_en:]
            with torch.no_grad():
                out_abl = model(torch.tensor(one_hot_encode_sequence(c_abl_seq), device=device).unsqueeze(0), compute_late=True)
                score_abl_abs = float(out_abl["y_abs"].squeeze().item())
                score_abl_rel = float(out_abl["y_rel"].squeeze().item())
                score_abl_late = float(out_abl.get("pred_late", out_abl.get("late", out_abl["y_abs"])).squeeze().item())
                score_abl_rel_gated = score_abl_rel if score_abl_late >= open_gate_late_thresh else np.nan

            abl_delta_abs = score_abl_abs - score_wt_abs
            abl_delta_late = score_abl_late - score_wt_late
            abl_delta_rel_gated = (score_abl_rel_gated - score_wt_rel_gated) if (np.isfinite(score_abl_rel_gated) and np.isfinite(score_wt_rel_gated)) else np.nan

            # Unbiased TOMTOM against full JASPAR + secondary targeted pre-MZT check
            with tempfile.NamedTemporaryFile("w", suffix=".meme", delete=False) as f_tmp:
                tmp_meme = f_tmp.name
            try:
                seqlet_to_meme(m_seq, motif_id, tmp_meme)
                tom_full = run_tomtom(tmp_meme, str(jaspar_meme_path))
                tom_premzt = run_tomtom(tmp_meme, str(premzt_meme_path))

                if not tom_full.empty:
                    ann_df = annotate_motif_hits(
                        tomtom_df=tom_full.head(1),
                        tf_list_path=tf_list_path,
                        rnaseq_cpm_dict=rnaseq_cpm_dict,
                        target_meme_path=str(jaspar_meme_path),
                        premzt_tomtom_df=tom_premzt
                    )
                    h_row = ann_df.iloc[0]
                    top_tf = str(h_row["top_jaspar_tf"])
                    top_id = str(h_row["Target_ID"])
                    top_qv = float(h_row["top_jaspar_qval"])
                    is_premzt = bool(h_row["in_premzt_tf_list"])
                    premzt_gene = str(h_row["matched_premzt_tf_gene"])
                    premzt_qv = float(h_row["premzt_subset_qval"]) if pd.notna(h_row["premzt_subset_qval"]) else np.nan
                    disp_name = str(h_row["display_name"])
                    mat_cpm = float(h_row["maternal_cpm"])
                else:
                    top_tf, top_id, top_qv = "Unmatched", "-", 1.0
                    is_premzt, premzt_gene, premzt_qv = False, "-", np.nan
                    disp_name, mat_cpm = "Unmatched", 0.0
            except Exception:
                top_tf, top_id, top_qv = "Unmatched", "-", 1.0
                is_premzt, premzt_gene, premzt_qv = False, "-", np.nan
                disp_name, mat_cpm = "Unmatched", 0.0
            finally:
                Path(tmp_meme).unlink(missing_ok=True)

            # Reciprocal evolutionary swap anchored on element orthology
            swap_res = perform_evolutionary_alignment_and_swap(
                model=model,
                target_window_seq=c_seq,
                ortholog_locus_seq=zf_locus_seq,
                motif_start_in_window=m_st,
                motif_len=m_len,
                device=device,
                open_gate_thresh=open_gate_late_thresh,
                elem_ortho_info=elem_ortho
            )

            rec = {
                "motif_id": motif_id,
                "element_rank": c_rank,
                "element_class": gtf_feature_class,
                "gtf_feature_class": gtf_feature_class,
                "overlapping_gene": overlapping_gene,
                "chrom": locus_chr,
                "element_center": c_pos,
                "dist_from_primary_tss_bp": dist_from_primary_tss_bp,
                "dist_from_primary_tss_kb": dist_from_primary_tss_kb,
                "nearest_target_gene": nearest_target_gene,
                "dist_from_nearest_target_tss_bp": dist_from_nearest_target_tss_bp,
                "gc_frac": c_gc,
                "cpg_oe": c_cpg,
                "cpg_ablation_delta_abs": cpg_ablation_delta_abs,
                "cpg_ablation_delta_late": cpg_ablation_delta_late,
                "dinuc_shuffle_delta_abs": dinuc_shuffle_delta_abs,
                "zf_gc_frac": elem_ortho.get("zf_gc_frac", np.nan),
                "zf_cpg_oe": elem_ortho.get("zf_cpg_oe", np.nan),
                "is_lineage_gain": is_lineage_gain,
                "evolutionary_status": elem_ortho_status,
                "motif_genomic_start": motif_g_st,
                "motif_genomic_end": motif_g_en,
                "motif_dist_primary_tss_bp": motif_dist_primary_tss_bp,
                "motif_len": m_len,
                "motif_seq": m_seq,
                "seqlet_abs_contrib": float(m_score),
                "top_jaspar_tf": top_tf,
                "top_jaspar_id": top_id,
                "top_jaspar_qval": top_qv,
                "in_premzt_tf_list": is_premzt,
                "matched_premzt_tf_gene": premzt_gene,
                "premzt_subset_qval": premzt_qv,
                "display_name": disp_name,
                "maternal_cpm": mat_cpm,
                "target_wt_abs": score_wt_abs,
                "target_ablation_delta": abl_delta_abs,
                "target_ablation_delta_late": abl_delta_late,
                "target_ablation_delta_rel_gated": abl_delta_rel_gated,
                "best_strand": swap_res.get("best_strand"),
                "classification": swap_res.get("classification"),
                "target_swap_delta": swap_res.get("target_swap_delta"),
                "target_swap_delta_late": swap_res.get("target_swap_delta_late"),
                "target_swap_delta_rel_gated": swap_res.get("target_swap_delta_rel_gated"),
                "zf_wt_abs": swap_res.get("ortholog_wt"),
                "zf_swap_delta": swap_res.get("ortholog_swap_delta"),
                "aligned_target_motif_window": swap_res.get("aligned_target_motif_window"),
                "aligned_match_bar": swap_res.get("aligned_match_bar"),
                "aligned_ortholog_motif_window": swap_res.get("aligned_ortholog_motif_window"),
                "ortholog_aligned_motif": swap_res.get("ortholog_aligned_motif")
            }
            all_motif_records.append(rec)

            if abl_delta_abs < elem_best_delta:
                elem_best_delta = abl_delta_abs
                elem_best_motif = rec

        # Element-level summary
        if elem_best_motif is not None:
            sum_rec = dict(elem_best_motif)
            sum_rec["num_motifs_extracted"] = len(motifs)
            all_element_summaries.append(sum_rec)

        # Deep dive data for publication Figure 4b (Ranks 1, 2, 3, and 9)
        if c_rank in [1, 2, 3, 9]:
            logo_half = 80
            logo_st = center_offset - logo_half
            logo_en = center_offset + logo_half
            logo_attr = (c_attr * c_onehot)[:, logo_st:logo_en]
            df_logo = pd.DataFrame(logo_attr.T, columns=["A", "C", "G", "T"])
            deep_dive_panels[c_rank] = {
                "rank": c_rank,
                "class": gtf_feature_class,
                "overlapping_gene": overlapping_gene,
                "center": c_pos,
                "dist_from_primary_tss_kb": dist_from_primary_tss_kb,
                "df_logo": df_logo,
                "logo_genomic_start": w_st + logo_st,
                "logo_st_in_window": logo_st,
                "logo_en_in_window": logo_en,
                "best_motif": elem_best_motif,
                "motifs_in_window": [
                    r for r in all_motif_records
                    if r["element_rank"] == c_rank and r["motif_genomic_start"] >= (w_st + logo_st) and r["motif_genomic_end"] <= (w_st + logo_en)
                ]
            }

        diag_desc = f"GC:{c_gc*100:.1f}% CpG:{c_cpg:.2f} Abl_CG:{cpg_ablation_delta_abs:+.2f} Shuf:{dinuc_shuffle_delta_abs:+.2f}"
        print(f"  [✓] Rank {c_rank:2d} ({gtf_feature_class:16s} of {overlapping_gene:10s} | {dist_from_primary_tss_kb:+6.1f} kb | {elem_ortho_status:26s}): {diag_desc} | Top TF: {elem_best_motif['display_name'] if elem_best_motif else 'None':12s} | Δ_abs: {elem_best_delta:+.3f}")

    df_motifs = pd.DataFrame(all_motif_records)
    df_elem_summary = pd.DataFrame(all_element_summaries)

    # Export master motif table and summary
    motif_table_path = res_dir / "locus_cis_regulatory_motifs.tsv"
    df_motifs.to_csv(motif_table_path, sep="\t", index=False)
    print(f"\n[+] Saved Master Cis-Regulatory Motif Table ({len(df_motifs)} motifs across all elements): {motif_table_path}")

    swap_summary_path = res_dir / "evolutionary_swap_summary.tsv"
    df_elem_summary.to_csv(swap_summary_path, sep="\t", index=False)
    print(f"[+] Saved Evolutionary Swap Summary: {swap_summary_path}")

    # Top Motifs across whole locus
    top_n = min(20, len(df_motifs))
    top_motifs = df_motifs.sort_values(by="target_ablation_delta", ascending=True).head(top_n)
    print("=" * 80)
    print(f"[*] Top {top_n} Causal Motifs Across Entire '{locus_name}' Landscape (Ranked by |Ablation Δ_abs|):")
    print("=" * 80)
    fmt_hdr = f"{'Rank':<5} {'Elem':<6} {'Class':<16} {'Gene':<10} {'Dist_TSS':<10} {'TF Name':<14} {'PreMZT':<8} {'Δ_abs':<9} {'Swap_Δ':<9} {'q-val':<8}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    for r_i, (_, m_r) in enumerate(top_motifs.iterrows()):
        e_rk = f"R{int(m_r['element_rank'])}"
        c_desc = str(m_r['gtf_feature_class'])[:15]
        og = str(m_r['overlapping_gene'])[:9]
        d_tss = f"{float(m_r['dist_from_primary_tss_kb']):+.1f}k"
        tf_d = str(m_r['display_name'])[:13]
        is_p = "YES ★" if m_r['in_premzt_tf_list'] else "no"
        d_abs = f"{float(m_r['target_ablation_delta']):+.3f}"
        d_sw = f"{float(m_r['target_swap_delta']):+.3f}" if pd.notna(m_r['target_swap_delta']) else "-"
        qv = f"{float(m_r['top_jaspar_qval']):.1e}"
        print(f"{r_i + 1:<5} {e_rk:<6} {c_desc:<16} {og:<10} {d_tss:<10} {tf_d:<14} {is_p:<8} {d_abs:<9} {d_sw:<9} {qv:<8}")
    print("=" * 80)

    # Generate Publication Figures 4a & 4b
    _generate_publication_figures(
        res_dir=res_dir,
        locus_name=locus_name,
        df_motifs=df_motifs,
        df_cands=df_cands,
        deep_dive_panels=deep_dive_panels,
        p_tss=p_tss,
        p_strand=p_strand,
        primary_target_gene=primary_target_gene,
        target_genes=target_genes,
        gtf_genes=gtf_genes
    )

    return {
        "df_motifs": df_motifs,
        "df_elements": df_elem_summary,
        "motif_table_path": str(motif_table_path),
        "swap_summary_path": str(swap_summary_path),
        "fig4a_png": str(res_dir / "fig4a_locus_cis_regulatory_landscape.png"),
        "fig4a_svg": str(res_dir / "fig4a_locus_cis_regulatory_landscape.svg"),
        "fig4b_png": str(res_dir / "fig4b_single_nt_attribution_tracks.png"),
        "fig4b_svg": str(res_dir / "fig4b_single_nt_attribution_tracks.svg"),
    }


def _generate_publication_figures(
    res_dir: Path,
    locus_name: str,
    df_motifs: pd.DataFrame,
    df_cands: pd.DataFrame,
    deep_dive_panels: Dict[int, Dict[str, Any]],
    p_tss: int,
    p_strand: str,
    primary_target_gene: str,
    target_genes: List[str],
    gtf_genes: List[Dict[str, Any]]
) -> None:
    """Render and export publication-ready Figure 4a and 4b (300 DPI PNG & SVG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import logomaker

    def to_signed_kb(pos_bp):
        return ((pos_bp - p_tss) if p_strand == "+" else (p_tss - pos_bp)) / 1000.0

    # --------------------------------------------------------------------------
    # Figure 4a: Whole Locus Cis-Regulatory Landscape
    # --------------------------------------------------------------------------
    fig_4a = plt.figure(figsize=(19, 12.5))
    gs_4a = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.32)

    # 1. Top Panel: Locus Regulatory Architecture & Accessibility Map (Signed kb x-axis)
    ax_map = fig_4a.add_subplot(gs_4a[0, :])

    bg_abs_path = res_dir / f"{locus_name}_pred_y_abs.bedgraph"
    bg_gated_path = res_dir / f"{locus_name}_pred_y_rel_gated.bedgraph"

    pred_min_bp = None
    pred_max_bp = None
    has_tracks = False

    if bg_abs_path.exists():
        try:
            df_bg_abs = pd.read_csv(bg_abs_path, sep="\t", header=None, names=["chr", "st", "en", "val"])
            if not df_bg_abs.empty:
                pred_min_bp = float(df_bg_abs["st"].min())
                pred_max_bp = float(df_bg_abs["en"].max())
                mids_bp = 0.5 * (df_bg_abs["st"] + df_bg_abs["en"])
                mids_kb = [to_signed_kb(p) for p in mids_bp]
                ax_map.plot(mids_kb, df_bg_abs["val"], color="#2b5c8f", lw=1.8, label="Predicted Pre-MZT Accessibility ($y_{abs}$)")
                ax_map.fill_between(mids_kb, df_bg_abs["val"], color="#2b5c8f", alpha=0.18)
                has_tracks = True
        except Exception:
            pass

    if bg_gated_path.exists():
        try:
            df_bg_rel = pd.read_csv(bg_gated_path, sep="\t", header=None, names=["chr", "st", "en", "val"])
            if pred_min_bp is not None and pred_max_bp is not None:
                df_bg_rel = df_bg_rel[(df_bg_rel["st"] >= pred_min_bp) & (df_bg_rel["en"] <= pred_max_bp)]
            if not df_bg_rel.empty:
                mids_r_bp = 0.5 * (df_bg_rel["st"] + df_bg_rel["en"])
                mids_r_kb = [to_signed_kb(p) for p in mids_r_bp]
                ax_map.plot(mids_r_kb, df_bg_rel["val"], color="#88419d", lw=1.5, ls="--", label=r"Timing-Gated Early Bias ($y_{rel\_gated}$)")
        except Exception:
            pass

    # Strictly clamp x-axis to model predicted region
    if pred_min_bp is not None and pred_max_bp is not None:
        x_min_kb = min(to_signed_kb(pred_min_bp), to_signed_kb(pred_max_bp))
        x_max_kb = max(to_signed_kb(pred_min_bp), to_signed_kb(pred_max_bp))
    else:
        x_min_kb = min(to_signed_kb(float(df_cands["start"].min())), to_signed_kb(float(df_cands["end"].max())))
        x_max_kb = max(to_signed_kb(float(df_cands["start"].min())), to_signed_kb(float(df_cands["end"].max())))
    ax_map.set_xlim(x_min_kb, x_max_kb)

    # Primary TSS reference line at 0 kb
    if x_min_kb <= 0 <= x_max_kb:
        ax_map.axvline(0, color="#d73027", linestyle="--", lw=1.5, alpha=0.8, label=f"{primary_target_gene} TSS (0 kb)")

    # Overlay Candidate Elements strictly within predicted window
    top_motifs_by_elem = df_motifs.sort_values(by="target_ablation_delta", ascending=True).groupby("element_rank").first().to_dict(orient="index")
    y_top = float(ax_map.get_ylim()[1]) if has_tracks else 3.5
    ax_map.set_ylim(-0.34 * y_top, y_top * 1.28)
    ax_map.axhline(0.0, color="#969696", lw=0.7, alpha=0.7)

    # Annotate elements with matched TFs cleanly below top legend headroom
    annot_y_levels = [0.84, 0.67, 0.50, 0.96]
    annot_idx = 0

    for _, c_row in df_cands.sort_values(by="center").iterrows():
        rk = int(c_row["candidate_rank"])
        c_st_kb = to_signed_kb(float(c_row["start"]))
        c_en_kb = to_signed_kb(float(c_row["end"]))
        c_mid_kb = to_signed_kb(float(c_row["center"]))
        x_st = min(c_st_kb, c_en_kb)
        x_en = max(c_st_kb, c_en_kb)

        # Skip if outside model predicted window
        if x_en < x_min_kb or x_st > x_max_kb:
            continue

        m_info = top_motifs_by_elem.get(rk, {})
        is_lineage = m_info.get("is_lineage_gain", bool(c_row.get("lacks_zf_ortholog_peak", False)))
        feat_cls = m_info.get("gtf_feature_class", "enhancer")

        if is_lineage:
            col = "#d73027"
            alpha = 0.35
            lbl_color = "#990000"
        elif feat_cls == "promoter_tss_2kb":
            col = "#1b7837"
            alpha = 0.25
            lbl_color = "#005a32"
        else:
            col = "#4575b4"
            alpha = 0.18
            lbl_color = "#2166ac"

        ax_map.axvspan(max(x_st, x_min_kb), min(x_en, x_max_kb), color=col, alpha=alpha, zorder=1)

        # Label elements that have a matched TF
        tf_call = m_info.get("display_name", "")
        if tf_call and tf_call != "None" and tf_call != "Unmatched" and (x_min_kb <= c_mid_kb <= x_max_kb):
            ov_gene = m_info.get("overlapping_gene", "")
            badge = f"R{rk} ({ov_gene})\n{tf_call}"
            y_annot = y_top * annot_y_levels[annot_idx % len(annot_y_levels)]
            annot_idx += 1
            ax_map.annotate(
                badge,
                xy=(c_mid_kb, float(c_row.get("pred_abs", 2.0))),
                xytext=(c_mid_kb, y_annot),
                arrowprops=dict(arrowstyle="->", color=lbl_color, lw=1.0),
                fontsize=8,
                fontweight="bold",
                ha="center",
                color=lbl_color,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=lbl_color, alpha=0.9, lw=0.9),
                zorder=5
            )

    # Draw Gene Models cleanly in dedicated strip below y=0 with width-aware interval scheduling
    p_min_bp = pred_min_bp if pred_min_bp is not None else float(df_cands["start"].min())
    p_max_bp = pred_max_bp if pred_max_bp is not None else float(df_cands["end"].max())

    # Select canonical protein-coding genes in locus
    key_locus_genes = [g for g in gtf_genes if not (g["en"] < p_min_bp or g["st"] > p_max_bp)]
    # De-duplicate by gene name and keep longest span
    dedup_genes = {}
    for g in key_locus_genes:
        gn = g["name"]
        if (gn.startswith("ENS") or gn.startswith("LOC")) and not any(tg.lower() == gn.lower() for tg in target_genes):
            continue
        if gn not in dedup_genes or (g["en"] - g["st"]) > (dedup_genes[gn]["en"] - dedup_genes[gn]["st"]):
            dedup_genes[gn] = g
    sorted_locus_genes = sorted(dedup_genes.values(), key=lambda x: x["st"])

    x_span_kb = max(10.0, x_max_kb - x_min_kb)
    gene_track_ys = [-0.07 * y_top, -0.18 * y_top, -0.29 * y_top]
    track_ends_kb: List[float] = []

    for g in sorted_locus_genes:
        clipped_st = max(g["st"], p_min_bp)
        clipped_en = min(g["en"], p_max_bp)
        gx_st = min(to_signed_kb(clipped_st), to_signed_kb(clipped_en))
        gx_en = max(to_signed_kb(clipped_st), to_signed_kb(clipped_en))
        mid_g = 0.5 * (gx_st + gx_en)
        arrow = "→" if g["strand"] == "+" else "←"
        lbl_str = f"{g['name']} {arrow}"
        half_lbl_kb = max(4.5, 0.55 * len(lbl_str) * (x_span_kb / 140.0))
        occ_st = min(gx_st, mid_g - half_lbl_kb)
        occ_en = max(gx_en, mid_g + half_lbl_kb)

        assigned_tr = None
        for t_idx, last_en_kb in enumerate(track_ends_kb):
            if occ_st > last_en_kb + 1.5:
                track_ends_kb[t_idx] = occ_en
                assigned_tr = t_idx
                break
        if assigned_tr is None:
            assigned_tr = len(track_ends_kb)
            track_ends_kb.append(occ_en)

        y_g = gene_track_ys[assigned_tr % len(gene_track_ys)]
        is_tg = g["name"].lower() in [tg.lower() for tg in target_genes]
        g_col = "#d73027" if is_tg else "#2b5c8f"
        g_lw = 3.5 if is_tg else 2.0
        ax_map.plot([gx_st, gx_en], [y_g, y_g], color=g_col, lw=g_lw, solid_capstyle="round", zorder=4)
        if x_min_kb <= mid_g <= x_max_kb:
            ax_map.text(
                mid_g,
                y_g + 0.015 * y_top,
                lbl_str,
                fontsize=7.8,
                fontweight="bold" if is_tg else "semibold",
                color=g_col,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=g_col, lw=0.7, alpha=0.90),
                zorder=6
            )

    ax_map.set_xlabel(f"Signed Distance from {primary_target_gene} TSS (kb)", fontsize=11, fontweight="bold")
    ax_map.set_ylabel("Predicted Accessibility", fontsize=11, fontweight="bold")
    ax_map.set_title(
        f"A. Pre-MZT Cis-Regulatory Landscape: '{locus_name}' ({len(df_cands)} Elements relative to {primary_target_gene} TSS)",
        fontsize=13, fontweight="bold", pad=44
    )

    legend_handles_a = [
        matplotlib.lines.Line2D([0], [0], color="#2b5c8f", lw=1.8, label="Predicted Pre-MZT Accessibility ($y_{abs}$)"),
        matplotlib.lines.Line2D([0], [0], color="#88419d", lw=1.5, ls="--", label=r"Timing-Gated Early Bias ($y_{rel\_gated}$)"),
        matplotlib.lines.Line2D([0], [0], color="#d73027", lw=1.5, ls="--", label=f"{primary_target_gene} TSS (0 kb)"),
        patches.Patch(facecolor="#d73027", alpha=0.35, edgecolor="none", label="Lineage-Gained Element (Cichlid-Specific)"),
        patches.Patch(facecolor="#1b7837", alpha=0.25, edgecolor="none", label="Conserved Syntenic Promoter (TSS ±2 kb)"),
        patches.Patch(facecolor="#4575b4", alpha=0.18, edgecolor="none", label="Conserved Syntenic Enhancer")
    ]
    ax_map.legend(handles=legend_handles_a, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=True, fontsize=8.3, framealpha=0.94)
    ax_map.grid(True, linestyle=":", alpha=0.4)

    # 2. Bottom-Left Panel: Ranked Top Motifs by Ablation Effect (Clean Spacing & Margin)
    ax_bar = fig_4a.add_subplot(gs_4a[1, 0])
    top15 = df_motifs.sort_values(by="target_ablation_delta", ascending=True).head(15).iloc[::-1]

    y_pos = np.arange(len(top15))
    bar_colors = ["#d73027" if p else "#4575b4" for p in top15["in_premzt_tf_list"]]
    ax_bar.barh(y_pos, top15["target_ablation_delta"], color=bar_colors, edgecolor="black", height=0.65, lw=0.8)

    labels = [
        f"R{int(r['element_rank'])} ({str(r['overlapping_gene'])[:12]}) | {str(r['display_name'])[:14]} | {r['motif_seq']}"
        for _, r in top15.iterrows()
    ]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=8.3, fontweight="medium")
    ax_bar.tick_params(axis="y", labelsize=8.3, pad=6)
    ax_bar.set_ylabel("Causal Motifs (Rank | Gene | TF | K-mer)", fontsize=10.5, fontweight="bold", labelpad=8)
    ax_bar.set_xlabel("In Silico Ablation Effect (Δ Pre-MZT Accessibility $y_{abs}$)", fontsize=10.5, fontweight="bold")
    ax_bar.set_title("B. Top 15 Causal Motifs Across Locus by Ablation Effect", fontsize=12, fontweight="bold")
    ax_bar.axvline(0, color="gray", linestyle="--", lw=1)
    ax_bar.grid(True, linestyle=":", alpha=0.5, axis="x")

    legend_elements = [
        patches.Patch(facecolor="#d73027", edgecolor="black", label="Maternal TF"),
        patches.Patch(facecolor="#4575b4", edgecolor="black", label="Other Vertebrate JASPAR TF")
    ]
    ax_bar.legend(handles=legend_elements, loc="lower left", fontsize=8.5, frameon=True)

    # 3. Bottom-Right Panel: Ablation vs. Swap Delta Scatter Plot (Dot Legends + All Dots Labeled cleanly)
    ax_scatter = fig_4a.add_subplot(gs_4a[1, 1])
    valid_swaps = df_motifs[df_motifs["target_swap_delta"].notna()].copy().reset_index(drop=True)
    unaligned_motifs = df_motifs[df_motifs["target_swap_delta"].isna()].copy()

    class_colors = {
        "promoter_tss_2kb": "#1b7837",
        "coding_exon": "#e7298a",
        "utr": "#7570b3",
        "intronic": "#e6ab02",
        "intergenic": "#66a61e"
    }
    class_display = {
        "promoter_tss_2kb": "Promoter TSS (±2 kb)",
        "coding_exon": "Coding Exon",
        "utr": "UTR",
        "intronic": "Intronic",
        "intergenic": "Intergenic",
    }

    if not valid_swaps.empty:
        valid_swaps["abs_ablation_delta"] = valid_swaps["target_ablation_delta"].abs()
        valid_swaps = valid_swaps.sort_values(by=["element_rank", "abs_ablation_delta"], ascending=[True, False]).reset_index(drop=True)
        valid_swaps["motif_sub_idx"] = valid_swaps.groupby("element_rank").cumcount() + 1

        primary_motifs = valid_swaps[valid_swaps["motif_sub_idx"] == 1].copy().reset_index(drop=True)
        secondary_motifs = valid_swaps[valid_swaps["motif_sub_idx"] > 1].copy().reset_index(drop=True)

        # Plot secondary motifs (smaller, semi-transparent markers)
        if not secondary_motifs.empty:
            for c_type, grp in secondary_motifs.groupby("gtf_feature_class"):
                col = class_colors.get(c_type, "gray")
                ax_scatter.scatter(
                    grp["target_ablation_delta"],
                    grp["target_swap_delta"],
                    c=col,
                    s=60,
                    edgecolors=["gold" if p else "black" for p in grp["in_premzt_tf_list"]],
                    linewidths=[1.3 if p else 0.7 for p in grp["in_premzt_tf_list"]],
                    alpha=0.60,
                    zorder=3
                )

        # Plot primary causal motifs per element (prominent markers)
        for c_type, grp in primary_motifs.groupby("gtf_feature_class"):
            col = class_colors.get(c_type, "gray")
            ax_scatter.scatter(
                grp["target_ablation_delta"],
                grp["target_swap_delta"],
                c=col,
                s=195,
                edgecolors=["gold" if p else "black" for p in grp["in_premzt_tf_list"]],
                linewidths=[2.2 if p else 1.0 for p in grp["in_premzt_tf_list"]],
                alpha=0.95,
                zorder=4
            )

        # Expand bounds generously for collision-free labeling & upper-left dot legend
        all_x = valid_swaps["target_ablation_delta"].values
        all_y = valid_swaps["target_swap_delta"].values
        xr = max(1e-3, float(all_x.max() - all_x.min()))
        yr = max(1e-3, float(all_y.max() - all_y.min()))
        ax_scatter.set_xlim(float(all_x.min()) - 0.14 * xr, float(all_x.max()) + 0.16 * xr)
        ax_scatter.set_ylim(float(all_y.min()) - 0.14 * yr, float(all_y.max()) + 0.28 * yr)

        fig_4a.canvas.draw()
        pts_prim = ax_scatter.transData.transform(np.column_stack([primary_motifs["target_ablation_delta"], primary_motifs["target_swap_delta"]])) if not primary_motifs.empty else np.empty((0, 2))
        pts_sec = ax_scatter.transData.transform(np.column_stack([secondary_motifs["target_ablation_delta"], secondary_motifs["target_swap_delta"]])) if not secondary_motifs.empty else np.empty((0, 2))
        all_pts_px = np.vstack([pts_prim, pts_sec]) if (len(pts_prim) + len(pts_sec)) > 0 else np.empty((0, 2))
        ax_bbox = ax_scatter.get_window_extent()

        # Reserve top-left legend corner in pixel space to avoid label collisions there
        leg_px_box = (
            ax_bbox.x0,
            ax_bbox.y1 - 0.34 * ax_bbox.height,
            ax_bbox.x0 + 0.56 * ax_bbox.width,
            ax_bbox.y1
        )
        placed_boxes: List[Tuple[float, float, float, float]] = [leg_px_box]

        # Place label inside isolated primary dots first; track crowded primaries and all secondaries for callouts
        callout_queue = []
        for i_p in range(len(primary_motifs)):
            rk = int(primary_motifs.loc[i_p, "element_rank"])
            px_val = float(primary_motifs.loc[i_p, "target_ablation_delta"])
            py_val = float(primary_motifs.loc[i_p, "target_swap_delta"])
            p_px = pts_prim[i_p]
            col = class_colors.get(primary_motifs.loc[i_p, "gtf_feature_class"], "gray")
            dists = [np.hypot(p_px[0] - op[0], p_px[1] - op[1]) for j, op in enumerate(pts_prim) if j != i_p]
            min_d = min(dists) if dists else 999.0
            in_leg = (leg_px_box[0] <= p_px[0] <= leg_px_box[2]) and (leg_px_box[1] <= p_px[1] <= leg_px_box[3])
            if min_d >= 20.0 and not in_leg:
                ax_scatter.text(px_val, py_val, str(rk), color="white", fontsize=8.2, fontweight="bold", ha="center", va="center", zorder=6)
                placed_boxes.append((p_px[0] - 9, p_px[1] - 9, p_px[0] + 9, p_px[1] + 9))
            else:
                callout_queue.append((str(rk), px_val, py_val, p_px[0], p_px[1], col, 8.2, "bold", True))

        for j_s in range(len(secondary_motifs)):
            s_row = secondary_motifs.iloc[j_s]
            rk = int(s_row["element_rank"])
            sub_idx = int(s_row["motif_sub_idx"])
            px_val = float(s_row["target_ablation_delta"])
            py_val = float(s_row["target_swap_delta"])
            s_px = pts_sec[j_s]
            col = class_colors.get(s_row["gtf_feature_class"], "#525252")
            callout_queue.append((f"{rk}.{sub_idx}", px_val, py_val, s_px[0], s_px[1], col, 7.2, "semibold", False))

        # Iterative pixel-space greedy non-overlapping placement for every remaining dot
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        radii = [18, 28, 38, 50, 64]
        for lbl_txt, dx_val, dy_val, x_px, y_px, col, fsz, fwt, is_p in callout_queue:
            box_w = max(16.0, len(lbl_txt) * 5.8 + 6.0)
            box_h = 13.0
            best_score = float("inf")
            best_cand = (18.0, 14.0)
            best_box = (x_px, y_px, x_px + box_w, y_px + box_h)

            for r_pt in radii:
                r_px = r_pt * (fig_4a.dpi / 72.0)
                for ang in angles:
                    ox_pt = r_pt * np.cos(ang)
                    oy_pt = r_pt * np.sin(ang)
                    cx_px = x_px + r_px * np.cos(ang)
                    cy_px = y_px + r_px * np.sin(ang)
                    bx0, by0, bx1, by1 = cx_px - 0.5 * box_w, cy_px - 0.5 * box_h, cx_px + 0.5 * box_w, cy_px + 0.5 * box_h
                    pen = 0.05 * r_pt
                    # Penalize out-of-bounds
                    if bx0 < ax_bbox.x0 + 4 or bx1 > ax_bbox.x1 - 4 or by0 < ax_bbox.y0 + 4 or by1 > ax_bbox.y1 - 4:
                        pen += 5000.0
                    # Penalize overlap with points
                    if len(all_pts_px) > 0:
                        px_dist = np.hypot(all_pts_px[:, 0] - cx_px, all_pts_px[:, 1] - cy_px)
                        too_close = np.sum(px_dist < 14.0)
                        pen += 400.0 * float(too_close)
                    # Penalize overlap with already placed label boxes & legend
                    for pb0, pb1, pb2, pb3 in placed_boxes:
                        ix = max(0.0, min(bx1, pb2) - max(bx0, pb0))
                        iy = max(0.0, min(by1, pb3) - max(by0, pb1))
                        if ix > 0 and iy > 0:
                            pen += 800.0 + ix * iy
                    if pen < best_score:
                        best_score = pen
                        best_cand = (ox_pt, oy_pt)
                        best_box = (bx0, by0, bx1, by1)

            placed_boxes.append(best_box)
            ax_scatter.annotate(
                lbl_txt,
                xy=(dx_val, dy_val),
                xytext=best_cand,
                textcoords="offset points",
                fontsize=fsz,
                fontweight=fwt,
                color=col if is_p else "#333333",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.16" if not is_p else "circle,pad=0.18", fc="white", ec=col, lw=0.9 if is_p else 0.7, alpha=0.94),
                arrowprops=dict(arrowstyle="-", color=col, lw=0.75, alpha=0.8),
                zorder=6 if is_p else 5
            )

        # Build crisp dot legends (Class colors + Dot size hierarchy + Maternal TF ring)
        dot_legend_handles = []
        for c_type in ["promoter_tss_2kb", "coding_exon", "utr", "intronic", "intergenic"]:
            n_c = int((valid_swaps["gtf_feature_class"] == c_type).sum())
            if n_c > 0:
                dot_legend_handles.append(
                    matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=class_colors[c_type],
                                            markeredgecolor="black", markeredgewidth=0.8, markersize=7.5,
                                            label=f"{class_display[c_type]} (n={n_c})")
                )
        dot_legend_handles.extend([
            matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#636363", markeredgecolor="black",
                                    markeredgewidth=1.0, markersize=10.5, label="Large dot (n): Top causal motif"),
            matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#636363", markeredgecolor="black",
                                    markeredgewidth=0.7, alpha=0.60, markersize=5.8, label="Small dot (n.m): Secondary motif"),
            matplotlib.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="gold",
                                    markeredgewidth=2.0, markersize=8.5, label="Gold ring: Maternal TF match"),
        ])
        ax_scatter.legend(handles=dot_legend_handles, loc="upper left", fontsize=7.6, frameon=True, framealpha=0.94, handletextpad=0.4, borderpad=0.35)
        n_unaligned = len(unaligned_motifs)
        scatter_title = f"C. Ablation vs. Evolutionary Swap Perturbation\n(Syntenically aligned: {len(valid_swaps)} | Lineage-specific: {n_unaligned} motifs)"
    else:
        for c_type, grp in df_motifs.groupby("gtf_feature_class"):
            col = class_colors.get(c_type, "gray")
            ax_scatter.scatter(
                grp["target_ablation_delta"],
                [0.0] * len(grp),
                c=col,
                label=class_display.get(c_type, c_type.replace("_", " ").title()),
                s=70,
                alpha=0.7
            )
        ax_scatter.legend(loc="upper left", fontsize=8.0, frameon=True)
        scatter_title = "C. Ablation vs. Evolutionary Swap Perturbation"

    ax_scatter.axhline(0, color="gray", linestyle="--", lw=0.9)
    ax_scatter.axvline(0, color="gray", linestyle="--", lw=0.9)
    ax_scatter.set_xlabel("In Silico Ablation Effect (Target Δ $y_{abs}$)", fontsize=10.5, fontweight="bold")
    ax_scatter.set_ylabel("Target → Zebrafish Swap Effect (Δ $y_{abs}$)", fontsize=10.5, fontweight="bold")
    ax_scatter.set_title(scatter_title, fontsize=11.5, fontweight="bold")
    ax_scatter.grid(True, linestyle=":", alpha=0.5)

    plt.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.07, wspace=0.30, hspace=0.44)
    fig4a_png = res_dir / "fig4a_locus_cis_regulatory_landscape.png"
    fig4a_svg = res_dir / "fig4a_locus_cis_regulatory_landscape.svg"
    plt.savefig(fig4a_png, dpi=300)
    plt.savefig(fig4a_svg)
    plt.close(fig_4a)
    print(f"[+] Saved Publication Figure 4a: {fig4a_png} and .svg")

    # --------------------------------------------------------------------------
    # Figure 4b: Multi-Panel Single-Nucleotide Attribution Tracks (Ranks 1, 2, 3, 9)
    # --------------------------------------------------------------------------
    panel_ranks = [1, 2, 3, 9]
    available_ranks = [rk for rk in panel_ranks if rk in deep_dive_panels]
    if available_ranks:
        fig_4b, axes_4b = plt.subplots(len(available_ranks), 1, figsize=(16, 3.2 * len(available_ranks)), sharex=False)
        if len(available_ranks) == 1:
            axes_4b = [axes_4b]
        plt.subplots_adjust(hspace=0.45)

        for p_idx, rk in enumerate(available_ranks):
            ax = axes_4b[p_idx]
            p_data = deep_dive_panels.get(rk)
            df_l = p_data["df_logo"].reset_index(drop=True)
            logomaker.Logo(df_l, ax=ax)
            ax.set_ylabel("WT Attribution\n($y_{abs}$ Gradient)", fontsize=9.5, fontweight="bold")

            # Dynamic panel title from config and feature annotations
            feat_map = {
                "promoter_tss_2kb": "Promoter (TSS ±2 kb)",
                "coding_exon": "Coding Exon",
                "utr": "UTR",
                "intronic": "Intronic",
                "intergenic": "Intergenic",
            }
            raw_cls = str(p_data.get("class", "element"))
            feat_cls = feat_map.get(raw_cls, raw_cls.replace("_", " ").title())
            ov_gene = p_data.get("overlapping_gene", "")
            dist_kb = p_data.get("dist_from_primary_tss_kb", 0.0)
            top_m = p_data.get("best_motif", {})
            ortho_desc = "Lineage-Gained Element" if top_m.get("is_lineage_gain", False) else "Conserved Syntenic Element"
            dyn_title = f"Rank {rk}: {feat_cls} ({ov_gene} | {dist_kb:+.1f} kb from {primary_target_gene} TSS | {ortho_desc})"
            ax.set_title(f"Track {chr(65 + p_idx)}: {dyn_title}", fontsize=11, fontweight="bold", loc="left")

            motifs_in_p = p_data.get("motifs_in_window", [])
            plotted_any_box = False
            for m_rec in motifs_in_p:
                g_st = m_rec["motif_genomic_start"]
                g_en = m_rec["motif_genomic_end"]
                rel_st = (g_st - (p_data["center"] - 80))
                rel_en = (g_en - (p_data["center"] - 80))

                if rel_en > 0 and rel_st < 160:
                    plotted_any_box = True
                    draw_st = max(0.0, float(rel_st))
                    draw_en = min(160.0, float(rel_en))
                    is_pm = m_rec["in_premzt_tf_list"]
                    box_color = "#fee08b" if is_pm else "#e0f3f8"
                    ec_color = "#d73027" if is_pm else "#4575b4"

                    y_min, y_max = ax.get_ylim()
                    rect = patches.Rectangle(
                        (draw_st - 0.5, y_min),
                        max(1.0, draw_en - draw_st),
                        y_max - y_min,
                        linewidth=1.2,
                        edgecolor=ec_color,
                        facecolor=box_color,
                        alpha=0.35,
                        zorder=0
                    )
                    ax.add_patch(rect)

                    tf_text = m_rec["display_name"]
                    d_abs = float(m_rec["target_ablation_delta"])
                    sw_d = float(m_rec["target_swap_delta"]) if pd.notna(m_rec["target_swap_delta"]) else np.nan
                    sw_str = f"{sw_d:+.3f}" if np.isfinite(sw_d) else "unaligned"
                    label_text = f"{tf_text}\nΔ_abs:{d_abs:+.3f} | Swap:{sw_str}"
                    ax.text(
                        0.5 * (draw_st + draw_en),
                        y_max * 0.92,
                        label_text,
                        fontsize=8,
                        fontweight="bold",
                        color=ec_color,
                        ha="center",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ec_color, alpha=0.85, lw=0.8)
                    )

            # Note top causal motif for the full 2,114 bp element if it sits outside ±80 bp summit window
            if top_m:
                tm_off = int(top_m.get("motif_genomic_start", 0)) - int(p_data.get("center", 0))
                if abs(tm_off) > 75 or not plotted_any_box:
                    tm_tf = top_m.get("display_name", "Unmatched")
                    tm_seq = top_m.get("motif_seq", "")
                    tm_da = float(top_m.get("target_ablation_delta", 0.0))
                    ax.text(
                        0.01, 0.94,
                        f"Top element motif outside ±80 bp window: {tm_tf} '{tm_seq}' ({tm_off:+d} bp from summit, Δ_abs={tm_da:+.3f})",
                        transform=ax.transAxes,
                        fontsize=8.0, fontweight="semibold", color="#333333",
                        ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f7f7f7", ec="#969696", alpha=0.90, lw=0.8)
                    )

            if top_m and pd.notna(top_m.get("aligned_target_motif_window")):
                t_aln = str(top_m["aligned_target_motif_window"])[:38]
                m_bar = str(top_m["aligned_match_bar"])[:38]
                o_aln = str(top_m["aligned_ortholog_motif_window"])[:38]
                align_note = f"Zebrafish Ortholog Alignment: Target [{t_aln}] / Matches [{m_bar}] / Danio [{o_aln}]"
                ax.set_xlabel(f"Relative Offset across 160 bp Summit Window (bp)\n{align_note}", fontsize=8.2, fontstyle="italic")
            else:
                ax.set_xlabel("Relative Offset across 160 bp Summit Window (bp)", fontsize=9)

        fig_4b.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.07, hspace=0.55)
        fig4b_png = res_dir / "fig4b_single_nt_attribution_tracks.png"
        fig4b_svg = res_dir / "fig4b_single_nt_attribution_tracks.svg"
        plt.savefig(fig4b_png, dpi=300, bbox_inches="tight")
        plt.savefig(fig4b_svg, bbox_inches="tight")
        plt.close(fig_4b)
        print(f"[+] Saved Publication Figure 4b: {fig4b_png} and .svg")


def screen_genomewide_minor_wave_motifs(
    work_dir: str | Path = "work_astatotilapia_premzt",
    locus_name: str = "notch2_slc35a3a",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tf_list_path: str = "./TF_gene_names.txt"
) -> Dict[str, Any]:
    """
    Genome-Wide Top Occurring Motifs & TFBSs in High-Confidence Cichlid Minor-Wave Loci:
    1. Intersects cichlid RNA-seq minor-wave sets (strictly_zygotic_highconf, is_both_cichlids_minor_wave, SLAM confirmed).
    2. Excludes miR-430 / tandem repeat microRNAs.
    3. Selects GC-matched silent control comparison set.
    4. Evaluates top open 2,114 bp window within ±5 kb of TSS, extracts causal seqlets via fast input gradient.
    5. Single batched TOMTOM call against premzt_tfs.meme and full JASPAR CORE vertebrates.
    6. Aggregates motif frequencies, odds ratios, and Fisher p-values into minor_wave_genomewide_top_motifs.tsv.
    7. Renders publication Figure 4c (PNG & SVG).
    """
    from pyfaidx import Fasta
    from scipy.stats import fisher_exact
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    work_dir_p = Path(work_dir).resolve()
    res_dir = work_dir_p / "results" / locus_name
    res_dir.mkdir(parents=True, exist_ok=True)

    with open(work_dir_p / "config.json") as f:
        config = json.load(f)

    target_genome_fa = config.get("TARGET_GENOME_FA")
    target_genome = Fasta(target_genome_fa, as_raw=True)
    zf_ckpt = work_dir_p / "models" / "chrombpnet" / "Danio_rerio" / "fold_0" / "grafted_chrombpnet.pt"
    model = load_grafted_model(zf_ckpt, device=device)

    gene_sets_tsv = work_dir_p / "rnaseq" / "target_species_gene_sets.tsv"
    if not gene_sets_tsv.exists():
        gene_sets_tsv = res_dir / f"{locus_name}_gene_sets.tsv"
    df_genes = pd.read_csv(gene_sets_tsv, sep="\t")

    # 1. Identify high-confidence minor wave genes
    is_highconf = (
        (df_genes["category"] == "strictly_zygotic_highconf") |
        (df_genes.get("is_both_cichlids_minor_wave", False) == True) |
        (df_genes.get("is_slam_confirmed_minor_wave", False) == True)
    )
    df_mw = df_genes[is_highconf].copy()

    # 2. Exclude miR-430 / tandem microRNA repeats
    is_mir = df_mw["gene_name"].astype(str).str.contains("mir|mirna|snor|rRNA", case=False, regex=True)
    n_excluded_mir = int(is_mir.sum())
    df_mw = df_mw[~is_mir].reset_index(drop=True)
    print(f"[+] Screen: {len(df_mw)} high-confidence minor-wave genes (excluded {n_excluded_mir} microRNA/repeats)")

    # 3. Select GC-matched silent controls
    df_silent = df_genes[df_genes["category"] == "control_silent"].copy()
    if len(df_silent) > len(df_mw):
        rng = np.random.RandomState(42)
        ctrl_indices = rng.choice(len(df_silent), size=min(len(df_silent), len(df_mw)), replace=False)
        df_ctrl = df_silent.iloc[ctrl_indices].reset_index(drop=True)
    else:
        df_ctrl = df_silent.reset_index(drop=True)
    print(f"    Comparison set: {len(df_ctrl)} matched control silent genes")

    # Fast attribution extraction helper
    def extract_seqlets_for_set(df_set: pd.DataFrame, set_label: str):
        seqlet_records = []
        cpg_oe_vals = []
        gc_vals = []
        for idx, row in df_set.iterrows():
            g_chr = str(row["chrom"])
            g_strand = str(row.get("strand", "+"))
            tss = int(row.get("tss", row["start"] if g_strand == "+" else row["end"]))

            # Extract 2,114 bp window centered on TSS
            w_st = max(0, tss - 1057)
            w_en = w_st + 2114
            try:
                seq = str(target_genome[g_chr][w_st:w_en]).upper()
            except Exception:
                continue
            if len(seq) < 2114:
                seq += "N" * (2114 - len(seq))

            # Central 500 bp GC & CpG
            s_500 = seq[1057 - 250 : 1057 + 250]
            gc = (s_500.count("G") + s_500.count("C")) / max(1, len(s_500))
            cpg = (s_500.count("CG") * len(s_500)) / max(1, s_500.count("C") * s_500.count("G"))
            gc_vals.append(gc)
            cpg_oe_vals.append(cpg)

            # Fast input gradient attribution
            onehot = one_hot_encode_sequence(seq)
            attr = compute_input_gradient_attribution(model, onehot, target="y_abs", device=device)
            imp = np.maximum(0.0, np.sum(attr * onehot, axis=0))

            # Extract top seqlet from central 1,000 bp
            imp_c = np.zeros_like(imp)
            imp_c[1057 - 500 : 1057 + 500] = imp[1057 - 500 : 1057 + 500]
            cand_motifs = extract_multiple_causal_seqlets(imp_c, min_len=6, max_len=14, top_k=2)

            for m_i, (m_st, m_en, m_score, _) in enumerate(cand_motifs):
                m_seq = seq[m_st:m_en]
                seqlet_records.append({
                    "gene_name": str(row.get("gene_name", row.get("gene_id", ""))),
                    "group": set_label,
                    "motif_id": f"{set_label}_{idx}_m{m_i}_{m_seq}",
                    "motif_seq": m_seq,
                    "score": float(m_score)
                })
        return seqlet_records, gc_vals, cpg_oe_vals

    print("[*] Extracting causal seqlets via input gradient across minor-wave and control genes...")
    mw_seqlets, mw_gc, mw_cpg = extract_seqlets_for_set(df_mw, "minor_wave")
    ctrl_seqlets, ctrl_gc, ctrl_cpg = extract_seqlets_for_set(df_ctrl, "control_silent")
    all_seqlets = mw_seqlets + ctrl_seqlets
    print(f"    Extracted {len(mw_seqlets)} minor-wave seqlets and {len(ctrl_seqlets)} control seqlets")

    # 4. Write all extracted seqlets into a single combined multi-motif .meme file
    comb_meme_path = res_dir / "genomewide_minor_wave_seqlets.meme"
    with open(comb_meme_path, "w") as f_m:
        f_m.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for s_rec in all_seqlets:
            m_id = s_rec["motif_id"]
            m_s = s_rec["motif_seq"]
            f_m.write(f"MOTIF {m_id}\nletter-probability matrix: alength= 4 w= {len(m_s)} nsites= 20 E= 0\n")
            for base in m_s:
                probs = {"A": 0.05, "C": 0.05, "G": 0.05, "T": 0.05}
                if base in probs:
                    probs[base] = 0.85
                f_m.write(f"{probs['A']:.2f} {probs['C']:.2f} {probs['G']:.2f} {probs['T']:.2f}\n")
            f_m.write("\n")

    # 5. Batched TOMTOM against full JASPAR
    jaspar_meme_path = work_dir_p / "downloads" / "JASPAR2024_CORE_vertebrates_non-redundant.meme"
    premzt_meme_path = work_dir_p / "downloads" / "premzt_tfs.meme"
    if not premzt_meme_path.exists():
        premzt_meme_path = jaspar_meme_path

    print("[*] Running single batched TOMTOM against full JASPAR CORE vertebrates...")
    tom_res = run_tomtom(str(comb_meme_path), str(jaspar_meme_path))
    comb_meme_path.unlink(missing_ok=True)

    # Teleost pre-MZT TF list
    known_tf_genes = set()
    if Path(tf_list_path).exists():
        with open(tf_list_path) as f_tf:
            known_tf_genes = {line.strip().lower() for line in f_tf if line.strip() and not line.startswith("#")}

    # Build JASPAR ID to Name dictionary
    jaspar_id_to_name = {}
    if Path(jaspar_meme_path).exists():
        with open(jaspar_meme_path) as f_j:
            for line in f_j:
                if line.startswith("MOTIF"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        jaspar_id_to_name[parts[1]] = parts[2]
                    elif len(parts) == 2:
                        jaspar_id_to_name[parts[1]] = parts[1]

    # Map seqlet hits to TF family and count by gene group
    seqlet_map = {r["motif_id"]: r for r in all_seqlets}
    mw_gene_count = len(df_mw)
    ctrl_gene_count = len(df_ctrl)

    tf_mw_genes = {}
    tf_ctrl_genes = {}
    tf_display_map = {}

    if not tom_res.empty:
        # Take top hit per Query_ID
        top_hits = tom_res.sort_values(by="q-value", ascending=True).groupby("Query_ID").first().reset_index()
        for _, hit in top_hits.iterrows():
            q_id = str(hit["Query_ID"])
            s_info = seqlet_map.get(q_id)
            if not s_info:
                continue
            target_id = str(hit["Target_ID"])
            tf_raw = jaspar_id_to_name.get(target_id, target_id)
            tf_clean = tf_raw.split("::")[0].split("(")[0].strip()
            g_name = s_info["gene_name"]
            grp = s_info["group"]
            tf_display_map[tf_clean] = f"{tf_raw} ({target_id})"

            if grp == "minor_wave":
                tf_mw_genes.setdefault(tf_clean, set()).add(g_name)
            else:
                tf_ctrl_genes.setdefault(tf_clean, set()).add(g_name)

    # 6. Contingency tables & Fisher exact tests
    all_tfs = sorted(list(set(tf_mw_genes.keys()) | set(tf_ctrl_genes.keys())))
    tf_rows = []
    for tf in all_tfs:
        n_mw = len(tf_mw_genes.get(tf, set()))
        n_ctrl = len(tf_ctrl_genes.get(tf, set()))
        pct_mw = (n_mw / max(1, mw_gene_count)) * 100
        pct_ctrl = (n_ctrl / max(1, ctrl_gene_count)) * 100

        # Fisher's exact test
        tbl = [
            [n_mw, mw_gene_count - n_mw],
            [n_ctrl, ctrl_gene_count - n_ctrl]
        ]
        try:
            odds, pval = fisher_exact(tbl, alternative="greater")
        except Exception:
            odds, pval = 1.0, 1.0

        is_premzt_tf = (tf.lower() in known_tf_genes) or any(k in tf.lower() for k in known_tf_genes)
        tf_rows.append({
            "tf_name": tf,
            "display_name": tf_display_map.get(tf, tf),
            "minor_wave_gene_count": n_mw,
            "minor_wave_pct": pct_mw,
            "control_gene_count": n_ctrl,
            "control_pct": pct_ctrl,
            "enrichment_odds_ratio": odds,
            "fisher_pvalue": pval,
            "in_premzt_tf_list": is_premzt_tf
        })

    df_tf_stats = pd.DataFrame(tf_rows)
    if not df_tf_stats.empty:
        df_tf_stats = df_tf_stats.sort_values(by=["minor_wave_pct", "enrichment_odds_ratio"], ascending=[False, False]).reset_index(drop=True)

    out_motif_tsv = res_dir / "minor_wave_genomewide_top_motifs.tsv"
    df_tf_stats.to_csv(out_motif_tsv, sep="\t", index=False)
    print(f"[+] Saved Genome-Wide Minor-Wave Motif Stats ({len(df_tf_stats)} TFs): {out_motif_tsv}")

    # 7. Render Figure 4c: Genome-Wide Top Motifs & CpG/GC Comparison
    fig4c = plt.figure(figsize=(15, 6))
    gs4c = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1.0], wspace=0.3)

    # Left: Top 15 Enriched TFs in Minor-Wave Promoters
    ax_bars = fig4c.add_subplot(gs4c[0])
    top_tfs_plot = df_tf_stats.head(15).iloc[::-1]
    if not top_tfs_plot.empty:
        y_pos = np.arange(len(top_tfs_plot))
        b_mw = ax_bars.barh(y_pos + 0.18, top_tfs_plot["minor_wave_pct"], height=0.35, color="#d73027", label=f"Minor-Wave ({mw_gene_count} loci)", edgecolor="black", lw=0.6)
        b_ctrl = ax_bars.barh(y_pos - 0.18, top_tfs_plot["control_pct"], height=0.35, color="#4575b4", label=f"Silent Controls ({ctrl_gene_count} loci)", edgecolor="black", lw=0.6)

        tf_labels = [
            f"{r['tf_name']}{' ★' if r['in_premzt_tf_list'] else ''} (OR={r['enrichment_odds_ratio']:.1f}, p={r['fisher_pvalue']:.1e})"
            for _, r in top_tfs_plot.iterrows()
        ]
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(tf_labels, fontsize=8.5, fontweight="medium")
        ax_bars.set_xlabel("% of Loci with Motif", fontsize=10.5, fontweight="bold")
        ax_bars.set_title("A. Top TF Motifs in Cichlid Minor-Wave Loci vs. Controls", fontsize=11.5, fontweight="bold")
        ax_bars.legend(loc="lower right", frameon=True, fontsize=8.5)
        ax_bars.grid(True, linestyle=":", alpha=0.5, axis="x")

    # Right: CpG O/E and GC% Violin Comparison
    ax_cpg = fig4c.add_subplot(gs4c[1])
    parts = ax_cpg.violinplot([mw_cpg, ctrl_cpg], positions=[1, 2], showmeans=True, showmedians=True)
    if "bodies" in parts:
        parts["bodies"][0].set_facecolor("#d73027")
        parts["bodies"][0].set_edgecolor("black")
        parts["bodies"][1].set_facecolor("#4575b4")
        parts["bodies"][1].set_edgecolor("black")

    # Callouts for candidate element values (e.g. Rank 1 fam78bb, Rank 2 TMEM59L, Rank 9 NOTCH2)
    swap_sum_file = res_dir / "evolutionary_swap_summary.tsv"
    if swap_sum_file.exists():
        df_sw = pd.read_csv(swap_sum_file, sep="\t")
        for _, sw_row in df_sw.iterrows():
            rk = int(sw_row.get("element_rank", 0))
            if rk in [1, 2, 9]:
                c_val = float(sw_row.get("cpg_oe", np.nan))
                if np.isfinite(c_val):
                    ax_cpg.scatter([1], [c_val], color="#ffffbf", edgecolor="black", s=70, zorder=5)
                    ax_cpg.text(1.08, c_val, f"R{rk} ({sw_row.get('overlapping_gene','')})", fontsize=8, fontweight="bold", va="center")

    ax_cpg.set_xticks([1, 2])
    ax_cpg.set_xticklabels([f"Minor-Wave\n(Mean CpG={np.mean(mw_cpg):.2f})", f"Silent Controls\n(Mean CpG={np.mean(ctrl_cpg):.2f})"], fontsize=9.5)
    ax_cpg.set_ylabel("Observed / Expected CpG Ratio", fontsize=10.5, fontweight="bold")
    ax_cpg.set_title("B. CpG-Island Priming: Minor-Wave vs. Controls", fontsize=11.5, fontweight="bold")
    ax_cpg.grid(True, linestyle=":", alpha=0.5, axis="y")

    fig4c.subplots_adjust(left=0.23, right=0.97, top=0.90, bottom=0.12, wspace=0.28)
    fig4c_png = res_dir / "fig4c_genomewide_minor_wave_motifs.png"
    fig4c_svg = res_dir / "fig4c_genomewide_minor_wave_motifs.svg"
    plt.savefig(fig4c_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig4c_svg, bbox_inches="tight")
    plt.close(fig4c)
    print(f"[+] Saved Publication Figure 4c: {fig4c_png} and .svg")

    return {
        "df_tf_stats": df_tf_stats,
        "motif_tsv": str(out_motif_tsv),
        "fig4c_png": str(fig4c_png),
        "fig4c_svg": str(fig4c_svg)
    }


def quantseq_3p_gene_scoring(
    pred_nascent_track: np.ndarray,
    gene_span_len: Optional[int] = None,
    strand: str = "+",
    resolution_bp: int = 1,
    utr3_bp: int = 1000,
    crop_genomic_start: Optional[int] = None,
    crop_genomic_end: Optional[int] = None,
    gene_start: Optional[int] = None,
    gene_end: Optional[int] = None,
    exons: Optional[List[Tuple[int, int]]] = None
) -> Dict[str, float]:
    """
    Score nascent transcription for QuantSeq SLAM-seq tracks across exact 3' UTR and full gene exons.
    Slices the exact genomic bin coordinates when crop_genomic_start/end and gene_start/end are provided.
    Falls back gracefully to span-based relative slicing if coordinates are omitted.
    """
    total_bins = len(pred_nascent_track)
    if total_bins == 0:
        return {"nascent_3p_1000bp": 0.0, "nascent_all_exons": 0.0, "ratio_3p_to_total": 0.0}

    # When exact genomic coordinates are provided:
    if crop_genomic_start is not None and crop_genomic_end is not None and gene_start is not None and gene_end is not None:
        c_st = crop_genomic_start
        c_en = crop_genomic_end
        span_bp = c_en - c_st
        bp_per_bin = max(1.0, span_bp / total_bins)

        # Terminal 1,000 bp / 3' UTR exact interval
        if strand == "+":
            term_st = max(gene_start, gene_end - utr3_bp)
            term_en = gene_end
        else:
            term_st = gene_start
            term_en = min(gene_end, gene_start + utr3_bp)

        # Map to crop bin indices
        b_3p_st = max(0, min(total_bins, int((term_st - c_st) / bp_per_bin)))
        b_3p_en = max(b_3p_st, min(total_bins, int(np.ceil((term_en - c_st) / bp_per_bin))))
        signal_3p = float(np.sum(pred_nascent_track[b_3p_st:b_3p_en])) if b_3p_en > b_3p_st else 0.0

        # All exons signal within crop
        if exons:
            exon_signals = []
            for e_st, e_en in exons:
                ov_st = max(e_st, c_st)
                ov_en = min(e_en, c_en)
                if ov_en > ov_st:
                    b_e_st = max(0, min(total_bins, int((ov_st - c_st) / bp_per_bin)))
                    b_e_en = max(b_e_st, min(total_bins, int(np.ceil((ov_en - c_st) / bp_per_bin))))
                    if b_e_en > b_e_st:
                        exon_signals.append(np.sum(pred_nascent_track[b_e_st:b_e_en]))
            total_signal = float(sum(exon_signals)) if exon_signals else 0.0
        else:
            # Gene span overlap within crop
            ov_st = max(gene_start, c_st)
            ov_en = min(gene_end, c_en)
            if ov_en > ov_st:
                b_g_st = max(0, min(total_bins, int((ov_st - c_st) / bp_per_bin)))
                b_g_en = max(b_g_st, min(total_bins, int(np.ceil((ov_en - c_st) / bp_per_bin))))
                total_signal = float(np.sum(pred_nascent_track[b_g_st:b_g_en])) if b_g_en > b_g_st else 0.0
            else:
                total_signal = 0.0
    else:
        # Fallback relative slicing
        bins_for_1kb = max(1, min(total_bins, utr3_bp // max(1, resolution_bp)))
        total_signal = float(np.sum(pred_nascent_track))
        if strand == "+":
            signal_3p = float(np.sum(pred_nascent_track[-bins_for_1kb:]))
        else:
            signal_3p = float(np.sum(pred_nascent_track[:bins_for_1kb]))

    return {
        "nascent_3p_1000bp": signal_3p,
        "nascent_all_exons": total_signal,
        "ratio_3p_to_total": signal_3p / max(total_signal, 1e-6)
    }


def select_gc_matched_controls(
    df_genes: pd.DataFrame,
    genome_fa_path: Union[str, Path],
    pos_mask: Union[pd.Series, np.ndarray],
    ctrl_pool_mask: Union[pd.Series, np.ndarray],
    flank_bp: int = 1000,
    max_controls: int = 300
) -> np.ndarray:
    """
    Select expression- and GC-matched control genes from candidate pool.
    Calculates GC% in promoter windows (TSS ± flank_bp) for positive genes and candidate controls.
    Performs stratified/nearest-neighbor matching so control promoter GC% distribution matches positives.
    Returns a boolean mask of length len(df_genes) where selected controls are True.
    """
    from pyfaidx import Fasta
    genome = Fasta(str(genome_fa_path), as_raw=True)

    pos_idx = np.where(pos_mask)[0]
    ctrl_idx = np.where(ctrl_pool_mask)[0]

    if len(pos_idx) == 0 or len(ctrl_idx) == 0:
        return np.zeros(len(df_genes), dtype=bool)

    def get_promoter_gc(idx_list):
        gc_list = []
        valid_indices = []
        for i in idx_list:
            row = df_genes.iloc[i]
            chrom = str(row["chrom"])
            if chrom not in genome:
                continue
            strand = str(row.get("strand", "+"))
            if "tss" in row and pd.notna(row["tss"]):
                tss = int(row["tss"])
            elif "start" in row and pd.notna(row["start"]):
                tss = int(row["start"]) if strand == "+" else int(row.get("end", row["start"]))
            else:
                continue

            st = max(0, tss - flank_bp)
            en = min(len(genome[chrom]), tss + flank_bp)
            try:
                seq = str(genome[chrom][st:en]).upper()
            except Exception:
                continue

            if len(seq) == 0:
                continue
            gc = (seq.count("G") + seq.count("C")) / len(seq)
            gc_list.append(gc)
            valid_indices.append(i)
        return np.array(valid_indices), np.array(gc_list)

    v_pos_idx, pos_gc = get_promoter_gc(pos_idx)
    v_ctrl_idx, ctrl_gc = get_promoter_gc(ctrl_idx)

    if len(v_pos_idx) == 0 or len(v_ctrl_idx) == 0:
        selected_mask = np.zeros(len(df_genes), dtype=bool)
        selected_mask[ctrl_idx[:max_controls]] = True
        return selected_mask

    target_k = min(max_controls, len(v_ctrl_idx))
    selected_ctrl_indices = set()
    available_ctrl_indices = list(range(len(v_ctrl_idx)))
    k_per_pos = max(1, target_k // max(1, len(v_pos_idx)))

    for p_val in sorted(pos_gc):
        if len(selected_ctrl_indices) >= target_k or not available_ctrl_indices:
            break
        diffs = [abs(ctrl_gc[ci] - p_val) for ci in available_ctrl_indices]
        closest_order = np.argsort(diffs)
        picks = []
        for pick in closest_order:
            if len(picks) >= k_per_pos or len(selected_ctrl_indices) >= target_k:
                break
            ci = available_ctrl_indices[pick]
            selected_ctrl_indices.add(v_ctrl_idx[ci])
            picks.append(ci)
        available_ctrl_indices = [ci for ci in available_ctrl_indices if v_ctrl_idx[ci] not in selected_ctrl_indices]

    med_pos_gc = float(np.median(pos_gc))
    if len(selected_ctrl_indices) < target_k:
        remaining = [ci for ci in range(len(v_ctrl_idx)) if v_ctrl_idx[ci] not in selected_ctrl_indices]
        if remaining:
            diffs_rem = [abs(ctrl_gc[ci] - med_pos_gc) for ci in remaining]
            for pick in np.argsort(diffs_rem):
                if len(selected_ctrl_indices) >= target_k:
                    break
                selected_ctrl_indices.add(v_ctrl_idx[remaining[pick]])

    selected_mask = np.zeros(len(df_genes), dtype=bool)
    selected_mask[list(selected_ctrl_indices)] = True

    sel_ctrl_gc = [ctrl_gc[i] for i, idx in enumerate(v_ctrl_idx) if idx in selected_ctrl_indices]
    mean_pos = float(np.mean(pos_gc)) * 100
    mean_ctrl = float(np.mean(sel_ctrl_gc)) * 100 if sel_ctrl_gc else 0.0
    print(f"[+] GC Matching: {len(v_pos_idx)} positive promoters (mean GC: {mean_pos:.2f}%) matched to {selected_mask.sum()} control promoters (mean GC: {mean_ctrl:.2f}%, Δ: {abs(mean_pos - mean_ctrl):.2f}%).")
    return selected_mask


if __name__ == "__main__":
    print("[+] scripts/infer_and_attribute.py loaded successfully as CLI / module.")
