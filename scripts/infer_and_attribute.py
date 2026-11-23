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
    payload = torch.load(str(p), map_location=device, weights_only=True)

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

    # Process in batches
    for start_idx in range(0, n_summits, batch_size):
        end_idx = min(start_idx + batch_size, n_summits)
        sub_df = df_summits.iloc[start_idx:end_idx]

        batch_tensors = []
        for _, row in sub_df.iterrows():
            chrom = str(row["chrom"])
            summit = int(row["summit"])
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

def normalize_teleost_tf_name(name: str) -> str:
    """Normalize teleost gene symbols: strip trailing 'a'/'b' and map common aliases."""
    raw = name.strip()
    upper = raw.upper()

    alias_map = {
        "POU5F3": "POU5F1", "OCT4": "POU5F1",
        "MXTX2": "MIXL1",
        "VOX": "VENTX", "VENT": "VENTX", "VED": "VENTX",
        "SOX19B": "SOX2", "SOX19A": "SOX2", "SOXB1": "SOX2",
        "NANOG": "NANOG", "FOXH1": "FOXH1",
        "NOTCH2": "NOTCH2", "SLC35A3A": "SLC35A3", "SLC35A3B": "SLC35A3",
        "KLF17": "KLF17", "KLF4": "KLF4"
    }
    if upper in alias_map:
        return alias_map[upper]

    match = re.match(r"^([A-Za-z0-9]+?)[abAB]$", raw)
    if match and len(match.group(1)) >= 3:
        stem = match.group(1).upper()
        return alias_map.get(stem, stem)

    return upper


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

    x = torch.tensor(one_hot_seq, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad = True

    out = grafted_model(x, rc_average=False)
    score = out[target].sum()
    score.backward()

    grad = x.grad.squeeze(0).cpu().numpy()
    attr_fwd = one_hot_seq * grad  # shape (4, L)

    if not rc_average:
        return attr_fwd

    grafted_model.zero_grad()
    seq_rc = np.flip(one_hot_seq, axis=-1)[[3, 2, 1, 0], :]
    x_rc = torch.tensor(seq_rc, dtype=torch.float32, device=device).unsqueeze(0)
    x_rc.requires_grad = True

    out_rc = grafted_model(x_rc, rc_average=False)
    score_rc = out_rc[target].sum()
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

        diff = x_np - x_base
        grads = []
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            x_step = x_base + alpha * diff
            x_tensor = torch.tensor(x_step, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
            out = grafted_model(x_tensor, rc_average=False)
            score = out[target].sum()
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
    min_len: int = 8,
    max_len: int = 18,
    peak_center: Optional[int] = None
) -> Tuple[int, int, float]:
    """
    Select the contiguous causal seqlet of length between min_len and max_len
    that captures the core motif footprint without truncating heterodimers (e.g. Pou5f3::Sox19b).
    Returns (start_idx, end_idx, score).
    """
    importance = np.array(wt_importance, dtype=np.float32)
    L = len(importance)
    if peak_center is None:
        peak_center = int(np.argmax(importance))

    best_score = -float("inf")
    best_start = max(0, peak_center - min_len // 2)
    best_end = min(L, best_start + min_len)

    for w in range(min_len, min(max_len + 1, L + 1)):
        s_min = max(0, peak_center - w + 1)
        s_max = min(L - w, peak_center)
        for s in range(s_min, s_max + 1):
            e = s + w
            sub = importance[s:e]
            score = float(np.sum(sub) / np.sqrt(w))
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    return best_start, best_end, best_score


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

    x_wt = torch.tensor(wt_one_hot, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        score_wt = float(grafted_model(x_wt)[target].item())

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
            out = grafted_model(b_x)
            mut_scores.extend(out[target].cpu().numpy().tolist())

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
    ppm_or_pwm: np.ndarray,
    motif_id: str,
    out_meme_path: str | Path,
    pseudocount: float = 0.01,
    beta: Optional[float] = None
) -> str:
    """
    Write a motif probability matrix (shape 4 x L or L x 4) to minimal standard MEME format.
    If beta is supplied or values are signed/negative, builds PPM directly from signed ISM matrix:
    PPM = softmax(beta * ism_delta).
    """
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


def annotate_motif_hits(
    tomtom_df: pd.DataFrame,
    tf_list_path: Optional[str | Path] = None,
    rnaseq_cpm_dict: Optional[Dict[str, float]] = None,
    target_meme_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Annotate TOMTOM motif hits with pre-MZT TF status and maternal RNA expression.
    Parses MOTIF <matrix_id> <tf_name> from target_meme_path to map JASPAR IDs to TF symbols,
    and checks all partners in :: heterodimers.
    """
    if tomtom_df.empty:
        df = tomtom_df.copy()
        df["TF_name"] = pd.Series(dtype=str)
        df["is_premzt_tf"] = pd.Series(dtype=bool)
        df["maternal_cpm"] = pd.Series(dtype=float)
        return df

    df = tomtom_df.copy()
    tf_list_set = set()
    if tf_list_path and Path(tf_list_path).exists():
        with open(tf_list_path) as f:
            for line in f:
                s = line.strip()
                if s:
                    tf_list_set.add(normalize_teleost_tf_name(s))
                    tf_list_set.add(s.upper())

    # Build matrix_id -> tf_name map from target .meme file if available
    motif_to_tf: Dict[str, str] = {}
    if target_meme_path and Path(target_meme_path).exists():
        with open(target_meme_path) as mf:
            for line in mf:
                if line.startswith("MOTIF"):
                    p = line.strip().split()
                    if len(p) >= 3:
                        motif_to_tf[p[1]] = p[2]
                        motif_to_tf[p[1].upper()] = p[2]
                        motif_to_tf[f"{p[1]}_{p[2]}"] = p[2]
                        motif_to_tf[f"{p[1]}_{p[2]}".upper()] = p[2]
                    elif len(p) == 2:
                        motif_to_tf[p[1]] = p[1]
                        motif_to_tf[p[1].upper()] = p[1]

    # Build a normalized lookup map from rnaseq_cpm_dict indexing BOTH raw lowercase gene symbols
    # AND normalize_teleost_tf_name(k).lower(), taking MAX CPM across paralogs (e.g. sox19a/b -> sox2, pou5f3 -> pou5f1)
    norm_cpm_dict: Dict[str, float] = {}
    if rnaseq_cpm_dict:
        for k, v in rnaseq_cpm_dict.items():
            k_raw = str(k).strip().lower()
            k_norm = normalize_teleost_tf_name(str(k)).strip().lower()
            val = float(v)
            norm_cpm_dict[k_raw] = max(norm_cpm_dict.get(k_raw, 0.0), val)
            norm_cpm_dict[k_norm] = max(norm_cpm_dict.get(k_norm, 0.0), val)

    tf_names = []
    is_premzt_list = []
    maternal_cpms = []
    for _, row in df.iterrows():
        t_id = str(row.get("Target_ID", ""))
        mapped_name = motif_to_tf.get(t_id, motif_to_tf.get(t_id.upper(), t_id))
        if mapped_name == t_id and "_" in t_id:
            parts = t_id.split("_", 1)
            if parts[0].startswith("MA") or parts[0].startswith("UN"):
                mapped_name = parts[1]
        tf_names.append(mapped_name)

        # Check all partners in heterodimers (e.g. POU5F1::SOX2 or Pou5f3::Sox19b)
        partners = [p.strip() for p in mapped_name.split("::")]

        is_premzt = False
        cpm_vals = []
        for p in partners:
            p_norm = normalize_teleost_tf_name(p)
            if (p_norm in tf_list_set) or (p.upper() in tf_list_set) or (p_norm.upper() in tf_list_set) or (p.lower() in [s.lower() for s in tf_list_set]):
                is_premzt = True
            if norm_cpm_dict:
                c = max(
                    norm_cpm_dict.get(p.lower(), 0.0),
                    norm_cpm_dict.get(p_norm.lower(), 0.0)
                )
                cpm_vals.append(c)

        is_premzt_list.append(is_premzt)
        maternal_cpms.append(max(cpm_vals) if cpm_vals else 0.0)

    df["TF_name"] = tf_names
    df["is_premzt_tf"] = is_premzt_list
    df["maternal_cpm"] = maternal_cpms
    return df


def filter_jaspar_meme(
    jaspar_meme_path: str,
    tf_list_path: Optional[str],
    out_filtered_meme: str
) -> Tuple[str, List[str], List[str]]:
    """
    Filter full JASPAR .meme file against pre-MZT TF list, logging matched vs. unmatched TFs.
    Checks case-insensitively and inspects all partners in :: heterodimers.
    """
    if not tf_list_path or not os.path.exists(tf_list_path):
        return jaspar_meme_path, [], []

    with open(tf_list_path, "r") as f:
        tf_symbols = [line.strip() for line in f if line.strip()]

    target_tfs_lower = set(normalize_teleost_tf_name(s).lower() for s in tf_symbols)
    target_raw_lower = set(s.lower() for s in tf_symbols)

    matched_tfs = set()
    filtered_motifs = []

    with open(jaspar_meme_path, "r") as f:
        content = f.read()

    motif_splits = re.split(r"(?=MOTIF\s+)", content)
    header_lines = motif_splits[0]

    for block in motif_splits[1:]:
        match = re.search(r"MOTIF\s+([^\s]+)\s+([^\s\n]+)", block)
        if match:
            motif_id, motif_name = match.group(1), match.group(2)
            partners = motif_name.split("::")
            matched = False
            for p in partners:
                p_norm = normalize_teleost_tf_name(p).lower()
                p_raw = p.lower()
                if p_norm in target_tfs_lower or p_raw in target_raw_lower:
                    matched = True
                    break

            if matched:
                filtered_motifs.append(block)
                matched_tfs.add(motif_name)

    with open(out_filtered_meme, "w") as f:
        f.write(header_lines)
        for m in filtered_motifs:
            f.write(m)

    matched_all_partners_lower = set(
        normalize_teleost_tf_name(p).lower()
        for m in matched_tfs
        for p in m.split("::")
    ) | set(p.lower() for m in matched_tfs for p in m.split("::"))

    unmatched_tfs = [
        s for s in tf_symbols
        if normalize_teleost_tf_name(s).lower() not in matched_all_partners_lower
        and s.lower() not in matched_all_partners_lower
    ]
    print(f"[+] Built filtered MEME: {out_filtered_meme} ({len(filtered_motifs)} motifs matched across {len(matched_tfs)} TFs).")
    return out_filtered_meme, sorted(list(matched_tfs)), sorted(list(set(unmatched_tfs)))


def perform_evolutionary_alignment_and_swap(
    model: GraftedChromBPNet,
    target_window_seq: str,
    ortholog_locus_seq: str,
    motif_start_in_window: int,
    motif_len: int,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Perform sequence alignment of the target window against the orthologous locus.
    Searches BOTH forward and reverse-complement strands of ortholog_locus_seq.
    Uses minimap2 / blastn / binned k-mer density seeding to locate the best syntenic window
    before fine alignment with PairwiseAligner.
    Unifies return key 'ortholog_swap_delta'.
    """
    from Bio import Align
    model.to(device)
    model.eval()

    def seq_to_tensor(s):
        mat = one_hot_encode_sequence(s)
        return torch.tensor(mat, dtype=torch.float32, device=device).unsqueeze(0)

    # 1. Score Wild-Type target window (both y_abs and y_rel)
    with torch.no_grad():
        out_wt = model(seq_to_tensor(target_window_seq), compute_late=True)
        score_target_wt = float(out_wt["y_abs"].squeeze().cpu().item())
        score_target_wt_rel = float(out_wt["y_rel"].squeeze().cpu().item())

    # 2. Strict in silico knockout / ablation control (N-masking)
    target_ablated = list(target_window_seq)
    target_ablated[motif_start_in_window:motif_start_in_window + motif_len] = ["N"] * motif_len
    target_ablated_seq = "".join(target_ablated)
    with torch.no_grad():
        out_abl = model(seq_to_tensor(target_ablated_seq), compute_late=True)
        score_target_ablation = float(out_abl["y_abs"].squeeze().cpu().item())
        score_target_ablation_rel = float(out_abl["y_rel"].squeeze().cpu().item())

    ablation_delta = score_target_ablation - score_target_wt
    ablation_delta_rel = score_target_ablation_rel - score_target_wt_rel

    # 3. Align target flanking region against orthologous locus sequence (both strands)
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

    # Detect syntenic homology using minimap2 / blastn with sensitive parameters
    best_strand = "fwd"
    search_seq = ortholog_locus_seq
    ortho_rc = None
    slice_offset = 0
    found_homology = False

    # Try minimap2 first
    minimap2_bin = shutil.which("minimap2") or str(Path(sys.prefix) / "bin" / "minimap2")
    if minimap2_bin and Path(minimap2_bin).exists() and os.access(minimap2_bin, os.X_OK):
        import tempfile
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_ref, \
                 tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_qry:
                ref_path = f_ref.name
                qry_path = f_qry.name
                f_ref.write(f">ortho_locus\n{ortholog_locus_seq}\n")
                f_qry.write(f">target_flank\n{target_flank}\n")

            mm2_cmd = [minimap2_bin, "-c", "-k", "9", "-w", "4", "-A", "2", "-B", "2", "-O", "3", "-E", "1", ref_path, qry_path]
            res_mm2 = subprocess.run(mm2_cmd, capture_output=True, text=True, check=False)
            if res_mm2.returncode == 0 and res_mm2.stdout.strip():
                paf_lines = res_mm2.stdout.strip().split("\n")
                best_hit = None
                max_matches = 0
                for pl in paf_lines:
                    parts = pl.split("\t")
                    if len(parts) >= 11:
                        matches = int(parts[9])
                        if matches > max_matches:
                            max_matches = matches
                            best_hit = parts
                if best_hit and max_matches >= 15:
                    strand_hit = best_hit[4]
                    t_st = int(best_hit[7])
                    t_en = int(best_hit[8])
                    center_pos = (t_st + t_en) // 2
                    if strand_hit == "-":
                        best_strand = "rev"
                        ortho_rc = revcomp(ortholog_locus_seq)
                        center_rc = len(ortholog_locus_seq) - center_pos
                        st = max(0, center_rc - 1500)
                        en = min(len(ortho_rc), center_rc + 1500)
                        search_seq = ortho_rc[st:en]
                        slice_offset = st
                    else:
                        best_strand = "fwd"
                        st = max(0, center_pos - 1500)
                        en = min(len(ortholog_locus_seq), center_pos + 1500)
                        search_seq = ortholog_locus_seq[st:en]
                        slice_offset = st
                    found_homology = True
        except Exception:
            pass
        finally:
            if 'ref_path' in locals():
                Path(ref_path).unlink(missing_ok=True)
            if 'qry_path' in locals():
                Path(qry_path).unlink(missing_ok=True)

    # Fallback to blastn if minimap2 produced no hit
    if not found_homology:
        blastn_bin = shutil.which("blastn") or str(Path(sys.prefix) / "bin" / "blastn")
        if blastn_bin and Path(blastn_bin).exists() and os.access(blastn_bin, os.X_OK):
            import tempfile
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_ref, \
                     tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f_qry:
                    ref_path = f_ref.name
                    qry_path = f_qry.name
                    f_ref.write(f">ortho_locus\n{ortholog_locus_seq}\n")
                    f_qry.write(f">target_flank\n{target_flank}\n")

                cmd_blast = [blastn_bin, "-query", qry_path, "-subject", ref_path, "-task", "blastn", "-word_size", "7", "-evalue", "10", "-outfmt", "6 sstart send sstrand score length"]
                res_b = subprocess.run(cmd_blast, capture_output=True, text=True, check=False)
                if res_b.returncode == 0 and res_b.stdout.strip():
                    first_hit = res_b.stdout.strip().split("\n")[0].split("\t")
                    if len(first_hit) >= 4:
                        s_st = min(int(first_hit[0]), int(first_hit[1]))
                        s_en = max(int(first_hit[0]), int(first_hit[1]))
                        s_strand = first_hit[2]
                        center_pos = (s_st + s_en) // 2
                        if "minus" in s_strand.lower():
                            best_strand = "rev"
                            ortho_rc = revcomp(ortholog_locus_seq)
                            center_rc = len(ortholog_locus_seq) - center_pos
                            st = max(0, center_rc - 1500)
                            en = min(len(ortho_rc), center_rc + 1500)
                            search_seq = ortho_rc[st:en]
                            slice_offset = st
                        else:
                            best_strand = "fwd"
                            st = max(0, center_pos - 1500)
                            en = min(len(ortholog_locus_seq), center_pos + 1500)
                            search_seq = ortholog_locus_seq[st:en]
                            slice_offset = st
                        found_homology = True
            except Exception:
                pass
            finally:
                if 'ref_path' in locals():
                    Path(ref_path).unlink(missing_ok=True)
                if 'qry_path' in locals():
                    Path(qry_path).unlink(missing_ok=True)

    # Sensitive k-mer search fallback (mode density bin instead of global median, k=7 for ~220 MYA teleost divergence)
    if not found_homology:
        k_len = 7
        kmers = [
            target_flank[i:i + k_len]
            for i in range(0, len(target_flank) - k_len, 2)
            if "N" not in target_flank[i:i + k_len]
        ]
        ortho_rc = revcomp(ortholog_locus_seq)
        hits_fwd = []
        hits_rev = []
        for km in kmers:
            for m in re.finditer(re.escape(km), ortholog_locus_seq, re.IGNORECASE):
                hits_fwd.append(m.start())
            for m in re.finditer(re.escape(km), ortho_rc, re.IGNORECASE):
                hits_rev.append(m.start())

        def _best_density_center(hits: List[int], seq_len: int, bin_bp: int = 1000) -> Tuple[int, int]:
            if not hits:
                return 0, seq_len // 2
            bins = np.array(hits, dtype=np.int64) // bin_bp
            uniq, counts = np.unique(bins, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_bin = int(uniq[best_idx])
            in_bin = [h for h in hits if (h // bin_bp) == best_bin]
            return int(counts[best_idx]), int(np.median(in_bin))

        fwd_peak_cnt, fwd_center = _best_density_center(hits_fwd, len(ortholog_locus_seq))
        rev_peak_cnt, rev_center = _best_density_center(hits_rev, len(ortho_rc))

        if rev_peak_cnt > fwd_peak_cnt and rev_peak_cnt > 0:
            best_strand = "rev"
            st = max(0, rev_center - 1500)
            en = min(len(ortho_rc), rev_center + 1500)
            search_seq = ortho_rc[st:en]
            slice_offset = st
        elif fwd_peak_cnt > 0:
            best_strand = "fwd"
            st = max(0, fwd_center - 1500)
            en = min(len(ortholog_locus_seq), fwd_center + 1500)
            search_seq = ortholog_locus_seq[st:en]
            slice_offset = st
        else:
            mid = len(ortholog_locus_seq) // 2
            st = max(0, mid - 5000)
            en = min(len(ortholog_locus_seq), mid + 5000)
            search_seq = ortholog_locus_seq[st:en]
            slice_offset = st

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
                match_bar.append(".")

        return "".join(aligned_t), "".join(match_bar), "".join(aligned_o)

    alignments = aligner.align(search_seq, target_flank)

    is_homologous = False
    best_aln = None
    if len(alignments) > 0:
        best_aln = alignments[0]
        idx_ortho = best_aln.indices[0]
        idx_target = best_aln.indices[1]
        aligned_cols = (idx_ortho >= 0) & (idx_target >= 0)
        aligned_len = int(np.sum(aligned_cols))
        n_matches = sum(
            1 for c in np.where(aligned_cols)[0]
            if search_seq[idx_ortho[c]].upper() == target_flank[idx_target[c]].upper()
        )
        identity = n_matches / max(1, aligned_len)
        if best_aln.score >= 60 and aligned_len >= 40 and identity >= 0.55:
            is_homologous = True

    if not is_homologous:
        aln_t, aln_bar, aln_o = _extract_aligned_motif_window(
            aln=best_aln,
            search_seq=search_seq,
            target_flank=target_flank,
            motif_rel_start=motif_rel_start,
            motif_len=motif_len,
            half_window=25
        )
        is_lineage_gain = bool(ablation_delta < -0.05)
        return {
            "status": "insertion_in_target_novel_element",
            "alignment_score": float(best_aln.score) if best_aln is not None else 0.0,
            "target_wt": score_target_wt,
            "target_wt_rel": score_target_wt_rel,
            "target_ablation_score": score_target_ablation,
            "target_ablation_score_rel": score_target_ablation_rel,
            "target_ablation_delta": ablation_delta,
            "target_ablation_delta_rel": ablation_delta_rel,
            "classification": "Target-specific novel enhancer (ortholog lacks syntenic alignment)",
            "ortholog_aligned_motif": None,
            "aligned_target_motif_window": aln_t,
            "aligned_match_bar": aln_bar,
            "aligned_ortholog_motif_window": aln_o,
            "target_flank": aln_t,
            "ortholog_flank": aln_o,
            "best_strand": best_strand,
            "is_lineage_gain": is_lineage_gain,
            "target_swapped_with_ortholog": None,
            "target_swapped_with_ortholog_rel": None,
            "target_swap_delta": None,
            "target_swap_delta_rel": None,
            "ortholog_wt": None,
            "ortholog_wt_rel": None,
            "ortholog_swapped_with_target": None,
            "ortholog_swapped_with_target_rel": None,
            "ortholog_swap_delta": None,
            "ortholog_swap_delta_rel": None
        }

    aligned_ortholog_seq = None
    o_motif_start_in_locus = None
    is_indel_swap = False
    indel_diff = 0

    try:
        ortho_coords = best_aln.aligned[0]
        target_coords = best_aln.aligned[1]
        for (o_st, o_en), (t_st, t_en) in zip(ortho_coords, target_coords):
            if t_st <= motif_rel_start and motif_rel_start + motif_len <= t_en:
                offset_st = motif_rel_start - t_st
                local_motif_start = o_st + offset_st
                o_motif_start_in_locus = slice_offset + local_motif_start
                if best_strand == "fwd":
                    aligned_ortholog_seq = ortholog_locus_seq[o_motif_start_in_locus:o_motif_start_in_locus + motif_len]
                else:
                    if ortho_rc is None:
                        ortho_rc = revcomp(ortholog_locus_seq)
                    aligned_ortholog_seq = ortho_rc[o_motif_start_in_locus:o_motif_start_in_locus + motif_len]
                break
    except Exception:
        aligned_ortholog_seq = None

    # Indel-tolerant motif swap: if flank aligns cleanly (>= 20 bp on at least one side) but 1-3 bp indel inside motif
    if aligned_ortholog_seq is None or len(aligned_ortholog_seq) != motif_len:
        motif_rel_end = motif_rel_start + motif_len
        left_flank_aligned = int(np.sum((idx_ortho >= 0) & (idx_target >= 0) & (idx_target < motif_rel_start)))
        right_flank_aligned = int(np.sum((idx_ortho >= 0) & (idx_target >= 0) & (idx_target >= motif_rel_end)))
        if left_flank_aligned >= 20 or right_flank_aligned >= 20:
            motif_cols = np.where((idx_target >= motif_rel_start) & (idx_target < motif_rel_end))[0]
            if len(motif_cols) > 0:
                min_c = min(motif_cols)
                max_c = max(motif_cols)
                ortho_indices_in_motif = [idx_ortho[c] for c in range(min_c, max_c + 1) if idx_ortho[c] >= 0]
                if ortho_indices_in_motif:
                    o_st_local = min(ortho_indices_in_motif)
                    o_en_local = max(ortho_indices_in_motif) + 1
                    indel_span_len = o_en_local - o_st_local
                    if abs(indel_span_len - motif_len) <= 3 and indel_span_len >= 4:
                        o_motif_start_in_locus = slice_offset + o_st_local
                        if best_strand == "fwd":
                            aligned_ortholog_seq = ortholog_locus_seq[o_motif_start_in_locus:o_motif_start_in_locus + indel_span_len]
                        else:
                            if ortho_rc is None:
                                ortho_rc = revcomp(ortholog_locus_seq)
                            aligned_ortholog_seq = ortho_rc[o_motif_start_in_locus:o_motif_start_in_locus + indel_span_len]
                        is_indel_swap = True
                        indel_diff = abs(indel_span_len - motif_len)

    if aligned_ortholog_seq is None:
        aln_t, aln_bar, aln_o = _extract_aligned_motif_window(
            aln=best_aln,
            search_seq=search_seq,
            target_flank=target_flank,
            motif_rel_start=motif_rel_start,
            motif_len=motif_len,
            half_window=25
        )
        return {
            "status": "partial_alignment_motif_gapped",
            "alignment_score": float(best_aln.score),
            "target_wt": score_target_wt,
            "target_wt_rel": score_target_wt_rel,
            "target_ablation_score": score_target_ablation,
            "target_ablation_score_rel": score_target_ablation_rel,
            "target_ablation_delta": ablation_delta,
            "target_ablation_delta_rel": ablation_delta_rel,
            "classification": f"Orthologous locus aligned ({best_strand}) with gap or indel across motif site",
            "ortholog_aligned_motif": None,
            "aligned_target_motif_window": aln_t,
            "aligned_match_bar": aln_bar,
            "aligned_ortholog_motif_window": aln_o,
            "target_flank": aln_t,
            "ortholog_flank": aln_o,
            "best_strand": best_strand,
            "is_lineage_gain": False,
            "target_swapped_with_ortholog": None,
            "target_swapped_with_ortholog_rel": None,
            "target_swap_delta": None,
            "target_swap_delta_rel": None,
            "ortholog_wt": None,
            "ortholog_wt_rel": None,
            "ortholog_swapped_with_target": None,
            "ortholog_swapped_with_target_rel": None,
            "ortholog_swap_delta": None,
            "ortholog_swap_delta_rel": None
        }

    def _adjust_window_len(seq: str, target_len: int = 2114) -> str:
        cur_len = len(seq)
        if cur_len == target_len:
            return seq
        elif cur_len > target_len:
            extra = cur_len - target_len
            left_trim = extra // 2
            right_trim = extra - left_trim
            return seq[left_trim:cur_len - right_trim]
        else:
            short = target_len - cur_len
            left_pad = short // 2
            right_pad = short - left_pad
            return ("N" * left_pad) + seq + ("N" * right_pad)

    raw_swapped = target_window_seq[:motif_start_in_window] + aligned_ortholog_seq + target_window_seq[motif_start_in_window + motif_len:]
    target_swapped_seq = _adjust_window_len(raw_swapped, len(target_window_seq))

    with torch.no_grad():
        out_swapped = model(seq_to_tensor(target_swapped_seq), compute_late=True)
        score_target_swapped = float(out_swapped["y_abs"].squeeze().cpu().item())
        score_target_swapped_rel = float(out_swapped["y_rel"].squeeze().cpu().item())
        target_swap_delta = score_target_swapped - score_target_wt
        target_swap_delta_rel = score_target_swapped_rel - score_target_wt_rel

    # Evaluate reciprocal swap in ortholog window
    w_len = len(target_window_seq)
    real_motif_pos = o_motif_start_in_locus if best_strand == "fwd" else (len(ortholog_locus_seq) - (o_motif_start_in_locus + len(aligned_ortholog_seq)))
    ortho_win_start = max(0, real_motif_pos - w_len // 2)
    ortho_win_end = ortho_win_start + w_len

    if ortho_win_end <= len(ortholog_locus_seq):
        ortho_win_seq = ortholog_locus_seq[ortho_win_start:ortho_win_end]
        ortho_motif_pos = real_motif_pos - ortho_win_start
        target_motif_seq = target_window_seq[motif_start_in_window:motif_start_in_window + motif_len]
        insert_motif = revcomp(target_motif_seq) if best_strand == "rev" else target_motif_seq
        raw_ortho_swapped = ortho_win_seq[:ortho_motif_pos] + insert_motif + ortho_win_seq[ortho_motif_pos + len(aligned_ortholog_seq):]
        ortho_swapped_seq = _adjust_window_len(raw_ortho_swapped, w_len)

        with torch.no_grad():
            out_ortho_wt = model(seq_to_tensor(ortho_win_seq), compute_late=True)
            out_ortho_swapped = model(seq_to_tensor(ortho_swapped_seq), compute_late=True)
            score_ortho_wt = float(out_ortho_wt["y_abs"].squeeze().cpu().item())
            score_ortho_wt_rel = float(out_ortho_wt["y_rel"].squeeze().cpu().item())
            score_ortho_swapped = float(out_ortho_swapped["y_abs"].squeeze().cpu().item())
            score_ortho_swapped_rel = float(out_ortho_swapped["y_rel"].squeeze().cpu().item())
            ortho_delta = score_ortho_swapped - score_ortho_wt
            ortho_delta_rel = score_ortho_swapped_rel - score_ortho_wt_rel
    else:
        score_ortho_wt = None
        score_ortho_wt_rel = None
        score_ortho_swapped = None
        score_ortho_swapped_rel = None
        ortho_delta = None
        ortho_delta_rel = None

    aln_t, aln_bar, aln_o = _extract_aligned_motif_window(
        aln=best_aln,
        search_seq=search_seq,
        target_flank=target_flank,
        motif_rel_start=motif_rel_start,
        motif_len=motif_len,
        half_window=25
    )
    is_lineage_gain = bool(target_swap_delta < -0.05)
    class_desc = (
        f"Syntenic local alignment found ({best_strand} strand): reciprocal swap of aligned ortholog bases ({indel_diff} bp indel)"
        if is_indel_swap
        else f"Syntenic local alignment found ({best_strand} strand): reciprocal swap of aligned ortholog bases"
    )

    return {
        "status": "aligned_motif_turnover",
        "alignment_score": float(best_aln.score),
        "target_wt": score_target_wt,
        "target_wt_rel": score_target_wt_rel,
        "target_ablation_score": score_target_ablation,
        "target_ablation_score_rel": score_target_ablation_rel,
        "target_ablation_delta": ablation_delta,
        "target_ablation_delta_rel": ablation_delta_rel,
        "classification": class_desc,
        "ortholog_aligned_motif": aligned_ortholog_seq,
        "aligned_target_motif_window": aln_t,
        "aligned_match_bar": aln_bar,
        "aligned_ortholog_motif_window": aln_o,
        "target_flank": aln_t,
        "ortholog_flank": aln_o,
        "best_strand": best_strand,
        "is_lineage_gain": is_lineage_gain,
        "target_swapped_with_ortholog": score_target_swapped,
        "target_swapped_with_ortholog_rel": score_target_swapped_rel,
        "target_swap_delta": target_swap_delta,
        "target_swap_delta_rel": target_swap_delta_rel,
        "ortholog_wt": score_ortho_wt,
        "ortholog_wt_rel": score_ortho_wt_rel,
        "ortholog_swapped_with_target": score_ortho_swapped,
        "ortholog_swapped_with_target_rel": score_ortho_swapped_rel,
        "ortholog_swap_delta": ortho_delta,
        "ortholog_swap_delta_rel": ortho_delta_rel
    }


def perform_evolutionary_swap(
    model: GraftedChromBPNet,
    target_species_seq: str,
    ortholog_species_seq: str,
    motif_start: int,
    motif_len: int,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Perform real alignment-based evolutionary swap (delegates to perform_evolutionary_alignment_and_swap)."""
    return perform_evolutionary_alignment_and_swap(
        model=model,
        target_window_seq=target_species_seq,
        ortholog_locus_seq=ortholog_species_seq,
        motif_start_in_window=motif_start,
        motif_len=motif_len,
        device=device
    )


# ==============================================================================
# 6. Block 7: QuantSeq 3'-End Scoring
# ==============================================================================

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
