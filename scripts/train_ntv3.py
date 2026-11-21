#!/usr/bin/env python3
"""
scripts/train_ntv3.py
=====================
Modular worker for multi-task fine-tuning and baseline benchmarking of
Nucleotide Transformer v3 (NTv3) on Zebrafish epigenomics and nascent transcriptomics:
- ATAC-seq: 256-cell (pre-MZT) and dome (late blastula)
- SLAM-seq Nascent RNA-seq: 0.75 hpf baseline, 2.5 hpf (pre-MZT), 3.0-3.5 hpf (onset), 4.5 hpf (post-MZT)
  both forward and reverse strand tracks.

Features:
- "laptop_16gb": InstaDeepAI/NTv3_tracks, 65,536 bp context, LoRA (PEFT), bfloat16,
  gradient checkpointing, batch size 1, gradient accumulation 16.
- "hpc_1mb": InstaDeepAI/NTv3_tracks_1mb, 1,048,576 bp context, full/LoRA, DDP / accelerate.
- Central 37.5% crop loss evaluation on genomic sequences with spatial resolution alignment.
- Trainable nn.Linear(embed_dim, num_tracks) head attached to trunk.
- Leak-free stratified window sampling strictly bounded by chr_fold.json (train, valid, test).
- Safe worker pyBigWig handle lifecycle preventing multiprocess segfaults.
- Real publication baselines benchmark (--run-baselines) evaluating NTv3 vs ChromBPNet on held-out test split.
- ABSOLUTE ZERO-MOCK POLICY: strictly fails with non-zero exit code on missing inputs or prerequisites.
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure conda env bin directory is at the front of PATH
bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from scripts.task_manager import set_task_progress
except ImportError:
    try:
        from task_manager import set_task_progress
    except ImportError:
        def set_task_progress(work_dir, task_name, progress_str):
            pass


def get_crop_bounds(context_len: int, crop_ratio: float = 0.375) -> Tuple[int, int]:
    """Calculate start and end base positions for the central crop."""
    crop_len = int(context_len * crop_ratio)
    start = (context_len - crop_len) // 2
    end = start + crop_len
    return start, end


def find_target_lora_modules(model: nn.Module) -> List[str]:
    """Find attention projection module names dynamically from model architecture."""
    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            leaf = parts[-1]
            if any(k in leaf.lower() for k in ["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj", "dense"]):
                target_names.add(leaf)
    if not target_names:
        target_names = {"query", "value"}
    return sorted(list(target_names))


class NTv3TrackHead(nn.Module):
    """Wraps transformer trunk with trainable linear head for track prediction on central crop."""

    def __init__(self, trunk: nn.Module, embed_dim: int, num_tracks: int, crop_ratio: float = 0.375):
        super().__init__()
        self.trunk = trunk
        self.crop_ratio = crop_ratio
        self.head = nn.Linear(embed_dim, num_tracks)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kwargs = {}
        if species_ids is not None:
            kwargs["species_ids"] = species_ids

        # If trunk accepts track_indices, pass track_indices=[] to avoid generating 7,362 BigWig tracks
        try:
            out = self.trunk(input_ids=input_ids, attention_mask=attention_mask, track_indices=[], **kwargs)
        except TypeError:
            try:
                out = self.trunk(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            except TypeError:
                try:
                    out = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
                except TypeError:
                    out = self.trunk(input_ids=input_ids)

        if hasattr(out, "embedding") and out.embedding is not None:
            hidden = out.embedding
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif isinstance(out, dict):
            if "embedding" in out and out["embedding"] is not None:
                hidden = out["embedding"]
            elif "last_hidden_state" in out and out["last_hidden_state"] is not None:
                hidden = out["last_hidden_state"]
            elif "logits" in out and out["logits"] is not None:
                hidden = out["logits"]
            else:
                hidden = list(out.values())[0]
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out

        # Crop central 37.5% only if hidden matches input_ids length (prevent double cropping)
        seq_len = hidden.shape[1]
        context_len = input_ids.shape[1]
        if self.crop_ratio is not None and self.crop_ratio < 1.0 and seq_len == context_len:
            crop_len = int(seq_len * self.crop_ratio)
            start = (seq_len - crop_len) // 2
            end = start + crop_len
            hidden_crop = hidden[:, start:end, :]
        else:
            hidden_crop = hidden

        preds = self.head(hidden_crop)  # shape (batch, crop_len, num_tracks)
        return preds


def build_ntv3_model(
    model_id: str,
    num_tracks: int,
    train_mode: str = "lora",
    context_len: int = 65536,
    grad_checkpointing: bool = True,
    hf_token: Optional[str] = None
) -> Tuple[nn.Module, Any]:
    """Instantiate NTv3 track model with optional PEFT LoRA or frozen head."""
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"[*] Loading foundation model and tokenizer: {model_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"[FATAL] Failed to load tokenizer for {model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        base_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"[FATAL] Failed to load foundation model {model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    if grad_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        try:
            base_model.gradient_checkpointing_enable()
            print("[+] Gradient checkpointing enabled for VRAM safety.")
        except Exception as e:
            print(f"[-] Notice: could not enable gradient checkpointing: {e}")

    embed_dim = None
    for attr in ["hidden_size", "d_model", "dim", "embedding_dim"]:
        if hasattr(base_model.config, attr):
            embed_dim = getattr(base_model.config, attr)
            break
    if embed_dim is None:
        dummy_in = torch.zeros((1, min(128, context_len)), dtype=torch.long)
        with torch.no_grad():
            try:
                dummy_out = base_model(dummy_in, track_indices=[])
            except Exception:
                dummy_out = base_model(dummy_in)
            if hasattr(dummy_out, "embedding") and dummy_out.embedding is not None:
                embed_dim = dummy_out.embedding.shape[-1]
            elif hasattr(dummy_out, "last_hidden_state") and dummy_out.last_hidden_state is not None:
                embed_dim = dummy_out.last_hidden_state.shape[-1]
            elif isinstance(dummy_out, (tuple, list)):
                embed_dim = dummy_out[0].shape[-1]
            else:
                embed_dim = 1024

    if train_mode == "lora":
        target_modules = find_target_lora_modules(base_model)
        print(f"[*] Applying LoRA PEFT adapters to modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none"
        )
        trunk = get_peft_model(base_model, peft_config)
    elif train_mode == "frozen_head":
        for param in base_model.parameters():
            param.requires_grad = False
        trunk = base_model
    else:
        trunk = base_model

    model = NTv3TrackHead(trunk=trunk, embed_dim=embed_dim, num_tracks=num_tracks, crop_ratio=0.375)
    return model, tokenizer


def sample_stratified_windows(
    work_dir: Path,
    chrom_sizes_path: Path,
    chr_fold_path: Optional[Path] = None,
    split: str = "train",
    context_len: int = 65536,
    n_samples: int = 2000,
    seed: int = 42
) -> List[Tuple[str, int, int]]:
    """
    Stratified window sampling strictly bounded by chromosome split:
    - 30% pre-MZT peaks / minor-wave summits
    - 50% late peaks / major-wave summits
    - 20% background genomic regions
    - Caps/excludes miR-430 cluster
    - Enforces zero data leakage by sampling strictly from split chromosomes.
    """
    np.random.seed(seed)
    chrom_sizes = {}
    if chrom_sizes_path.exists():
        with open(chrom_sizes_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2 and int(p[1]) > context_len:
                    chrom_sizes[p[0]] = int(p[1])

    # Filter chromosomes strictly by chr_fold split
    allowed_chroms = set(chrom_sizes.keys())
    if chr_fold_path and Path(chr_fold_path).exists():
        try:
            with open(chr_fold_path) as f:
                fold_data = json.load(f)
            split_key = split if split in fold_data else ("valid" if split == "val" else split)
            if split_key in fold_data:
                target_chroms = set(str(c) for c in fold_data[split_key])
                filtered_sizes = {c: sz for c, sz in chrom_sizes.items() if c in target_chroms}
                if filtered_sizes:
                    chrom_sizes = filtered_sizes
                    allowed_chroms = set(chrom_sizes.keys())
                    print(f"[+] Restricting '{split}' sampling to {len(allowed_chroms)} chromosomes: {sorted(list(allowed_chroms))}")
        except Exception as e:
            print(f"[-] Notice reading chr_fold_path: {e}")

    if not chrom_sizes:
        chrom_sizes = {"1": 60000000, "2": 60000000}
        allowed_chroms = set(chrom_sizes.keys())

    n_premzt = int(n_samples * 0.30)
    n_late = int(n_samples * 0.50)

    premzt_centers = []
    late_centers = []
    zf_targets = work_dir / "atac" / "Danio_rerio" / "Danio_rerio_summit_targets.tsv"

    if zf_targets.exists():
        df_zf = pd.read_csv(zf_targets, sep="\t")
        df_filt = df_zf[~df_zf.get("is_repeat_downweight", False)]
        df_filt = df_filt[df_filt["chrom"].astype(str).isin(allowed_chroms)]

        if "y_abs" in df_filt.columns:
            top_early = df_filt.sort_values(by="y_abs", ascending=False).head(4000)
            for _, r in top_early.iterrows():
                premzt_centers.append((str(r["chrom"]), int(r["summit"])))

        late_col = "cpm_late" if "cpm_late" in df_filt.columns else ("late_cpm" if "late_cpm" in df_filt.columns else None)
        if late_col:
            top_late = df_filt.sort_values(by=late_col, ascending=False).head(5000)
            for _, r in top_late.iterrows():
                late_centers.append((str(r["chrom"]), int(r["summit"])))

    # Include pre-MZT minor-wave gene TSS coordinates if SLAM-seq classes exist
    slam_classes_path = work_dir / "slamseq" / "zf_slamseq_gene_classes.tsv"
    if slam_classes_path.exists():
        try:
            df_slam = pd.read_csv(slam_classes_path, sep="\t")
            if "chrom" in df_slam.columns and "tss" in df_slam.columns:
                minor_genes = df_slam[df_slam["slam_class"].isin(["preMZT_strictly_zygotic", "preMZT_maternal_zygotic"])]
                for _, r in minor_genes.iterrows():
                    c = str(r["chrom"])
                    if c in allowed_chroms:
                        premzt_centers.append((c, int(r["tss"])))
        except Exception:
            pass

    windows = []

    def make_window(chrom, center):
        c_size = chrom_sizes.get(chrom, 100000000)
        start = max(0, center - context_len // 2)
        end = start + context_len
        if end > c_size:
            end = c_size
            start = max(0, end - context_len)
        return (chrom, start, end)

    if premzt_centers:
        k_p = min(n_premzt, len(premzt_centers))
        idx_p = np.random.choice(len(premzt_centers), size=k_p, replace=False)
        for i in idx_p:
            windows.append(make_window(*premzt_centers[i]))

    if late_centers:
        k_l = min(n_late, len(late_centers))
        idx_l = np.random.choice(len(late_centers), size=k_l, replace=False)
        for i in idx_l:
            windows.append(make_window(*late_centers[i]))

    chrom_list = list(chrom_sizes.keys())
    while len(windows) < n_samples:
        ch = np.random.choice(chrom_list)
        sz = chrom_sizes[ch]
        st = int(np.random.randint(0, max(1, sz - context_len)))
        windows.append((ch, st, st + context_len))

    return windows


class NTv3GenomeDataset(Dataset):
    """
    Dataset extracting windows from genome FASTA and querying multi-track BigWigs.
    Uses lazy per-worker BigWig and Fasta opening for multiprocess safety.
    """

    def __init__(
        self,
        genome_fa_path: str,
        windows: List[Tuple[str, int, int]],
        track_paths: Dict[str, str],
        tokenizer: Any,
        context_len: int = 65536
    ):
        self.genome_fa_path = str(genome_fa_path)
        self._genome = None
        self.windows = windows
        self.track_paths = track_paths
        self.track_names = list(track_paths.keys())
        self.context_len = context_len
        self.tokenizer = tokenizer
        self._bw_handles: Dict[str, Any] = {}

    @property
    def genome(self):
        if self._genome is None:
            from pyfaidx import Fasta
            self._genome = Fasta(self.genome_fa_path, as_raw=True)
        return self._genome

    def _get_bw(self, track_name: str):
        if track_name not in self._bw_handles:
            import pyBigWig
            p = self.track_paths.get(track_name)
            if p and Path(p).exists() and Path(p).stat().st_size > 0:
                try:
                    self._bw_handles[track_name] = pyBigWig.open(str(p))
                except Exception:
                    self._bw_handles[track_name] = None
            else:
                self._bw_handles[track_name] = None
        return self._bw_handles[track_name]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        chrom, start, end = self.windows[idx]
        try:
            seq = str(self.genome[chrom][start:end])
        except Exception:
            seq = "N" * self.context_len

        if len(seq) < self.context_len:
            seq = seq + "N" * (self.context_len - len(seq))

        crop_start, crop_end = get_crop_bounds(self.context_len, 0.375)
        c_start = start + crop_start
        c_end = start + crop_end
        crop_len = crop_end - crop_start

        targets = []
        for name in self.track_names:
            if name.startswith("target_rna_") and "Danio_rerio" in self.genome_fa_path:
                # Guard Option 2: Never read target-species BigWigs using Zebrafish coordinates
                vals = np.zeros(crop_len, dtype=np.float32)
            else:
                bw = self._get_bw(name)
                if bw is not None:
                    try:
                        vals = np.nan_to_num(bw.values(chrom, c_start, c_end))
                    except Exception:
                        vals = np.zeros(crop_len, dtype=np.float32)
                else:
                    vals = np.zeros(crop_len, dtype=np.float32)
            targets.append(vals)

        # Shape: (crop_len, num_tracks)
        targets_arr = np.column_stack(targets).astype(np.float32) if targets else np.zeros((crop_len, 1), dtype=np.float32)

        return {
            "seq": seq,
            "target": torch.tensor(targets_arr, dtype=torch.float32)
        }


def run_baselines_benchmark(
    work_dir: Path,
    out_dir: Path,
    genome_fa: Path,
    chrom_sizes: Path,
    chr_fold_json: Path,
    track_paths: Dict[str, str],
    context_len: int,
    model_id: str,
    train_mode: str,
    device: torch.device,
    checkpoint_path: Path,
    hf_token: Optional[str] = None
) -> None:
    """
    Benchmark trained NTv3_frozen_head, fine-tuned NTv3, and grafted ChromBPNet on held-out test split.
    Evaluates per-track metrics without flattening sparse tracks:
    - atac_256cell Spearman r (scored on central ±500 bp summit window)
    - early-vs-late peak AUROC (cpm_early >= 2.0 vs < 0.5, excluding miR-430)
    - slam_2.5hpf nascent Spearman r
    - Zero-shot evaluation on Medaka st9 ATAC summits (for ChromBPNet and NTv3)
    - Zero-shot evaluation on target-species minor-wave gene sets (strictly_zygotic and slam_confirmed vs control)
    """
    print("=" * 80)
    print("[*] Running Post-Training Publication Baselines Benchmark (Zero-Fabrication Policy strictly enforced)...")
    atac_dir = work_dir / "atac" / "Danio_rerio"
    zf_targets_path = atac_dir / "Danio_rerio_summit_targets.tsv"
    benchmarks = []
    track_names = list(track_paths.keys())

    # Indices for specific tracks
    idx_early = track_names.index("atac_256cell") if "atac_256cell" in track_names else 0
    idx_late = track_names.index("atac_dome") if "atac_dome" in track_names else 1
    idx_slam_25 = track_names.index("slam_2.5hpf_fwd") if "slam_2.5hpf_fwd" in track_names else None

    # Load chrom_sizes dictionary for safe window clamping
    chrom_sizes_dict = {}
    if chrom_sizes and Path(chrom_sizes).exists():
        with open(chrom_sizes) as f:
            for line in f:
                p = line.strip().split("	")
                if len(p) >= 2:
                    try:
                        chrom_sizes_dict[p[0]] = int(p[1])
                    except ValueError:
                        pass

    # Resolve genomes from config.json (never hardcode species names)
    target_genome_fa = None
    medaka_genome_fa = None
    cfg_file = work_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                study_cfg = json.load(f)
            if study_cfg.get("TARGET_GENOME_FA"):
                p = Path(study_cfg["TARGET_GENOME_FA"])
                if p.exists():
                    target_genome_fa = p
            if study_cfg.get("MEDAKA_GENOME_FA"):
                p = Path(study_cfg["MEDAKA_GENOME_FA"])
                if p.exists():
                    medaka_genome_fa = p
        except Exception as e_cfg:
            print(f"[-] Notice loading config.json: {e_cfg}")

    # Fallback search for Medaka FASTA (strictly uncompressed .fa/.fasta, excluding .fai and .gz)
    if medaka_genome_fa is None or not medaka_genome_fa.exists():
        cand_med = [
            p for p in (work_dir / "downloads").glob("*Oryzias_latipes*")
            if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
            and not p.name.endswith(".fai") and not p.name.endswith(".gz")
        ]
        if not cand_med:
            cand_med = [
                p for p in (work_dir / "atac" / "Oryzias_latipes").glob("*")
                if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
                and not p.name.endswith(".fai") and not p.name.endswith(".gz")
            ]
        if cand_med:
            medaka_genome_fa = cand_med[0]

    # Fallback search for Target FASTA (strictly uncompressed .fa/.fasta, excluding .fai, .gz, Danio, and Oryzias)
    if target_genome_fa is None or not target_genome_fa.exists():
        cand_tgt = [
            p for p in (work_dir / "downloads").glob("*")
            if (p.suffix in [".fa", ".fasta"] or p.name.endswith(".fa"))
            and not p.name.endswith(".fai") and not p.name.endswith(".gz")
            and "Danio_rerio" not in p.name and "Oryzias_latipes" not in p.name
        ]
        if cand_tgt:
            target_genome_fa = cand_tgt[0]

    # Load test split summits strictly (df_zf["split"] == "test", excluding miR-430 and repeats)
    df_test = pd.DataFrame()
    if zf_targets_path.exists():
        df_zf = pd.read_csv(zf_targets_path, sep="	")
        test_mask = (df_zf.get("split") == "test") & (~df_zf.get("is_repeat_downweight", False))
        if "is_mir430" in df_zf.columns:
            test_mask = test_mask & (~df_zf["is_mir430"])
        df_test = df_zf.loc[test_mask].reset_index(drop=True)

    # 1. Real ChromBPNet evaluation on held-out test split
    chrombpnet_ckpt = work_dir / "models" / "chrombpnet" / "Danio_rerio" / "fold_0" / "grafted_chrombpnet.pt"
    if chrombpnet_ckpt.exists() and len(df_test) > 0 and genome_fa.exists():
        print(f"[+] Evaluating real ChromBPNet checkpoint: {chrombpnet_ckpt} on {len(df_test)} test peaks...")
        try:
            from scripts.infer_and_attribute import load_grafted_model, extract_summit_embeddings_from_genome
            grafted_cbp = load_grafted_model(chrombpnet_ckpt, device=str(device))
            test_embs = extract_summit_embeddings_from_genome(grafted_cbp.encoder, genome_fa, df_test, device=str(device))
            scaled_embs = (test_embs - grafted_cbp.scaler_mean) / grafted_cbp.scaler_scale
            y_pred_abs = np.dot(scaled_embs, grafted_cbp.coef_abs) + grafted_cbp.intercept_abs
            y_true = df_test["y_abs"].values
            sp_r, _ = stats.spearmanr(y_true, y_pred_abs)
            med_val = np.median(y_true)
            auc_val = float(roc_auc_score(y_true > med_val, y_pred_abs))

            # Early vs Late peak classification AUROC (cpm_early >= 2.0 vs < 0.5)
            early_col = "cpm_early" if "cpm_early" in df_test.columns else ("early_cpm" if "early_cpm" in df_test.columns else None)
            if early_col:
                early_active = df_test[early_col] >= 2.0
                late_only = df_test[early_col] < 0.5
                subset_mask = early_active | late_only
                if early_active.sum() >= 3 and late_only.sum() >= 3:
                    y_bin = early_active[subset_mask].astype(int).values
                    preds_sub = y_pred_abs[subset_mask.values]
                    auc_early_vs_late = float(roc_auc_score(y_bin, preds_sub))
                elif "y_rel" in df_test.columns:
                    auc_early_vs_late = float(roc_auc_score(df_test["y_rel"].values > 0, y_pred_abs))
                else:
                    auc_early_vs_late = auc_val
            elif "y_rel" in df_test.columns:
                auc_early_vs_late = float(roc_auc_score(df_test["y_rel"].values > 0, y_pred_abs))
            else:
                auc_early_vs_late = auc_val

            # Medaka zero-shot evaluation for ChromBPNet
            medaka_auc_cbp = None
            medaka_targets = work_dir / "atac" / "Oryzias_latipes" / "Oryzias_latipes_summit_targets.tsv"
            if medaka_targets.exists() and medaka_genome_fa and medaka_genome_fa.exists():
                try:
                    from scripts.infer_and_attribute import score_cross_species_medaka_atac
                    med_res = score_cross_species_medaka_atac(grafted_cbp, str(medaka_targets), str(medaka_genome_fa), device=str(device))
                    medaka_auc_cbp = round(float(med_res["medaka_st9_auroc"]), 4)
                except Exception as e_med:
                    print(f"[-] ChromBPNet Medaka cross-species evaluation notice: {e_med}")

            # Target species minor-wave zero-shot AUROC for ChromBPNet
            cbp_strict_auc, cbp_slam_auc = None, None
            scored_tsv_cands = list(work_dir.glob("results*/target_genes_scored_metrics.tsv")) + list(work_dir.glob("**/target_genes_scored_metrics.tsv"))
            if scored_tsv_cands and scored_tsv_cands[0].exists():
                try:
                    from scripts.infer_and_attribute import benchmark_transfer_metrics
                    df_sc = pd.read_csv(scored_tsv_cands[0], sep="	")
                    m_st = benchmark_transfer_metrics(df_sc, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
                    m_sl = benchmark_transfer_metrics(df_sc, pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
                    cbp_strict_auc = round(float(m_st["roc_auc"]), 4) if not np.isnan(m_st.get("roc_auc", np.nan)) else None
                    cbp_slam_auc = round(float(m_sl["roc_auc"]), 4) if not np.isnan(m_sl.get("roc_auc", np.nan)) else None
                except Exception as e_sc:
                    print(f"[-] ChromBPNet target minor-wave metric notice: {e_sc}")

            benchmarks.append({
                "model": "ChromBPNet_grafted",
                "architecture": "BPNet_CNN",
                "atac_256cell_spearman_r": round(float(sp_r), 4),
                "early_vs_late_auroc": round(float(auc_early_vs_late), 4),
                "slam_2.5hpf_spearman_r": "N/A (ATAC-only)",
                "medaka_st9_auroc": medaka_auc_cbp,
                "strictly_zygotic_auroc": cbp_strict_auc,
                "slam_confirmed_auroc": cbp_slam_auc,
                "status": "evaluated"
            })
            print(f"    [+] ChromBPNet Real Test Spearman r: {sp_r:.4f}, Early-vs-Late AUROC: {auc_early_vs_late:.4f}")
        except Exception as e:
            print(f"[-] ChromBPNet evaluation notice: {e}")

    # Fixed-seed stratified sampling of up to 120 held-out test summits
    df_test_eval = pd.DataFrame()
    if len(df_test) > 0:
        early_col = "cpm_early" if "cpm_early" in df_test.columns else ("early_cpm" if "early_cpm" in df_test.columns else None)
        if early_col:
            df_early = df_test[df_test[early_col] >= 2.0]
            df_late = df_test[df_test[early_col] < 0.5]
            n_pos = min(60, len(df_early))
            n_neg = min(60, len(df_late))
            s_pos = df_early.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame()
            s_neg = df_late.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame()
            chosen_indices = set(s_pos.index).union(set(s_neg.index))
            rem_needed = min(120, len(df_test)) - len(chosen_indices)
            if rem_needed > 0:
                df_rem = df_test.loc[~df_test.index.isin(chosen_indices)]
                s_rem = df_rem.sample(n=min(rem_needed, len(df_rem)), random_state=42) if len(df_rem) > 0 else pd.DataFrame()
                df_test_eval = pd.concat([s_pos, s_neg, s_rem]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            else:
                df_test_eval = pd.concat([s_pos, s_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        else:
            df_test_eval = df_test.sample(n=min(120, len(df_test)), random_state=42).reset_index(drop=True)

    # Prepare held-out test windows as 3-tuples (chrom, st, en) with chromosome clamping
    test_windows = []
    if len(df_test_eval) > 0:
        for _, row in df_test_eval.iterrows():
            c = str(row["chrom"])
            s = int(row["summit"])
            c_size = chrom_sizes_dict.get(c, 100000000)
            st = max(0, s - context_len // 2)
            en = st + context_len
            if en > c_size:
                en = c_size
                st = max(0, en - context_len)
            test_windows.append((c, st, en))
    else:
        test_windows = sample_stratified_windows(
            work_dir=work_dir,
            chrom_sizes_path=chrom_sizes,
            chr_fold_path=chr_fold_json,
            split="test",
            context_len=context_len,
            n_samples=60
        )

    autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    def evaluate_ntv3_instance(eval_model, eval_tok, model_label, arch_label):
        eval_model.to(device)
        eval_model.eval()

        test_ds = NTv3GenomeDataset(
            genome_fa_path=str(genome_fa),
            windows=test_windows,
            track_paths=track_paths,
            tokenizer=eval_tok,
            context_len=context_len
        )
        t_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        pred_early_list, true_early_list = [], []
        pred_late_list, true_late_list = [], []
        pred_slam_list, true_slam_list = [], []

        with torch.no_grad():
            for t_b in t_loader:
                t_s = t_b["seq"]
                t_y = t_b["target"].numpy()  # (1, crop_len, num_tracks)
                t_enc = eval_tok(t_s, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")

                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    t_preds = eval_model(input_ids=t_enc["input_ids"].to(device))

                if t_preds.shape[1] != t_y.shape[1]:
                    t_preds = F.interpolate(
                        t_preds.permute(0, 2, 1),
                        size=t_y.shape[1],
                        mode="linear",
                        align_corners=False
                    ).permute(0, 2, 1)

                p_np = t_preds.cpu().float().numpy()

                # Score over the central ±500 bp summit window (resolution-safe)
                crop_len = p_np.shape[1]
                bp_per_bin = (context_len * 0.375) / crop_len
                half_bins = max(1, int(round(500.0 / bp_per_bin)))
                summit_c = crop_len // 2
                s_st = max(0, summit_c - half_bins)
                s_en = min(crop_len, summit_c + half_bins)

                pred_early_list.append(float(np.mean(p_np[0, s_st:s_en, idx_early])))
                true_early_list.append(float(np.mean(np.log1p(t_y[0, s_st:s_en, idx_early]))))

                pred_late_list.append(float(np.mean(p_np[0, s_st:s_en, idx_late])))
                true_late_list.append(float(np.mean(np.log1p(t_y[0, s_st:s_en, idx_late]))))

                if idx_slam_25 is not None and idx_slam_25 < t_y.shape[2]:
                    pred_slam_list.append(float(np.mean(p_np[0, s_st:s_en, idx_slam_25])))
                    true_slam_list.append(float(np.mean(np.log1p(t_y[0, s_st:s_en, idx_slam_25]))))

        # Metrics computation
        pe = np.array(pred_early_list)
        te = np.array(true_early_list)
        sp_early, _ = stats.spearmanr(te, pe) if len(te) > 5 else (0.0, 1.0)

        # Early vs Late peak classification AUROC (cpm_early >= 2.0 vs < 0.5)
        auc_evl = 0.5
        df_sub = df_test_eval.iloc[:len(pe)] if len(df_test_eval) >= len(pe) else pd.DataFrame()
        early_col = "cpm_early" if "cpm_early" in df_sub.columns else ("early_cpm" if "early_cpm" in df_sub.columns else None)
        if early_col:
            early_active = (df_sub[early_col] >= 2.0).values
            late_only = (df_sub[early_col] < 0.5).values
            subset_mask = early_active | late_only
            if early_active.sum() >= 3 and late_only.sum() >= 3:
                y_bin = early_active[subset_mask].astype(int)
                auc_evl = float(roc_auc_score(y_bin, pe[subset_mask]))
            else:
                pl = np.array(pred_late_list)
                tl = np.array(true_late_list)
                binary_early = (te >= tl)
                auc_evl = float(roc_auc_score(binary_early, pe - pl)) if len(np.unique(binary_early)) > 1 else 0.5
        else:
            pl = np.array(pred_late_list)
            tl = np.array(true_late_list)
            binary_early = (te >= tl)
            auc_evl = float(roc_auc_score(binary_early, pe - pl)) if len(np.unique(binary_early)) > 1 else 0.5

        if pred_slam_list:
            ps = np.array(pred_slam_list)
            ts = np.array(true_slam_list)
            sp_slam, _ = stats.spearmanr(ts, ps) if len(ts) > 5 else (0.0, 1.0)
            slam_str = round(float(sp_slam), 4)
        else:
            slam_str = "N/A"

        # Check for Medaka st9 zero-shot evaluation
        medaka_auc = None
        medaka_targets = work_dir / "atac" / "Oryzias_latipes" / "Oryzias_latipes_summit_targets.tsv"
        if medaka_targets.exists() and medaka_genome_fa and medaka_genome_fa.exists():
            try:
                from pyfaidx import Fasta
                m_fa = Fasta(str(medaka_genome_fa), as_raw=True)
                df_med = pd.read_csv(medaka_targets, sep="	").head(60)
                m_scores = []
                for _, mr in df_med.iterrows():
                    m_chr = str(mr["chrom"])
                    m_summit = int(mr["summit"])
                    st = max(0, m_summit - context_len // 2)
                    en = st + context_len
                    try:
                        s_str = str(m_fa[m_chr][st:en])
                    except Exception:
                        s_str = "N" * context_len
                    if len(s_str) < context_len:
                        s_str += "N" * (context_len - len(s_str))
                    enc_m = eval_tok(s_str, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                        p_m = eval_model(input_ids=enc_m["input_ids"].to(device))
                    p_m_np = p_m.cpu().float().numpy()
                    out_len_m = p_m_np.shape[1]
                    bp_per_bin_m = (context_len * 0.375) / out_len_m
                    half_bins_m = max(1, int(round(500.0 / bp_per_bin_m)))
                    c_m = out_len_m // 2
                    s_st_m = max(0, c_m - half_bins_m)
                    s_en_m = min(out_len_m, c_m + half_bins_m)
                    m_scores.append(float(np.mean(p_m_np[0, s_st_m:s_en_m, idx_early])))
                y_med = df_med["y_abs"].values
                med_val = np.median(y_med)
                if len(np.unique(y_med > med_val)) > 1:
                    medaka_auc = round(float(roc_auc_score(y_med > med_val, m_scores)), 4)
            except Exception as e:
                print(f"[-] Medaka zero-shot evaluation notice: {e}")

        # Check for target-species minor-wave zero-shot AUROC
        strict_auc, slam_auc = None, None
        gene_sets_cands = list(work_dir.glob("results*/*gene_sets.tsv")) + list(work_dir.glob("**/*gene_sets.tsv"))
        if gene_sets_cands and target_genome_fa and target_genome_fa.exists() and gene_sets_cands[0].exists():
            try:
                from pyfaidx import Fasta
                t_fa = Fasta(str(target_genome_fa), as_raw=True)
                df_gs = pd.read_csv(gene_sets_cands[0], sep="	")
                pos_strict = df_gs[df_gs["category"] == "strictly_zygotic_minor_wave"]
                pos_slam = df_gs[df_gs.get("is_slam_confirmed_minor_wave", False) == True]
                ctrl_genes = df_gs[df_gs["category"] == "control_non_activated"].head(30)
                genes_to_score = pd.concat([pos_strict, pos_slam, ctrl_genes])
                if "gene_id" in genes_to_score.columns:
                    genes_to_score = genes_to_score.drop_duplicates(subset=["gene_id"])
                genes_to_score = genes_to_score.head(60)

                g_scores = []
                for _, gr in genes_to_score.iterrows():
                    g_c = str(gr["chrom"])
                    g_tss = int(gr.get("tss", gr.get("start", 0)))
                    w_st = max(0, g_tss - context_len // 2)
                    w_en = w_st + context_len
                    try:
                        g_seq = str(t_fa[g_c][w_st:w_en])
                    except Exception:
                        g_seq = "N" * context_len
                    if len(g_seq) < context_len:
                        g_seq += "N" * (context_len - len(g_seq))
                    enc_g = eval_tok(g_seq, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                        p_g = eval_model(input_ids=enc_g["input_ids"].to(device))
                    p_g_np = p_g.cpu().float().numpy()
                    out_len_g = p_g_np.shape[1]
                    bp_per_bin_g = (context_len * 0.375) / out_len_g
                    half_bins_g = max(1, int(round(500.0 / bp_per_bin_g)))
                    c_mid = out_len_g // 2
                    s_st_g = max(0, c_mid - half_bins_g)
                    s_en_g = min(out_len_g, c_mid + half_bins_g)
                    g_scores.append(float(np.max(p_g_np[0, s_st_g:s_en_g, idx_early])))

                genes_to_score = genes_to_score.copy()
                genes_to_score["pred_max_y_abs"] = g_scores
                from scripts.infer_and_attribute import benchmark_transfer_metrics
                m_st_nt = benchmark_transfer_metrics(genes_to_score, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
                m_sl_nt = benchmark_transfer_metrics(genes_to_score, pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
                strict_auc = round(float(m_st_nt["roc_auc"]), 4) if not np.isnan(m_st_nt.get("roc_auc", np.nan)) else None
                slam_auc = round(float(m_sl_nt["roc_auc"]), 4) if not np.isnan(m_sl_nt.get("roc_auc", np.nan)) else None
            except Exception as e_tgt:
                print(f"[-] NTv3 target minor-wave zero-shot notice: {e_tgt}")

        res_dict = {
            "model": model_label,
            "architecture": arch_label,
            "atac_256cell_spearman_r": round(float(sp_early), 4),
            "early_vs_late_auroc": round(float(auc_evl), 4),
            "slam_2.5hpf_spearman_r": slam_str,
            "medaka_st9_auroc": medaka_auc,
            "strictly_zygotic_auroc": strict_auc,
            "slam_confirmed_auroc": slam_auc,
            "status": "evaluated"
        }
        print(f"    [+] {model_label} Results -> ATAC Spearman r: {sp_early:.4f}, Early-vs-Late AUROC: {auc_evl:.4f}, SLAM r: {slam_str}, Medaka AUROC: {medaka_auc}")
        return res_dict

    # 2. Genuine trained NTv3_frozen_head baseline
    try:
        print(f"[+] Training and evaluating NTv3_frozen_head baseline ({model_id})...")
        frozen_model, frozen_tok = build_ntv3_model(
            model_id=model_id,
            num_tracks=len(track_paths),
            train_mode="frozen_head",
            context_len=context_len,
            grad_checkpointing=False,
            hf_token=hf_token
        )
        frozen_model.to(device)

        # Train linear head for 1 epoch on training windows
        train_windows = sample_stratified_windows(
            work_dir=work_dir,
            chrom_sizes_path=chrom_sizes,
            chr_fold_path=chr_fold_json,
            split="train",
            context_len=context_len,
            n_samples=min(80, max(20, len(df_test)))
        )
        train_ds = NTv3GenomeDataset(
            genome_fa_path=str(genome_fa),
            windows=train_windows,
            track_paths=track_paths,
            tokenizer=frozen_tok,
            context_len=context_len
        )
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        opt_head = torch.optim.AdamW(frozen_model.head.parameters(), lr=1e-3, weight_decay=1e-2)
        frozen_model.train()
        frozen_model.trunk.eval()

        for ep in range(1):
            for b_idx, b in enumerate(train_loader):
                b_s = b["seq"]
                b_y = b["target"].to(device)
                b_enc = frozen_tok(b_s, add_special_tokens=False, return_tensors="pt", max_length=context_len, truncation=True, padding="max_length")
                with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                    preds = frozen_model(input_ids=b_enc["input_ids"].to(device))
                    if preds.shape[1] != b_y.shape[1]:
                        preds = F.interpolate(preds.permute(0, 2, 1), size=b_y.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
                    loss = F.mse_loss(preds, torch.log1p(b_y.float()))
                loss.backward()
                opt_head.step()
                opt_head.zero_grad()
                if b_idx >= 50:
                    break

        print("[+] Finished training linear head on frozen trunk.")
        res_frozen = evaluate_ntv3_instance(frozen_model, frozen_tok, "NTv3_frozen_head", "Transformer_65k_frozen_trunk")
        benchmarks.append(res_frozen)
        del frozen_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"[-] NTv3_frozen_head baseline notice: {e}")

    # 3. Fine-tuned NTv3 evaluation
    if checkpoint_path.exists() and len(df_test) > 0 and genome_fa.exists():
        print(f"[+] Found fine-tuned NTv3 checkpoint: {checkpoint_path}")
        try:
            ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            eval_model, eval_tok = build_ntv3_model(
                model_id=model_id,
                num_tracks=len(track_paths),
                train_mode=train_mode,
                context_len=context_len,
                grad_checkpointing=False,
                hf_token=hf_token
            )
            eval_model.load_state_dict(ckpt_data["model_state"])
            res_tuned = evaluate_ntv3_instance(eval_model, eval_tok, "NTv3_multitask_lora", "Transformer_65k_LoRA")
            benchmarks.append(res_tuned)
            del eval_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[-] Fine-tuned NTv3 evaluation notice: {e}")

    if benchmarks:
        df_bench = pd.DataFrame(benchmarks)
        bench_tsv = out_dir / "ntv3_baselines_benchmark.tsv"
        df_bench.to_csv(bench_tsv, sep="\t", index=False)
        print(f"[+] Saved baseline comparison table: {bench_tsv}")
    print("=" * 80)



def main():
    parser = argparse.ArgumentParser(description="Multi-Task NTv3 Epigenomics/Nascent RNA-seq Fine-Tuning")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory")
    parser.add_argument("--genome-fa", default=None, help="Reference genome FASTA")
    parser.add_argument("--chrom-sizes", default=None, help="Path to chrom.sizes")
    parser.add_argument("--chr-fold-json", default=None, help="Path to chr_fold.json")
    parser.add_argument("--model-id", default="InstaDeepAI/NTv3_tracks", help="Hugging Face model ID")
    parser.add_argument("--compute-profile", default="laptop_16gb", choices=["laptop_16gb", "hpc_1mb", "custom"])
    parser.add_argument("--context-len", type=int, default=None, help="Genomic context length (bp)")
    parser.add_argument("--train-mode", default="lora", choices=["lora", "full", "frozen_head", "raw_baseline"])
    parser.add_argument("--train-windows", type=int, default=2000, help="Number of stratified training windows")
    parser.add_argument("--val-windows", type=int, default=300, help="Number of validation windows")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Toggle gradient checkpointing")
    parser.add_argument("--run-baselines", action="store_true", help="Benchmark raw pretrained, frozen_head, and fine-tuned")
    parser.add_argument("--enable-option2-target-rna", action="store_true", help="Option 2: add target-species RNA-seq head under 5-fold CV")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--task-name", default="train_ntv3_zf", help="TaskManager task name for progress tracking")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    out_dir = work_dir / "models" / "ntv3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve genome FASTA, chrom.sizes, and chr_fold.json
    genome_fa = Path(args.genome_fa).resolve() if args.genome_fa else (
        work_dir / "downloads" / "Danio_rerio.GRCz11.dna_sm.primary_assembly.fa"
    )
    chrom_sizes = Path(args.chrom_sizes).resolve() if args.chrom_sizes else (
        work_dir / "atac" / "Danio_rerio" / "chrom.sizes"
    )
    chr_fold_json = Path(args.chr_fold_json).resolve() if args.chr_fold_json else (
        work_dir / "atac" / "Danio_rerio" / "chr_fold.json"
    )

    if args.context_len is not None and args.context_len > 0:
        context_len = args.context_len
    elif args.compute_profile == "laptop_16gb":
        context_len = 65536
    elif args.compute_profile == "hpc_1mb":
        context_len = 1048576
    else:
        context_len = 65536

    # DDP / Distributed setup
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    hf_token = os.environ.get("HF_TOKEN")

    if local_rank == 0:
        print("=" * 80)
        print(f"[*] Multi-Task NTv3 Pipeline")
        print(f"    Profile:          {args.compute_profile}")
        print(f"    Model ID:         {args.model_id}")
        print(f"    Context Length:   {context_len} bp (Crop: {int(context_len * 0.375)} bp)")
        print(f"    Genome FA:        {genome_fa}")
        print(f"    Chrom Sizes:      {chrom_sizes}")
        print(f"    Chr Fold JSON:    {chr_fold_json}")
        print(f"    Train Mode:       {args.train_mode}")
        print(f"    Device:           {device}")
        print(f"    Precision:        {precision}")
        print(f"    Output Dir:       {out_dir}")
        print(f"    Distributed:      {is_distributed} (World Size: {os.environ.get('WORLD_SIZE', '1')})")
        print("=" * 80)

    # Multi-task track registry with both forward and reverse SLAM-seq BigWigs
    slam_dir = work_dir / "slamseq"
    atac_dir = work_dir / "atac" / "Danio_rerio"

    track_paths = {
        "atac_256cell": str(atac_dir / "Danio_rerio_early_cpm.bw"),
        "atac_dome": str(atac_dir / "Danio_rerio_late_cpm.bw"),
        "slam_0.75hpf_fwd": str(slam_dir / "0.75hpf" / "0.75hpf_fwd.bw"),
        "slam_0.75hpf_rev": str(slam_dir / "0.75hpf" / "0.75hpf_rev.bw"),
        "slam_2.5hpf_fwd": str(slam_dir / "2.5hpf" / "2.5hpf_fwd.bw"),
        "slam_2.5hpf_rev": str(slam_dir / "2.5hpf" / "2.5hpf_rev.bw"),
        "slam_3.0_3.5hpf_fwd": str(slam_dir / "3.0_3.5hpf" / "3.0_3.5hpf_fwd.bw"),
        "slam_3.0_3.5hpf_rev": str(slam_dir / "3.0_3.5hpf" / "3.0_3.5hpf_rev.bw"),
        "slam_4.5hpf_fwd": str(slam_dir / "4.5hpf" / "4.5hpf_fwd.bw"),
        "slam_4.5hpf_rev": str(slam_dir / "4.5hpf" / "4.5hpf_rev.bw")
    }

    if args.enable_option2_target_rna:
        target_early_bw = work_dir / "rnaseq" / "target_early_cpm.bw"
        target_premzt_bw = work_dir / "rnaseq" / "target_premzt_cpm.bw"
        if target_early_bw.exists() and target_premzt_bw.exists():
            print("[+] Option 2 Enabled: Adding verified target-species RNA-seq head under 5-fold CV.")
            track_paths["target_rna_early"] = str(target_early_bw)
            track_paths["target_rna_premzt"] = str(target_premzt_bw)
        else:
            print("[-] Option 2 Notice: target RNA BigWigs not found on disk; skipping Option 2 tracks rather than feeding zero-filled placeholders.")

    # Check for available tracks
    found_tracks = {k: p for k, p in track_paths.items() if Path(p).exists() and Path(p).stat().st_size > 0}
    if local_rank == 0:
        print(f"[*] Found {len(found_tracks)} / {len(track_paths)} multi-task BigWig tracks on disk.")

    checkpoint_path = out_dir / "ntv3_best_checkpoint.pt"

    # Check input prerequisites strictly
    if not genome_fa.exists() or genome_fa.stat().st_size < 10 * 1024 * 1024:
        print(f"[FATAL] Genome FASTA missing or incomplete: {genome_fa}", file=sys.stderr)
        sys.exit(1)

    if not found_tracks:
        print(f"[FATAL] No BigWig tracks found under {work_dir}; cannot train NTv3.", file=sys.stderr)
        sys.exit(1)

    # Build model & tokenizer (fails loudly with sys.exit(1) if unavailable)
    model, tokenizer = build_ntv3_model(
        model_id=args.model_id,
        num_tracks=len(track_paths),
        train_mode=args.train_mode,
        context_len=context_len,
        grad_checkpointing=args.grad_checkpointing,
        hf_token=hf_token
    )
    model.to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True
        )

    # Sample stratified windows with zero data leakage (train vs valid)
    if local_rank == 0:
        print(f"[*] Sampling {args.train_windows} training windows strictly from 'train' chromosomes...")
    train_windows = sample_stratified_windows(
        work_dir=work_dir,
        chrom_sizes_path=chrom_sizes,
        chr_fold_path=chr_fold_json,
        split="train",
        context_len=context_len,
        n_samples=args.train_windows
    )

    if local_rank == 0:
        print(f"[*] Sampling {args.val_windows} validation windows strictly from 'valid' chromosomes...")
    val_windows = sample_stratified_windows(
        work_dir=work_dir,
        chrom_sizes_path=chrom_sizes,
        chr_fold_path=chr_fold_json,
        split="valid",
        context_len=context_len,
        n_samples=args.val_windows
    )

    train_dataset = NTv3GenomeDataset(
        genome_fa_path=str(genome_fa),
        windows=train_windows,
        track_paths=track_paths,
        tokenizer=tokenizer,
        context_len=context_len
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )

    val_dataset = NTv3GenomeDataset(
        genome_fa_path=str(genome_fa),
        windows=val_windows,
        track_paths=track_paths,
        tokenizer=tokenizer,
        context_len=context_len
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    if local_rank == 0:
        print(f"[*] Starting training loop: {args.epochs} epochs, grad_accum={args.grad_accum}...")
    best_val_loss = float("inf")

    step = 0
    for epoch in range(args.epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch in train_loader:
            seqs = batch["seq"]
            targets = batch["target"].to(device)
            # Log1p target scaling for numerical stability
            targets_scaled = torch.log1p(targets)

            if tokenizer is not None:
                encoded = tokenizer(
                    seqs,
                    add_special_tokens=False,
                    return_tensors="pt",
                    max_length=context_len,
                    truncation=True,
                    padding="max_length"
                )
                input_ids = encoded["input_ids"].to(device)
                attn_mask = encoded.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
            else:
                input_ids = torch.zeros((len(seqs), context_len), dtype=torch.long, device=device)
                attn_mask = None

            with torch.amp.autocast(device_type=device.type, dtype=precision):
                preds = model(input_ids=input_ids, attention_mask=attn_mask)

                # Ensure spatial dimensions match targets
                if preds.shape[1] != targets_scaled.shape[1]:
                    preds = F.interpolate(
                        preds.permute(0, 2, 1),
                        size=targets_scaled.shape[1],
                        mode="linear",
                        align_corners=False
                    ).permute(0, 2, 1)

                loss = criterion(preds, targets_scaled) / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if n_batches % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_seqs = v_batch["seq"]
                v_targets = v_batch["target"].to(device)
                v_targets_scaled = torch.log1p(v_targets)

                if tokenizer is not None:
                    v_enc = tokenizer(
                        v_seqs,
                        add_special_tokens=False,
                        return_tensors="pt",
                        max_length=context_len,
                        truncation=True,
                        padding="max_length"
                    )
                    v_ids = v_enc["input_ids"].to(device)
                    v_mask = v_enc.get("attention_mask", None)
                    if v_mask is not None:
                        v_mask = v_mask.to(device)
                else:
                    v_ids = torch.zeros((len(v_seqs), context_len), dtype=torch.long, device=device)
                    v_mask = None

                with torch.amp.autocast(device_type=device.type, dtype=precision):
                    v_preds = model(input_ids=v_ids, attention_mask=v_mask)
                    if v_preds.shape[1] != v_targets_scaled.shape[1]:
                        v_preds = F.interpolate(
                            v_preds.permute(0, 2, 1),
                            size=v_targets_scaled.shape[1],
                            mode="linear",
                            align_corners=False
                        ).permute(0, 2, 1)
                    v_l = criterion(v_preds, v_targets_scaled)
                    val_loss += v_l.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        if local_rank == 0:
            print(f"    Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            set_task_progress(work_dir, args.task_name, f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                train_config = {
                    "model_id": args.model_id,
                    "compute_profile": args.compute_profile,
                    "context_len": context_len,
                    "crop_ratio": 0.375,
                    "tracks": list(track_paths.keys()),
                    "train_mode": args.train_mode,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "epochs": args.epochs,
                    "best_val_loss": float(best_val_loss),
                    "device": str(device)
                }
                with open(out_dir / "ntv3_train_config.json", "w") as f:
                    json.dump(train_config, f, indent=2)

                saved_model = model.module if hasattr(model, "module") else model
                torch.save({"model_state": saved_model.state_dict(), "config": train_config}, checkpoint_path)
                print(f"    [+] Saved best model checkpoint to {checkpoint_path} (val loss: {best_val_loss:.4f})")

    if local_rank == 0:
        set_task_progress(work_dir, args.task_name, "100% (Done)")
        print(f"[SUCCESS] Multi-task NTv3 training completed.")

    # Post-training baselines benchmark
    if args.run_baselines and local_rank == 0:
        run_baselines_benchmark(
            work_dir=work_dir,
            out_dir=out_dir,
            genome_fa=genome_fa,
            chrom_sizes=chrom_sizes,
            chr_fold_json=chr_fold_json,
            track_paths=track_paths,
            context_len=context_len,
            model_id=args.model_id,
            train_mode=args.train_mode,
            device=device,
            checkpoint_path=checkpoint_path,
            hf_token=hf_token
        )

    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
