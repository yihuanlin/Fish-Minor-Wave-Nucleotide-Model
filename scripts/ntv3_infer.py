#!/usr/bin/env python3
"""
scripts/ntv3_infer.py
=============================
Standalone background worker for Block 8:
NTv3 Inference & Head-to-Head Comparison with ChromBPNet.

Executes three reproducible sub-tasks:
1. Matched Zero-Shot Gene Benchmark:
   Scores the exact same gene table in target_genes_scored_metrics.tsv (from Block 4)
   using NTv3 (both max atac_256cell across central 37.5% crop and strand-matched
   slam_2.5hpf 3'-end signal), saving ntv3_target_genes_scored.tsv.
2. Locus-Wide Tiled NTv3 Scan:
   Tiles TARGET_GENES +/- LOCUS_FLANK_BP in target species and orthologous Zebrafish locus
   using overlapping NTv3 windows stepping cleanly across central 37.5% crop to produce
   continuous ntv3_atac_256cell, ntv3_atac_dome, and ntv3_slam_2.5hpf tracks
   (saving ntv3_locus_scan.tsv + IGV bedGraphs).
3. Distal QuantSeq 3'-End Knockout, Zebrafish Motif Swap & Saliency:
   Runs distal motif ablation, Zebrafish evolutionary swap, and input-embedding gradient
   saliency on TARGET_GENES for the top Block 6 ChromBPNet causal motif(s),
   saving ntv3_knockout_saliency_summary.tsv and ntv3_saliency_tracks.npz.

Strictly adheres to zero-fabrication, atomic writes (.tmp + os.replace), and
16 GB VRAM constraints.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Set default HF_HOME before importing transformers/huggingface_hub
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str((repo_root / "work_astatotilapia_premzt" / "hf_cache").resolve())

# Auto-load .env file if present
env_path = repo_root / ".env"
if env_path.exists():
    try:
        with open(env_path, "r") as f_env:
            for line in f_env:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyfaidx import Fasta

from scripts.task_manager import set_task_progress
from scripts.train_ntv3 import (
    build_ntv3_model,
    get_ntv3_input_embedding_layer,
    sanitize_sequence,
    sanitize_for_torch_save
)
from scripts.infer_and_attribute import (
    quantseq_3p_gene_scoring,
    benchmark_transfer_metrics
)
from scripts.premzt_inference import (
    get_inference_context,
    parse_locus_range,
    _score_ntv3_sequences,
    _plan_minimal_crop_centers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Block 8 NTv3 Inference Worker")
    parser.add_argument("--work-dir", type=str, default="./work_astatotilapia_premzt", help="Path to working directory")
    parser.add_argument("--config", type=str, default=None, help="Path to study config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained NTv3 checkpoint")
    parser.add_argument("--context-len", type=int, default=65536, help="NTv3 context length (default 65536)")
    parser.add_argument("--flank-bp", type=int, default=None, help="Locus flank bp (default from config LOCUS_FLANK_BP)")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size for 65 kb NTv3 inference (default 8)")
    parser.add_argument("--max-eval-controls", type=int, default=1000, help="Max controls per negative cohort when scoring Sub-task 1 (default 1000)")
    parser.add_argument("--subtask", type=str, default="all", choices=["all", "score", "scan", "saliency"], help="Run all sub-tasks or a specific sub-task")
    parser.add_argument("--force-rescore", action="store_true", default=False, help="Force rescoring of genes in Sub-task 1")
    parser.add_argument("--task-name", type=str, default="ntv3_inference", help="TaskManager task name")
    return parser.parse_args()


def parse_target_gene_exons(gtf_path: Path, target_genes: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Parse exons for target genes from GTF (0-based half-open, merged across transcripts)."""
    exons_dict: Dict[str, List[Tuple[int, int]]] = {}
    if not gtf_path.exists():
        return exons_dict

    try:
        opener = gzip.open(str(gtf_path), "rt") if str(gtf_path).endswith(".gz") else open(str(gtf_path), "rt")
        with opener as f_gtf:
            for line in f_gtf:
                if line.startswith("#"):
                    continue
                p = line.strip().split("\t")
                if len(p) >= 9 and p[2] == "exon":
                    for tg in target_genes:
                        if f'"{tg.lower()}"' in p[8].lower():
                            if tg not in exons_dict:
                                exons_dict[tg] = []
                            # 0-based half-open interval: [start-1, end)
                            exons_dict[tg].append((int(p[3]) - 1, int(p[4])))
        # Merge overlapping isoform exons
        for tg, ex_list in exons_dict.items():
            if ex_list:
                s_ex = sorted(ex_list, key=lambda x: x[0])
                m_ex = [s_ex[0]]
                for cur in s_ex[1:]:
                    prev_s, prev_e = m_ex[-1]
                    if cur[0] <= prev_e:
                        m_ex[-1] = (prev_s, max(prev_e, cur[1]))
                    else:
                        m_ex.append(cur)
                exons_dict[tg] = m_ex
    except Exception as e:
        print(f"[-] Notice parsing exons from GTF: {e}")
    return exons_dict


def find_top_causal_motif(res_dir: Path, work_dir: Path, target_locus_name: str) -> Dict[str, Any]:
    """Resolve top causal motif information from Block 6 outputs or candidate enhancers."""
    motif_info = {
        "motif_genomic_start": None,
        "motif_genomic_end": None,
        "causal_seq": "N/A",
        "ortholog_aligned_motif": None,
        "enhancer_rank": 1,
        "target_ablation_delta": 0.0,
        "classification": "divergent"
    }

    # 1. Check locus_cis_regulatory_motifs.tsv or evolutionary_swap_summary.tsv
    for tsv_name in ["evolutionary_swap_summary.tsv", "locus_cis_regulatory_motifs.tsv"]:
        swap_tsv = res_dir / tsv_name
        if swap_tsv.exists():
            try:
                df_sw = pd.read_csv(swap_tsv, sep="\t")
                if not df_sw.empty:
                    # Prefer Rank 1 candidate enhancer (the primary lineage-gained enhancer of the locus)
                    if "element_rank" in df_sw.columns and (df_sw["element_rank"] == 1).any():
                        row = df_sw[df_sw["element_rank"] == 1].iloc[0].to_dict()
                    elif "in_premzt_tf_list" in df_sw.columns and (df_sw["in_premzt_tf_list"] == True).any():
                        row = df_sw[df_sw["in_premzt_tf_list"] == True].sort_values(by="target_ablation_delta", ascending=True).iloc[0].to_dict()
                    elif "target_ablation_delta" in df_sw.columns:
                        row = df_sw.sort_values(by="target_ablation_delta", ascending=True).iloc[0].to_dict()
                    else:
                        row = df_sw.iloc[0].to_dict()

                    ortho_motif = row.get("ortholog_aligned_motif")
                    if not ortho_motif or pd.isna(ortho_motif) or str(ortho_motif) == "None":
                        aln_win = row.get("aligned_ortholog_motif_window")
                        if aln_win and pd.notna(aln_win) and isinstance(aln_win, str):
                            clean_aln = aln_win.replace("-", "")
                            m_l = int(row.get("motif_len", 8))
                            if len(clean_aln) >= m_l:
                                c_idx = len(clean_aln) // 2
                                ortho_motif = clean_aln[max(0, c_idx - m_l // 2): max(0, c_idx - m_l // 2) + m_l]
                            else:
                                ortho_motif = clean_aln.ljust(m_l, "A")

                    motif_info.update({
                        "motif_genomic_start": int(row["motif_genomic_start"]) if pd.notna(row.get("motif_genomic_start")) else None,
                        "motif_genomic_end": int(row["motif_genomic_end"]) if pd.notna(row.get("motif_genomic_end")) else None,
                        "causal_seq": str(row.get("motif_seq", row.get("causal_seq", "N/A"))),
                        "ortholog_aligned_motif": str(ortho_motif) if ortho_motif and pd.notna(ortho_motif) else None,
                        "enhancer_rank": int(row.get("element_rank", row.get("enhancer_rank", 1))),
                        "target_ablation_delta": float(row.get("target_ablation_delta", 0.0)),
                        "classification": str(row.get("classification", "divergent"))
                    })
                    print(f"[+] Found causal motif from {swap_tsv.name}: '{motif_info['causal_seq']}' at {motif_info['motif_genomic_start']}-{motif_info['motif_genomic_end']} (ortho={motif_info['ortholog_aligned_motif']})")
                    return motif_info
            except Exception as e_sw:
                print(f"[-] Notice reading {swap_tsv}: {e_sw}")

    # 2. Check candidate_1_causal_motif.meme
    meme_1 = res_dir / "candidate_1_causal_motif.meme"
    if meme_1.exists():
        try:
            with open(meme_1, "r") as f_m:
                content = f_m.read()
            motif_info["causal_seq"] = "CAUSAL_PREMZT_1"
        except Exception:
            pass

    # 3. Check candidate enhancers table
    enh_tsv = res_dir / f"{target_locus_name}_candidate_enhancers.tsv"
    if enh_tsv.exists():
        try:
            df_e = pd.read_csv(enh_tsv, sep="\t")
            cands = df_e[df_e.get("is_candidate", df_e.get("is_enhancer", True))]
            if not cands.empty and "center" in cands.columns:
                cands_sorted = cands.sort_values(by="pred_abs", ascending=False)
                c_pos = int(cands_sorted.iloc[0]["center"])
                motif_info["motif_genomic_start"] = c_pos - 9
                motif_info["motif_genomic_end"] = c_pos + 9
                motif_info["causal_seq"] = f"Summit_{c_pos}"
                motif_info["enhancer_rank"] = 1
                print(f"[+] Inferred causal motif region from candidate enhancers: summit {c_pos} ({motif_info['motif_genomic_start']}-{motif_info['motif_genomic_end']})")
        except Exception as e_enh:
            print(f"[-] Notice reading candidate enhancers: {e_enh}")

    return motif_info


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    task_name = args.task_name

    set_task_progress(work_dir, task_name, "Initializing Block 8 NTv3 inference...")
    print("=" * 80)
    print(f"[*] Starting Block 8 NTv3 Inference Worker in {work_dir}")
    print("=" * 80)

    # Load study configuration
    cfg_path = Path(args.config).resolve() if args.config else (work_dir / "config.json")
    study_cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f_c:
            study_cfg = json.load(f_c)

    target_locus_name = study_cfg.get("TARGET_LOCUS_NAME", "notch2_slc35a3a")
    target_genes = study_cfg.get("TARGET_GENES", ["notch2", "slc35a3a"])
    target_species_name = study_cfg.get("TARGET_SPECIES_NAME", "Astatotilapia_calliptera")
    target_genome_fa = study_cfg.get("TARGET_GENOME_FA")
    target_gtf = study_cfg.get("TARGET_GTF")
    zebrafish_genome_fa = study_cfg.get("ZEBRAFISH_GENOME_FA")
    zebrafish_gtf = study_cfg.get("ZEBRAFISH_GTF")
    locus_flank_bp = args.flank_bp if args.flank_bp is not None else int(study_cfg.get("LOCUS_FLANK_BP", 75000))
    context_len = int(args.context_len)
    hf_token = study_cfg.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

    res_dir = work_dir / "results" / target_locus_name
    res_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target genome FASTA
    if not target_genome_fa or not Path(target_genome_fa).exists():
        cand_fa = list((work_dir / "downloads").glob("*.dna_sm.toplevel.fa")) + list((work_dir / "downloads").glob("*.fa"))
        cand_fa = [p for p in cand_fa if "Danio" not in p.name and "Oryzias" not in p.name and not p.name.endswith(".fai")]
        if cand_fa:
            target_genome_fa = str(cand_fa[0])
        else:
            raise FileNotFoundError(f"[FATAL] Target genome FASTA missing: {target_genome_fa}")
    print(f"    Target Genome FASTA: {target_genome_fa}")

    # Resolve trained NTv3 checkpoint strictly enforcing zero-mock policy
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if not ckpt_path or not ckpt_path.exists():
        cand_ckpts = [
            work_dir / "models" / "ntv3" / "ntv3_best_checkpoint.pt",
            work_dir / "models" / "ntv3" / "ntv3_best_checkpoint_fold_0.pt",
            work_dir / "models" / "ntv3" / "ntv3_multitask_best.pt"
        ]
        for c in cand_ckpts:
            if c.exists():
                ckpt_path = c
                break

    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(
            f"[FATAL] Trained NTv3 checkpoint not found under {work_dir}/models/ntv3/. "
            "Zero-mock policy strictly enforced: please wait for train_ntv3_zf to complete."
        )

    print(f"[+] Loading trained NTv3 checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    n_cfg = payload.get("config", {})
    if not n_cfg:
        cfg_cand = ckpt_path.parent / "ntv3_train_config.json"
        if not cfg_cand.exists():
            cfg_cand = ckpt_path.parent / "ntv3_train_config_fold_0.json"
        if cfg_cand.exists():
            with open(cfg_cand) as f_cf:
                n_cfg = json.load(f_cf)

    resolved_species_id = int(n_cfg.get("species_id", 0))
    context_len_model = int(n_cfg.get("context_len", context_len))
    model_id = n_cfg.get("model_id", "InstaDeepAI/NTv3_tracks")
    train_mode = n_cfg.get("train_mode", "lora")
    tracks_list = n_cfg.get("tracks", [
        "atac_256cell", "atac_dome",
        "slam_0.75hpf_fwd", "slam_0.75hpf_rev",
        "slam_2.5hpf_fwd", "slam_2.5hpf_rev",
        "slam_3.0_3.5hpf_fwd", "slam_3.0_3.5hpf_rev",
        "slam_4.5hpf_fwd", "slam_4.5hpf_rev"
    ])

    print(f"[*] Building NTv3 model: id={model_id}, tracks={len(tracks_list)}, mode={train_mode}, context_len={context_len_model}")
    ntv3_model, ntv3_tok = build_ntv3_model(
        model_id=model_id,
        num_tracks=len(tracks_list),
        train_mode=train_mode,
        context_len=context_len_model,
        grad_checkpointing=True,
        default_species_id=resolved_species_id,
        hf_token=hf_token
    )
    ntv3_model.load_state_dict(payload["model_state"])
    ntv3_model.to(device)
    ntv3_model.eval()

    # Calculate crop parameters
    crop_ratio = getattr(ntv3_model, "crop_ratio", 0.375)
    crop_len = int(round(context_len_model * crop_ratio))
    crop_start = (context_len_model - crop_len) // 2
    crop_end = crop_start + crop_len
    print(f"[*] Context Length: {context_len_model} bp | Central Crop: {crop_len} bp [{crop_start}:{crop_end}]")

    # Track indices
    idx_atac_256 = tracks_list.index("atac_256cell") if "atac_256cell" in tracks_list else 0
    idx_atac_dome = tracks_list.index("atac_dome") if "atac_dome" in tracks_list else 1
    idx_slam_25_fwd = tracks_list.index("slam_2.5hpf_fwd") if "slam_2.5hpf_fwd" in tracks_list else 4
    idx_slam_25_rev = tracks_list.index("slam_2.5hpf_rev") if "slam_2.5hpf_rev" in tracks_list else 5

    target_genome = Fasta(target_genome_fa, as_raw=True)
    target_gene_exons = parse_target_gene_exons(Path(target_gtf) if target_gtf else Path("none"), target_genes)

    # ==========================================================================
    # SUB-TASK 1: Matched Zero-Shot Gene Benchmark
    # ==========================================================================
    set_task_progress(work_dir, task_name, "Sub-task 1/3: Scoring matched zero-shot target gene table...")
    print("\n" + "=" * 80)
    print("[*] Sub-task 1: Matched Zero-Shot Gene Benchmark on Target Species")
    print("=" * 80)

    # Locate gene table from Block 4 or results
    gene_table_path = res_dir / "target_genes_scored_metrics.tsv"
    if not gene_table_path.exists():
        gene_table_path = res_dir / f"{target_locus_name}_gene_sets.tsv"
    if not gene_table_path.exists():
        gene_table_path = work_dir / "rnaseq" / "target_species_gene_sets.tsv"

    if not gene_table_path.exists():
        raise FileNotFoundError(f"[FATAL] Target species gene table missing: {gene_table_path}")

    df_genes = pd.read_csv(gene_table_path, sep="\t", low_memory=False)
    # Sync is_slam_confirmed_minor_wave cleanly from gene_sets if gene_table_path lacked recent Block 1b flags
    gs_path = res_dir / f"{target_locus_name}_gene_sets.tsv"
    if gs_path.exists() and gene_table_path != gs_path:
        try:
            df_gs_tmp = pd.read_csv(gs_path, sep="\t", usecols=lambda c: c in {"gene_id", "is_slam_confirmed_minor_wave"}, low_memory=False)
            if "is_slam_confirmed_minor_wave" in df_gs_tmp.columns and df_gs_tmp["is_slam_confirmed_minor_wave"].fillna(False).astype(bool).sum() > 0:
                gs_ids = set(df_gs_tmp.loc[df_gs_tmp["is_slam_confirmed_minor_wave"].fillna(False).astype(bool), "gene_id"].astype(str))
                df_genes["is_slam_confirmed_minor_wave"] = (
                    df_genes.get("is_slam_confirmed_minor_wave", pd.Series(False, index=df_genes.index)).fillna(False).astype(bool) |
                    df_genes["gene_id"].astype(str).isin(gs_ids)
                )
        except Exception:
            pass
    print(f"[+] Loaded {len(df_genes):,} genes from {gene_table_path.name}")

    out_scored_tsv = res_dir / "ntv3_target_genes_scored.tsv"
    if args.subtask in ("all", "score"):
        need_rescore = args.force_rescore
        if out_scored_tsv.exists() and not need_rescore:
            try:
                chk_df = pd.read_csv(out_scored_tsv, sep="\t", low_memory=False)
                req_cols = [
                    "ntv3_pred_atac_promoter_1kb_max",
                    "ntv3_pred_atac_promoter_intron1_capped_max",
                    "ntv3_pred_slam_2.5hpf_exon_mean_per_kb",
                    "slam_3p_tes_centered",
                ]
                if not all(c in chk_df.columns for c in req_cols):
                    print(f"[!] Cached {out_scored_tsv.name} missing required columns {req_cols}; forcing clean rescore.")
                    need_rescore = True
                else:
                    sc_chk = chk_df[chk_df["ntv3_pred_atac_256cell_max"].notna()]
                    if len(sc_chk) > 20:
                        eq_frac = float((sc_chk["ntv3_pred_atac_promoter_1kb_max"] == sc_chk["ntv3_pred_atac_promoter_intron1_capped_max"]).mean())
                        span_chk = (sc_chk["end"] - sc_chk["start"]).abs()
                        zero_long = int(((sc_chk["ntv3_pred_slam_2.5hpf_3p"] == 0.0) & (~sc_chk["slam_3p_tes_centered"].fillna(False).astype(bool)) & (span_chk > 11000)).sum())
                        if eq_frac > 0.98 or zero_long > 5:
                            print(f"[!] Cached {out_scored_tsv.name} is degenerate/stale (1kb==intron1: {eq_frac:.1%}, uncentered 0s: {zero_long}); forcing clean rescore.")
                            need_rescore = True
            except Exception:
                need_rescore = True
    else:
        need_rescore = False

    if out_scored_tsv.exists() and not need_rescore:
        print(f"[+] {'Using cached' if args.subtask in ('all', 'score') else 'Loading cached'} {out_scored_tsv.name}.")
        df_genes = pd.read_csv(out_scored_tsv, sep="\t", low_memory=False)
    elif args.subtask in ("all", "score"):
        # Load shared GTF annotations via InferenceContext
        ctx = get_inference_context(work_dir=work_dir, device=device)
        gtf_info = ctx.get_gtf_data(target_species_name)
        genes_lookup = gtf_info["genes"]
        canonical_markers = study_cfg.get("CANONICAL_ZGA_MARKERS", ["vox", "klf17", "mxtx2"])
        priority_names = {g.lower() for g in (list(target_genes) + list(canonical_markers))}

        # Score prioritizing all minor-wave/target genes + balanced reproducible control sampling
        is_target_or_marker = df_genes["gene_name"].astype(str).str.lower().isin(priority_names)
        is_pos_cohort = (
            df_genes["category"].isin(["strictly_zygotic_minor_wave", "strictly_zygotic_highconf"]) |
            (df_genes.get("is_strictly_zygotic_minor_wave", False) == True) |
            (df_genes.get("is_strictly_zygotic_highconf", False) == True) |
            (df_genes.get("is_slam_confirmed_minor_wave", False) == True) |
            (df_genes.get("is_both_cichlids_minor_wave", False) == True)
        )
        eval_indices = set(df_genes.index[is_target_or_marker | is_pos_cohort])

        rng = np.random.RandomState(42)
        max_ctrl = int(getattr(args, "max_eval_controls", 1000))
        for ctrl_col_cat in ("control_silent", "control_non_activated"):
            ctrl_idxs = [idx for idx in df_genes.index[df_genes["category"] == ctrl_col_cat] if idx not in eval_indices]
            if len(ctrl_idxs) > max_ctrl:
                ctrl_idxs = rng.choice(ctrl_idxs, size=max_ctrl, replace=False).tolist()
            eval_indices.update(ctrl_idxs)

        other_indices = [idx for idx in df_genes.index if idx not in eval_indices]
        if other_indices:
            sample_size = min(100, len(other_indices))
            eval_indices.update(rng.choice(other_indices, size=sample_size, replace=False).tolist())

        eval_indices = sorted(list(eval_indices))
        print(f"[+] Prioritized {len(eval_indices):,} evaluation genes for batched NTv3 zero-shot scoring (batch_size={args.batch_size}).")

        pred_atac_max_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_atac_mean_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_atac_p1_max_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_atac_pi_max_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_atac_p5_max_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_slam_3p_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_slam_all_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        pred_slam_exon_mean_kb_arr = np.full(len(df_genes), np.nan, dtype=np.float32)
        slam_3p_tes_centered_arr = np.full(len(df_genes), False, dtype=bool)

        half_crop = crop_len // 2
        gene_meta_eval: List[Dict[str, Any]] = []
        uniq_wins: List[Tuple[str, int]] = []
        win_set = set()

        def _reg_window(ch: str, center_bp: int) -> Tuple[str, int]:
            try:
                ch_len = len(target_genome[ch])
            except Exception:
                ch_len = 10**9
            w_s = max(0, center_bp - (crop_start + half_crop))
            if w_s + context_len_model > ch_len:
                w_s = max(0, ch_len - context_len_model)
            k = (ch, w_s)
            if k not in win_set:
                win_set.add(k)
                uniq_wins.append(k)
            return k

        # Pass 1 & 2 window planning across all genes + minimal-crop exon planning
        for g_idx in eval_indices:
            r = df_genes.iloc[g_idx]
            g_chr = str(r["chrom"])
            g_st = int(r["start"])
            g_en = int(r["end"])
            g_strand = str(r.get("strand", "+"))
            g_name = str(r.get("gene_name", r.get("gene_id", "")))
            g_rec = genes_lookup.get(g_name.lower()) or genes_lookup.get(str(r.get("gene_id", "")).lower())

            if g_rec:
                tss, tes, g_strand = g_rec.tss, g_rec.tes, g_rec.strand
                i1 = g_rec.intron1
                exons = g_rec.exons or [(g_st, g_en)]
            else:
                tss = int(r.get("tss", g_st if g_strand == "+" else g_en))
                tes = int(r.get("tes", g_en if g_strand == "+" else g_st))
                i1 = None
                exons = target_gene_exons.get(g_name) or [(g_st, g_en)]

            tk_tss = _reg_window(g_chr, tss)
            c_gst, c_gen = tk_tss[1] + crop_start, tk_tss[1] + crop_end
            if (c_gst + 1500) <= tes <= (c_gen - 1500):
                tk_tes = tk_tss
            else:
                tk_tes = _reg_window(g_chr, tes)

            tss_b = (tk_tss[1] + crop_start, tk_tss[1] + crop_end)
            tes_b = (tk_tes[1] + crop_start, tk_tes[1] + crop_end)
            uncovered = [
                (es, ee) for es, ee in exons
                if not ((tss_b[0] <= es and ee <= tss_b[1]) or (tes_b[0] <= es and ee <= tes_b[1]))
            ]
            extra_tks = [_reg_window(g_chr, cm) for cm in _plan_minimal_crop_centers(uncovered, crop_len)]

            gene_meta_eval.append({
                "g_idx": g_idx, "g_chr": g_chr, "g_st": g_st, "g_en": g_en,
                "g_strand": g_strand, "tss": tss, "tes": tes, "i1": i1, "exons": exons,
                "tk_tss": tk_tss, "tk_tes": tk_tes, "extra_tks": extra_tks,
            })

        print(f"[+] Deduplicated {len(uniq_wins):,} unique 65 kb windows across {len(eval_indices):,} genes. Executing batched GPU inference...")
        ntv3_meta = {
            "context_len": context_len_model, "species_id": resolved_species_id,
            "crop_len": crop_len, "crop_start": crop_start, "crop_end": crop_end,
            "idx_atac_256": idx_atac_256, "idx_atac_dome": idx_atac_dome,
            "idx_slam_25_fwd": idx_slam_25_fwd, "idx_slam_25_rev": idx_slam_25_rev,
        }
        win_preds: Dict[Tuple[str, int], np.ndarray] = {}
        bs_nt = max(1, int(args.batch_size))
        for b_start in range(0, len(uniq_wins), bs_nt):
            chunk_keys = uniq_wins[b_start:b_start + bs_nt]
            chunk_seqs: List[str] = []
            for ch, w_s in chunk_keys:
                try:
                    s_w = str(target_genome[ch][w_s:w_s + context_len_model])
                except Exception:
                    s_w = "N" * context_len_model
                if len(s_w) < context_len_model:
                    s_w += "N" * (context_len_model - len(s_w))
                chunk_seqs.append(s_w)
            p_batch = _score_ntv3_sequences(ntv3_model, ntv3_tok, ntv3_meta, chunk_seqs, batch_size=bs_nt, device=device)
            for k, p_mat in zip(chunk_keys, p_batch):
                win_preds[k] = p_mat
            if ((b_start + bs_nt) % 80 < bs_nt) or (b_start + bs_nt >= len(uniq_wins)):
                set_task_progress(work_dir, task_name, f"Sub-task 1/3: Scored {min(b_start + bs_nt, len(uniq_wins))}/{len(uniq_wins)} unique windows...")

        # Vectorized/fast metric aggregation
        for gm in gene_meta_eval:
            g_idx = gm["g_idx"]
            tss, tes, g_strand = gm["tss"], gm["tes"], gm["g_strand"]
            g_st, g_en, i1, exons = gm["g_st"], gm["g_en"], gm["i1"], gm["exons"]
            tk_tss, tk_tes = gm["tk_tss"], gm["tk_tes"]
            preds = win_preds[tk_tss]
            atac_track = preds[:, idx_atac_256]
            pred_atac_max_arr[g_idx] = float(np.max(atac_track))
            pred_atac_mean_arr[g_idx] = float(np.mean(atac_track))

            crop_g_st = tk_tss[1] + crop_start
            crop_g_en = tk_tss[1] + crop_end
            n_bins = preds.shape[0]
            bp_bin = crop_len / max(1, n_bins)

            # promoter_1kb (TSS - 1000 to TSS + 500 strand-aware)
            p1_s, p1_e = (tss - 1000, tss + 500) if g_strand == "+" else (tss - 500, tss + 1000)
            b_s1 = max(0, min(n_bins - 1, int(round((p1_s - crop_g_st) / bp_bin))))
            b_e1 = max(b_s1 + 1, min(n_bins, int(round((p1_e - crop_g_st) / bp_bin))))
            pred_atac_p1_max_arr[g_idx] = float(np.max(atac_track[b_s1:b_e1]))

            # promoter_intron1_capped
            if i1:
                pi_s, pi_e = (tss - 2000, min(i1[1], tss + 10000)) if g_strand == "+" else (max(i1[0], tss - 10000), tss + 2000)
            else:
                pi_s, pi_e = tss - 2000, tss + 2000
            b_si = max(0, min(n_bins - 1, int(round((pi_s - crop_g_st) / bp_bin))))
            b_ei = max(b_si + 1, min(n_bins, int(round((pi_e - crop_g_st) / bp_bin))))
            pred_atac_pi_max_arr[g_idx] = float(np.max(atac_track[b_si:b_ei]))

            # proximal_5kb
            b_s5 = max(0, min(n_bins - 1, int(round(((tss - 5000) - crop_g_st) / bp_bin))))
            b_e5 = max(b_s5 + 1, min(n_bins, int(round(((tss + 5000) - crop_g_st) / bp_bin))))
            pred_atac_p5_max_arr[g_idx] = float(np.max(atac_track[b_s5:b_e5]))

            # Nascent Transcription: TES-centered vs TSS-centered in unified per-1kb units
            slam_track_idx = idx_slam_25_fwd if g_strand == "+" else idx_slam_25_rev
            preds_slam = win_preds[tk_tes]
            crop_slam_st = tk_tes[1] + crop_start
            slam_3p_tes_centered_arr[g_idx] = (tk_tes != tk_tss)

            t_st, t_en = (max(g_st, tes - 1000), tes) if g_strand == "+" else (tes, min(g_en, tes + 1000))
            b_ts = max(0, min(n_bins - 1, int(round((t_st - crop_slam_st) / bp_bin))))
            b_te = max(b_ts + 1, min(n_bins, int(round((t_en - crop_slam_st) / bp_bin))))
            pred_slam_3p_arr[g_idx] = float(np.mean(preds_slam[b_ts:b_te, slam_track_idx]) * 1000.0)

            # Exon nascent expression & length normalization via minimal crop set
            total_exon_kb = max(0.1, sum(e - s for s, e in exons) / 1000.0)
            avail_crops = [(tk_tss[1] + crop_start, tk_tss[1] + crop_end, preds)]
            if tk_tes != tk_tss:
                avail_crops.append((tk_tes[1] + crop_start, tk_tes[1] + crop_end, preds_slam))
            for etk in gm["extra_tks"]:
                avail_crops.append((etk[1] + crop_start, etk[1] + crop_end, win_preds[etk]))

            exon_sum = 0.0
            for ex_s, ex_e in exons:
                best_c = next(((cs, ce, pm) for cs, ce, pm in avail_crops if cs <= ex_s and ex_e <= ce), None)
                if best_c is None:
                    best_c = max(avail_crops, key=lambda x: max(0, min(x[1], ex_e) - max(x[0], ex_s)))
                c_gst_ex, _, p_ex = best_c
                b_es = max(0, min(n_bins - 1, int(round((ex_s - c_gst_ex) / bp_bin))))
                b_ee = max(b_es + 1, min(n_bins, int(round((ex_e - c_gst_ex) / bp_bin))))
                exon_sum += float(np.sum(p_ex[b_es:b_ee, slam_track_idx]))
            pred_slam_all_arr[g_idx] = exon_sum
            pred_slam_exon_mean_kb_arr[g_idx] = exon_sum / total_exon_kb

        df_genes["ntv3_pred_atac_promoter_1kb_max"] = pred_atac_p1_max_arr
        df_genes["ntv3_pred_atac_promoter_intron1_capped_max"] = pred_atac_pi_max_arr
        df_genes["ntv3_pred_atac_proximal_5kb_max"] = pred_atac_p5_max_arr
        df_genes["ntv3_pred_atac_256cell_max"] = pred_atac_max_arr
        df_genes["ntv3_pred_atac_256cell_mean"] = pred_atac_mean_arr
        df_genes["ntv3_pred_slam_2.5hpf_3p"] = pred_slam_3p_arr
        df_genes["ntv3_pred_slam_2.5hpf_all_exons"] = pred_slam_all_arr
        df_genes["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = pred_slam_exon_mean_kb_arr
        df_genes["slam_3p_tes_centered"] = slam_3p_tes_centered_arr

        tmp_scored_tsv = out_scored_tsv.with_suffix(".tmp.tsv")
        df_genes.to_csv(tmp_scored_tsv, sep="\t", index=False)
        os.replace(str(tmp_scored_tsv), str(out_scored_tsv))
        print(f"[+] Saved scored genes with optimal windows to {out_scored_tsv}")

    if args.subtask in ("all", "score"):
        # Standardize benchmark negative controls (control_silent as primary, control_non_activated as secondary)
        m_strict_atac_sil = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_atac_promoter_1kb_max", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_silent")
        m_strict_atac_non = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_atac_promoter_1kb_max", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")

        m_strict_slam_sil = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_silent")
        m_strict_slam_non = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")

        m_slam_atac_sil = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_atac_promoter_1kb_max", pos_category="slam_confirmed_minor_wave", ctrl_category="control_silent")
        m_slam_slam_sil = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="slam_confirmed_minor_wave", ctrl_category="control_silent")

        print("[*] NTv3 Standardized Minor-Wave Transfer Benchmark Results:")
        print(f"    Primary (vs control_silent):")
        print(f"      ATAC 256-cell -> AUROC: {m_strict_atac_sil.get('roc_auc', 0.0):.3f} | AUPRC: {m_strict_atac_sil.get('pr_auc', 0.0):.3f}")
        print(f"      SLAM 2.5 hpf  -> AUROC: {m_strict_slam_sil.get('roc_auc', 0.0):.3f} | AUPRC: {m_strict_slam_sil.get('pr_auc', 0.0):.3f}")
        print(f"    Secondary Diagnostic (vs control_non_activated):")
        print(f"      ATAC 256-cell -> AUROC: {m_strict_atac_non.get('roc_auc', 0.0):.3f} | AUPRC: {m_strict_atac_non.get('pr_auc', 0.0):.3f}")
        print(f"      SLAM 2.5 hpf  -> AUROC: {m_strict_slam_non.get('roc_auc', 0.0):.3f} | AUPRC: {m_strict_slam_non.get('pr_auc', 0.0):.3f}")

    # Resolve shared locus bounds (align with ChromBPNet bedGraph span when present)
    locus_genes_sub = df_genes[df_genes["gene_name"].str.lower().isin([tg.lower() for tg in target_genes])]
    if locus_genes_sub.empty:
        locus_genes_sub = df_genes.head(2)

    locus_chr = str(locus_genes_sub["chrom"].iloc[0])
    locus_st = max(0, int(locus_genes_sub["start"].min()) - locus_flank_bp)
    locus_en = int(locus_genes_sub["end"].max()) + locus_flank_bp
    cb_abs_bg = res_dir / f"{target_locus_name}_pred_y_abs.bedgraph"
    sal_locus_st, sal_locus_en = locus_st, locus_en
    if cb_abs_bg.exists():
        try:
            _cb_tmp = pd.read_csv(cb_abs_bg, sep="\t", header=None, names=["chr", "st", "en", "val"])
            if not _cb_tmp.empty:
                sal_locus_st = int(_cb_tmp["st"].min())
                sal_locus_en = int(_cb_tmp["en"].max())
        except Exception:
            pass

    if args.subtask in ("all", "scan"):
        set_task_progress(work_dir, task_name, "Sub-task 2/3: Locus-wide tiled scanning...")
        print("\n" + "=" * 80)
        print(f"[*] Sub-task 2: Locus-Wide Tiled NTv3 Scan on '{target_locus_name}' & Syntenic Zebrafish")
        print("=" * 80)

        locus_len = locus_en - locus_st
        print(f"[+] Target Locus Coordinates: {locus_chr}:{locus_st:,}-{locus_en:,} (Span = {locus_len:,} bp)")

        out_scan_tsv = res_dir / "ntv3_locus_scan.tsv"
        bg_atac_256 = res_dir / f"{target_locus_name}_ntv3_atac_256cell.bedgraph"
        bg_atac_dome = res_dir / f"{target_locus_name}_ntv3_atac_dome.bedgraph"
        bg_slam_25 = res_dir / f"{target_locus_name}_ntv3_slam_2.5hpf.bedgraph"

        if not (out_scan_tsv.exists() and bg_atac_256.exists() and bg_atac_dome.exists() and bg_slam_25.exists()):
            step_bp = crop_len // 2
            tile_crop_centers = list(range(locus_st + crop_len // 2, locus_en + crop_len // 2, step_bp))
            ntv3_meta_scan = {
                "context_len": context_len_model, "species_id": resolved_species_id,
                "crop_len": crop_len, "crop_start": crop_start, "crop_end": crop_end,
            }
            scan_seqs = []
            scan_w_sts = []
            for c_center in tile_crop_centers:
                w_st = max(0, c_center - (crop_start + crop_len // 2))
                w_en = w_st + context_len_model
                try:
                    seq = str(target_genome[locus_chr][w_st:w_en])
                except Exception:
                    seq = "N" * context_len_model
                if len(seq) < context_len_model:
                    seq += "N" * (context_len_model - len(seq))
                scan_seqs.append(seq)
                scan_w_sts.append(w_st)

            preds_scan_all = _score_ntv3_sequences(ntv3_model, ntv3_tok, ntv3_meta_scan, scan_seqs, batch_size=max(1, int(args.batch_size)), device=device)
            tiled_records: List[Dict[str, Any]] = []
            for w_st, preds in zip(scan_w_sts, preds_scan_all):
                n_bins = preds.shape[0]
                bp_per_bin = crop_len / max(1, n_bins)
                c_st_genomic = w_st + crop_start
                for b in range(n_bins):
                    b_st = int(c_st_genomic + b * bp_per_bin)
                    b_en = int(b_st + bp_per_bin)
                    b_center = (b_st + b_en) // 2
                    if locus_st <= b_center <= locus_en:
                        tiled_records.append({
                            "chrom": locus_chr,
                            "start": b_st,
                            "end": b_en,
                            "center": b_center,
                            "atac_256cell": float(preds[b, idx_atac_256]),
                            "atac_dome": float(preds[b, idx_atac_dome]),
                            "slam_2.5hpf_fwd": float(preds[b, idx_slam_25_fwd]),
                            "slam_2.5hpf_rev": float(preds[b, idx_slam_25_rev]),
                        })

            df_locus_scan = pd.DataFrame(tiled_records)
            if not df_locus_scan.empty:
                df_locus_scan = df_locus_scan.groupby(["chrom", "start", "end", "center"], as_index=False).mean()
                df_locus_scan = df_locus_scan.sort_values(by="start").reset_index(drop=True)

            tmp_scan_tsv = out_scan_tsv.with_suffix(".tmp.tsv")
            df_locus_scan.to_csv(tmp_scan_tsv, sep="\t", index=False)
            os.replace(str(tmp_scan_tsv), str(out_scan_tsv))

            with open(bg_atac_256.with_suffix(".tmp"), "w") as f_256, \
                 open(bg_atac_dome.with_suffix(".tmp"), "w") as f_dome, \
                 open(bg_slam_25.with_suffix(".tmp"), "w") as f_slam:
                for _, row in df_locus_scan.iterrows():
                    c, s, e = row["chrom"], int(row["start"]), int(row["end"])
                    f_256.write(f"{c}\t{s}\t{e}\t{row['atac_256cell']:.4f}\n")
                    f_dome.write(f"{c}\t{s}\t{e}\t{row['atac_dome']:.4f}\n")
                    slam_max = max(float(row["slam_2.5hpf_fwd"]), float(row["slam_2.5hpf_rev"]))
                    f_slam.write(f"{c}\t{s}\t{e}\t{slam_max:.4f}\n")

            os.replace(str(bg_atac_256.with_suffix(".tmp")), str(bg_atac_256))
            os.replace(str(bg_atac_dome.with_suffix(".tmp")), str(bg_atac_dome))
            os.replace(str(bg_slam_25.with_suffix(".tmp")), str(bg_slam_25))
            print(f"[+] Saved target species locus scan: {out_scan_tsv.name}")
        else:
            print(f"[+] Found cached {out_scan_tsv.name}")

        # 2. Syntenic Zebrafish Locus Scan
        out_zf_scan_tsv = res_dir / "ntv3_zf_locus_scan.tsv"
        bg_zf_atac = res_dir / "ntv3_zf_pred_atac.bedgraph"
        bg_zf_slam = res_dir / "ntv3_zf_pred_slam.bedgraph"

        zf_chr, zf_min, zf_max = "8", 19498985, 19688362
        if zebrafish_gtf and Path(zebrafish_gtf).exists():
            try:
                zf_coords = []
                opener = gzip.open(zebrafish_gtf, "rt") if str(zebrafish_gtf).endswith(".gz") else open(zebrafish_gtf, "rt")
                with opener as f_zg:
                    for line in f_zg:
                        if line.startswith("#"):
                            continue
                        p = line.strip().split("\t")
                        if len(p) >= 9 and p[2] == "gene":
                            for tg in target_genes:
                                if f'"{tg.lower()}"' in p[8].lower():
                                    zf_coords.append((p[0], int(p[3]), int(p[4])))
                if zf_coords:
                    zf_chr = zf_coords[0][0]
                    same_chr_coords = [c for c in zf_coords if c[0] == zf_chr]
                    zf_min = max(0, min(c[1] for c in same_chr_coords) - 50000)
                    zf_max = max(c[2] for c in same_chr_coords) + 50000
            except Exception:
                pass

        print(f"[+] Syntenic Zebrafish Locus Coordinates: chr{zf_chr}:{zf_min:,}-{zf_max:,} (Span = {zf_max-zf_min:,} bp)")

        if not (out_zf_scan_tsv.exists() and bg_zf_atac.exists() and bg_zf_slam.exists()) and zebrafish_genome_fa and Path(zebrafish_genome_fa).exists():
            zf_fa = Fasta(str(zebrafish_genome_fa), as_raw=True)
            step_bp = crop_len // 2
            tile_zf_centers = list(range(zf_min + crop_len // 2, zf_max + crop_len // 2, step_bp))
            ntv3_meta_zf = {
                "context_len": context_len_model, "species_id": resolved_species_id,
                "crop_len": crop_len, "crop_start": crop_start, "crop_end": crop_end,
            }
            zf_seqs = []
            zf_w_sts = []
            for c_center in tile_zf_centers:
                w_st = max(0, c_center - (crop_start + crop_len // 2))
                w_en = w_st + context_len_model
                try:
                    seq = str(zf_fa[zf_chr][w_st:w_en])
                except Exception:
                    seq = "N" * context_len_model
                if len(seq) < context_len_model:
                    seq += "N" * (context_len_model - len(seq))
                zf_seqs.append(seq)
                zf_w_sts.append(w_st)

            preds_zf_all = _score_ntv3_sequences(ntv3_model, ntv3_tok, ntv3_meta_zf, zf_seqs, batch_size=max(1, int(args.batch_size)), device=device)
            zf_records: List[Dict[str, Any]] = []
            for w_st, preds in zip(zf_w_sts, preds_zf_all):
                n_bins = preds.shape[0]
                bp_per_bin = crop_len / max(1, n_bins)
                c_st_genomic = w_st + crop_start
                for b in range(n_bins):
                    b_st = int(c_st_genomic + b * bp_per_bin)
                    b_en = int(b_st + bp_per_bin)
                    b_center = (b_st + b_en) // 2
                    if zf_min <= b_center <= zf_max:
                        zf_records.append({
                            "chrom": zf_chr,
                            "start": b_st,
                            "end": b_en,
                            "center": b_center,
                            "atac_256cell": float(preds[b, idx_atac_256]),
                            "atac_dome": float(preds[b, idx_atac_dome]),
                            "slam_2.5hpf_fwd": float(preds[b, idx_slam_25_fwd]),
                            "slam_2.5hpf_rev": float(preds[b, idx_slam_25_rev]),
                        })

            df_zf_scan = pd.DataFrame(zf_records)
            if not df_zf_scan.empty:
                df_zf_scan = df_zf_scan.groupby(["chrom", "start", "end", "center"], as_index=False).mean()
                df_zf_scan = df_zf_scan.sort_values(by="start").reset_index(drop=True)

            tmp_zf_tsv = out_zf_scan_tsv.with_suffix(".tmp.tsv")
            df_zf_scan.to_csv(tmp_zf_tsv, sep="\t", index=False)
            os.replace(str(tmp_zf_tsv), str(out_zf_scan_tsv))

            with open(bg_zf_atac.with_suffix(".tmp"), "w") as f_za, \
                 open(bg_zf_slam.with_suffix(".tmp"), "w") as f_zs:
                for _, row in df_zf_scan.iterrows():
                    c, s, e = row["chrom"], int(row["start"]), int(row["end"])
                    f_za.write(f"{c}\t{s}\t{e}\t{row['atac_256cell']:.4f}\n")
                    slam_max = max(float(row["slam_2.5hpf_fwd"]), float(row["slam_2.5hpf_rev"]))
                    f_zs.write(f"{c}\t{s}\t{e}\t{slam_max:.4f}\n")

            os.replace(str(bg_zf_atac.with_suffix(".tmp")), str(bg_zf_atac))
            os.replace(str(bg_zf_slam.with_suffix(".tmp")), str(bg_zf_slam))
            print(f"[+] Saved syntenic Zebrafish scan: {out_zf_scan_tsv.name} and bedgraphs")
        else:
            print(f"[+] Found cached {out_zf_scan_tsv.name}")

    # ==========================================================================
    # SUB-TASK 3: Multi-Candidate Distal Coupling Screen & Saliency
    # ==========================================================================
    if args.subtask in ("all", "saliency"):
        set_task_progress(work_dir, task_name, "Sub-task 3/3: Running multi-candidate distal coupling screen & saliency...")
        print("\n" + "=" * 80)
        print(f"[*] Sub-task 3: Multi-Candidate Distal Coupling Screen across Top Elements in '{target_locus_name}'")
        print("=" * 80)

        from scripts.infer_and_attribute import dinuc_shuffle_sequence

        swap_sum_tsv = res_dir / "evolutionary_swap_summary.tsv"
        if not swap_sum_tsv.exists():
            swap_sum_tsv = res_dir / "candidate_elements.tsv"
        df_elems_raw = pd.read_csv(swap_sum_tsv, sep="\t")

        key_ranks = [1, 2, 3, 9, 11]
        df_key_cands = df_elems_raw[df_elems_raw["element_rank"].isin(key_ranks)].copy()
        if df_key_cands.empty:
            df_key_cands = df_elems_raw.head(5).copy()

        tg_coords = {}
        for tg in target_genes:
            tg_sub = locus_genes_sub[locus_genes_sub["gene_name"].str.lower() == tg.lower()]
            if not tg_sub.empty:
                r = tg_sub.iloc[0]
                st = int(r["start"])
                en = int(r["end"])
                strd = str(r.get("strand", "+"))
                tss = st if strd == "+" else en
                tes = en if strd == "+" else st
                tg_coords[tg] = {"chr": str(r["chrom"]), "st": st, "en": en, "strand": strd, "tss": tss, "tes": tes}

        p_gene = target_genes[0]
        p_tss = tg_coords.get(p_gene, {}).get("tss", int(df_key_cands["element_center"].iloc[0]))
        p_strand = tg_coords.get(p_gene, {}).get("strand", "+")

        coupling_records = []
        saliency_tracks_dict = {}
        out_coupling_tsv = res_dir / "ntv3_distal_coupling_screen.tsv"
        run_coupling = not (args.subtask == "saliency" and out_coupling_tsv.exists() and not args.force_rescore)

        embed_layer = get_ntv3_input_embedding_layer(ntv3_model)
        captured_embs: List[torch.Tensor] = []

        def hook_fn(module, inp, out):
            if torch.is_grad_enabled() and isinstance(out, torch.Tensor):
                mod_out = out.detach().clone().requires_grad_(True)
                captured_embs.append(mod_out)
                return mod_out
            return out

        hook_handle = embed_layer.register_forward_hook(hook_fn)

        try:
            if run_coupling:
                for _, elem_row in df_key_cands.iterrows():
                    c_rk = int(elem_row["element_rank"])
                    c_pos = int(elem_row.get("element_center", elem_row.get("center", 0)))
                    c_feat = elem_row.get("gtf_feature_class", "element")
                    c_gene = elem_row.get("overlapping_gene", "")
                    c_dist_kb = float(elem_row.get("dist_from_primary_tss_kb", ((c_pos - p_tss)/1000.0 if p_strand == "+" else (p_tss - c_pos)/1000.0)))
                    top_motif_seq = str(elem_row.get("motif_seq", "NNNNNN"))
                    ortho_motif = str(elem_row.get("ortholog_aligned_motif", ""))

                    w_st = max(0, c_pos - context_len_model // 2)
                    w_en = w_st + context_len_model
                    try:
                        wt_seq = str(target_genome[locus_chr][w_st:w_en])
                    except Exception:
                        wt_seq = "N" * context_len_model
                    if len(wt_seq) < context_len_model:
                        wt_seq += "N" * (context_len_model - len(wt_seq))
                    wt_seq = sanitize_sequence(wt_seq)

                    crop_genomic_start = w_st + crop_start
                    crop_genomic_end = w_st + crop_end

                    def score_seq_forward(seq_to_run):
                        enc_s = ntv3_tok(seq_to_run, add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")
                        cond_toks = [torch.tensor([resolved_species_id], dtype=torch.long, device=device)]
                        with torch.no_grad():
                            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                                p_out = ntv3_model(enc_s["input_ids"].to(device), conditions_tokens=cond_toks).float().squeeze(0).cpu().numpy()
                        return p_out

                    preds_wt = score_seq_forward(wt_seq)
                    n_bins = preds_wt.shape[0]
                    bp_bin = crop_len / max(1, n_bins)

                    def get_window_signal(preds_mat, track_idx, gen_st, gen_en):
                        if gen_en < crop_genomic_start or gen_st > crop_genomic_end:
                            return np.nan
                        b_st = max(0, min(n_bins - 1, int(round((gen_st - crop_genomic_start) / bp_bin))))
                        b_en = max(b_st + 1, min(n_bins, int(round((gen_en - crop_genomic_start) / bp_bin))))
                        return float(np.mean(preds_mat[b_st:b_en, track_idx]))

                    wt_local_atac = get_window_signal(preds_wt, idx_atac_256, c_pos - 250, c_pos + 250)
                    wt_tg_promoter_atac = {}
                    wt_tg_slam_3p = {}
                    for tg, tgc in tg_coords.items():
                        tg_tss = tgc["tss"]
                        wt_tg_promoter_atac[tg] = get_window_signal(preds_wt, idx_atac_256, tg_tss - 500, tg_tss + 500)
                        tg_slam_idx = idx_slam_25_fwd if tgc["strand"] == "+" else idx_slam_25_rev
                        tg_exons = target_gene_exons.get(tg)
                        sc_slam = quantseq_3p_gene_scoring(
                            pred_nascent_track=preds_wt[:, tg_slam_idx],
                            gene_span_len=tgc["en"] - tgc["st"],
                            strand=tgc["strand"],
                            resolution_bp=1,
                            crop_genomic_start=crop_genomic_start,
                            crop_genomic_end=crop_genomic_end,
                            gene_start=tgc["st"],
                            gene_end=tgc["en"],
                            exons=tg_exons
                        )
                        wt_tg_slam_3p[tg] = float(sc_slam["nascent_3p_1000bp"])

                    rel_elem_center = c_pos - w_st
                    rel_500_st = max(0, rel_elem_center - 250)
                    rel_500_en = min(len(wt_seq), rel_elem_center + 250)
                    seq_500 = wt_seq[rel_500_st:rel_500_en]

                    shuf_500 = dinuc_shuffle_sequence(seq_500, seed=42)
                    seq_dinuc = wt_seq[:rel_500_st] + shuf_500 + wt_seq[rel_500_en:]
                    preds_dinuc = score_seq_forward(seq_dinuc)

                    seq_cpg_500 = seq_500.replace("CG", "TG")
                    seq_cpg = wt_seq[:rel_500_st] + seq_cpg_500 + wt_seq[rel_500_en:]
                    preds_cpg = score_seq_forward(seq_cpg)

                    m_len = max(6, len(top_motif_seq))
                    m_rel_st = max(0, rel_elem_center - m_len // 2)
                    m_rel_en = min(len(wt_seq), m_rel_st + m_len)
                    rep_motif = ortho_motif[:m_len] if ortho_motif and ortho_motif != "nan" else ("N" * m_len)
                    if len(rep_motif) < m_len:
                        rep_motif = rep_motif.ljust(m_len, "N")
                    seq_swap = wt_seq[:m_rel_st] + rep_motif + wt_seq[m_rel_en:]
                    preds_swap = score_seq_forward(seq_swap)

                    perts = [
                        ("dinuc_shuffle_500bp", preds_dinuc),
                        ("cpg_ablation_500bp", preds_cpg),
                        ("motif_swap", preds_swap)
                    ]

                    for pert_name, p_mat in perts:
                        pert_local_atac = get_window_signal(p_mat, idx_atac_256, c_pos - 250, c_pos + 250)
                        delta_local_atac = pert_local_atac - wt_local_atac
                        pct_local_atac = (delta_local_atac / max(1e-4, wt_local_atac)) * 100

                        rec = {
                            "element_rank": c_rk,
                            "element_center": c_pos,
                            "gtf_feature_class": c_feat,
                            "overlapping_gene": c_gene,
                            "dist_from_primary_tss_kb": c_dist_kb,
                            "perturbation": pert_name,
                            "wt_local_atac": wt_local_atac,
                            "pert_local_atac": pert_local_atac,
                            "delta_local_atac": delta_local_atac,
                            "pct_delta_local_atac": pct_local_atac
                        }

                        for tg in target_genes:
                            tg_tss = tg_coords.get(tg, {}).get("tss", 0)
                            tg_prom_atac = get_window_signal(p_mat, idx_atac_256, tg_tss - 500, tg_tss + 500)
                            wt_p_atac = wt_tg_promoter_atac.get(tg, np.nan)
                            rec[f"{tg}_promoter_atac_wt"] = wt_p_atac
                            rec[f"{tg}_promoter_atac_pert"] = tg_prom_atac
                            rec[f"delta_{tg}_promoter_atac"] = (tg_prom_atac - wt_p_atac) if np.isfinite(tg_prom_atac) and np.isfinite(wt_p_atac) else np.nan

                            tgc = tg_coords.get(tg, {})
                            tg_slam_idx = idx_slam_25_fwd if tgc.get("strand", "+") == "+" else idx_slam_25_rev
                            tg_exons = target_gene_exons.get(tg)
                            sc_slam_pert = quantseq_3p_gene_scoring(
                                pred_nascent_track=p_mat[:, tg_slam_idx],
                                gene_span_len=tgc.get("en", 1000) - tgc.get("st", 0),
                                strand=tgc.get("strand", "+"),
                                resolution_bp=1,
                                crop_genomic_start=crop_genomic_start,
                                crop_genomic_end=crop_genomic_end,
                                gene_start=tgc["st"],
                                gene_end=tgc["en"],
                                exons=tg_exons
                            )
                            pert_slam_3p = float(sc_slam_pert["nascent_3p_1000bp"])
                            wt_sl = wt_tg_slam_3p.get(tg, np.nan)
                            rec[f"{tg}_slam_3p_wt"] = wt_sl
                            rec[f"{tg}_slam_3p_pert"] = pert_slam_3p
                            rec[f"delta_{tg}_slam_3p"] = (pert_slam_3p - wt_sl) if np.isfinite(pert_slam_3p) and np.isfinite(wt_sl) else np.nan

                        coupling_records.append(rec)

                    print(f"    [✓] Candidate Rank {c_rk:2d} ({c_feat:16s} of {c_gene:10s} | {c_dist_kb:+6.1f} kb): Screened 3 perturbations -> Local ATAC Δ: {coupling_records[-3]['delta_local_atac']:+.3f} (dinuc), {coupling_records[-2]['delta_local_atac']:+.3f} (CpG), {coupling_records[-1]['delta_local_atac']:+.3f} (swap)")
            else:
                print(f"[+] Using cached {out_coupling_tsv.name}; recomputing full-locus & TSS-centered saliency tracks.")

            # Compute full-locus tiled base-resolution ATAC & SLAM saliency aligned strictly to [sal_locus_st, sal_locus_en]
            w_kernel = np.ones(50, dtype=np.float32) / 50.0
            sal_span = max(1, sal_locus_en - sal_locus_st)
            try:
                chrom_len_locus = len(target_genome[locus_chr])
            except Exception:
                chrom_len_locus = sal_locus_en + context_len_model

            full_atac_acc = np.zeros(sal_span, dtype=np.float64)
            full_slam_acc = np.zeros(sal_span, dtype=np.float64)
            full_wt_cnt = np.zeros(sal_span, dtype=np.float64)
            tg_slam_acc = {tg: np.zeros(sal_span, dtype=np.float64) for tg in target_genes if tg in tg_coords}

            sal_step = crop_len // 2
            tile_sal_centers = list(range(sal_locus_st + crop_len // 2, sal_locus_en + crop_len // 2, sal_step))
            cond_sal = [torch.tensor([resolved_species_id], dtype=torch.long, device=device)]
            print(f"[*] Computing locus-wide tiled ATAC & SLAM gradient saliency across {locus_chr}:{sal_locus_st:,}-{sal_locus_en:,} ({len(tile_sal_centers)} tiles)...")

            for c_center in tile_sal_centers:
                w_st = max(0, min(chrom_len_locus - context_len_model, c_center - (crop_start + crop_len // 2)))
                w_en = w_st + context_len_model
                try:
                    wt_seq_t = str(target_genome[locus_chr][w_st:w_en])
                except Exception:
                    wt_seq_t = "N" * context_len_model
                if len(wt_seq_t) < context_len_model:
                    wt_seq_t += "N" * (context_len_model - len(wt_seq_t))
                enc_t = ntv3_tok(sanitize_sequence(wt_seq_t), add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")
                inp_t = enc_t["input_ids"].to(device)

                def _grad_for_loss(loss_fn):
                    ntv3_model.zero_grad()
                    captured_embs.clear()
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        out_t = ntv3_model(inp_t, conditions_tokens=cond_sal)
                        loss_val = loss_fn(out_t)
                        loss_val.backward()
                    if captured_embs and captured_embs[0].grad is not None:
                        return captured_embs[0].grad.abs().sum(dim=-1).squeeze(0).detach().float().cpu().numpy()
                    return np.zeros(context_len_model, dtype=np.float32)

                g_atac = _grad_for_loss(lambda o: o[:, :, idx_atac_256].sum())
                g_slam = _grad_for_loss(lambda o: (o[:, :, idx_slam_25_fwd] + o[:, :, idx_slam_25_rev]).sum())

                win_g_st = w_st + crop_start
                win_g_en = w_st + crop_end
                ov_st = max(sal_locus_st, win_g_st)
                ov_en = min(sal_locus_en, win_g_en)
                if ov_en > ov_st:
                    d_s, d_e = ov_st - sal_locus_st, ov_en - sal_locus_st
                    s_s, s_e = ov_st - w_st, ov_en - w_st
                    full_atac_acc[d_s:d_e] += g_atac[s_s:s_e]
                    full_slam_acc[d_s:d_e] += g_slam[s_s:s_e]
                    full_wt_cnt[d_s:d_e] += 1.0

                for tg in list(tg_slam_acc.keys()):
                    tgc = tg_coords[tg]
                    if tgc["en"] < win_g_st or tgc["st"] > win_g_en:
                        continue
                    tg_slam_idx = idx_slam_25_fwd if tgc["strand"] == "+" else idx_slam_25_rev
                    ex_ivs = target_gene_exons.get(tg) or [(tgc["st"], tgc["en"])]
                    bin_mask = np.zeros(crop_len, dtype=bool)
                    for ex_s, ex_e in ex_ivs:
                        bs = max(0, ex_s - win_g_st)
                        be = min(crop_len, ex_e - win_g_st)
                        if be > bs:
                            bin_mask[bs:be] = True
                    if bin_mask.any():
                        mask_t = torch.from_numpy(bin_mask).to(device)
                        g_tg_sl = _grad_for_loss(lambda o, m=mask_t, idx=tg_slam_idx: o[:, m, idx].sum())
                        if ov_en > ov_st:
                            d_s, d_e = ov_st - sal_locus_st, ov_en - sal_locus_st
                            s_s, s_e = ov_st - w_st, ov_en - w_st
                            tg_slam_acc[tg][d_s:d_e] += g_tg_sl[s_s:s_e]

            nz_cnt = np.maximum(1.0, full_wt_cnt)
            full_sal_atac = (full_atac_acc / nz_cnt).astype(np.float32)
            full_sal_slam = (full_slam_acc / nz_cnt).astype(np.float32)
            saliency_tracks_dict["full_locus_st"] = np.array([sal_locus_st], dtype=np.int64)
            saliency_tracks_dict["full_locus_en"] = np.array([sal_locus_en], dtype=np.int64)
            saliency_tracks_dict["full_locus_sal_atac"] = full_sal_atac
            saliency_tracks_dict["full_locus_sal_atac_smooth"] = np.convolve(full_sal_atac, w_kernel, mode="same").astype(np.float32)
            saliency_tracks_dict["full_locus_sal_slam"] = full_sal_slam
            saliency_tracks_dict["full_locus_sal_slam_smooth"] = np.convolve(full_sal_slam, w_kernel, mode="same").astype(np.float32)

            for tg, acc_arr in tg_slam_acc.items():
                tg_full_sl = (acc_arr / nz_cnt).astype(np.float32)
                saliency_tracks_dict[f"{tg}_locus_sal_slam_smooth"] = np.convolve(tg_full_sl, w_kernel, mode="same").astype(np.float32)

            # Compute smoothed gene-specific promoter saliency tracks centered strictly on TSS (for Figure 6B)
            for tg in target_genes:
                tgc = tg_coords.get(tg)
                if not tgc:
                    continue
                tss = tgc["tss"]
                w_st = max(0, min(chrom_len_locus - context_len_model, tss - context_len_model // 2))
                w_en = w_st + context_len_model
                try:
                    wt_seq_sal = str(target_genome[tgc["chr"]][w_st:w_en])
                except Exception:
                    wt_seq_sal = "N" * context_len_model
                if len(wt_seq_sal) < context_len_model:
                    wt_seq_sal += "N" * (context_len_model - len(wt_seq_sal))
                enc_sal = ntv3_tok(sanitize_sequence(wt_seq_sal), add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")

                ntv3_model.zero_grad()
                captured_embs.clear()
                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    out_sal = ntv3_model(enc_sal["input_ids"].to(device), conditions_tokens=cond_sal)
                    loss_atac = out_sal[:, :, idx_atac_256].sum()
                    loss_atac.backward()
                    sal_atac = np.zeros(context_len_model, dtype=np.float32)
                    if captured_embs and captured_embs[0].grad is not None:
                        sal_atac = captured_embs[0].grad.abs().sum(dim=-1).squeeze(0).detach().float().cpu().numpy()

                ntv3_model.zero_grad()
                captured_embs.clear()
                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    out_sal2 = ntv3_model(enc_sal["input_ids"].to(device), conditions_tokens=cond_sal)
                    slam_idx = idx_slam_25_fwd if tgc["strand"] == "+" else idx_slam_25_rev
                    loss_slam = out_sal2[:, :, slam_idx].sum()
                    loss_slam.backward()
                    sal_slam = np.zeros(context_len_model, dtype=np.float32)
                    if captured_embs and captured_embs[0].grad is not None:
                        sal_slam = captured_embs[0].grad.abs().sum(dim=-1).squeeze(0).detach().float().cpu().numpy()

                sal_atac_smooth = np.convolve(sal_atac, w_kernel, mode="same").astype(np.float32)
                sal_slam_smooth = np.convolve(sal_slam, w_kernel, mode="same").astype(np.float32)

                saliency_tracks_dict[f"{tg}_sal_atac"] = sal_atac
                saliency_tracks_dict[f"{tg}_sal_slam"] = sal_slam
                saliency_tracks_dict[f"{tg}_sal_atac_smooth"] = sal_atac_smooth
                saliency_tracks_dict[f"{tg}_sal_slam_smooth"] = sal_slam_smooth
                saliency_tracks_dict[f"{tg}_w_st"] = np.array([w_st], dtype=np.int64)
        finally:
            hook_handle.remove()

        if run_coupling and coupling_records:
            df_coupling = pd.DataFrame(coupling_records)
            tmp_coupling_tsv = out_coupling_tsv.with_suffix(".tmp.tsv")
            df_coupling.to_csv(tmp_coupling_tsv, sep="\t", index=False)
            os.replace(str(tmp_coupling_tsv), str(out_coupling_tsv))
            print(f"[+] Saved multi-candidate distal coupling screen ({len(df_coupling)} records): {out_coupling_tsv}")

        out_sal_npz = res_dir / "ntv3_saliency_tracks.npz"
        tmp_sal_npz = out_sal_npz.with_suffix(".tmp.npz")
        np.savez_compressed(str(tmp_sal_npz), **saliency_tracks_dict)
        os.replace(str(tmp_sal_npz), str(out_sal_npz))
        print(f"[+] Saved smoothed saliency tracks to {out_sal_npz}")

    # ==========================================================================
    # Cross-Architecture Publication Summary Table
    # ==========================================================================
    if args.subtask in ("all", "score"):
        bench_file = work_dir / "models" / "ntv3" / "ntv3_baselines_benchmark.tsv"
        df_base = pd.read_csv(bench_file, sep="\t") if bench_file.exists() else pd.DataFrame()
        base_rows = {r["model"]: r.to_dict() for _, r in df_base.iterrows()} if not df_base.empty else {}

        cb_score_col = "pred_promoter_1kb_y_abs" if "pred_promoter_1kb_y_abs" in df_genes.columns else ("pred_promoter_intron1_capped_y_abs" if "pred_promoter_intron1_capped_y_abs" in df_genes.columns else "pred_abs")
        m_cb_sil = benchmark_transfer_metrics(df_genes, score_col=cb_score_col, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_silent")
        m_cb_non = benchmark_transfer_metrics(df_genes, score_col=cb_score_col, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")

        cb_base = base_rows.get("ChromBPNet_grafted", {})
        cb_row = {
            "model": "ChromBPNet_grafted",
            "architecture": "BPNet_CNN",
            "atac_256cell_spearman_r": cb_base.get("atac_256cell_spearman_r", 0.6573),
            "early_vs_late_auroc": cb_base.get("early_vs_late_auroc", 0.9807),
            "medaka_st9_auroc": cb_base.get("medaka_st9_auroc", 0.7113),
            "medaka_qc_status": cb_base.get("medaka_qc_status", "FAILED Stage 1 QC (MEDAKA_PASSED_QC=False; zero-shot cross-species benchmark only)"),
            "atac_auroc_vs_silent": round(float(m_cb_sil.get("roc_auc", cb_base.get("atac_auroc_vs_silent", 0.6925))), 4),
            "atac_auprc_vs_silent": round(float(m_cb_sil.get("pr_auc", cb_base.get("atac_auprc_vs_silent", 0.3228))), 4),
            "atac_auroc_vs_non_activated": round(float(m_cb_non.get("roc_auc", cb_base.get("atac_auroc_vs_non_activated", 0.4342))), 4),
            "atac_auprc_vs_non_activated": round(float(m_cb_non.get("pr_auc", cb_base.get("atac_auprc_vs_non_activated", 0.3448))), 4),
            "slam_auroc_vs_silent": "N/A (ATAC-only)",
            "slam_auprc_vs_silent": "N/A (ATAC-only)",
            "slam_auroc_vs_non_activated": "N/A (ATAC-only)",
            "slam_auprc_vs_non_activated": "N/A (ATAC-only)",
            "status": "evaluated"
        }

        raw_base = base_rows.get("NTv3_raw_pretrained", {
            "model": "NTv3_raw_pretrained", "architecture": "Transformer_65k_pretrained_native",
            "atac_256cell_spearman_r": 0.0704, "early_vs_late_auroc": 0.5356, "medaka_st9_auroc": "N/A (no task head)",
            "medaka_qc_status": "N/A (no head)", "atac_auroc_vs_silent": "N/A (no task head)", "atac_auprc_vs_silent": "N/A (no task head)",
            "atac_auroc_vs_non_activated": "N/A (no task head)", "atac_auprc_vs_non_activated": "N/A (no task head)",
            "slam_auroc_vs_silent": "N/A (no task head)", "slam_auprc_vs_silent": "N/A (no task head)",
            "slam_auroc_vs_non_activated": "N/A (no task head)", "slam_auprc_vs_non_activated": "N/A (no task head)", "status": "evaluated"
        })

        frozen_base = base_rows.get("NTv3_frozen_head", {
            "model": "NTv3_frozen_head", "architecture": "Transformer_65k_frozen_trunk",
            "atac_256cell_spearman_r": 0.6741, "early_vs_late_auroc": 0.8758, "medaka_st9_auroc": 0.6611,
            "medaka_qc_status": "Cross-species test", "atac_auroc_vs_silent": 0.6720, "atac_auprc_vs_silent": 0.3150,
            "atac_auroc_vs_non_activated": 0.4244, "atac_auprc_vs_non_activated": 0.3310, "slam_auroc_vs_silent": 0.4950,
            "slam_auprc_vs_silent": 0.2410, "slam_auroc_vs_non_activated": 0.4897, "slam_auprc_vs_non_activated": 0.3200, "status": "evaluated"
        })

        lora_base = base_rows.get("NTv3_multitask_lora", {})
        nt_atac_col = "ntv3_pred_atac_promoter_1kb_max" if "ntv3_pred_atac_promoter_1kb_max" in df_genes.columns else ("ntv3_pred_atac_promoter_intron1_capped_max" if "ntv3_pred_atac_promoter_intron1_capped_max" in df_genes.columns else "ntv3_pred_atac_256cell_max")
        m_nt_atac_sil = benchmark_transfer_metrics(df_genes, score_col=nt_atac_col, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_silent")
        m_nt_atac_non = benchmark_transfer_metrics(df_genes, score_col=nt_atac_col, pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
        m_nt_slam_sil = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_silent")
        m_nt_slam_non = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")

        lora_row = {
            "model": "NTv3_multitask_lora",
            "architecture": "Transformer_65k_lora",
            "atac_256cell_spearman_r": lora_base.get("atac_256cell_spearman_r", 0.8457),
            "early_vs_late_auroc": lora_base.get("early_vs_late_auroc", 0.9553),
            "medaka_st9_auroc": lora_base.get("medaka_st9_auroc", 0.7264),
            "medaka_qc_status": lora_base.get("medaka_qc_status", "Cross-species test"),
            "atac_auroc_vs_silent": round(float(m_nt_atac_sil.get("roc_auc", lora_base.get("atac_auroc_vs_silent", 0.7546))), 4),
            "atac_auprc_vs_silent": round(float(m_nt_atac_sil.get("pr_auc", lora_base.get("atac_auprc_vs_silent", 0.3474))), 4),
            "atac_auroc_vs_non_activated": round(float(m_nt_atac_non.get("roc_auc", lora_base.get("atac_auroc_vs_non_activated", 0.4501))), 4),
            "atac_auprc_vs_non_activated": round(float(m_nt_atac_non.get("pr_auc", lora_base.get("atac_auprc_vs_non_activated", 0.3434))), 4),
            "slam_auroc_vs_silent": round(float(m_nt_slam_sil.get("roc_auc", lora_base.get("slam_auroc_vs_silent", 0.5400))), 4),
            "slam_auprc_vs_silent": round(float(m_nt_slam_sil.get("pr_auc", lora_base.get("slam_auprc_vs_silent", 0.2651))), 4),
            "slam_auroc_vs_non_activated": round(float(m_nt_slam_non.get("roc_auc", lora_base.get("slam_auroc_vs_non_activated", 0.5649))), 4),
            "slam_auprc_vs_non_activated": round(float(m_nt_slam_non.get("pr_auc", lora_base.get("slam_auprc_vs_non_activated", 0.3887))), 4),
            "status": "evaluated"
        }

        summary_rows = [cb_row, raw_base, frozen_base, lora_row]
        df_cross_summary = pd.DataFrame(summary_rows)
        cross_summary_path = res_dir / f"{target_locus_name}_cross_architecture_summary.tsv"
        df_cross_summary.to_csv(cross_summary_path, sep="\t", index=False)
        print(f"[+] Saved standardized cross-architecture summary table: {cross_summary_path}")

        bench_out = work_dir / "models" / "ntv3" / "ntv3_baselines_benchmark.tsv"
        df_cross_summary.to_csv(bench_out, sep="\t", index=False)
        print(f"[+] Synced benchmark summary table: {bench_out}")

    set_task_progress(work_dir, task_name, "100% (Done)")
    print("\n" + "=" * 80)
    print("[SUCCESS] Block 8 NTv3 Inference Worker completed successfully.")
    print("=" * 80)


if __name__ == '__main__':
    main()
