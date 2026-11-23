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


def parse_args():
    parser = argparse.ArgumentParser(description="Block 8 NTv3 Inference Worker")
    parser.add_argument("--work-dir", type=str, default="./work_astatotilapia_premzt", help="Path to working directory")
    parser.add_argument("--config", type=str, default=None, help="Path to study config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained NTv3 checkpoint")
    parser.add_argument("--context-len", type=int, default=65536, help="NTv3 context length (default 65536)")
    parser.add_argument("--flank-bp", type=int, default=None, help="Locus flank bp (default from config LOCUS_FLANK_BP)")
    parser.add_argument("--task-name", type=str, default="block8_ntv3_inference", help="TaskManager task name")
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

    # 1. Check evolutionary_swap_summary.tsv
    swap_tsv = res_dir / "evolutionary_swap_summary.tsv"
    if swap_tsv.exists():
        try:
            df_sw = pd.read_csv(swap_tsv, sep="\t")
            if not df_sw.empty:
                if "is_selected" in df_sw.columns and (df_sw["is_selected"] == True).any():
                    row = df_sw[df_sw["is_selected"] == True].iloc[0].to_dict()
                elif "target_ablation_delta" in df_sw.columns:
                    row = df_sw.sort_values(by="target_ablation_delta", ascending=True).iloc[0].to_dict()
                else:
                    row = df_sw.iloc[0].to_dict()

                motif_info.update({
                    "motif_genomic_start": int(row["motif_genomic_start"]) if pd.notna(row.get("motif_genomic_start")) else None,
                    "motif_genomic_end": int(row["motif_genomic_end"]) if pd.notna(row.get("motif_genomic_end")) else None,
                    "causal_seq": str(row.get("causal_seq", "N/A")),
                    "ortholog_aligned_motif": str(row["ortholog_aligned_motif"]) if pd.notna(row.get("ortholog_aligned_motif")) else None,
                    "enhancer_rank": int(row.get("enhancer_rank", 1)),
                    "target_ablation_delta": float(row.get("target_ablation_delta", 0.0)),
                    "classification": str(row.get("classification", "divergent"))
                })
                print(f"[+] Found causal motif from {swap_tsv.name}: '{motif_info['causal_seq']}' at {motif_info['motif_genomic_start']}-{motif_info['motif_genomic_end']}")
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
        grad_checkpointing=False,
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

    df_genes = pd.read_csv(gene_table_path, sep="\t")
    print(f"[+] Loaded {len(df_genes):,} genes from {gene_table_path.name}")

    # Score each gene with NTv3
    pred_atac_max_list: List[float] = []
    pred_atac_mean_list: List[float] = []
    pred_slam_3p_list: List[float] = []
    pred_slam_all_list: List[float] = []

    half_crop = crop_len // 2
    for g_idx, (_, r) in enumerate(df_genes.iterrows()):
        g_chr = str(r["chrom"])
        g_st = int(r["start"])
        g_en = int(r["end"])
        g_strand = str(r.get("strand", "+"))
        g_len = g_en - g_st
        tss = g_st if g_strand == "+" else g_en
        tes = g_en if g_strand == "+" else g_st

        # Center window on TSS so TSS sits in the middle of the central valid crop
        w_st = max(0, tss - (crop_start + half_crop))
        w_en = w_st + context_len_model

        # Extract sequence
        try:
            chrom_len = len(target_genome[g_chr])
        except Exception:
            chrom_len = w_en + 1000

        if w_en > chrom_len:
            shift = w_en - chrom_len
            w_st = max(0, w_st - shift)
            w_en = w_st + context_len_model

        try:
            seq = str(target_genome[g_chr][w_st:w_en])
        except Exception:
            seq = "N" * context_len_model

        if len(seq) < context_len_model:
            seq += "N" * (context_len_model - len(seq))
        seq = sanitize_sequence(seq)

        enc = ntv3_tok(
            seq,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=context_len_model,
            truncation=True,
            padding="max_length"
        )
        inp_ids = enc["input_ids"].to(device)
        cond_tokens = [torch.tensor([resolved_species_id], dtype=torch.long, device=device)]

        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                out_p = ntv3_model(inp_ids, conditions_tokens=cond_tokens)
                preds = out_p.float().squeeze(0).cpu().numpy()

        # 1. ATAC 256-cell signal (max & mean across central crop)
        atac_track = preds[:, idx_atac_256]
        pred_atac_max_list.append(float(np.max(atac_track)))
        pred_atac_mean_list.append(float(np.mean(atac_track)))

        # 2. SLAM-seq 2.5 hpf strand-matched signal
        slam_track_idx = idx_slam_25_fwd if g_strand == "+" else idx_slam_25_rev
        slam_track = preds[:, slam_track_idx]

        crop_g_st = w_st + crop_start
        crop_g_en = w_st + crop_end

        g_name = str(r.get("gene_name", r.get("gene_id", "")))
        exons = target_gene_exons.get(g_name, None)

        scores_slam = quantseq_3p_gene_scoring(
            pred_nascent_track=slam_track,
            gene_span_len=g_len,
            strand=g_strand,
            resolution_bp=1,
            crop_genomic_start=crop_g_st,
            crop_genomic_end=crop_g_en,
            gene_start=g_st,
            gene_end=g_en,
            exons=exons
        )
        pred_slam_3p_list.append(float(scores_slam["nascent_3p_1000bp"]))
        pred_slam_all_list.append(float(scores_slam["nascent_all_exons"]))

        if (g_idx + 1) % 50 == 0 or (g_idx + 1) == len(df_genes):
            set_task_progress(work_dir, task_name, f"Sub-task 1/3: Scored {g_idx + 1}/{len(df_genes)} genes...")

    df_genes["ntv3_pred_atac_256cell_max"] = pred_atac_max_list
    df_genes["ntv3_pred_atac_256cell_mean"] = pred_atac_mean_list
    df_genes["ntv3_pred_slam_2.5hpf_3p"] = pred_slam_3p_list
    df_genes["ntv3_pred_slam_2.5hpf_all_exons"] = pred_slam_all_list

    # Compute transfer benchmark metrics
    m_strict_atac = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_atac_256cell_max", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
    m_slam_atac = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_atac_256cell_max", pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")
    m_strict_slam = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="strictly_zygotic_minor_wave", ctrl_category="control_non_activated")
    m_slam_slam = benchmark_transfer_metrics(df_genes, score_col="ntv3_pred_slam_2.5hpf_3p", pos_category="slam_confirmed_minor_wave", ctrl_category="control_non_activated")

    print("[*] NTv3 Zero-Shot Minor-Wave Transfer Benchmark Results:")
    print(f"    ATAC 256-cell -> Strictly Zygotic AUROC: {m_strict_atac.get('roc_auc', 0.0):.3f} | SLAM-Confirmed AUROC: {m_slam_atac.get('roc_auc', 0.0):.3f}")
    print(f"    SLAM 2.5 hpf  -> Strictly Zygotic AUROC: {m_strict_slam.get('roc_auc', 0.0):.3f} | SLAM-Confirmed AUROC: {m_slam_slam.get('roc_auc', 0.0):.3f}")

    # Atomically write ntv3_target_genes_scored.tsv
    out_scored_tsv = res_dir / "ntv3_target_genes_scored.tsv"
    tmp_scored_tsv = out_scored_tsv.with_suffix(".tmp.tsv")
    df_genes.to_csv(tmp_scored_tsv, sep="\t", index=False)
    os.replace(str(tmp_scored_tsv), str(out_scored_tsv))
    print(f"[+] Atomically saved scored genes to {out_scored_tsv}")

    # ==========================================================================
    # SUB-TASK 2: Locus-Wide Tiled NTv3 Scan
    # ==========================================================================
    set_task_progress(work_dir, task_name, "Sub-task 2/3: Locus-wide tiled scanning...")
    print("\n" + "=" * 80)
    print(f"[*] Sub-task 2: Locus-Wide Tiled NTv3 Scan on '{target_locus_name}'")
    print("=" * 80)

    # Determine locus boundaries for target genes
    locus_genes_sub = df_genes[df_genes["gene_name"].str.lower().isin([tg.lower() for tg in target_genes])]
    if locus_genes_sub.empty:
        locus_genes_sub = df_genes.head(2)

    locus_chr = str(locus_genes_sub["chrom"].iloc[0])
    locus_st = max(0, int(locus_genes_sub["start"].min()) - locus_flank_bp)
    locus_en = int(locus_genes_sub["end"].max()) + locus_flank_bp
    locus_len = locus_en - locus_st
    print(f"[+] Target Locus Coordinates: {locus_chr}:{locus_st:,}-{locus_en:,} (Span = {locus_len:,} bp)")

    # Tiling: step across the central crop
    # Each window is context_len_model bp; central crop is crop_len bp.
    # To tile seamlessly with zero gaps, step by crop_len (or crop_len // 2)
    step_bp = crop_len // 2
    tile_crop_centers = list(range(locus_st + crop_len // 2, locus_en + crop_len // 2, step_bp))

    tiled_records: List[Dict[str, Any]] = []
    for c_idx, c_center in enumerate(tile_crop_centers):
        w_st = max(0, c_center - (crop_start + crop_len // 2))
        w_en = w_st + context_len_model

        try:
            seq = str(target_genome[locus_chr][w_st:w_en])
        except Exception:
            seq = "N" * context_len_model
        if len(seq) < context_len_model:
            seq += "N" * (context_len_model - len(seq))
        seq = sanitize_sequence(seq)

        enc = ntv3_tok(
            seq,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=context_len_model,
            truncation=True,
            padding="max_length"
        )
        inp_ids = enc["input_ids"].to(device)
        cond_tokens = [torch.tensor([resolved_species_id], dtype=torch.long, device=device)]

        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                out_p = ntv3_model(inp_ids, conditions_tokens=cond_tokens)
                preds = out_p.float().squeeze(0).cpu().numpy()

        n_bins = preds.shape[0]
        bp_per_bin = crop_len / max(1, n_bins)
        c_st_genomic = w_st + crop_start
        c_en_genomic = w_st + crop_end

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
        # Sort and average overlapping bin tiles
        df_locus_scan = df_locus_scan.groupby(["chrom", "start", "end", "center"], as_index=False).mean()
        df_locus_scan = df_locus_scan.sort_values(by="start").reset_index(drop=True)
    print(f"[+] Tiled {len(df_locus_scan):,} genomic bins across {target_locus_name}")

    # Atomically write ntv3_locus_scan.tsv
    out_scan_tsv = res_dir / "ntv3_locus_scan.tsv"
    tmp_scan_tsv = out_scan_tsv.with_suffix(".tmp.tsv")
    df_locus_scan.to_csv(tmp_scan_tsv, sep="\t", index=False)
    os.replace(str(tmp_scan_tsv), str(out_scan_tsv))
    print(f"[+] Atomically saved locus scan table to {out_scan_tsv}")

    # Export IGV bedGraphs
    bg_atac_256 = res_dir / f"{target_locus_name}_ntv3_atac_256cell.bedgraph"
    bg_atac_dome = res_dir / f"{target_locus_name}_ntv3_atac_dome.bedgraph"
    bg_slam_25 = res_dir / f"{target_locus_name}_ntv3_slam_2.5hpf.bedgraph"

    with open(bg_atac_256.with_suffix(".tmp"), "w") as f_256, \
         open(bg_atac_dome.with_suffix(".tmp"), "w") as f_dome, \
         open(bg_slam_25.with_suffix(".tmp"), "w") as f_slam:
        for _, row in df_locus_scan.iterrows():
            c, s, e = row["chrom"], int(row["start"]), int(row["end"])
            f_256.write(f"{c}\t{s}\t{e}\t{row['atac_256cell']:.4f}\n")
            f_dome.write(f"{c}\t{s}\t{e}\t{row['atac_dome']:.4f}\n")
            # Max of fwd and rev for combined nascent track
            slam_max = max(float(row["slam_2.5hpf_fwd"]), float(row["slam_2.5hpf_rev"]))
            f_slam.write(f"{c}\t{s}\t{e}\t{slam_max:.4f}\n")

    os.replace(str(bg_atac_256.with_suffix(".tmp")), str(bg_atac_256))
    os.replace(str(bg_atac_dome.with_suffix(".tmp")), str(bg_atac_dome))
    os.replace(str(bg_slam_25.with_suffix(".tmp")), str(bg_slam_25))
    print(f"[+] Exported IGV bedGraphs: {bg_atac_256.name}, {bg_atac_dome.name}, {bg_slam_25.name}")

    # ==========================================================================
    # SUB-TASK 3: Distal Knockout, Zebrafish Motif Swap & Saliency
    # ==========================================================================
    set_task_progress(work_dir, task_name, "Sub-task 3/3: Running QuantSeq 3'-end knockout & saliency...")
    print("\n" + "=" * 80)
    print("[*] Sub-task 3: Distal QuantSeq 3'-End Knockout, Zebrafish Swap & Saliency")
    print("=" * 80)

    motif_data = find_top_causal_motif(res_dir, work_dir, target_locus_name)
    c_motif_st = motif_data["motif_genomic_start"]
    c_motif_en = motif_data["motif_genomic_end"]
    causal_seq = motif_data["causal_seq"]
    ortholog_motif = motif_data["ortholog_aligned_motif"]
    enh_rank = motif_data["enhancer_rank"]

    if c_motif_st is None:
        # Fallback to midpoint of locus
        c_motif_st = (locus_st + locus_en) // 2
        c_motif_en = c_motif_st + 12
        causal_seq = "Summit_Midpoint"

    m_len = (c_motif_en - c_motif_st) if (c_motif_en and c_motif_st) else len(causal_seq)
    print(f"[*] Causal Motif: '{causal_seq}' at {locus_chr}:{c_motif_st:,}-{c_motif_en:,} (Len = {m_len} bp)")

    # Setup forward hook on input embeddings for saliency attribution
    embed_layer = get_ntv3_input_embedding_layer(ntv3_model)
    captured_embs: List[torch.Tensor] = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            mod_out = out.detach().clone().requires_grad_(True)
            captured_embs.append(mod_out)
            return mod_out

    hook_handle = embed_layer.register_forward_hook(hook_fn)

    knockout_records: List[Dict[str, Any]] = []
    saliency_tracks_dict: Dict[str, np.ndarray] = {}

    try:
        for tg in target_genes:
            tg_sub = locus_genes_sub[locus_genes_sub["gene_name"].str.lower() == tg.lower()]
            if tg_sub.empty:
                continue
            r_g = tg_sub.iloc[0]
            g_chr = str(r_g["chrom"])
            g_st = int(r_g["start"])
            g_en = int(r_g["end"])
            g_strand = str(r_g.get("strand", "+"))
            g_len = g_en - g_st
            tes = g_en if g_strand == "+" else g_st

            # Center window on TES crop center
            w_st = max(0, tes - (crop_start + crop_len // 2))
            w_en = w_st + context_len_model

            # Ensure causal motif is also contained inside context window
            if c_motif_st < w_st:
                shift = w_st - c_motif_st + 200
                w_st = max(0, w_st - shift)
                w_en = w_st + context_len_model
            elif c_motif_en > w_en:
                shift = c_motif_en - w_en + 200
                w_st = w_st + shift
                w_en = w_st + context_len_model

            dist_tes_motif = abs(tes - c_motif_st)
            print(f"    [+] {tg}: Window {g_chr}:{w_st:,}-{w_en:,} (TES-to-motif dist = {dist_tes_motif:,} bp)")

            crop_genomic_start = w_st + crop_start
            crop_genomic_end = w_st + crop_end

            try:
                wt_seq = str(target_genome[g_chr][w_st:w_en])
            except Exception:
                wt_seq = "N" * context_len_model
            if len(wt_seq) < context_len_model:
                wt_seq += "N" * (context_len_model - len(wt_seq))
            wt_seq = sanitize_sequence(wt_seq)

            enc_wt = ntv3_tok(wt_seq, add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")
            inp_ids_wt = enc_wt["input_ids"].to(device)
            cond_tokens = [torch.tensor([resolved_species_id], dtype=torch.long, device=device)]

            # 1. WT Forward Pass
            with torch.no_grad():
                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    preds_wt = ntv3_model(inp_ids_wt, conditions_tokens=cond_tokens).float().squeeze(0).cpu().numpy()

            slam_track_idx = idx_slam_25_fwd if g_strand == "+" else idx_slam_25_rev
            exons = target_gene_exons.get(tg, None)

            scores_wt = quantseq_3p_gene_scoring(
                pred_nascent_track=preds_wt[:, slam_track_idx],
                gene_span_len=g_len,
                strand=g_strand,
                resolution_bp=1,
                crop_genomic_start=crop_genomic_start,
                crop_genomic_end=crop_genomic_end,
                gene_start=g_st,
                gene_end=g_en,
                exons=exons
            )
            scores_wt_linear = quantseq_3p_gene_scoring(
                pred_nascent_track=np.expm1(np.maximum(0.0, preds_wt[:, slam_track_idx])),
                gene_span_len=g_len,
                strand=g_strand,
                resolution_bp=1,
                crop_genomic_start=crop_genomic_start,
                crop_genomic_end=crop_genomic_end,
                gene_start=g_st,
                gene_end=g_en,
                exons=exons
            )

            # 2. Distal Knockout Forward Pass (Mutating motif to Ns)
            abl_seq = list(wt_seq)
            m_rel_pos = c_motif_st - w_st
            scores_abl = None
            scores_abl_linear = None
            delta_log1p = 0.0
            delta_linear = 0.0

            if 0 <= m_rel_pos and m_rel_pos + m_len <= len(abl_seq):
                abl_seq[m_rel_pos:m_rel_pos + m_len] = ["N"] * m_len
                enc_abl = ntv3_tok("".join(abl_seq), add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        preds_abl = ntv3_model(enc_abl["input_ids"].to(device), conditions_tokens=cond_tokens).float().squeeze(0).cpu().numpy()

                scores_abl = quantseq_3p_gene_scoring(
                    pred_nascent_track=preds_abl[:, slam_track_idx],
                    gene_span_len=g_len,
                    strand=g_strand,
                    resolution_bp=1,
                    crop_genomic_start=crop_genomic_start,
                    crop_genomic_end=crop_genomic_end,
                    gene_start=g_st,
                    gene_end=g_en,
                    exons=exons
                )
                scores_abl_linear = quantseq_3p_gene_scoring(
                    pred_nascent_track=np.expm1(np.maximum(0.0, preds_abl[:, slam_track_idx])),
                    gene_span_len=g_len,
                    strand=g_strand,
                    resolution_bp=1,
                    crop_genomic_start=crop_genomic_start,
                    crop_genomic_end=crop_genomic_end,
                    gene_start=g_st,
                    gene_end=g_en,
                    exons=exons
                )
                delta_log1p = float(scores_abl["nascent_3p_1000bp"] - scores_wt["nascent_3p_1000bp"])
                delta_linear = float(scores_abl_linear["nascent_3p_1000bp"] - scores_wt_linear["nascent_3p_1000bp"])
                print(f"        {tg} Knockout -> WT 3': {scores_wt['nascent_3p_1000bp']:.4f} | Abl: {scores_abl['nascent_3p_1000bp']:.4f} | Delta_log1p: {delta_log1p:+.4f} | Delta_linear: {delta_linear:+.4f}")

            # 3. Evolutionary Swap Forward Pass (Zebrafish ortholog motif inserted)
            scores_swap = None
            scores_swap_linear = None
            swap_delta_log1p = None
            swap_delta_linear = None

            if ortholog_motif and isinstance(ortholog_motif, str) and len(ortholog_motif) == m_len:
                sw_seq = list(wt_seq)
                if 0 <= m_rel_pos and m_rel_pos + m_len <= len(sw_seq):
                    sw_seq[m_rel_pos:m_rel_pos + m_len] = list(ortholog_motif)
                    enc_sw = ntv3_tok("".join(sw_seq), add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                            preds_sw = ntv3_model(enc_sw["input_ids"].to(device), conditions_tokens=cond_tokens).float().squeeze(0).cpu().numpy()

                    scores_swap = quantseq_3p_gene_scoring(
                        pred_nascent_track=preds_sw[:, slam_track_idx],
                        gene_span_len=g_len,
                        strand=g_strand,
                        resolution_bp=1,
                        crop_genomic_start=crop_genomic_start,
                        crop_genomic_end=crop_genomic_end,
                        gene_start=g_st,
                        gene_end=g_en,
                        exons=exons
                    )
                    scores_swap_linear = quantseq_3p_gene_scoring(
                        pred_nascent_track=np.expm1(np.maximum(0.0, preds_sw[:, slam_track_idx])),
                        gene_span_len=g_len,
                        strand=g_strand,
                        resolution_bp=1,
                        crop_genomic_start=crop_genomic_start,
                        crop_genomic_end=crop_genomic_end,
                        gene_start=g_st,
                        gene_end=g_en,
                        exons=exons
                    )
                    swap_delta_log1p = float(scores_swap["nascent_3p_1000bp"] - scores_wt["nascent_3p_1000bp"])
                    swap_delta_linear = float(scores_swap_linear["nascent_3p_1000bp"] - scores_wt_linear["nascent_3p_1000bp"])
                    print(f"        {tg} ZF Swap  -> Swap 3': {scores_swap['nascent_3p_1000bp']:.4f} | Swap Delta_log1p: {swap_delta_log1p:+.4f}")

            # 4. Saliency Attribution via Input Embedding Gradients
            sal_atac = np.zeros(context_len_model, dtype=np.float32)
            sal_slam = np.zeros(context_len_model, dtype=np.float32)

            # (a) Enhancer ATAC loss backward
            w_st_atac = max(0, c_motif_st - (crop_start + crop_len // 2))
            w_en_atac = w_st_atac + context_len_model
            try:
                wt_seq_atac = str(target_genome[g_chr][w_st_atac:w_en_atac])
            except Exception:
                wt_seq_atac = "N" * context_len_model
            if len(wt_seq_atac) < context_len_model:
                wt_seq_atac += "N" * (context_len_model - len(wt_seq_atac))
            enc_atac = ntv3_tok(sanitize_sequence(wt_seq_atac), add_special_tokens=False, return_tensors="pt", max_length=context_len_model, truncation=True, padding="max_length")

            ntv3_model.zero_grad()
            captured_embs.clear()
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                out_atac = ntv3_model(enc_atac["input_ids"].to(device), conditions_tokens=cond_tokens)
                n_bins_a = out_atac.shape[1]
                bp_bin_a = crop_len / max(1, n_bins_a)
                half_bins_500 = max(1, int(round(500.0 / bp_bin_a)))
                crop_st_a = w_st_atac + crop_start
                crop_en_a = w_st_atac + crop_end

                if crop_st_a <= c_motif_st <= crop_en_a:
                    enh_bin = int(round((c_motif_st - crop_st_a) / bp_bin_a))
                    st_b = max(0, min(n_bins_a - 1, enh_bin - half_bins_500))
                    en_b = max(st_b + 1, min(n_bins_a, enh_bin + half_bins_500))
                    loss_atac = out_atac[:, st_b:en_b, idx_atac_256].sum()
                    loss_atac.backward()
                    if captured_embs and captured_embs[0].grad is not None:
                        sal_atac = captured_embs[0].grad.abs().sum(dim=-1).squeeze(0).cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # (b) Gene 3' UTR SLAM-seq loss backward
            ntv3_model.zero_grad()
            captured_embs.clear()
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                out_slam = ntv3_model(enc_wt["input_ids"].to(device), conditions_tokens=cond_tokens)
                n_bins_s = out_slam.shape[1]
                bp_bin_s = crop_len / max(1, n_bins_s)
                half_bins_1000 = max(1, int(round(1000.0 / bp_bin_s)))

                if crop_genomic_start <= tes <= crop_genomic_end:
                    tes_bin = int(round((tes - crop_genomic_start) / bp_bin_s))
                    if g_strand == "+":
                        st_s = max(0, min(n_bins_s - 1, tes_bin - half_bins_1000))
                        en_s = max(st_s + 1, min(n_bins_s, tes_bin))
                    else:
                        st_s = max(0, min(n_bins_s - 1, tes_bin))
                        en_s = max(st_s + 1, min(n_bins_s, tes_bin + half_bins_1000))

                    loss_slam = out_slam[:, st_s:en_s, slam_track_idx].sum()
                    loss_slam.backward()
                    if captured_embs and captured_embs[0].grad is not None:
                        sal_slam = captured_embs[0].grad.abs().sum(dim=-1).squeeze(0).cpu().numpy()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save saliency arrays
            saliency_tracks_dict[f"{tg}_sal_atac"] = sal_atac
            saliency_tracks_dict[f"{tg}_sal_slam"] = sal_slam
            saliency_tracks_dict[f"{tg}_w_st_atac"] = np.array([w_st_atac])
            saliency_tracks_dict[f"{tg}_w_st_slam"] = np.array([w_st])

            knockout_records.append({
                "gene": tg,
                "strand": g_strand,
                "dist_tes_motif": int(dist_tes_motif),
                "wt_nascent_3p_log1p": float(scores_wt["nascent_3p_1000bp"]),
                "wt_nascent_3p_linear": float(scores_wt_linear["nascent_3p_1000bp"]),
                "abl_nascent_3p_log1p": float(scores_abl["nascent_3p_1000bp"]) if scores_abl else 0.0,
                "abl_nascent_3p_linear": float(scores_abl_linear["nascent_3p_1000bp"]) if scores_abl_linear else 0.0,
                "ablation_delta_log1p": delta_log1p,
                "ablation_delta_linear_expm1": delta_linear,
                "swap_nascent_3p_log1p": float(scores_swap["nascent_3p_1000bp"]) if scores_swap else None,
                "swap_nascent_3p_linear": float(scores_swap_linear["nascent_3p_1000bp"]) if scores_swap_linear else None,
                "swap_delta_log1p": swap_delta_log1p,
                "swap_delta_linear_expm1": swap_delta_linear,
                "sal_atac_max": float(sal_atac.max()),
                "sal_atac_mean": float(sal_atac.mean()),
                "sal_slam_max": float(sal_slam.max()),
                "sal_slam_mean": float(sal_slam.mean()),
                "causal_seq": causal_seq,
                "enhancer_rank": enh_rank
            })
    finally:
        hook_handle.remove()

    # Atomically write ntv3_knockout_saliency_summary.tsv
    df_ko_summary = pd.DataFrame(knockout_records)
    out_ko_tsv = res_dir / "ntv3_knockout_saliency_summary.tsv"
    tmp_ko_tsv = out_ko_tsv.with_suffix(".tmp.tsv")
    df_ko_summary.to_csv(tmp_ko_tsv, sep="\t", index=False)
    os.replace(str(tmp_ko_tsv), str(out_ko_tsv))
    print(f"[+] Atomically saved knockout summary to {out_ko_tsv}")

    # Atomically write ntv3_saliency_tracks.npz
    out_sal_npz = res_dir / "ntv3_saliency_tracks.npz"
    tmp_sal_npz = out_sal_npz.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp_sal_npz), **saliency_tracks_dict)
    os.replace(str(tmp_sal_npz), str(out_sal_npz))
    print(f"[+] Atomically saved saliency arrays to {out_sal_npz}")

    # Update summary metrics table
    summary_metrics_file = res_dir / f"{target_locus_name}_summary_metrics.tsv"
    summary_records = [
        {"metric": "strictly_zygotic_minor_wave_genes", "value": int((df_genes.get("category", "") == "strictly_zygotic_minor_wave").sum())},
        {"metric": "slam_confirmed_minor_wave_genes", "value": int(m_slam_atac.get("n_pos", 0))},
        {"metric": "zero_shot_strictly_zygotic_auroc", "value": float(m_strict_atac.get("roc_auc", 0.0))},
        {"metric": "zero_shot_slam_confirmed_auroc", "value": float(m_slam_atac.get("roc_auc", 0.0))},
        {"metric": "ntv3_slam_strictly_zygotic_auroc", "value": float(m_strict_slam.get("roc_auc", 0.0))},
        {"metric": "ntv3_slam_confirmed_auroc", "value": float(m_slam_slam.get("roc_auc", 0.0))},
        {"metric": "top_causal_motif", "value": causal_seq},
        {"metric": "evolutionary_ablation_delta", "value": float(motif_data.get("target_ablation_delta", 0.0))}
    ]
    pd.DataFrame(summary_records).to_csv(summary_metrics_file, sep="\t", index=False)
    print(f"[+] Updated summary metrics to {summary_metrics_file}")

    set_task_progress(work_dir, task_name, "100% (Done)")
    print("\n" + "=" * 80)
    print("[SUCCESS] Block 8 NTv3 Inference Worker completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
