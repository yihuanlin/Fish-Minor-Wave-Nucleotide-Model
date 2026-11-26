#!/usr/bin/env python3
"""
scripts/target_gene_benchmark.py

Target Gene Statistical Enrichment & Minor-Wave Parity Benchmark (Block 8c).
Evaluates target genes and positive reference markers against genome-wide control
and confident minor-wave cohorts across ChromBPNet and NTv3 architectures.

Features:
- Generic, locus- and species-agnostic design (Block 0 driven).
- Strict leave-one-out testing (target genes excluded from reference cohorts).
- miR-430 tandem repeat exclusion.
- GC- and length-matched control sensitivity evaluation.
- Empirical p-values, robust z-scores (MAD), BH-FDR q-values, and minor-wave quartile parity.
- Multi-panel publication visualization (Figure 8, 300 DPI PNG & vector SVG).
- Tidy TSV export: {TARGET_LOCUS_NAME}_target_gene_ranking_stats.tsv.
"""

import os
import sys
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from pyfaidx import Fasta

from scripts.premzt_inference import infer_premzt


def _compute_promoter_gc(
    chrom: str,
    tss: int,
    fasta_obj: Fasta,
    half_window: int = 500
) -> float:
    """Calculate GC content of the promoter region (TSS +/- half_window)."""
    st = max(0, tss - half_window)
    en = st + 2 * half_window
    try:
        seq = str(fasta_obj[str(chrom)][st:en]).upper()
        if not seq:
            return np.nan
        gc = (seq.count("G") + seq.count("C")) / float(len(seq))
        return gc
    except Exception:
        return np.nan


def _bh_fdr_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    n = len(p_arr)
    if n == 0:
        return np.array([])
    order = np.argsort(p_arr)
    ranks = np.arange(1, n + 1)
    q_arr = np.zeros(n, dtype=np.float64)
    q_vals = (p_arr[order] * n) / ranks
    # Enforce monotonicity
    q_cum = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_cum = np.clip(q_cum, 0.0, 1.0)
    q_arr[order] = q_cum
    return q_arr


def run_target_gene_benchmark(
    work_dir: Union[str, Path],
    locus_name: str,
    target_genes: List[str],
    canonical_markers: Optional[List[str]] = None,
    target_species_name: Optional[str] = None,
    target_genome_fa: Optional[str] = None,
    res_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Execute Block 8c Target Gene Statistical Ranking & Minor-Wave Parity Test.
    """
    work_dir = Path(work_dir).resolve()
    if res_dir is None:
        res_dir = work_dir / "results" / locus_name
    else:
        res_dir = Path(res_dir).resolve()
    res_dir.mkdir(parents=True, exist_ok=True)

    if config_path is None:
        config_path = work_dir / "config.json"
    
    cfg = {}
    if Path(config_path).exists():
        with open(config_path) as f_cf:
            cfg = json.load(f_cf)

    if canonical_markers is None:
        canonical_markers = cfg.get("CANONICAL_ZGA_MARKERS", ["vox", "klf17", "mxtx2"])
    if target_species_name is None:
        target_species_name = cfg.get("TARGET_SPECIES_NAME", "Target_Species")
    if target_genome_fa is None:
        target_genome_fa = cfg.get("TARGET_GENOME_FA")

    print(f"[*] Running Target Gene Statistical Ranking & Minor-Wave Parity Benchmark:")
    print(f"    Target Locus:    {locus_name}")
    print(f"    Target Genes:    {target_genes}")
    print(f"    Canonical ZGA:   {canonical_markers}")

    # 1. Load Scored Tables
    cb_scored_tsv = res_dir / "target_genes_scored_metrics.tsv"
    nt_scored_tsv = res_dir / "ntv3_target_genes_scored.tsv"

    if not cb_scored_tsv.exists():
        raise FileNotFoundError(f"[FATAL] ChromBPNet scored table not found: {cb_scored_tsv}")
    if not nt_scored_tsv.exists():
        raise FileNotFoundError(f"[FATAL] NTv3 scored table not found: {nt_scored_tsv}")

    df_cb = pd.read_csv(cb_scored_tsv, sep="\t", low_memory=False)
    df_nt = pd.read_csv(nt_scored_tsv, sep="\t", low_memory=False)

    # Sync is_slam_confirmed_minor_wave cleanly from Block 1b gene_sets if present
    gs_tsv = res_dir / f"{locus_name}_gene_sets.tsv"
    if gs_tsv.exists():
        try:
            df_gs = pd.read_csv(gs_tsv, sep="\t", usecols=lambda c: c in {"gene_id", "is_slam_confirmed_minor_wave"}, low_memory=False)
            if "is_slam_confirmed_minor_wave" in df_gs.columns and df_gs["is_slam_confirmed_minor_wave"].fillna(False).astype(bool).sum() > 0:
                slam_gs_ids = set(df_gs.loc[df_gs["is_slam_confirmed_minor_wave"].fillna(False).astype(bool), "gene_id"].astype(str))
                cb_slam = df_cb.get("is_slam_confirmed_minor_wave", pd.Series(False, index=df_cb.index)).fillna(False).astype(bool)
                df_cb["is_slam_confirmed_minor_wave"] = cb_slam | df_cb["gene_id"].astype(str).isin(slam_gs_ids)
        except Exception as e_gs:
            print(f"[-] Notice syncing SLAM-confirmed flags from {gs_tsv.name}: {e_gs}")

    # Merge NTv3 Metrics into Master Gene Table (including TES-centering provenance flag)
    nt_cols = [
        "gene_id",
        "ntv3_pred_atac_256cell_max",
        "ntv3_pred_atac_256cell_mean",
        "ntv3_pred_atac_promoter_1kb_max",
        "ntv3_pred_atac_promoter_intron1_capped_max",
        "ntv3_pred_atac_proximal_5kb_max",
        "ntv3_pred_slam_2.5hpf_3p",
        "ntv3_pred_slam_2.5hpf_exon_mean_per_kb",
        "ntv3_pred_slam_2.5hpf_all_exons",
        "slam_3p_tes_centered",
        "is_slam_confirmed_minor_wave",
    ]
    avail_nt_cols = [c for c in nt_cols if c in df_nt.columns]
    df_nt_sub = df_nt[avail_nt_cols].copy()
    if "is_slam_confirmed_minor_wave" in df_nt_sub.columns:
        df_nt_sub = df_nt_sub.rename(columns={"is_slam_confirmed_minor_wave": "_nt_slam_confirmed"})
    df = pd.merge(df_cb, df_nt_sub, on="gene_id", how="left")
    if "_nt_slam_confirmed" in df.columns:
        df["is_slam_confirmed_minor_wave"] = (
            df.get("is_slam_confirmed_minor_wave", pd.Series(False, index=df.index)).fillna(False).astype(bool) |
            df["_nt_slam_confirmed"].fillna(False).astype(bool)
        )
        df = df.drop(columns=["_nt_slam_confirmed"])

    # 3. Filter miR-430 Tandem Repeats and Unmapped/Mitochondrial Contigs
    is_mir = (
        df["gene_name"].astype(str).str.contains("mir-430|miR-430", case=False) |
        df["Geneid"].astype(str).str.contains("mir-430|miR-430", case=False) |
        df.get("gene_id", pd.Series()).astype(str).str.contains("mir-430|miR-430", case=False)
    )
    chrom_s = df["chrom"].astype(str).str.strip().str.lower()
    is_unmapped_or_mt = chrom_s.isin({"mt", "chrm", "m", "chrmt", "unknown", "nan", ""})
    if "start" in df.columns and "end" in df.columns:
        is_unmapped_or_mt = is_unmapped_or_mt | ((df["start"].fillna(0) == 0) & (df["end"].fillna(0) == 0))
    n_mir_excluded = int(is_mir.sum())
    n_mt_excluded = int((~is_mir & is_unmapped_or_mt).sum())
    df = df[~(is_mir | is_unmapped_or_mt)].copy()
    print(f"[+] Excluded {n_mir_excluded} miR-430 repeat(s) and {n_mt_excluded} mitochondrial/unmapped zero-span gene(s) from benchmark cohorts.")

    # 4. Promoter GC Content & Length Calculation for Matched Control
    fa = None
    if target_genome_fa and Path(target_genome_fa).exists():
        try:
            fa = Fasta(target_genome_fa, as_raw=True)
            print(f"[+] Loaded genome FASTA for promoter GC matching: {Path(target_genome_fa).name}")
        except Exception as e_fa:
            print(f"[-] Warning loading FASTA: {e_fa}")

    if "gene_span_len" not in df.columns:
        if "Length" in df.columns:
            df["gene_span_len"] = df["Length"]
        elif "start" in df.columns and "end" in df.columns:
            df["gene_span_len"] = df["end"] - df["start"]
        else:
            df["gene_span_len"] = 5000

    # 5. Define Reference Cohorts (Leave-One-Out for TARGET_GENES)
    is_tg = df["gene_name"].astype(str).str.lower().isin([tg.lower() for tg in target_genes])
    
    # Negative controls
    ctrl_silent_df = df[~is_tg & (df["is_control_silent"] == True)].copy()
    ctrl_nonact_df = df[~is_tg & (df["is_control_non_activated"] == True)].copy()

    # Confident minor-wave cohorts
    mw_strict_df = df[~is_tg & (df["is_strictly_zygotic_minor_wave"] == True)].copy()
    mw_slam_df = df[~is_tg & (df.get("is_slam_confirmed_minor_wave", False) == True)].copy()
    # High confidence minor wave is SLAM-confirmed minor wave (or strictly zygotic highconf)
    if not mw_slam_df.empty:
        mw_highconf_df = mw_slam_df
    elif "is_strictly_zygotic_highconf" in df.columns and df["is_strictly_zygotic_highconf"].sum() > 0:
        mw_highconf_df = df[~is_tg & (df["is_strictly_zygotic_highconf"] == True)].copy()
    else:
        mw_highconf_df = mw_strict_df

    print(f"[+] Reference Cohorts (Leave-One-Out):")
    print(f"    - control_silent:                 n = {len(ctrl_silent_df):,}")
    print(f"    - control_non_activated:          n = {len(ctrl_nonact_df):,}")
    print(f"    - strictly_zygotic_minor_wave:    n = {len(mw_strict_df):,}")
    print(f"    - high_conf_minor_wave (SLAM-seq): n = {len(mw_highconf_df):,}")

    # Compute promoter GC for control_silent and target genes if FASTA available
    tg_gc_map = {}
    tg_len_map = {}
    ctrl_silent_gc = []
    if fa is not None:
        for idx, r in ctrl_silent_df.iterrows():
            ctrl_silent_gc.append(_compute_promoter_gc(str(r["chrom"]), int(r["tss"]), fa))
        ctrl_silent_df["promoter_gc"] = ctrl_silent_gc

        for tg in target_genes:
            tg_sub = df[df["gene_name"].astype(str).str.lower() == tg.lower()]
            if not tg_sub.empty:
                r_tg = tg_sub.iloc[0]
                tg_gc_map[tg.lower()] = _compute_promoter_gc(str(r_tg["chrom"]), int(r_tg["tss"]), fa)
                tg_len_map[tg.lower()] = float(r_tg.get("gene_span_len", 10000))

    # 6. Metrics to Evaluate Across Architectures
    metric_specs = [
        # (Architecture, Modality, Column Name, Display Name, Short Metric)
        ("ChromBPNet", "Accessibility", "pred_promoter_1kb_y_abs", "Core Promoter ±1 kb Pre-MZT ATAC", "promoter_1kb_abs"),
        ("ChromBPNet", "Accessibility", "pred_promoter_intron1_capped_y_abs", "Promoter + Intron 1 Pre-MZT ATAC", "promoter_intron1_abs"),
        ("ChromBPNet", "Accessibility", "pred_proximal_5kb_y_abs", "Proximal ±5 kb Pre-MZT ATAC", "proximal_5kb_abs"),
        ("ChromBPNet", "Accessibility", "pred_bounded_25kb_y_abs", "Gene Domain ±25 kb Pre-MZT ATAC", "bounded_25kb_abs"),
        ("ChromBPNet", "Timing Specificity", "pred_promoter_1kb_y_rel_gated", "Pre-MZT Timing-Gated Early Bias", "promoter_y_rel_gated"),
        ("ChromBPNet", "Baseline Openness", "pred_promoter_1kb_late", "Late Blastula Baseline Openness", "promoter_late_open"),
        ("NTv3", "Accessibility", "ntv3_pred_atac_promoter_1kb_max", "NTv3 Pre-MZT ATAC (Promoter ±1 kb)", "ntv3_atac_promoter_1kb"),
        ("NTv3", "Accessibility", "ntv3_pred_atac_promoter_intron1_capped_max", "NTv3 Pre-MZT ATAC (Promoter + Intron 1)", "ntv3_atac_promoter_intron1"),
        ("NTv3", "Accessibility", "ntv3_pred_atac_proximal_5kb_max", "NTv3 Pre-MZT ATAC (Proximal ±5 kb)", "ntv3_atac_proximal_5kb"),
        ("NTv3", "Accessibility", "ntv3_pred_atac_256cell_max", "NTv3 Pre-MZT ATAC (256-cell Max)", "ntv3_atac_256_max"),
        ("NTv3", "Nascent Expression", "ntv3_pred_slam_2.5hpf_3p", "NTv3 2.5 hpf 3'-end Nascent SLAM (TES)", "ntv3_slam_2.5hpf_3p"),
        ("NTv3", "Nascent Expression", "ntv3_pred_slam_2.5hpf_exon_mean_per_kb", "NTv3 2.5 hpf Nascent SLAM (Exon Mean/kb)", "ntv3_slam_exon_mean"),
        ("NTv3", "Nascent Expression", "ntv3_pred_slam_2.5hpf_all_exons", "NTv3 2.5 hpf Full-Exon Nascent SLAM", "ntv3_slam_exons")
    ]

    # Filter to available columns in table
    metric_specs = [m for m in metric_specs if m[2] in df.columns]

    # Refresh target_genes and any small un-TES-centered mw_highconf rows via live infer_premzt
    # so NTv3 scores use uniform TES-centered per-1kb units matching cohort baselines.
    try:
        from scripts.premzt_inference import infer_premzt_batch
        genes_to_refresh = list(target_genes)
        if not mw_highconf_df.empty and len(mw_highconf_df) <= 30:
            hc_need = []
            for _, h_r in mw_highconf_df.iterrows():
                val_s = h_r.get("ntv3_pred_slam_2.5hpf_3p", np.nan)
                tes_ok = bool(h_r.get("slam_3p_tes_centered", False))
                g_span = abs(float(h_r.get("end", 0)) - float(h_r.get("start", 0)))
                if pd.isna(val_s) or val_s < 1.0 or (val_s == 0.0 and not tes_ok and g_span > 10500):
                    hc_need.append(str(h_r["gene_name"]))
            for hn in hc_need:
                if hn.lower() not in [g.lower() for g in genes_to_refresh]:
                    genes_to_refresh.append(hn)

        if genes_to_refresh:
            df_live = infer_premzt_batch(
                genes=genes_to_refresh,
                species=target_species_name,
                model=["ntv3atac", "ntv3slam"],
                locus_range="auto",
                work_dir=work_dir,
            )
            refresh_cols = [
                "ntv3_pred_atac_256cell_max",
                "ntv3_pred_atac_256cell_mean",
                "ntv3_pred_atac_promoter_1kb_max",
                "ntv3_pred_atac_promoter_intron1_capped_max",
                "ntv3_pred_atac_proximal_5kb_max",
                "ntv3_pred_slam_2.5hpf_3p",
                "ntv3_pred_slam_2.5hpf_all_exons",
                "ntv3_pred_slam_2.5hpf_exon_mean_per_kb",
                "slam_3p_tes_centered",
            ]
            for _, lr in df_live.iterrows():
                if pd.notna(lr.get("error")):
                    continue
                g_key = str(lr.get("gene", "")).lower()
                m_mask = df["gene_name"].astype(str).str.lower() == g_key
                if m_mask.any():
                    for rc in refresh_cols:
                        if rc in lr and rc in df.columns and pd.notna(lr[rc]):
                            df.loc[m_mask, rc] = lr[rc]
            # Rebuild cohorts cleanly with refreshed target + high-conf values
            ctrl_silent_df = df[~is_tg & (df["is_control_silent"] == True)].copy()
            ctrl_nonact_df = df[~is_tg & (df["is_control_non_activated"] == True)].copy()
            mw_strict_df = df[~is_tg & (df["is_strictly_zygotic_minor_wave"] == True)].copy()
            mw_slam_df = df[~is_tg & (df.get("is_slam_confirmed_minor_wave", False) == True)].copy()
            if not mw_slam_df.empty:
                mw_highconf_df = mw_slam_df
            elif "is_strictly_zygotic_highconf" in df.columns and df["is_strictly_zygotic_highconf"].sum() > 0:
                mw_highconf_df = df[~is_tg & (df["is_strictly_zygotic_highconf"] == True)].copy()
            else:
                mw_highconf_df = mw_strict_df
            print(f"[+] Refreshed live NTv3 per-1kb TES/ATAC scores for {len(genes_to_refresh)} target/high-conf gene(s).")
    except Exception as e_live:
        print(f"[-] Notice refreshing target/high-conf NTv3 scores: {e_live}")

    # Filter legacy un-TES-centered / unscored 0.0 SLAM artifacts from background cohorts if present
    def _clean_slam_3p_vals(sub_df: pd.DataFrame) -> np.ndarray:
        if "ntv3_pred_slam_2.5hpf_3p" not in sub_df.columns:
            return np.array([], dtype=np.float64)
        s = sub_df["ntv3_pred_slam_2.5hpf_3p"].copy()
        if "slam_3p_tes_centered" in sub_df.columns:
            bad_legacy = (s == 0.0) & (~sub_df["slam_3p_tes_centered"].fillna(False).astype(bool))
            # Also ignore isolated <1.0 unscaled per-bp legacy rows when cohort scale is ~85 per 1kb
            med_pos = float(s[s > 1.0].median()) if (s > 1.0).any() else 0.0
            if med_pos > 10.0:
                bad_legacy = bad_legacy | ((s >= 0.0) & (s < 1.0))
            s = s[~bad_legacy]
        return s.dropna().values.astype(np.float64)

    # Pre-score Zebrafish orthologs via infer_premzt for counter-scoring
    zf_ortholog_df = None
    try:
        zf_ortholog_df = infer_premzt(
            species="Danio_rerio",
            gene=target_genes,
            model="all",
            locus_range="auto",
            work_dir=work_dir
        )
        print(f"[+] Successfully scored {len(target_genes)} Danio rerio ortholog(s) for counter-scoring benchmark.")
    except Exception as e_zf:
        print(f"[-] Notice: Could not score zebrafish orthologs via infer_premzt: {e_zf}")

    # 7. Execute Statistical Tests
    records = []
    p_vals_for_fdr = []
    fdr_record_indices = []

    for tg in target_genes:
        tg_sub = df[df["gene_name"].astype(str).str.lower() == tg.lower()]
        if tg_sub.empty:
            print(f"[-] Warning: Target gene '{tg}' not found in scored tables!")
            continue
        tg_row = tg_sub.iloc[0]
        tg_canonical_name = str(tg_row["gene_name"])

        # Construct gene-specific GC- and length-matched control subset
        tg_gc = tg_gc_map.get(tg.lower(), np.nan)
        tg_len = tg_len_map.get(tg.lower(), np.nan)
        if fa is not None and np.isfinite(tg_gc) and "promoter_gc" in ctrl_silent_df.columns:
            matched_ctrl_df = ctrl_silent_df[
                (ctrl_silent_df["promoter_gc"] >= tg_gc - 0.05) &
                (ctrl_silent_df["promoter_gc"] <= tg_gc + 0.05)
            ].copy()
            if np.isfinite(tg_len):
                matched_ctrl_df = matched_ctrl_df[
                    (matched_ctrl_df["gene_span_len"] >= tg_len * 0.30) &
                    (matched_ctrl_df["gene_span_len"] <= tg_len * 3.00)
                ]
            if len(matched_ctrl_df) < 15:
                # Fallback to GC-only if length restriction left too few
                matched_ctrl_df = ctrl_silent_df[
                    (ctrl_silent_df["promoter_gc"] >= tg_gc - 0.05) &
                    (ctrl_silent_df["promoter_gc"] <= tg_gc + 0.05)
                ].copy()
        else:
            matched_ctrl_df = ctrl_silent_df

        for arch, modality, col, disp_name, short_code in metric_specs:
            s_g = float(tg_row[col]) if pd.notna(tg_row[col]) else np.nan

            # Reference distribution values (cleanly filtered for TES-centered SLAM unit consistency)
            if col == "ntv3_pred_slam_2.5hpf_3p":
                c_sil_vals = _clean_slam_3p_vals(ctrl_silent_df)
                c_non_vals = _clean_slam_3p_vals(ctrl_nonact_df)
                c_mat_vals = _clean_slam_3p_vals(matched_ctrl_df)
                mw_high_vals = _clean_slam_3p_vals(mw_highconf_df)
                mw_str_vals = _clean_slam_3p_vals(mw_strict_df)
            else:
                c_sil_vals = ctrl_silent_df[col].dropna().values.astype(np.float64)
                c_non_vals = ctrl_nonact_df[col].dropna().values.astype(np.float64)
                c_mat_vals = matched_ctrl_df[col].dropna().values.astype(np.float64)
                mw_high_vals = mw_highconf_df[col].dropna().values.astype(np.float64)
                mw_str_vals = mw_strict_df[col].dropna().values.astype(np.float64)

            # A. Statistics vs control_silent
            if len(c_sil_vals) > 0 and np.isfinite(s_g):
                p_emp_sil = float(1.0 + np.sum(c_sil_vals >= s_g)) / (len(c_sil_vals) + 1.0)
                pct_sil = float(np.sum(c_sil_vals < s_g)) / len(c_sil_vals) * 100.0
                med_sil = float(np.median(c_sil_vals))
                mad_sil = float(np.median(np.abs(c_sil_vals - med_sil)))
                z_rob_sil = (s_g - med_sil) / (1.4826 * mad_sil) if mad_sil > 1e-6 else (s_g - float(np.mean(c_sil_vals))) / max(1e-6, float(np.std(c_sil_vals)))
            else:
                p_emp_sil, pct_sil, med_sil, mad_sil, z_rob_sil = np.nan, np.nan, np.nan, np.nan, np.nan

            # B. Statistics vs control_non_activated
            if len(c_non_vals) > 0 and np.isfinite(s_g):
                p_emp_non = float(1.0 + np.sum(c_non_vals >= s_g)) / (len(c_non_vals) + 1.0)
                pct_non = float(np.sum(c_non_vals < s_g)) / len(c_non_vals) * 100.0
            else:
                p_emp_non, pct_non = np.nan, np.nan

            # C. Statistics vs GC/length matched control
            if len(c_mat_vals) > 0 and np.isfinite(s_g):
                p_emp_mat = float(1.0 + np.sum(c_mat_vals >= s_g)) / (len(c_mat_vals) + 1.0)
                pct_mat = float(np.sum(c_mat_vals < s_g)) / len(c_mat_vals) * 100.0
                n_mat = len(c_mat_vals)
            else:
                p_emp_mat, pct_mat, n_mat = np.nan, np.nan, 0

            # D. Parity within High-Confidence Minor-Wave Genes (SLAM-confirmed)
            if len(mw_high_vals) > 0 and np.isfinite(s_g):
                rank_high = int(1 + np.sum(mw_high_vals > s_g))
                pct_high = float(np.sum(mw_high_vals <= s_g)) / len(mw_high_vals) * 100.0
                p_below_high = float(1.0 + np.sum(mw_high_vals <= s_g)) / (len(mw_high_vals) + 1.0)
                if pct_high >= 75.0:
                    tier_high = "Above Q75 (Top Quartile)"
                elif pct_high >= 25.0:
                    tier_high = "Within IQR (Q25–Q75, On Par)"
                elif pct_high >= 10.0:
                    tier_high = "Q10–Q25 (Lower Tail)"
                else:
                    tier_high = "Below Q10"
                rank_high_str = f"{rank_high} / {len(mw_high_vals) + 1}"
            else:
                rank_high, pct_high, p_below_high, tier_high, rank_high_str = np.nan, np.nan, np.nan, "N/A", "N/A"

            # E. Parity within Strictly Zygotic Minor-Wave Genes (Genome-wide)
            if len(mw_str_vals) > 0 and np.isfinite(s_g):
                rank_str = int(1 + np.sum(mw_str_vals > s_g))
                pct_str = float(np.sum(mw_str_vals <= s_g)) / len(mw_str_vals) * 100.0
                p_below_str = float(1.0 + np.sum(mw_str_vals <= s_g)) / (len(mw_str_vals) + 1.0)
                if pct_str >= 75.0:
                    tier_str = "Above Q75 (Top Quartile)"
                elif pct_str >= 25.0:
                    tier_str = "Within IQR (Q25–Q75, On Par)"
                elif pct_str >= 10.0:
                    tier_str = "Q10–Q25 (Lower Tail)"
                else:
                    tier_str = "Below Q10"
                rank_str_str = f"{rank_str} / {len(mw_str_vals) + 1}"
            else:
                rank_str, pct_str, p_below_str, tier_str, rank_str_str = np.nan, np.nan, np.nan, "N/A", "N/A"

            # F. Genuine Zebrafish Ortholog Counterpart via infer_premzt
            zf_score = np.nan
            if zf_ortholog_df is not None and isinstance(zf_ortholog_df, pd.DataFrame) and not zf_ortholog_df.empty:
                zf_match = zf_ortholog_df[zf_ortholog_df["gene"].astype(str).str.lower() == tg.lower()]
                if not zf_match.empty and col in zf_match.columns:
                    val_zf = zf_match[col].iloc[0]
                    if pd.notna(val_zf):
                        zf_score = float(val_zf)
            if not np.isfinite(zf_score):
                zf_col = f"{col}_zf"
                if zf_col in tg_row and pd.notna(tg_row[zf_col]):
                    zf_score = float(tg_row[zf_col])

            if np.isfinite(zf_score) and len(c_sil_vals) > 0:
                pct_zf_sil = float(np.sum(c_sil_vals < zf_score)) / len(c_sil_vals) * 100.0
                delta_target_vs_zf = s_g - zf_score
            else:
                pct_zf_sil, delta_target_vs_zf = np.nan, np.nan

            rec_idx = len(records)
            rec = {
                "target_gene": tg_canonical_name,
                "architecture": arch,
                "modality": modality,
                "metric_code": short_code,
                "metric_display_name": disp_name,
                "target_score": s_g,
                "ctrl_silent_median": med_sil,
                "ctrl_silent_mad": mad_sil,
                "p_emp_vs_ctrl_silent": p_emp_sil,
                "fdr_q_vs_ctrl_silent": np.nan,  # populated below
                "percentile_in_ctrl_silent": pct_sil,
                "robust_z_vs_ctrl_silent": z_rob_sil,
                "p_emp_vs_ctrl_non_activated": p_emp_non,
                "percentile_in_ctrl_non_activated": pct_non,
                "p_emp_vs_gc_len_matched": p_emp_mat,
                "percentile_in_gc_len_matched": pct_mat,
                "n_gc_len_matched_controls": n_mat,
                "highconf_minor_wave_rank": rank_high_str,
                "highconf_minor_wave_percentile": pct_high,
                "highconf_minor_wave_top_pct": (100.0 - pct_high) if np.isfinite(pct_high) else np.nan,
                "highconf_minor_wave_parity_tier": tier_high,
                "p_below_highconf_minor_wave": p_below_high,
                "strictly_zygotic_minor_wave_rank": rank_str_str,
                "strictly_zygotic_minor_wave_percentile": pct_str,
                "strictly_zygotic_minor_wave_parity_tier": tier_str,
                "p_below_strictly_zygotic_minor_wave": p_below_str,
                "zebrafish_ortholog_score": zf_score,
                "zebrafish_ctrl_percentile": pct_zf_sil,
                "delta_target_vs_zf": delta_target_vs_zf
            }
            records.append(rec)
            if np.isfinite(p_emp_sil):
                p_vals_for_fdr.append(p_emp_sil)
                fdr_record_indices.append(rec_idx)

    # Compute BH-FDR across all target gene x metric tests against control_silent
    if p_vals_for_fdr:
        q_vals = _bh_fdr_adjust(np.array(p_vals_for_fdr))
        for r_idx, q_v in zip(fdr_record_indices, q_vals):
            records[r_idx]["fdr_q_vs_ctrl_silent"] = float(q_v)

    df_stats = pd.DataFrame(records)

    # 8. Save Full Tidy TSV
    stats_tsv_path = res_dir / f"{locus_name}_target_gene_ranking_stats.tsv"
    df_stats.to_csv(stats_tsv_path, sep="\t", index=False)
    print(f"[+] Exported statistical ranking benchmark table: {stats_tsv_path} ({len(df_stats)} tests)")

    # 9. Build Concise Executive Summary Table
    summary_rows = []
    key_metrics_summary = [
        "promoter_1kb_abs",
        "promoter_intron1_abs",
        "proximal_5kb_abs",
        "bounded_25kb_abs",
        "ntv3_atac_promoter_1kb",
        "ntv3_atac_promoter_intron1",
        "ntv3_atac_256_max",
        "ntv3_slam_exon_mean",
        "ntv3_slam_2.5hpf_3p",
        "ntv3_slam_exons"
    ]
    sub_stats = df_stats[df_stats["metric_code"].isin(key_metrics_summary)].copy()
    for _, r in sub_stats.iterrows():
        p_str = f"{r['p_emp_vs_ctrl_silent']:.4f}" if np.isfinite(r["p_emp_vs_ctrl_silent"]) else "N/A"
        q_str = f"{r['fdr_q_vs_ctrl_silent']:.4f}" if np.isfinite(r["fdr_q_vs_ctrl_silent"]) else "N/A"
        mw_rank = str(r["strictly_zygotic_minor_wave_rank"])
        mw_pct = f"{r['strictly_zygotic_minor_wave_percentile']:.1f}%" if np.isfinite(r["strictly_zygotic_minor_wave_percentile"]) else "N/A"
        parity = str(r["strictly_zygotic_minor_wave_parity_tier"])
        zf_str = f"{r['zebrafish_ortholog_score']:.3f}" if np.isfinite(r["zebrafish_ortholog_score"]) else "N/A"
        delta_str = f"{r['delta_target_vs_zf']:+.3f}" if np.isfinite(r["delta_target_vs_zf"]) else "N/A"
        summary_rows.append({
            "Gene": r["target_gene"],
            "Architecture": r["architecture"],
            "Modality": r["modality"],
            "Metric": r["metric_display_name"],
            "Score": f"{r['target_score']:.3f}",
            "Ctrl Silent P-val": p_str,
            "FDR q-val": q_str,
            "Rank in Minor Wave": mw_rank,
            "% Rank in Minor Wave": mw_pct,
            "Minor-Wave Parity Status": parity,
            "ZF Ortholog Score": zf_str,
            "Delta (Target - ZF)": delta_str
        })
    df_summary = pd.DataFrame(summary_rows)

    # 10. Render Publication-Quality Figure 8 (300 DPI PNG & SVG)
    fig8_png = res_dir / "fig8_target_gene_parity_benchmark.png"
    fig8_svg = res_dir / "fig8_target_gene_parity_benchmark.svg"
    _render_publication_figure_8(
        df_master=df,
        target_genes=target_genes,
        canonical_markers=canonical_markers,
        ctrl_silent_df=ctrl_silent_df,
        ctrl_nonact_df=ctrl_nonact_df,
        mw_strict_df=mw_strict_df,
        mw_highconf_df=mw_highconf_df,
        df_stats=df_stats,
        locus_name=locus_name,
        target_species_name=target_species_name,
        fig_png_path=fig8_png,
        fig_svg_path=fig8_svg
    )

    return {
        "df_stats": df_stats,
        "df_summary": df_summary,
        "fig8_png": str(fig8_png),
        "fig8_svg": str(fig8_svg),
        "stats_tsv": str(stats_tsv_path)
    }


def _render_publication_figure_8(
    df_master: pd.DataFrame,
    target_genes: List[str],
    canonical_markers: List[str],
    ctrl_silent_df: pd.DataFrame,
    ctrl_nonact_df: pd.DataFrame,
    mw_strict_df: pd.DataFrame,
    mw_highconf_df: pd.DataFrame,
    df_stats: pd.DataFrame,
    locus_name: str,
    target_species_name: str,
    fig_png_path: Path,
    fig_svg_path: Path
) -> None:
    """Generate 2x2 multi-panel publication Figure 8."""
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(18, 14.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.15, 1.0], hspace=0.34, wspace=0.28)

    # Cohort setup
    cohort_keys = ["ctrl_silent", "ctrl_nonact", "mw_strict", "mw_highconf"]
    cohort_names = [
        f"Control Silent\n(n={len(ctrl_silent_df):,})",
        f"Control Maternal\n(n={len(ctrl_nonact_df):,})",
        f"Minor Wave Zygotic\n(n={len(mw_strict_df):,})",
        f"SLAM High-Conf\n(n={len(mw_highconf_df):,})"
    ]
    cohort_colors = ["#969696", "#807dba", "#41b6c4", "#0868ac"]

    target_colors = ["#d73027", "#fc8d59", "#7f0000"]
    target_markers = ["D", "s", "^"]

    # Retrieve target gene and marker rows
    tg_rows = {
        tg: df_master[df_master["gene_name"].astype(str).str.lower() == tg.lower()]
        for tg in target_genes
    }
    marker_rows = {
        m: df_master[df_master["gene_name"].astype(str).str.lower() == m.lower()]
        for m in canonical_markers
    }

    # --------------------------------------------------------------------------
    # Panel A: ChromBPNet Accessibility (Core Promoter y_abs)
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    col_a = "pred_promoter_1kb_y_abs"
    data_a = [
        ctrl_silent_df[col_a].dropna().values,
        ctrl_nonact_df[col_a].dropna().values,
        mw_strict_df[col_a].dropna().values,
        mw_highconf_df[col_a].dropna().values
    ]

    parts_a = ax_a.violinplot(data_a, positions=[1, 2, 3, 4], showmeans=False, showextrema=False, widths=0.7)
    for pc, col in zip(parts_a["bodies"], cohort_colors):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.35)

    bp_a = ax_a.boxplot(data_a, positions=[1, 2, 3, 4], widths=0.22, patch_artist=True, showfliers=False)
    for patch, col in zip(bp_a["boxes"], cohort_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
    for median in bp_a["medians"]:
        median.set(color="black", linewidth=2.0)

    # Overlay Target Genes
    for i_tg, tg in enumerate(target_genes):
        sub_g = tg_rows.get(tg)
        if sub_g is not None and not sub_g.empty and col_a in sub_g:
            val = float(sub_g[col_a].iloc[0])
            g_col = target_colors[i_tg % len(target_colors)]
            g_mkr = target_markers[i_tg % len(target_markers)]
            ax_a.axhline(val, color=g_col, linestyle="--", lw=1.6, alpha=0.85)
            # Find stats for percentile label
            st_sub = df_stats[(df_stats["target_gene"].str.lower() == tg.lower()) & (df_stats["metric_code"] == "promoter_1kb_abs")]
            p_pct_lbl = f"{tg} ({val:.2f})"
            if not st_sub.empty:
                pct_val = st_sub["strictly_zygotic_minor_wave_percentile"].iloc[0]
                p_pct_lbl += f" | {pct_val:.0f}%ile MW"
            ax_a.scatter([3.0], [val], color=g_col, s=140, marker=g_mkr, edgecolor="black", lw=1.2, zorder=5, label=p_pct_lbl)

    # Staggered marker labels to avoid collisions
    def _plot_staggered_markers(ax_t, m_items):
        if not m_items:
            return
        y0, y1 = ax_t.get_ylim()
        min_dy = max(1e-4, 0.045 * (y1 - y0))
        m_items_sorted = sorted(m_items, key=lambda x: x[1])
        placed_ys = []
        for m_lbl, m_val in m_items_sorted:
            ax_t.scatter([3.0], [m_val], color="#252525", s=50, marker="x", alpha=0.75, zorder=4)
            y_txt = m_val
            for py in placed_ys:
                if abs(y_txt - py) < min_dy:
                    y_txt = py + min_dy
            placed_ys.append(y_txt)
            ax_t.annotate(
                m_lbl,
                xy=(3.0, m_val),
                xytext=(3.28, y_txt),
                fontsize=7.8,
                color="#404040",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.80),
                arrowprops=dict(arrowstyle="-", color="#969696", lw=0.6) if (abs(y_txt - m_val) > 0.25 * min_dy or True) else None,
                zorder=6
            )

    _plot_staggered_markers(ax_a, [
        (m, float(m_sub[col_a].iloc[0]))
        for m, m_sub in marker_rows.items()
        if not m_sub.empty and col_a in m_sub and pd.notna(m_sub[col_a].iloc[0])
    ])

    ax_a.set_xticks([1, 2, 3, 4])
    ax_a.set_xticklabels(cohort_names, fontsize=9.0, fontweight="medium")
    ax_a.set_xlim(0.45, 4.55)
    ax_a.set_ylabel("ChromBPNet Accessibility ($y_{abs}$)", fontsize=10.5, fontweight="bold")
    ax_a.set_title("A. ChromBPNet Core Promoter Accessibility (TSS ±1 kb)", fontsize=12, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_a.legend(loc="upper left", frameon=True, fontsize=8.5, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel B: NTv3 Pre-MZT Accessibility (Promoter + Intron 1 Capped)
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    col_b = "ntv3_pred_atac_promoter_intron1_capped_max" if "ntv3_pred_atac_promoter_intron1_capped_max" in df_master.columns else "ntv3_pred_atac_256cell_max"
    disp_b = "NTv3 ATAC (Promoter + Intron 1)" if col_b == "ntv3_pred_atac_promoter_intron1_capped_max" else "NTv3 ATAC (256-cell Max)"
    code_b = "ntv3_atac_promoter_intron1" if col_b == "ntv3_pred_atac_promoter_intron1_capped_max" else "ntv3_atac_256_max"
    data_b = [
        ctrl_silent_df[col_b].dropna().values,
        ctrl_nonact_df[col_b].dropna().values,
        mw_strict_df[col_b].dropna().values,
        mw_highconf_df[col_b].dropna().values
    ]

    parts_b = ax_b.violinplot(data_b, positions=[1, 2, 3, 4], showmeans=False, showextrema=False, widths=0.7)
    for pc, col in zip(parts_b["bodies"], cohort_colors):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.35)

    bp_b = ax_b.boxplot(data_b, positions=[1, 2, 3, 4], widths=0.22, patch_artist=True, showfliers=False)
    for patch, col in zip(bp_b["boxes"], cohort_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
    for median in bp_b["medians"]:
        median.set(color="black", linewidth=2.0)

    for i_tg, tg in enumerate(target_genes):
        sub_g = tg_rows.get(tg)
        if sub_g is not None and not sub_g.empty and col_b in sub_g:
            val = float(sub_g[col_b].iloc[0])
            g_col = target_colors[i_tg % len(target_colors)]
            g_mkr = target_markers[i_tg % len(target_markers)]
            ax_b.axhline(val, color=g_col, linestyle="--", lw=1.6, alpha=0.85)
            st_sub = df_stats[(df_stats["target_gene"].str.lower() == tg.lower()) & (df_stats["metric_code"] == code_b)]
            p_pct_lbl = f"{tg} ({val:.3f})"
            if not st_sub.empty:
                pct_val = st_sub["strictly_zygotic_minor_wave_percentile"].iloc[0]
                p_pct_lbl += f" | {pct_val:.0f}%ile MW"
            ax_b.scatter([3.0], [val], color=g_col, s=140, marker=g_mkr, edgecolor="black", lw=1.2, zorder=5, label=p_pct_lbl)

    _plot_staggered_markers(ax_b, [
        (m, float(m_sub[col_b].iloc[0]))
        for m, m_sub in marker_rows.items()
        if not m_sub.empty and col_b in m_sub and pd.notna(m_sub[col_b].iloc[0])
    ])

    ax_b.set_xticks([1, 2, 3, 4])
    ax_b.set_xticklabels(cohort_names, fontsize=9.0, fontweight="medium")
    ax_b.set_xlim(0.45, 4.55)
    ax_b.set_ylabel(disp_b, fontsize=10.5, fontweight="bold")
    ax_b.set_title(f"B. {disp_b}", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_b.legend(loc="upper left", frameon=True, fontsize=8.5, framealpha=0.92)

    # Panel C: NTv3 Pre-MZT Nascent Expression (log1p SLAM-seq, Signal per kb)
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    col_c = "ntv3_pred_slam_2.5hpf_3p" if "ntv3_pred_slam_2.5hpf_3p" in df_master.columns else "ntv3_pred_slam_2.5hpf_all_exons"
    disp_c = "NTv3 Nascent SLAM 3'-End (Signal/kb at TES)" if col_c == "ntv3_pred_slam_2.5hpf_3p" else "NTv3 Nascent SLAM (2.5 hpf Exons)"
    code_c = "ntv3_slam_2.5hpf_3p" if col_c == "ntv3_pred_slam_2.5hpf_3p" else "ntv3_slam_exons"

    def _filt_panel_c(cdf: pd.DataFrame) -> np.ndarray:
        vals = cdf[col_c].dropna() if col_c in cdf.columns else pd.Series(dtype=float)
        if col_c == "ntv3_pred_slam_2.5hpf_3p" and "slam_3p_tes_centered" in cdf.columns:
            bad = (cdf[col_c] == 0.0) & (~cdf["slam_3p_tes_centered"].fillna(False).astype(bool))
            med_pos = float(vals[vals > 1.0].median()) if (vals > 1.0).any() else 0.0
            if med_pos > 10.0:
                bad = bad | ((cdf[col_c] >= 0.0) & (cdf[col_c] < 1.0))
            vals = cdf.loc[~bad, col_c].dropna()
        return np.log1p(np.maximum(0, vals.values))

    # Log1p transform for visual clarity across orders of magnitude
    data_c = [
        _filt_panel_c(ctrl_silent_df),
        _filt_panel_c(ctrl_nonact_df),
        _filt_panel_c(mw_strict_df),
        _filt_panel_c(mw_highconf_df),
    ]
    cohort_names_c = [
        f"Control Silent\n(n={len(data_c[0]):,})",
        f"Control Maternal\n(n={len(data_c[1]):,})",
        f"Minor Wave Zygotic\n(n={len(data_c[2]):,})",
        f"SLAM High-Conf\n(n={len(data_c[3]):,})"
    ]

    parts_c = ax_c.violinplot(data_c, positions=[1, 2, 3, 4], showmeans=False, showextrema=False, widths=0.7)
    for pc, col in zip(parts_c["bodies"], cohort_colors):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.35)

    bp_c = ax_c.boxplot(data_c, positions=[1, 2, 3, 4], widths=0.22, patch_artist=True, showfliers=False)
    for patch, col in zip(bp_c["boxes"], cohort_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
    for median in bp_c["medians"]:
        median.set(color="black", linewidth=2.0)

    for i_tg, tg in enumerate(target_genes):
        sub_g = tg_rows.get(tg)
        if sub_g is not None and not sub_g.empty and col_c in sub_g:
            raw_val = float(sub_g[col_c].iloc[0])
            log_val = np.log1p(max(0, raw_val))
            g_col = target_colors[i_tg % len(target_colors)]
            g_mkr = target_markers[i_tg % len(target_markers)]
            ax_c.axhline(log_val, color=g_col, linestyle="--", lw=1.6, alpha=0.85)
            st_sub = df_stats[(df_stats["target_gene"].str.lower() == tg.lower()) & (df_stats["metric_code"] == code_c)]
            p_pct_lbl = f"{tg} ({raw_val:.3f})"
            if not st_sub.empty:
                pct_val = st_sub["strictly_zygotic_minor_wave_percentile"].iloc[0]
                p_pct_lbl += f" | {pct_val:.0f}%ile MW"
            ax_c.scatter([3.0], [log_val], color=g_col, s=140, marker=g_mkr, edgecolor="black", lw=1.2, zorder=5, label=p_pct_lbl)

    _plot_staggered_markers(ax_c, [
        (m, np.log1p(max(0, float(m_sub[col_c].iloc[0]))))
        for m, m_sub in marker_rows.items()
        if not m_sub.empty and col_c in m_sub and pd.notna(m_sub[col_c].iloc[0])
    ])

    ax_c.set_xticks([1, 2, 3, 4])
    ax_c.set_xticklabels(cohort_names_c, fontsize=9.0, fontweight="medium")
    ax_c.set_xlim(0.45, 4.55)
    ax_c.set_ylabel(f"{disp_c}\nlog(1 + Signal/kb)", fontsize=10.0, fontweight="bold")
    ax_c.set_title(f"C. {disp_c}", fontsize=12, fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax_c.legend(loc="upper left", frameon=True, fontsize=8.5, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel D: Minor-Wave Parity vs. Control-Silent Calibration (Quantile-CDF)
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])

    def _clean_panel_d_slam(sdf: pd.DataFrame) -> np.ndarray:
        if col_c not in sdf.columns:
            return np.array([])
        s = sdf[col_c].dropna()
        if col_c == "ntv3_pred_slam_2.5hpf_3p" and "slam_3p_tes_centered" in sdf.columns:
            bad = (sdf[col_c] == 0.0) & (~sdf["slam_3p_tes_centered"].fillna(False).astype(bool))
            med_pos = float(s[s > 1.0].median()) if (s > 1.0).any() else 0.0
            if med_pos > 10.0:
                bad = bad | ((sdf[col_c] >= 0.0) & (sdf[col_c] < 1.0))
            s = sdf.loc[~bad, col_c].dropna()
        return s.values

    sil_cb = ctrl_silent_df["pred_promoter_1kb_y_abs"].dropna().values if "pred_promoter_1kb_y_abs" in ctrl_silent_df.columns else np.array([])
    sil_nt_at = ctrl_silent_df[col_b].dropna().values if col_b in ctrl_silent_df.columns else np.array([])
    sil_nt_sl = _clean_panel_d_slam(ctrl_silent_df)

    mw_cb = mw_strict_df["pred_promoter_1kb_y_abs"].dropna().values if "pred_promoter_1kb_y_abs" in mw_strict_df.columns else np.array([])
    mw_nt_at = mw_strict_df[col_b].dropna().values if col_b in mw_strict_df.columns else np.array([])
    mw_nt_sl = _clean_panel_d_slam(mw_strict_df)

    # Diagonal y = x represents zero shift between Minor Wave and Control Silent (AUROC = 0.50)
    ax_d.plot([0, 100], [0, 100], color="#636363", linestyle="--", lw=1.2, alpha=0.8, label="No Enrichment vs. Silent (AUROC = 0.50)")

    # Shaded quartile regions
    ax_d.axhspan(25, 75, color="#31a354", alpha=0.12, label="Minor-Wave Parity IQR (Q25–Q75)")
    ax_d.axhspan(75, 100, color="#0868ac", alpha=0.08, label="Top Minor-Wave Quartile (>Q75)")

    track_curves = [
        ("ChromBPNet ATAC (1kb)", mw_cb, sil_cb, "#2b5c8f", "promoter_1kb_abs"),
        ("NTv3 ATAC (Intron 1)", mw_nt_at, sil_nt_at, "#d73027", code_b),
        ("NTv3 SLAM (TES 3')", mw_nt_sl, sil_nt_sl, "#1b7837", code_c)
    ]

    for label_t, mw_vals, sil_vals, col_t, m_code in track_curves:
        if len(mw_vals) > 0 and len(sil_vals) > 0:
            s_mw = np.sort(mw_vals)
            s_sil = np.sort(sil_vals)
            x_sil_pct = np.searchsorted(s_sil, s_mw, side="right") / float(len(s_sil)) * 100.0
            y_mw_pct = np.arange(1, len(s_mw) + 1) / float(len(s_mw)) * 100.0
            ax_d.plot(x_sil_pct, y_mw_pct, color=col_t, lw=2.2, label=f"{label_t}")

    # Plot each target gene directly at (Control-Silent %ile, Minor-Wave %ile) across the 3 architectures
    parity_badges = []
    pt_entries = []
    for i_tg, tg in enumerate(target_genes):
        g_col = target_colors[i_tg % len(target_colors)]
        g_mkr = target_markers[i_tg % len(target_markers)]
        st_tg = df_stats[df_stats["target_gene"].str.lower() == tg.lower()]
        tg_pcts = {}
        for short_m, t_col in [("ChromBPNet", "promoter_1kb_abs"), ("NTv3_ATAC", code_b), ("SLAM", code_c)]:
            row_m = st_tg[st_tg["metric_code"] == t_col]
            if not row_m.empty:
                x_s = float(row_m["percentile_in_ctrl_silent"].iloc[0])
                y_m = float(row_m["strictly_zygotic_minor_wave_percentile"].iloc[0])
                tg_pcts[short_m] = y_m
                c_pt = "#2b5c8f" if short_m == "ChromBPNet" else ("#d73027" if short_m == "NTv3_ATAC" else "#1b7837")
                if np.isfinite(x_s) and np.isfinite(y_m):
                    ax_d.scatter([x_s], [y_m], facecolor=g_col, edgecolor=c_pt, marker=g_mkr, s=115, linewidth=2.2, zorder=7)
                    pt_entries.append((x_s, y_m, f"{tg}\n({short_m.replace('_', ' ')})", g_col, c_pt))
        # Key legend item for target marker shape
        ax_d.scatter([], [], facecolor=g_col, edgecolor="black", marker=g_mkr, s=90, label=f"{tg} (points on curves)")
        parity_badges.append(
            f"{tg}: ChromBPNet={tg_pcts.get('ChromBPNet', np.nan):.0f}%, NTv3_ATAC={tg_pcts.get('NTv3_ATAC', np.nan):.0f}%, SLAM={tg_pcts.get('SLAM', np.nan):.0f}%"
        )

    # Greedy collision-free label placement for (x_s, y_m) points on curves
    placed_boxes = []
    cand_offsets = [(-14, 16), (-64, 2), (-14, -22), (-72, -16), (14, 12), (14, -18), (-85, 22), (-85, -28)]
    for x_s, y_m, lbl_txt, g_col, c_pt in sorted(pt_entries, key=lambda t: (-t[1], -t[0])):
        best_off = cand_offsets[0]
        best_score = 1e9
        for ox, oy in cand_offsets:
            # Approximate label anchor center in 0-100 data units (~11 x 9 units per box)
            ha = "right" if ox < 0 else "left"
            cx = x_s + ox * 0.24 - (9.0 if ha == "right" else -9.0)
            cy = y_m + oy * 0.24
            score = 0.0
            if cx < 10.0 or cx > 90.0 or cy < 8.0 or cy > 90.0:
                score += 100.0
            for px, py in placed_boxes:
                if abs(cx - px) < 20.0 and abs(cy - py) < 11.5:
                    score += 20.0 - abs(cy - py)
            if score < best_score:
                best_score = score
                best_off = (ox, oy)
        ha = "right" if best_off[0] < 0 else "left"
        placed_boxes.append((x_s + best_off[0] * 0.24 - (9.0 if ha == "right" else -9.0), y_m + best_off[1] * 0.24))
        ax_d.annotate(
            lbl_txt,
            xy=(x_s, y_m),
            xytext=best_off,
            textcoords="offset points",
            fontsize=7.3,
            fontweight="bold",
            color=g_col,
            ha=ha,
            va="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=c_pt, lw=0.75, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=c_pt, lw=0.8, alpha=0.85),
            zorder=8
        )

    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0, 102)
    ax_d.set_xlabel("Control-Silent Percentile Rank (0–100%, Right = Stronger Enrichment)", fontsize=10.2, fontweight="bold")
    ax_d.set_ylabel("Percentile Rank in Minor Wave (0–100%)", fontsize=10.2, fontweight="bold")
    ax_d.set_title("D. Cross-Architecture Parity vs. Silent Calibration", fontsize=12, fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.4)
    ax_d.legend(loc="lower right", frameon=True, fontsize=7.6, framealpha=0.92)

    # Text annotation box with exact consensus percentiles
    ax_d.text(
        0.03, 0.96,
        "Minor-Wave Percentiles:\n" + "\n".join(parity_badges),
        transform=ax_d.transAxes,
        fontsize=8.3,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.92)
    )

    # Subplot adjust & atomic exports
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.96, hspace=0.34, wspace=0.28)
    fig.suptitle(
        f"Target Gene Statistical Enrichment & Minor-Wave Parity Benchmark: '{locus_name}' ({target_species_name})",
        fontsize=14.5, fontweight="bold", y=0.98
    )

    plt.savefig(fig_png_path, dpi=300)
    plt.savefig(fig_svg_path)
    plt.close(fig)
    print(f"[+] Successfully exported Figure 8: {fig_png_path} and .svg")
