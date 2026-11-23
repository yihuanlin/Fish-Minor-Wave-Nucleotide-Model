#!/usr/bin/env python3
"""
scripts/zero_shoot_validation.py
================================
Standalone background worker for Zero-Shot Validation:
Zero-Shot Validation on Target-Species Minor-Wave Genes, Zebrafish In-Species
Ceiling Benchmark, and Cross-Species Medaka ATAC.
"""

import os
import re
import sys
import gzip
import json
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import pyfaidx
except ImportError:
    pyfaidx = None

from scripts.task_manager import set_task_progress
from scripts.infer_and_attribute import (
    load_grafted_model,
    score_tss_regions,
    benchmark_transfer_metrics,
    score_cross_species_medaka_atac,
    sanitize_for_torch_save,
)

# ==============================================================================
# CLI Arguments
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Cross-Species Validation Worker"
    )
    parser.add_argument(
        "--work-dir", type=str, default="./work_astatotilapia_premzt",
        help="Path to working directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to study config.json")
    parser.add_argument(
        "--folds", type=str, default="0",
        help="Comma-separated fold indices to evaluate (e.g. '0' or '0,1,2')"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="GPU inference batch size")
    parser.add_argument(
        "--enable-medaka", dest="enable_medaka", action="store_true", default=None,
        help="Enable Medaka model evaluation"
    )
    parser.add_argument(
        "--no-enable-medaka", dest="enable_medaka", action="store_false",
        help="Disable Medaka model evaluation"
    )
    parser.add_argument(
        "--task-name", type=str, default="zero_shoot_zeroshot_validation",
        help="TaskManager task name"
    )
    # Upgraded Block 4 controls, windows, and ceiling options
    parser.add_argument(
        "--primary-control", type=str, default="control_silent",
        help="Primary negative control category (default: control_silent)"
    )
    parser.add_argument(
        "--diagnostic-control", type=str, default="control_non_activated",
        help="Diagnostic negative control category (default: control_non_activated)"
    )
    parser.add_argument(
        "--silent-max-tpm", type=float, default=0.1,
        help="Max TPM in both early and pre-MZT stages to define silent controls"
    )
    parser.add_argument(
        "--max-silent-controls", type=int, default=1000,
        help="Max number of GC-matched silent control genes to sample"
    )
    parser.add_argument(
        "--windows", type=str, default="promoter_1kb,proximal_5kb,bounded_25kb",
        help="Comma-separated scoring windows: promoter_1kb,proximal_5kb,bounded_25kb"
    )
    parser.add_argument(
        "--strict-early-max-tpm", type=float, default=0.2,
        help="Max early-stage TPM for high-confidence strictly zygotic genes"
    )
    parser.add_argument(
        "--strict-premzt-min-tpm", type=float, default=5.0,
        help="Min pre-MZT TPM for high-confidence strictly zygotic genes"
    )
    parser.add_argument(
        "--case-insensitive-orthologs", action="store_true", default=True,
        help="Use case-insensitive + teleost paralog-normalized matching for SLAM-seq orthologs"
    )
    parser.add_argument(
        "--run-zebrafish-ceiling", action="store_true", default=False,
        help="Run in-species Zebrafish minor-wave ceiling benchmark with identical windowing"
    )
    return parser.parse_args()


# ==============================================================================
# FASTA / GTF / Gene-Symbol Utilities
# ==============================================================================
class GenomeAccessor:
    """Thin wrapper over pyfaidx (with pure-Python fallback) + chr-prefix resolution."""

    def __init__(self, fa_path: str):
        self.fa_path = str(fa_path)
        self._fa = None
        self._seqs: Dict[str, str] = {}
        self.chrom_lengths: Dict[str, int] = {}
        if pyfaidx is not None:
            self._fa = pyfaidx.Fasta(self.fa_path, as_raw=True, sequence_always_upper=True)
            for k in self._fa.keys():
                self.chrom_lengths[k] = len(self._fa[k])
        else:
            opener = gzip.open if self.fa_path.endswith(".gz") else open
            cur_name, chunks = None, []
            with opener(self.fa_path, "rt") as fh:
                for line in fh:
                    if line.startswith(">"):
                        if cur_name is not None:
                            s = "".join(chunks).upper()
                            self._seqs[cur_name] = s
                            self.chrom_lengths[cur_name] = len(s)
                        cur_name = line[1:].strip().split()[0]
                        chunks = []
                    else:
                        chunks.append(line.strip())
                if cur_name is not None:
                    s = "".join(chunks).upper()
                    self._seqs[cur_name] = s
                    self.chrom_lengths[cur_name] = len(s)

    def resolve_chrom(self, chrom: str) -> Optional[str]:
        c = str(chrom)
        if c in self.chrom_lengths:
            return c
        if c.startswith("chr") and c[3:] in self.chrom_lengths:
            return c[3:]
        if f"chr{c}" in self.chrom_lengths:
            return f"chr{c}"
        return None

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch [start, end) 0-based uppercase sequence, N-padding out-of-bounds edges."""
        length = end - start
        if length <= 0:
            return ""
        rc = self.resolve_chrom(chrom)
        if rc is None:
            return "N" * length
        clen = self.chrom_lengths[rc]
        pad_left = max(0, -start)
        pad_right = max(0, end - clen)
        s0 = max(0, start)
        e0 = min(clen, end)
        if s0 >= e0:
            return "N" * length
        if self._fa is not None:
            core = str(self._fa[rc][s0:e0]).upper()
        else:
            core = self._seqs[rc][s0:e0]
        return ("N" * pad_left) + core + ("N" * pad_right)


def parse_gtf_genes(gtf_path: Optional[str]) -> pd.DataFrame:
    """Parse gene-level TSS coordinates (chrom, tss, strand, gene_id, gene_name) from GTF."""
    if not gtf_path or not Path(gtf_path).exists():
        return pd.DataFrame(columns=["gene_id", "gene_name", "chrom", "tss", "strand"])

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    rows = []
    seen = set()
    id_re = re.compile(r'gene_id "([^"]+)"')
    name_re = re.compile(r'gene_name "([^"]+)"')

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom, _, _, start_s, end_s, _, strand, _, attrs = parts
            m_id = id_re.search(attrs)
            if not m_id:
                continue
            gid = m_id.group(1)
            if gid in seen:
                continue
            seen.add(gid)
            m_name = name_re.search(attrs)
            gname = m_name.group(1) if m_name else gid
            start, end = int(start_s) - 1, int(end_s)
            tss = start if strand == "+" else end
            rows.append({
                "gene_id": gid,
                "gene_name": gname,
                "chrom": chrom,
                "tss": int(tss),
                "strand": strand if strand in ("+", "-") else "+",
            })
    return pd.DataFrame(rows)


def normalize_gene_symbol(sym: Any) -> List[str]:
    """Return normalized lowercase symbol variants (exact + stripped teleost a/b paralog)."""
    if sym is None or (isinstance(sym, float) and np.isnan(sym)):
        return []
    s = str(sym).strip().lower()
    if not s or s == "nan":
        return []
    out = [s]
    # Strip Ensembl version suffix if present
    base = re.sub(r"\.\d+$", "", s)
    if base != s:
        out.append(base)
    # Strip trailing teleost a/b paralog suffix for symbols longer than 3 chars (e.g., slc35a3a -> slc35a3)
    if len(base) > 3 and base[-1] in ("a", "b") and base[-2].isalnum():
        out.append(base[:-1])
    return list(dict.fromkeys(out))


def seq_gc_fraction(seq: str) -> float:
    if not seq:
        return 0.5
    valid = seq.count("A") + seq.count("C") + seq.count("G") + seq.count("T")
    if valid == 0:
        return 0.5
    return (seq.count("G") + seq.count("C")) / float(valid)


def one_hot_encode_batch(seqs: List[str]) -> np.ndarray:
    """Encode list of equal-length DNA strings to float32 array of shape (N, 4, L)."""
    n = len(seqs)
    L = len(seqs[0])
    arr = np.zeros((n, 4, L), dtype=np.float32)
    lut = np.full(256, -1, dtype=np.int8)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3
    for i, s in enumerate(seqs):
        b = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8)
        idx = lut[b[:L]]
        valid = idx >= 0
        arr[i, idx[valid], np.nonzero(valid)[0]] = 1.0
    return arr


def reverse_complement_onehot(x_4xl: np.ndarray) -> np.ndarray:
    """Reverse complement of (N, 4, L) one-hot array (A<->T: 0<->3, C<->G: 1<->2)."""
    return x_4xl[:, ::-1, ::-1].copy()


# ==============================================================================
# Gene Set Enrichment: Silent Controls, High-Conf Strictly Zygotic, SLAM Orthologs
# ==============================================================================
def _find_tpm_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    early_cands = [
        "tpm_early", "early_tpm", "tpm_2cs", "mean_tpm_2cs", "tpm_2c",
        "mean_tpm_early", "2cs_tpm", "early_mean_tpm",
        "cpm_early", "early_cpm", "cpm_2cs", "mean_cpm_2cs", "cpm_2c",
        "mean_cpm_early", "2cs_cpm", "early_mean_cpm",
    ]
    premzt_cands = [
        "tpm_premzt", "premzt_tpm", "tpm_1000cs", "mean_tpm_1000cs", "tpm_1k",
        "mean_tpm_premzt", "1000cs_tpm", "premzt_mean_tpm", "late_tpm", "tpm_late",
        "cpm_premzt", "premzt_cpm", "cpm_1000cs", "mean_cpm_1000cs", "cpm_1k",
        "mean_cpm_premzt", "1000cs_cpm", "premzt_mean_cpm", "late_cpm", "cpm_late",
    ]
    early_col = next((cols_lower[k] for k in early_cands if k in cols_lower), None)
    premzt_col = next((cols_lower[k] for k in premzt_cands if k in cols_lower), None)
    return early_col, premzt_col


def _load_full_rnaseq_table(work_dir: Path) -> Optional[pd.DataFrame]:
    """Locate Block 1 full gene-level RNA-seq table (including silent genes) if saved."""
    rnaseq_dir = work_dir / "rnaseq"
    if not rnaseq_dir.exists():
        return None
    patterns = [
        "*all_genes*.tsv", "*deseq*.tsv", "*de_results*.tsv",
        "*tpm*.tsv", "*expression*.tsv", "*target_species_gene_sets*.tsv", "*.tsv"
    ]
    best_df = None
    best_len = 0
    for pat in patterns:
        for p in sorted(rnaseq_dir.glob(pat)):
            if "scored" in p.name:
                continue
            try:
                cand = pd.read_csv(p, sep="\t")
                if len(cand) > best_len and any(
                    k in [c.lower() for c in cand.columns] for k in ("gene_id", "gene", "gene_name", "geneid")
                ):
                    best_df = cand
                    best_len = len(cand)
            except Exception:
                continue
    return best_df


def _load_slam_minor_wave_symbols(work_dir: Path, study_cfg: Optional[Dict[str, Any]] = None) -> set:
    """Collect normalized gene symbols of empirical SLAM-seq minor-wave genes directly from prep_slamseq outputs."""
    slam_dir = Path(work_dir) / "slamseq"
    primary_tsv = slam_dir / "zebrafish_slamseq_minor_wave_genes.tsv"
    summary_tsv = slam_dir / "zebrafish_slamseq_gene_summary.tsv"

    target_tsv = None
    if primary_tsv.exists() and primary_tsv.stat().st_size > 0:
        target_tsv = primary_tsv
    elif summary_tsv.exists() and summary_tsv.stat().st_size > 0:
        target_tsv = summary_tsv

    if target_tsv is None:
        print(
            f"[!] SLAM-seq minor-wave gene table not found in {slam_dir} "
            f"(expected {primary_tsv.name}). Please run 'scripts/prep_slamseq.py' first. "
            f"No hardcoded fallback will be used (n=0 confirmed orthologs).",
            flush=True,
        )
        return set()

    symbols = set()
    try:
        df_s = pd.read_csv(target_tsv, sep="\t")
        if "is_slam_minor_wave" in df_s.columns:
            df_s = df_s[df_s["is_slam_minor_wave"].fillna(False).astype(bool)]
        elif "slam_class" in df_s.columns:
            df_s = df_s[df_s["slam_class"].astype(str).str.startswith("preMZT")]

        for col in ("gene_name", "gene_id", "gene_symbol", "symbol"):
            if col in df_s.columns:
                for v in df_s[col].dropna():
                    for tok in normalize_gene_symbol(v):
                        symbols.add(tok)
        print(f"[+] Loaded {len(symbols):,} normalized empirical SLAM-seq minor-wave symbols from {target_tsv.name}", flush=True)
    except Exception as e:
        print(f"[-] Error reading SLAM-seq minor-wave table {target_tsv}: {e}", flush=True)
        return set()

    return symbols


def gc_match_controls(
    pos_gc: np.ndarray,
    cand_df: pd.DataFrame,
    cand_gc: np.ndarray,
    n_target: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Select up to n_target candidate controls matched to pos_gc distribution without replacement."""
    if len(cand_df) <= n_target or len(pos_gc) == 0:
        return cand_df.copy()

    rng = np.random.default_rng(seed)
    # Sample target GC values from positive empirical distribution
    target_gcs = rng.choice(pos_gc, size=n_target, replace=True)
    # Sort candidates by GC for fast nearest-available lookup
    order = np.argsort(cand_gc)
    sorted_gc = cand_gc[order]
    used = np.zeros(len(sorted_gc), dtype=bool)
    picked_orig_idx = []

    for g in target_gcs:
        pos = int(np.searchsorted(sorted_gc, g))
        # Expand outward to find closest unused candidate
        best_j = -1
        best_dist = np.inf
        left, right = pos - 1, pos
        while left >= 0 or right < len(sorted_gc):
            if left >= 0 and not used[left]:
                d = abs(sorted_gc[left] - g)
                if d < best_dist:
                    best_dist, best_j = d, left
                left = -1  # no need to move further left
            elif left >= 0:
                left -= 1
            if right < len(sorted_gc) and not used[right]:
                d = abs(sorted_gc[right] - g)
                if d < best_dist:
                    best_dist, best_j = d, right
                right = len(sorted_gc)
            elif right < len(sorted_gc):
                right += 1
        if best_j >= 0:
            used[best_j] = True
            picked_orig_idx.append(int(order[best_j]))

    return cand_df.iloc[picked_orig_idx].copy()


def enrich_target_gene_sets(
    df_gene_sets: pd.DataFrame,
    df_gtf: pd.DataFrame,
    genome: GenomeAccessor,
    work_dir: Path,
    study_cfg: Dict[str, Any],
    silent_max_tpm: float = 0.1,
    max_silent_controls: int = 1000,
    strict_early_max_tpm: float = 0.2,
    strict_premzt_min_tpm: float = 5.0,
    case_insensitive_orthologs: bool = True,
) -> pd.DataFrame:
    """
    Ensure df_gene_sets contains:
      - is_strictly_zygotic_minor_wave
      - is_strictly_zygotic_highconf
      - is_all_upregulated
      - is_slam_confirmed_minor_wave
      - is_control_non_activated
      - is_control_silent (GC-matched transcriptionally silent genes)
    """
    df = df_gene_sets.copy()

    # Standardize key column names if needed
    if "chrom" not in df.columns and "chr" in df.columns:
        df["chrom"] = df["chr"]
    if "strand" not in df.columns:
        df["strand"] = "+"
    if "gene_id" not in df.columns:
        df["gene_id"] = df.get("gene_name", pd.Series([f"gene_{i}" for i in range(len(df))]))
    if "gene_name" not in df.columns:
        df["gene_name"] = df["gene_id"]
    if "category" not in df.columns:
        df["category"] = "unknown"

    # Merge TPM columns from full Block 1 RNA-seq table if missing
    early_col, premzt_col = _find_tpm_columns(df)
    full_rna = _load_full_rnaseq_table(work_dir)
    if full_rna is not None and (early_col is None or premzt_col is None):
        f_early, f_premzt = _find_tpm_columns(full_rna)
        if f_early and f_premzt:
            join_key = "gene_id" if "gene_id" in full_rna.columns else full_rna.columns[0]
            tmp_m = full_rna[[join_key, f_early, f_premzt]].drop_duplicates(subset=[join_key])
            tmp_m = tmp_m.rename(columns={join_key: "gene_id", f_early: "tpm_early", f_premzt: "tpm_premzt"})
            df = df.merge(tmp_m, on="gene_id", how="left")
            early_col, premzt_col = "tpm_early", "tpm_premzt"

    # Existing boolean flags
    def _existing_flag(col_name: str, cats: List[str]) -> pd.Series:
        s = df["category"].isin(cats)
        if col_name in df.columns:
            s = s | df[col_name].fillna(False).astype(bool)
        return s

    df["is_strictly_zygotic_minor_wave"] = _existing_flag(
        "is_strictly_zygotic_minor_wave", ["strictly_zygotic_minor_wave"]
    )
    df["is_all_upregulated"] = _existing_flag(
        "is_all_upregulated", ["all_upregulated", "strictly_zygotic_minor_wave"]
    )
    df["is_control_non_activated"] = _existing_flag(
        "is_control_non_activated", ["control_non_activated"]
    )
    df["is_control_silent"] = _existing_flag("is_control_silent", ["control_silent"])

    # High-confidence strictly zygotic subset
    if early_col and premzt_col:
        e_vals = pd.to_numeric(df[early_col], errors="coerce")
        p_vals = pd.to_numeric(df[premzt_col], errors="coerce")
        df["is_strictly_zygotic_highconf"] = (
            (df["is_strictly_zygotic_minor_wave"] | df["is_all_upregulated"])
            & (e_vals < strict_early_max_tpm)
            & (p_vals >= strict_premzt_min_tpm)
        )
    else:
        # Fallback: top half of strictly zygotic by log2FoldChange if TPM columns absent
        lfc_col = next((c for c in df.columns if "log2f" in c.lower()), None)
        sz_mask = df["is_strictly_zygotic_minor_wave"]
        if lfc_col and sz_mask.sum() > 0:
            med_lfc = pd.to_numeric(df.loc[sz_mask, lfc_col], errors="coerce").median()
            df["is_strictly_zygotic_highconf"] = sz_mask & (
                pd.to_numeric(df[lfc_col], errors="coerce") >= med_lfc
            )
        else:
            df["is_strictly_zygotic_highconf"] = sz_mask.copy()

    # Case-insensitive / paralog-normalized SLAM-seq confirmed ortholog matching
    existing_slam = _existing_flag("is_slam_confirmed_minor_wave", ["slam_confirmed_minor_wave"])
    if "is_slam_confirmed" in df.columns:
        existing_slam = existing_slam | df["is_slam_confirmed"].fillna(False).astype(bool)
    slam_syms = _load_slam_minor_wave_symbols(work_dir, study_cfg)
    if slam_syms and case_insensitive_orthologs:
        matched = []
        for gname, gid, is_up in zip(df["gene_name"], df["gene_id"], df["is_all_upregulated"]):
            if not is_up:
                matched.append(False)
                continue
            toks = normalize_gene_symbol(gname) + normalize_gene_symbol(gid)
            matched.append(any(t in slam_syms for t in toks))
        df["is_slam_confirmed_minor_wave"] = existing_slam | pd.Series(matched, index=df.index)
    else:
        df["is_slam_confirmed_minor_wave"] = existing_slam

    # Build GC-matched transcriptionally silent primary controls if not already present
    if df["is_control_silent"].sum() < 50:
        silent_cands = pd.DataFrame()
        silent_from_df = False

        non_pos_mask = (
            (~df["is_strictly_zygotic_minor_wave"])
            & (~df["is_all_upregulated"])
            & (~df["is_control_non_activated"])
            & (~df["is_slam_confirmed_minor_wave"])
        )

        # Source 1: Check if candidate silent genes already exist directly in df
        if early_col and premzt_col:
            e_num = pd.to_numeric(df[early_col], errors="coerce").fillna(0.0)
            p_num = pd.to_numeric(df[premzt_col], errors="coerce").fillna(0.0)
            cand_mask = non_pos_mask & (e_num < silent_max_tpm) & (p_num < silent_max_tpm)
            if cand_mask.sum() >= 50:
                silent_cands = df[cand_mask].copy()
                silent_from_df = True
        elif "baseMean" in df.columns:
            bm_num = pd.to_numeric(df["baseMean"], errors="coerce").fillna(0.0)
            cand_mask = non_pos_mask & (bm_num < 1.0)
            if cand_mask.sum() >= 50:
                silent_cands = df[cand_mask].copy()
                silent_from_df = True

        # Source 2: If df only has expressed/positive genes, extract silent candidates from df_gtf & full_rna
        if len(silent_cands) < 50:
            silent_from_df = False
            existing_ids = set(df["gene_id"].astype(str))
            if full_rna is not None and len(df_gtf) > 0:
                f_early, f_premzt = _find_tpm_columns(full_rna)
                join_key = "gene_id" if "gene_id" in full_rna.columns else full_rna.columns[0]
                if f_early and f_premzt:
                    e_num = pd.to_numeric(full_rna[f_early], errors="coerce").fillna(0.0)
                    p_num = pd.to_numeric(full_rna[f_premzt], errors="coerce").fillna(0.0)
                    sil_ids = set(full_rna.loc[(e_num < silent_max_tpm) & (p_num < silent_max_tpm), join_key].astype(str))
                    silent_cands = df_gtf[
                        df_gtf["gene_id"].astype(str).isin(sil_ids)
                        & (~df_gtf["gene_id"].astype(str).isin(existing_ids))
                    ].copy()
                elif "baseMean" in full_rna.columns:
                    bm = pd.to_numeric(full_rna["baseMean"], errors="coerce").fillna(0.0)
                    sil_ids = set(full_rna.loc[bm < 1.0, join_key].astype(str))
                    silent_cands = df_gtf[
                        df_gtf["gene_id"].astype(str).isin(sil_ids)
                        & (~df_gtf["gene_id"].astype(str).isin(existing_ids))
                    ].copy()

            # Fallback: GTF genes completely absent from expressed gene table (0 counts in RNA-seq)
            if len(silent_cands) < 50 and len(df_gtf) > 0:
                expressed_ids = set(df.loc[~non_pos_mask, "gene_id"].astype(str))
                if full_rna is not None:
                    join_key = "gene_id" if "gene_id" in full_rna.columns else full_rna.columns[0]
                    f_early, f_premzt = _find_tpm_columns(full_rna)
                    if f_early and f_premzt:
                        e_num = pd.to_numeric(full_rna[f_early], errors="coerce").fillna(0.0)
                        p_num = pd.to_numeric(full_rna[f_premzt], errors="coerce").fillna(0.0)
                        expressed_ids |= set(
                            full_rna.loc[(e_num >= silent_max_tpm) | (p_num >= silent_max_tpm), join_key].astype(str)
                        )
                    elif "baseMean" in full_rna.columns:
                        bm = pd.to_numeric(full_rna["baseMean"], errors="coerce").fillna(0.0)
                        expressed_ids |= set(full_rna.loc[bm >= 1.0, join_key].astype(str))
                silent_cands = df_gtf[~df_gtf["gene_id"].astype(str).isin(expressed_ids)].copy()

        # Filter to contigs present in genome FASTA
        if len(silent_cands) > 0:
            silent_cands = silent_cands[
                silent_cands["chrom"].apply(lambda c: genome.resolve_chrom(c) is not None)
            ].copy()

        if len(silent_cands) > 0:
            pos_ref = df[df["is_strictly_zygotic_minor_wave"]]
            if len(pos_ref) == 0:
                pos_ref = df[df["is_all_upregulated"]]
            if len(pos_ref) == 0:
                pos_ref = df[df["is_slam_confirmed_minor_wave"]]

            def _prom_gc(sub_df: pd.DataFrame) -> np.ndarray:
                gcs = []
                for c, t, st in zip(sub_df["chrom"], sub_df["tss"], sub_df["strand"]):
                    t_int = int(t)
                    s, e = (t_int - 1000, t_int + 500) if str(st) == "+" else (t_int - 500, t_int + 1000)
                    gcs.append(seq_gc_fraction(genome.fetch(c, s, e)))
                return np.asarray(gcs, dtype=np.float32)

            pos_gc = _prom_gc(pos_ref)
            # Subsample huge candidate pool slightly before GC computation if > 15k for speed
            if len(silent_cands) > 15000:
                silent_cands = silent_cands.sample(n=15000, random_state=42)
            cand_gc = _prom_gc(silent_cands)
            n_pick = min(max_silent_controls, max(300, 5 * len(pos_ref)), len(silent_cands))
            picked = gc_match_controls(pos_gc, silent_cands, cand_gc, n_target=n_pick, seed=42)

            if silent_from_df:
                picked_ids = set(picked["gene_id"].astype(str))
                is_picked = df["gene_id"].astype(str).isin(picked_ids)
                df.loc[is_picked, "category"] = "control_silent"
                df.loc[is_picked, "is_control_silent"] = True
                df.loc[is_picked, "is_control_non_activated"] = False
                df.loc[is_picked, "is_strictly_zygotic_minor_wave"] = False
                df.loc[is_picked, "is_strictly_zygotic_highconf"] = False
                df.loc[is_picked, "is_all_upregulated"] = False
                df.loc[is_picked, "is_slam_confirmed_minor_wave"] = False
            else:
                picked["category"] = "control_silent"
                picked["is_control_silent"] = True
                picked["is_control_non_activated"] = False
                picked["is_strictly_zygotic_minor_wave"] = False
                picked["is_strictly_zygotic_highconf"] = False
                picked["is_all_upregulated"] = False
                picked["is_slam_confirmed_minor_wave"] = False
                df = pd.concat([df, picked], ignore_index=True, sort=False)

            for flag_col in [
                "is_control_silent", "is_control_non_activated",
                "is_strictly_zygotic_minor_wave", "is_strictly_zygotic_highconf",
                "is_all_upregulated", "is_slam_confirmed_minor_wave",
            ]:
                df[flag_col] = df[flag_col].fillna(False).astype(bool)

            print(
                f"[+] Added {len(picked):,} GC-matched silent controls "
                f"(mean promoter GC: pos={pos_gc.mean():.3f} vs silent={_prom_gc(picked).mean():.3f})"
            )
        else:
            print("[!] Warning: Could not construct 'control_silent' (GTF or full RNA-seq table unavailable).")

    print(
        f"[*] Target Gene Set Counts: "
        f"strictly_zygotic_highconf={df['is_strictly_zygotic_highconf'].sum()} | "
        f"strictly_zygotic={df['is_strictly_zygotic_minor_wave'].sum()} | "
        f"slam_confirmed={df['is_slam_confirmed_minor_wave'].sum()} | "
        f"all_upreg={df['is_all_upregulated'].sum()} | "
        f"control_silent={df['is_control_silent'].sum()} | "
        f"control_non_activated={df['is_control_non_activated'].sum()}"
    )
    return df


# ==============================================================================
# Grafted Model Interface Auto-Detection & Multi-Window Scoring
# ==============================================================================
def _extract_three_heads(out: Any) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[List[np.ndarray]]]:
    """Parse grafted model forward output into either named dict or list of 3 1D numpy arrays."""
    def _to_1d(t):
        arr = t.detach().float().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[1] == 1:
                arr = arr[:, 0]
        return arr

    if isinstance(out, dict):
        mapped = {}
        for k, v in out.items():
            kl = k.lower()
            arr = _to_1d(v)
            if "abs" in kl:
                mapped["y_abs"] = arr
            elif "rel" in kl:
                mapped["y_rel"] = arr
            elif "late" in kl or "base" in kl or "count" in kl:
                mapped["late"] = arr
        if len(mapped) == 3:
            return mapped, None
        return None, [_to_1d(v) for v in list(out.values())[:3]]

    if isinstance(out, (tuple, list)):
        arrs = [_to_1d(v) for v in out]
        flat = []
        for a in arrs:
            if a.ndim == 1:
                flat.append(a)
            elif a.ndim == 2 and a.shape[1] <= 3:
                for j in range(a.shape[1]):
                    flat.append(a[:, j])
        if len(flat) >= 3:
            return None, flat[:3]
        return None, None

    if torch.is_tensor(out):
        arr = out.detach().float().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return None, [arr[:, 0], arr[:, 1], arr[:, 2]]
    return None, None


def detect_grafted_interface(
    grafted_model: torch.nn.Module,
    genome: GenomeAccessor,
    genome_fa_path: str,
    sample_gene_df: pd.DataFrame,
    window_bp: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    """
    Auto-detect input layout ((B,4,L) vs (B,L,4)), output head order, and
    forward/RC averaging by cross-checking 1 TSS window against score_tss_regions.
    """
    # Pick a single valid gene away from contig edge
    test_row = None
    for _, row in sample_gene_df.iterrows():
        rc = genome.resolve_chrom(row["chrom"])
        if rc is None:
            continue
        tss = int(row["tss"])
        if tss > 5000 and tss + 5000 < genome.chrom_lengths[rc]:
            test_row = row
            break
    if test_row is None:
        return None

    half = window_bp // 2
    seq = genome.fetch(test_row["chrom"], int(test_row["tss"]) - half, int(test_row["tss"]) - half + window_bp)
    x_4xl = one_hot_encode_batch([seq])
    x_rc_4xl = reverse_complement_onehot(x_4xl)

    grafted_model.eval()
    layout = None
    raw_fwd, raw_rc = None, None
    for cand_layout in ("NCL", "NLC"):
        try:
            tf = torch.from_numpy(x_4xl if cand_layout == "NCL" else np.transpose(x_4xl, (0, 2, 1))).to(device)
            tr = torch.from_numpy(x_rc_4xl if cand_layout == "NCL" else np.transpose(x_rc_4xl, (0, 2, 1))).to(device)
            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        of = grafted_model(tf)
                        or_ = grafted_model(tr)
                else:
                    of = grafted_model(tf)
                    or_ = grafted_model(tr)
            d_f, l_f = _extract_three_heads(of)
            d_r, l_r = _extract_three_heads(or_)
            if d_f is not None or l_f is not None:
                layout = cand_layout
                raw_fwd = (d_f, l_f)
                raw_rc = (d_r, l_r)
                break
        except Exception:
            continue

    if layout is None:
        return None

    # Default mapping if dict keys were explicit
    if raw_fwd[0] is not None:
        return {"layout": layout, "mode": "dict", "rc_avg": True}

    # Tuple/tensor output: match order against score_tss_regions on the same gene
    ref_vals = None
    try:
        one_df = pd.DataFrame([test_row])
        ref_df = score_tss_regions(
            grafted_model=grafted_model,
            genome_fa_path=genome_fa_path,
            gene_df=one_df,
            flank_bp=0,
            window_bp=window_bp,
            stride_bp=250,
            batch_size=16,
            device=device,
        )
        if np.isfinite(ref_df["pred_max_y_abs"].iloc[0]):
            ref_vals = {
                "y_abs": float(ref_df["pred_max_y_abs"].iloc[0]),
                "y_rel": float(ref_df["pred_max_y_rel"].iloc[0]),
                "late": float(ref_df["pred_max_late"].iloc[0]),
            }
    except Exception:
        ref_vals = None

    fwd_vec = [float(a[0]) for a in raw_fwd[1]]
    avg_vec = [0.5 * (float(a[0]) + float(b[0])) for a, b in zip(raw_fwd[1], raw_rc[1])]

    if ref_vals is not None:
        import itertools
        best_err = np.inf
        best_cfg = None
        for rc_flag, vec in [(True, avg_vec), (False, fwd_vec)]:
            for perm in itertools.permutations([0, 1, 2]):
                err = (
                    abs(vec[perm[0]] - ref_vals["y_abs"])
                    + abs(vec[perm[1]] - ref_vals["y_rel"])
                    + abs(vec[perm[2]] - ref_vals["late"])
                )
                if err < best_err:
                    best_err = err
                    best_cfg = {"layout": layout, "mode": "index", "indices": perm, "rc_avg": rc_flag}
        if best_cfg is not None and best_err < 0.15:
            return best_cfg

    # Standard fallback order (y_abs, y_rel, late)
    return {"layout": layout, "mode": "index", "indices": (0, 1, 2), "rc_avg": True}


def predict_sequences_batched(
    grafted_model: torch.nn.Module,
    seqs: List[str],
    iface: Dict[str, Any],
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched forward (and optional RC-averaged) inference returning (y_abs, y_rel, late)."""
    n = len(seqs)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    out_abs = np.empty(n, dtype=np.float32)
    out_rel = np.empty(n, dtype=np.float32)
    out_late = np.empty(n, dtype=np.float32)
    grafted_model.eval()

    def _run_x(x_4xl: np.ndarray):
        x_in = x_4xl if iface["layout"] == "NCL" else np.transpose(x_4xl, (0, 2, 1))
        xt = torch.from_numpy(x_in).to(device, non_blocking=True)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o = grafted_model(xt)
            else:
                o = grafted_model(xt)
        d, l = _extract_three_heads(o)
        if iface["mode"] == "dict" and d is not None:
            return d["y_abs"], d["y_rel"], d["late"]
        idx = iface["indices"]
        return l[idx[0]], l[idx[1]], l[idx[2]]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = one_hot_encode_batch(seqs[start:end])
        a1, r1, l1 = _run_x(x)
        if iface.get("rc_avg", True):
            a2, r2, l2 = _run_x(reverse_complement_onehot(x))
            out_abs[start:end] = 0.5 * (a1 + a2)
            out_rel[start:end] = 0.5 * (r1 + r2)
            out_late[start:end] = 0.5 * (l1 + l2)
        else:
            out_abs[start:end] = a1
            out_rel[start:end] = r1
            out_late[start:end] = l1

    return out_abs, out_rel, out_late


def build_neighbor_tss_index(df_all_genes: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Map normalized chrom -> sorted unique TSS positions for intergenic bounding."""
    idx: Dict[str, List[int]] = {}
    for c, t in zip(df_all_genes["chrom"], df_all_genes["tss"]):
        c_s = str(c)
        idx.setdefault(c_s, []).append(int(t))
    return {c: np.unique(np.asarray(ts, dtype=np.int64)) for c, ts in idx.items()}


def get_window_centers(
    chrom: str,
    tss: int,
    strand: str,
    window_name: str,
    neighbor_idx: Dict[str, np.ndarray],
    stride_bp: int = 250,
) -> List[int]:
    """Return 0-based center coordinates to score for a gene under the requested window spec."""
    if window_name == "promoter_1kb":
        # Strand-aware -1000..+500 bp around canonical TSS (centers tightly covering core promoter)
        if str(strand) == "-":
            lo, hi = tss - 250, tss + 500
        else:
            lo, hi = tss - 500, tss + 250
        centers = list(range(lo, hi + 1, stride_bp))
        if tss not in centers:
            centers.append(tss)
        return sorted(set(centers))

    if window_name == "proximal_5kb":
        centers = list(range(tss - 5000, tss + 5000 + 1, stride_bp))
        if tss not in centers:
            centers.append(tss)
        return sorted(set(centers))

    if window_name == "bounded_25kb":
        lo, hi = tss - 25000, tss + 25000
        arr = neighbor_idx.get(str(chrom))
        if arr is not None and len(arr) > 1:
            pos = int(np.searchsorted(arr, tss))
            # Nearest neighbor strictly left of tss
            l_idx = pos - 1
            while l_idx >= 0 and arr[l_idx] >= tss:
                l_idx -= 1
            if l_idx >= 0:
                mid_l = int((tss + int(arr[l_idx])) // 2)
                lo = max(lo, mid_l)
            # Nearest neighbor strictly right of tss
            r_idx = pos
            while r_idx < len(arr) and arr[r_idx] <= tss:
                r_idx += 1
            if r_idx < len(arr):
                mid_r = int((tss + int(arr[r_idx])) // 2)
                hi = min(hi, mid_r)
        if lo > hi:
            return [tss]
        centers = list(range(lo, hi + 1, stride_bp))
        if tss not in centers:
            centers.append(tss)
        return sorted(set(centers))

    raise ValueError(f"Unsupported window name: {window_name}")


def score_multi_windows(
    grafted_model: torch.nn.Module,
    genome: GenomeAccessor,
    genome_fa_path: str,
    gene_df: pd.DataFrame,
    neighbor_idx: Dict[str, np.ndarray],
    windows: List[str],
    window_bp: int,
    stride_bp: int,
    batch_size: int,
    device: str,
    work_dir: Optional[Path] = None,
    task_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score all genes across every requested window using a single pass over unique
    (chrom, center) coordinates for efficiency.
    """
    iface = detect_grafted_interface(
        grafted_model, genome, genome_fa_path, gene_df, window_bp=window_bp, device=device
    )

    out_df = gene_df.copy().reset_index(drop=True)

    # Graceful fallback to score_tss_regions if model interface cannot be probed
    if iface is None:
        print("[!] Direct model interface probe failed; falling back to score_tss_regions per window.")
        fallback_flanks = {"promoter_1kb": 1200, "proximal_5kb": 5000, "bounded_25kb": 25000}
        for w in windows:
            df_w = score_tss_regions(
                grafted_model=grafted_model,
                genome_fa_path=genome_fa_path,
                gene_df=out_df,
                flank_bp=fallback_flanks.get(w, 5000),
                window_bp=window_bp,
                stride_bp=stride_bp,
                batch_size=batch_size,
                device=device,
                work_dir=work_dir,
                task_name=task_name,
            )
            out_df[f"pred_{w}_y_abs"] = df_w["pred_max_y_abs"].values
            out_df[f"pred_{w}_y_rel"] = df_w["pred_max_y_rel"].values
            out_df[f"pred_{w}_late"] = df_w["pred_max_late"].values
        return out_df

    # Collect unique (chrom, center) windows across all requested window specs
    coord_to_idx: Dict[Tuple[str, int], int] = {}
    unique_coords: List[Tuple[str, int]] = []
    gene_window_indices: Dict[str, List[List[int]]] = {w: [] for w in windows}

    for _, row in out_df.iterrows():
        chrom = str(row["chrom"])
        tss = int(row["tss"])
        strand = str(row.get("strand", "+"))
        for w in windows:
            centers = get_window_centers(chrom, tss, strand, w, neighbor_idx, stride_bp=stride_bp)
            idxs = []
            for c in centers:
                key = (chrom, c)
                if key not in coord_to_idx:
                    coord_to_idx[key] = len(unique_coords)
                    unique_coords.append(key)
                idxs.append(coord_to_idx[key])
            gene_window_indices[w].append(idxs)

    n_unique = len(unique_coords)
    print(f"    [*] Scoring {n_unique:,} unique {window_bp} bp windows across {len(out_df):,} genes ({', '.join(windows)})...")

    half = window_bp // 2
    all_abs = np.empty(n_unique, dtype=np.float32)
    all_rel = np.empty(n_unique, dtype=np.float32)
    all_late = np.empty(n_unique, dtype=np.float32)

    chunk_size = max(batch_size * 16, 4096)
    for start in range(0, n_unique, chunk_size):
        end = min(n_unique, start + chunk_size)
        seqs = [genome.fetch(ch, ctr - half, ctr - half + window_bp) for ch, ctr in unique_coords[start:end]]
        a, r, l = predict_sequences_batched(grafted_model, seqs, iface, batch_size=batch_size, device=device)
        all_abs[start:end] = a
        all_rel[start:end] = r
        all_late[start:end] = l
        if work_dir and task_name and (end == n_unique or (start // chunk_size) % 5 == 0):
            pct = 100.0 * end / max(1, n_unique)
            set_task_progress(work_dir, task_name, f"Scoring multi-window TSS tiles ({pct:.1f}%)")

    # Aggregate max per gene per window
    for w in windows:
        w_abs = np.full(len(out_df), np.nan, dtype=np.float32)
        w_rel = np.full(len(out_df), np.nan, dtype=np.float32)
        w_late = np.full(len(out_df), np.nan, dtype=np.float32)
        for g_i, idxs in enumerate(gene_window_indices[w]):
            if idxs:
                idx_arr = np.asarray(idxs, dtype=np.int64)
                w_abs[g_i] = float(np.nanmax(all_abs[idx_arr]))
                w_rel[g_i] = float(np.nanmax(all_rel[idx_arr]))
                w_late[g_i] = float(np.nanmax(all_late[idx_arr]))
        out_df[f"pred_{w}_y_abs"] = w_abs
        out_df[f"pred_{w}_y_rel"] = w_rel
        out_df[f"pred_{w}_late"] = w_late

    return out_df


# ==============================================================================
# Metrics Evaluation & Zebrafish In-Species Ceiling
# ==============================================================================
def compute_binary_metrics(
    df: pd.DataFrame, pos_mask: pd.Series, ctrl_mask: pd.Series, score_col: str
) -> Dict[str, Any]:
    if score_col not in df.columns:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "p_value": float("nan"), "n_pos": 0, "n_ctrl": 0}
    pos_s = df.loc[pos_mask, score_col].dropna().values
    ctrl_s = df.loc[ctrl_mask, score_col].dropna().values
    if len(pos_s) == 0 or len(ctrl_s) == 0:
        return {
            "roc_auc": float("nan"), "pr_auc": float("nan"), "p_value": float("nan"),
            "n_pos": int(len(pos_s)), "n_ctrl": int(len(ctrl_s)),
        }
    y_true = np.concatenate([np.ones(len(pos_s)), np.zeros(len(ctrl_s))])
    y_score = np.concatenate([pos_s, ctrl_s])
    _, p_val = mannwhitneyu(pos_s, ctrl_s, alternative="greater")
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "p_value": float(p_val),
        "n_pos": int(len(pos_s)),
        "n_ctrl": int(len(ctrl_s)),
    }


def run_zebrafish_ceiling_benchmark(
    grafted_zf_models: List[torch.nn.Module],
    fold_indices: List[int],
    study_cfg: Dict[str, Any],
    work_dir: Path,
    windows: List[str],
    batch_size: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate Zebrafish-trained grafted models on empirical Zebrafish minor-wave vs. silent genes
    strictly on held-out test chromosomes (leak-free in-species ATAC -> RNA ceiling benchmark).
    """
    zf_fa = study_cfg.get("ZEBRAFISH_GENOME_FA")
    zf_gtf = study_cfg.get("ZEBRAFISH_GTF")
    if not zf_fa or not Path(zf_fa).exists():
        cands = sorted((work_dir / "downloads").glob("*Danio*fa")) + sorted((work_dir / "downloads").glob("*danRer*fa"))
        cands = [p for p in cands if not p.name.endswith(".fai")]
        zf_fa = str(cands[0]) if cands else None
    if not zf_gtf or not Path(zf_gtf).exists():
        cands = sorted((work_dir / "downloads").glob("*Danio*gtf*")) + sorted((work_dir / "downloads").glob("*danRer*gtf*"))
        zf_gtf = str(cands[0]) if cands else None

    if not zf_fa or not zf_gtf:
        print("[!] Skipping Zebrafish ceiling: Zebrafish FASTA or GTF not found.", flush=True)
        return None

    # Load empirical SLAM-seq positives strictly from prep_slamseq.py outputs
    slam_dir = Path(work_dir) / "slamseq"
    pos_path = slam_dir / "zebrafish_slamseq_minor_wave_genes.tsv"
    silent_path = slam_dir / "zebrafish_slamseq_silent_genes.tsv"
    summary_path = slam_dir / "zebrafish_slamseq_gene_summary.tsv"

    df_pos_raw = None
    if pos_path.exists() and pos_path.stat().st_size > 0:
        df_pos_raw = pd.read_csv(pos_path, sep="\t")
    elif summary_path.exists() and summary_path.stat().st_size > 0:
        df_summ = pd.read_csv(summary_path, sep="\t")
        if "is_slam_minor_wave" in df_summ.columns:
            df_pos_raw = df_summ[df_summ["is_slam_minor_wave"].fillna(False).astype(bool)].copy()

    if df_pos_raw is None or len(df_pos_raw) < 10:
        n_p = len(df_pos_raw) if df_pos_raw is not None else 0
        print(
            f"[!] Skipping Zebrafish ceiling: Empirical SLAM-seq minor-wave table missing or has < 10 genes (found {n_p}). "
            f"Please run 'scripts/prep_slamseq.py' first. Never falling back to hardcoded gene lists.",
            flush=True,
        )
        return None

    # Load empirical silent negative pool strictly from zebrafish_slamseq_silent_genes.tsv
    df_neg_raw = None
    if silent_path.exists() and silent_path.stat().st_size > 0:
        df_neg_raw = pd.read_csv(silent_path, sep="\t")
    elif summary_path.exists() and summary_path.stat().st_size > 0:
        df_summ = pd.read_csv(summary_path, sep="\t")
        if "is_slam_silent" in df_summ.columns:
            df_neg_raw = df_summ[df_summ["is_slam_silent"].fillna(False).astype(bool)].copy()

    if df_neg_raw is None or len(df_neg_raw) < 10:
        n_n = len(df_neg_raw) if df_neg_raw is not None else 0
        print(
            f"[!] Skipping Zebrafish ceiling: Empirical SLAM-seq silent gene pool missing or has < 10 genes (found {n_n}). "
            f"Please run 'scripts/prep_slamseq.py'.",
            flush=True,
        )
        return None

    # Load leak-free test split chromosomes strictly from chr_folds.json / chr_fold.json
    chr_fold_file = work_dir / "atac" / "Danio_rerio" / "chr_folds.json"
    if not chr_fold_file.exists():
        chr_fold_file = work_dir / "atac" / "Danio_rerio" / "chr_fold.json"

    test_chroms = set()
    if chr_fold_file.exists():
        try:
            with open(chr_fold_file, "r") as ff:
                fold_info = json.load(ff)
            if len(fold_indices) == 1:
                f_key = f"fold_{fold_indices[0]}"
                if f_key in fold_info:
                    test_chroms = set(str(c) for c in fold_info[f_key].get("test", []))
                elif "test" in fold_info:
                    test_chroms = set(str(c) for c in fold_info.get("test", []))
            else:
                for f_idx in fold_indices:
                    f_key = f"fold_{f_idx}"
                    if f_key in fold_info:
                        test_chroms.update(str(c) for c in fold_info[f_key].get("test", []))
                    elif "test" in fold_info:
                        test_chroms.update(str(c) for c in fold_info.get("test", []))
        except Exception as e_f:
            print(f"[-] Notice loading chr_fold info: {e_f}", flush=True)

    if not test_chroms:
        test_chroms = {"8", "17", "18", "11", "7", "19"}

    # Filter evaluation strictly to held-out test chromosomes for zero data leakage
    df_pos = df_pos_raw[df_pos_raw["chrom"].astype(str).isin(test_chroms)].copy()
    df_neg_pool = df_neg_raw[df_neg_raw["chrom"].astype(str).isin(test_chroms)].copy()

    if len(df_pos) < 10:
        print(
            f"[!] Skipping Zebrafish ceiling: Only {len(df_pos)} positive minor-wave genes on held-out test chromosomes "
            f"{sorted(list(test_chroms))} (minimum 10 required for leak-free evaluation).",
            flush=True,
        )
        return None

    zf_genome = GenomeAccessor(zf_fa)
    df_pos = df_pos[df_pos["chrom"].apply(lambda c: zf_genome.resolve_chrom(c) is not None)].copy()
    df_neg_pool = df_neg_pool[df_neg_pool["chrom"].apply(lambda c: zf_genome.resolve_chrom(c) is not None)].copy()

    if len(df_neg_pool) > 8000:
        df_neg_pool = df_neg_pool.sample(n=8000, random_state=42)

    def _prom_gc(sub_df: pd.DataFrame) -> np.ndarray:
        vals = []
        for c, t, st in zip(sub_df["chrom"], sub_df["tss"], sub_df["strand"]):
            ti = int(t)
            s, e = (ti - 1000, ti + 500) if str(st) == "+" else (ti - 500, ti + 1000)
            vals.append(seq_gc_fraction(zf_genome.fetch(c, s, e)))
        return np.asarray(vals, dtype=np.float32)

    pos_gc = _prom_gc(df_pos)
    neg_gc = _prom_gc(df_neg_pool)
    df_neg = gc_match_controls(pos_gc, df_neg_pool, neg_gc, n_target=min(600, 5 * len(df_pos)), seed=42)

    df_pos["is_zf_pos"] = True
    df_neg["is_zf_pos"] = False
    df_zf_eval = pd.concat([df_pos, df_neg], ignore_index=True)

    df_zf_gtf = parse_gtf_genes(zf_gtf)
    neighbor_idx = build_neighbor_tss_index(
        df_zf_gtf if len(df_zf_gtf) > 0 else df_zf_eval[["chrom", "tss"]]
    )

    source_tag = "empirical_slamseq_test_chroms"
    print(
        f"[*] Running Zebrafish In-Species Ceiling ({source_tag}): "
        f"{len(df_pos)} empirical positives vs. {len(df_neg)} GC-matched silent controls on held-out test chroms {sorted(list(test_chroms))}...",
        flush=True,
    )

    fold_dfs = []
    for m_fold in grafted_zf_models:
        fold_dfs.append(
            score_multi_windows(
                grafted_model=m_fold,
                genome=zf_genome,
                genome_fa_path=zf_fa,
                gene_df=df_zf_eval,
                neighbor_idx=neighbor_idx,
                windows=windows,
                window_bp=2114,
                stride_bp=250,
                batch_size=batch_size,
                device=device,
            )
        )

    df_zf_scored = fold_dfs[0].copy()
    for w in windows:
        for s_type in ("y_abs", "y_rel", "late"):
            col = f"pred_{w}_{s_type}"
            df_zf_scored[col] = np.mean([f[col].values for f in fold_dfs], axis=0)

    ceiling_res: Dict[str, Any] = {
        "_source": source_tag,
        "_test_chroms": sorted(list(test_chroms)),
        "_n_pos": len(df_pos),
        "_n_neg": len(df_neg)
    }
    pos_m = df_zf_scored["is_zf_pos"] == True
    neg_m = df_zf_scored["is_zf_pos"] == False
    for w in windows:
        ceiling_res[w] = {}
        for s_type in ("y_abs", "y_rel", "late"):
            ceiling_res[w][s_type] = compute_binary_metrics(df_zf_scored, pos_m, neg_m, f"pred_{w}_{s_type}")
        print(
            f"    [ZF Ceiling | {w:<12}] "
            f"y_abs AUC={ceiling_res[w]['y_abs']['roc_auc']:.3f} | "
            f"y_rel AUC={ceiling_res[w]['y_rel']['roc_auc']:.3f} | "
            f"late AUC={ceiling_res[w]['late']['roc_auc']:.3f}",
            flush=True,
        )
    return ceiling_res


# ==============================================================================
# Main Worker
# ==============================================================================
def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    task_name = args.task_name

    set_task_progress(work_dir, task_name, "Initializing validation...")
    print(f"[*] Starting Zero-Shot Validation in {work_dir}")

    # Load configuration
    cfg_path = Path(args.config).resolve() if args.config else (work_dir / "config.json")
    study_cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            study_cfg = json.load(f)

    target_locus_name = study_cfg.get("TARGET_LOCUS_NAME", "notch2_slc35a3a")
    target_genome_fa = study_cfg.get("TARGET_GENOME_FA")
    target_gtf = study_cfg.get("TARGET_GTF")
    medaka_genome_fa = study_cfg.get("MEDAKA_GENOME_FA")
    ensemble_medaka = bool(study_cfg.get("ENSEMBLE_MEDAKA_WITH_ZF", False))
    enable_medaka = args.enable_medaka if args.enable_medaka is not None else bool(study_cfg.get("ENABLE_MEDAKA", True))

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    if not windows:
        windows = ["promoter_1kb", "proximal_5kb", "bounded_25kb"]

    res_dir = work_dir / "results" / target_locus_name
    res_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target genome FASTA
    if not target_genome_fa or not Path(target_genome_fa).exists():
        cand_fa = list((work_dir / "downloads").glob("*.dna_sm.toplevel.fa")) + list((work_dir / "downloads").glob("*.fa"))
        cand_fa = [p for p in cand_fa if "Danio" not in p.name and "Oryzias" not in p.name and not p.name.endswith(".fai")]
        if cand_fa:
            target_genome_fa = str(cand_fa[0])
        else:
            raise FileNotFoundError(f"[FATAL] Target genome FASTA not found in config or downloads: {target_genome_fa}")

    # Resolve target gene sets table
    gene_sets_tsv = res_dir / f"{target_locus_name}_gene_sets.tsv"
    if not gene_sets_tsv.exists():
        gene_sets_tsv = work_dir / "rnaseq" / "target_species_gene_sets.tsv"
    if not gene_sets_tsv.exists():
        cands = list(work_dir.glob("results*/*gene_sets.tsv")) + list(work_dir.glob("**/*gene_sets.tsv"))
        if cands:
            gene_sets_tsv = cands[0]
        else:
            raise FileNotFoundError(f"[FATAL] Target gene sets table not found at {gene_sets_tsv}")

    df_gene_sets = pd.read_csv(gene_sets_tsv, sep="\t")
    print(f"[+] Loaded {len(df_gene_sets)} initial gene definitions from {gene_sets_tsv}")

    # Parse full target GTF (for silent control extraction + exact neighbor TSS bounding)
    target_genome = GenomeAccessor(target_genome_fa)
    df_target_gtf = parse_gtf_genes(target_gtf)
    if len(df_target_gtf) > 0:
        print(f"[+] Parsed {len(df_target_gtf):,} genes from target GTF for silent controls & neighbor bounding")

    df_gene_sets = enrich_target_gene_sets(
        df_gene_sets=df_gene_sets,
        df_gtf=df_target_gtf,
        genome=target_genome,
        work_dir=work_dir,
        study_cfg=study_cfg,
        silent_max_tpm=args.silent_max_tpm,
        max_silent_controls=args.max_silent_controls,
        strict_early_max_tpm=args.strict_early_max_tpm,
        strict_premzt_min_tpm=args.strict_premzt_min_tpm,
        case_insensitive_orthologs=args.case_insensitive_orthologs,
    )

    # Persist updated flags and categories back to disk if new controls were added
    try:
        df_gene_sets.to_csv(gene_sets_tsv, sep="\t", index=False)
        alt_tsv = work_dir / "rnaseq" / "target_species_gene_sets.tsv"
        if alt_tsv.exists():
            df_gene_sets.to_csv(alt_tsv, sep="\t", index=False)
    except Exception as e_save:
        print(f"[-] Notice: could not persist enriched gene sets table: {e_save}")

    # Build genome-wide neighbor TSS index (prefer full GTF, merged with df_gene_sets)
    tss_source = (
        pd.concat([df_target_gtf[["chrom", "tss"]], df_gene_sets[["chrom", "tss"]]], ignore_index=True)
        if len(df_target_gtf) > 0
        else df_gene_sets[["chrom", "tss"]]
    )
    neighbor_idx = build_neighbor_tss_index(tss_source)

    # Parse folds
    fold_indices = [int(f.strip()) for f in str(args.folds).split(",") if f.strip().isdigit()]
    if not fold_indices:
        fold_indices = [0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Device: {device} | Autocast: bf16 | Batch size: {args.batch_size} | Folds: {fold_indices} | Windows: {windows}")

    # Load Zebrafish grafted models
    grafted_zf_models = []
    for f_idx in fold_indices:
        zf_ckpt = work_dir / "models" / "chrombpnet" / "Danio_rerio" / f"fold_{f_idx}" / "grafted_chrombpnet.pt"
        if zf_ckpt.exists():
            grafted_zf_models.append(load_grafted_model(zf_ckpt, device=device))
    if not grafted_zf_models:
        raise FileNotFoundError(f"[FATAL] No Zebrafish grafted ChromBPNet checkpoints found in folds {fold_indices}")
    print(f"[+] Loaded {len(grafted_zf_models)} grafted Zebrafish model(s)")

    # Score target genes across all windows with Zebrafish models
    set_task_progress(work_dir, task_name, "Scoring target genes with Zebrafish grafted model...")
    scored_zf_folds = []
    for g_idx, m_fold in enumerate(grafted_zf_models):
        print(f"[*] Scoring with Zebrafish model fold {fold_indices[g_idx]}...")
        df_f = score_multi_windows(
            grafted_model=m_fold,
            genome=target_genome,
            genome_fa_path=str(target_genome_fa),
            gene_df=df_gene_sets,
            neighbor_idx=neighbor_idx,
            windows=windows,
            window_bp=2114,
            stride_bp=250,
            batch_size=args.batch_size,
            device=device,
            work_dir=work_dir,
            task_name=task_name,
        )
        scored_zf_folds.append(df_f)

    df_scored = scored_zf_folds[0].copy()
    for w in windows:
        for s_type in ("y_abs", "y_rel", "late"):
            col = f"pred_{w}_{s_type}"
            df_scored[col] = np.mean([f[col].values for f in scored_zf_folds], axis=0)
            df_scored[f"{col}_zf"] = df_scored[col]

    # Optional Medaka evaluation
    grafted_medaka_models = []
    if enable_medaka:
        for f_idx in fold_indices:
            med_ckpt = work_dir / "models" / "chrombpnet" / "Oryzias_latipes" / f"fold_{f_idx}" / "grafted_chrombpnet.pt"
            if med_ckpt.exists():
                grafted_medaka_models.append(load_grafted_model(med_ckpt, device=device))

    if grafted_medaka_models:
        print(f"[+] Loaded {len(grafted_medaka_models)} grafted Medaka model(s)")
        set_task_progress(work_dir, task_name, "Scoring target genes with Medaka grafted model...")
        med_folds_scored = []
        for m_idx, m_fold in enumerate(grafted_medaka_models):
            print(f"[*] Scoring with Medaka model fold {fold_indices[m_idx]}...")
            df_mf = score_multi_windows(
                grafted_model=m_fold,
                genome=target_genome,
                genome_fa_path=str(target_genome_fa),
                gene_df=df_gene_sets,
                neighbor_idx=neighbor_idx,
                windows=windows,
                window_bp=2114,
                stride_bp=250,
                batch_size=args.batch_size,
                device=device,
                work_dir=work_dir,
                task_name=task_name,
            )
            med_folds_scored.append(df_mf)

        for w in windows:
            for s_type in ("y_abs", "y_rel", "late"):
                col = f"pred_{w}_{s_type}"
                med_vals = np.mean([f[col].values for f in med_folds_scored], axis=0)
                df_scored[f"{col}_medaka"] = med_vals
                df_scored[f"{col}_consensus"] = 0.5 * (df_scored[f"{col}_zf"] + med_vals)
                if ensemble_medaka:
                    df_scored[col] = df_scored[f"{col}_consensus"]
                else:
                    df_scored[col] = df_scored[f"{col}_zf"]

    # Populate legacy pred_max_* columns from primary window (promoter_1kb if present, else windows[0])
    primary_w = "promoter_1kb" if "promoter_1kb" in windows else windows[0]
    for s_type in ("y_abs", "y_rel", "late"):
        df_scored[f"pred_max_{s_type}"] = df_scored[f"pred_{primary_w}_{s_type}"]
        df_scored[f"pred_max_{s_type}_zf"] = df_scored[f"pred_{primary_w}_{s_type}_zf"]
        if grafted_medaka_models:
            df_scored[f"pred_max_{s_type}_medaka"] = df_scored[f"pred_{primary_w}_{s_type}_medaka"]
            df_scored[f"pred_max_{s_type}_consensus"] = df_scored[f"pred_{primary_w}_{s_type}_consensus"]

    # Compute validation metrics across windows, score types, and primary vs. diagnostic controls
    set_task_progress(work_dir, task_name, "Computing transfer benchmark metrics...")

    pos_masks = {
        "strictly_zygotic_highconf": df_scored["is_strictly_zygotic_highconf"] == True,
        "strictly_zygotic": df_scored["is_strictly_zygotic_minor_wave"] == True,
        "slam_confirmed": df_scored["is_slam_confirmed_minor_wave"] == True,
        "all_upregulated": df_scored["is_all_upregulated"] == True,
    }
    ctrl_masks = {
        "control_silent": df_scored["is_control_silent"] == True,
        "control_non_activated": df_scored["is_control_non_activated"] == True,
    }
    active_primary_ctrl = (
        args.primary_control
        if args.primary_control in ctrl_masks and ctrl_masks[args.primary_control].sum() > 0
        else args.diagnostic_control
    )

    eval_models = [("Zebrafish_only", "_zf")]
    if grafted_medaka_models:
        eval_models.append(("Medaka_only", "_medaka"))
        eval_models.append(("Consensus", "_consensus"))

    metrics_by_window: Dict[str, Any] = {}
    metrics_report: Dict[str, Any] = {}  # backward-compatible top-level summary on primary_w vs active_primary_ctrl

    print("\n" + "=" * 80)
    print(f"[*] Zero-Shot Cross-Species Transfer Report (Primary Control: {active_primary_ctrl})")
    for m_label, suffix in eval_models:
        metrics_by_window[m_label] = {}
        print(f"  === Model: {m_label} ===")
        for w in windows:
            metrics_by_window[m_label][w] = {}
            for ctrl_name, c_mask in ctrl_masks.items():
                if c_mask.sum() == 0:
                    continue
                metrics_by_window[m_label][w][ctrl_name] = {}
                for s_type in ("y_abs", "y_rel", "late"):
                    col = f"pred_{w}_{s_type}{suffix}"
                    res_s = {
                        p_name: compute_binary_metrics(df_scored, p_mask, c_mask, col)
                        for p_name, p_mask in pos_masks.items()
                    }
                    metrics_by_window[m_label][w][ctrl_name][s_type] = res_s
            # Print concise summary for primary control
            pc = metrics_by_window[m_label][w].get(active_primary_ctrl, {})
            if pc:
                print(
                    f"    [{w:<12} | vs {active_primary_ctrl}] "
                    f"SZ-HighConf y_abs={pc['y_abs']['strictly_zygotic_highconf']['roc_auc']:.3f}, "
                    f"y_rel={pc['y_rel']['strictly_zygotic_highconf']['roc_auc']:.3f}, "
                    f"late={pc['late']['strictly_zygotic_highconf']['roc_auc']:.3f} | "
                    f"SZ-All y_abs={pc['y_abs']['strictly_zygotic']['roc_auc']:.3f}"
                )
        metrics_report[m_label] = metrics_by_window[m_label][primary_w].get(active_primary_ctrl, {})
    print("=" * 80)

    # Optional Zebrafish in-species ceiling benchmark
    zebrafish_ceiling = None
    if args.run_zebrafish_ceiling:
        set_task_progress(work_dir, task_name, "Running Zebrafish in-species ceiling benchmark...")
        try:
            zebrafish_ceiling = run_zebrafish_ceiling_benchmark(
                grafted_zf_models=grafted_zf_models,
                fold_indices=fold_indices,
                study_cfg=study_cfg,
                work_dir=work_dir,
                windows=windows,
                batch_size=args.batch_size,
                device=device,
            )
        except Exception as e_zf:
            print(f"[-] Notice evaluating Zebrafish ceiling: {e_zf}")

    # Optional cross-species Medaka ATAC benchmark
    medaka_atac_metrics = None
    medaka_st9_summary = None
    medaka_targets = work_dir / "atac" / "Oryzias_latipes" / "Oryzias_latipes_summit_targets.tsv"
    if medaka_targets.exists() and medaka_genome_fa and Path(medaka_genome_fa).exists():
        try:
            print("[*] Evaluating Zebrafish -> Medaka cross-species ATAC benchmark...")
            med_res = score_cross_species_medaka_atac(
                grafted_model=grafted_zf_models[0],
                medaka_targets_tsv=str(medaka_targets),
                medaka_fa_path=str(medaka_genome_fa),
                device=device,
            )
            medaka_atac_metrics = med_res
            medaka_st9_summary = {
                "spearman_r": med_res.get("medaka_st9_spearman_r", med_res.get("spearman_r", float("nan"))),
                "roc_auc": med_res.get("medaka_st9_auroc", med_res.get("roc_auc", float("nan"))),
            }
            print(
                f"[+] Zero-Shot Medaka st9 Pre-MZT ATAC: "
                f"Spearman r = {medaka_st9_summary['spearman_r']}, AUROC = {medaka_st9_summary['roc_auc']}"
            )
        except Exception as e_med:
            print(f"[-] Notice evaluating Medaka cross-species ATAC: {e_med}")

    final_payload = {
        "target_locus_name": target_locus_name,
        "n_genes_scored": int(len(df_scored)),
        "windows": windows,
        "primary_window": primary_w,
        "primary_control": active_primary_ctrl,
        "diagnostic_control": args.diagnostic_control,
        "ensemble_medaka_with_zf": ensemble_medaka,
        "metrics": metrics_report,
        "metrics_by_window": metrics_by_window,
        "zebrafish_ceiling": zebrafish_ceiling,
        "medaka_st9_atac": medaka_st9_summary,
        "medaka_cross_species_atac": medaka_atac_metrics,
    }

    # Atomically write target_genes_scored_metrics.tsv
    out_tsv = res_dir / "target_genes_scored_metrics.tsv"
    tmp_tsv = out_tsv.with_suffix(".tmp.tsv")
    df_scored.to_csv(tmp_tsv, sep="\t", index=False)
    os.replace(str(tmp_tsv), str(out_tsv))
    print(f"[+] Atomically saved scored target genes metrics table to {out_tsv}")

    # Atomically write zero_shoot_validation_metrics.json
    out_json = res_dir / "zero_shoot_validation_metrics.json"
    tmp_json = out_json.with_suffix(".tmp.json")
    with open(tmp_json, "w") as f_json:
        json.dump(sanitize_for_torch_save(final_payload), f_json, indent=2)
    os.replace(str(tmp_json), str(out_json))
    print(f"[+] Atomically saved validation metrics JSON to {out_json}")

    set_task_progress(work_dir, task_name, "100% (Done)")
    print("[SUCCESS] Zero-Shot Validation completed successfully.")


if __name__ == "__main__":
    main()