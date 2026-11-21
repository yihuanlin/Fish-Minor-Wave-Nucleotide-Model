#!/usr/bin/env python3
"""
scripts/prep_atac.py
====================
Modular worker for ATAC-seq preprocessing, peak calling, leak-free chromosome splitting,
and continuous pre-MZT target generation (y_abs, y_rel) across blastula summits.

Key Capabilities:
- End-to-end FASTQ -> BAM processing (fastp trimming, bowtie2 alignment, filtering chrM/MT,
  deduplication via samtools markdup, and merging stage replicates).
- MACS3 peak calling with auto-detection of paired-end vs single-end layout.
  Parameters: --nomodel --shift -100 --extsize 200 --keep-dup all --call-summits.
  Uses -c control_bam when naked gDNA control is available.
- Dynamic detection of miR-430 tandem repeat locus from GTF (falling back to chr4:27-30.5 Mb on GRCz11).
- Dynamic determination of held_out_chr from TARGET_GENES orthologs in the species GTF.
- Primary chromosomes only in chr_fold.json (length >= 1 Mb, excluding unplaced scaffolds and chrM/MT).
- Guaranteed non-empty test split (at least 1-2 primary chromosomes, strictly containing held_out_chr).
- Support for generating multi-fold splits (--cv-folds).
- Exports both <species>_summit_targets.tsv and chr_fold.json (train, valid, test).
- Flagging and exclusion of housekeeping promoters (--exclude-housekeeping-promoters).
- ABSOLUTE ZERO-MOCK POLICY: strictly fails loudly if required files or tools fail.
"""

import os
import sys
import math
import json
import gzip
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

# Ensure conda env bin directory is at the front of PATH
bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import pandas as pd
import numpy as np

try:
    from scripts.task_manager import set_task_progress
except ImportError:
    try:
        from task_manager import set_task_progress
    except ImportError:
        def set_task_progress(work_dir, task_name, progress_str):
            try:
                prog_file = Path(work_dir) / "tasks" / task_name / "progress.txt"
                prog_file.parent.mkdir(parents=True, exist_ok=True)
                prog_file.write_text(str(progress_str).strip())
            except Exception:
                pass


def run_cmd(cmd: List[str] | str, desc: str = "") -> None:
    """Run a shell command with live stdout/stderr streaming and strict error checking."""
    if desc:
        print(f"[*] {desc}", flush=True)
    cmd_str = cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)
    print(f"    CMD: {cmd_str}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, text=True, stdout=None, stderr=None, env=os.environ)


def is_bam_paired_end(bam_path: Path) -> bool:
    """Check if BAM contains paired-end reads using samtools view."""
    cmd = ["samtools", "view", "-c", "-f", "1", str(bam_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    count = int(res.stdout.strip() or 0)
    return count > 0


def get_bam_total_reads(bam_path: Path) -> int:
    """Return total mapped reads in BAM via samtools idxstats."""
    res = subprocess.run(["samtools", "idxstats", str(bam_path)], capture_output=True, text=True, check=True)
    total = 0
    for line in res.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and not (parts[0].startswith("chrM") or parts[0] == "MT"):
            total += int(parts[2])
    return max(total, 1)


def align_and_process_fastq(
    fastq_r1: Path,
    fastq_r2: Optional[Path],
    genome_fa: Path,
    bowtie2_index_prefix: str,
    output_bam: Path,
    threads: int = 8
) -> Path:
    """Trim with fastp, align with bowtie2, filter MT/chrM, sort, deduplicate, and index."""
    output_bam.parent.mkdir(parents=True, exist_ok=True)
    if output_bam.exists() and output_bam.stat().st_size > 10 * 1024 * 1024:
        print(f"[+] Processed BAM exists & verified: {output_bam}")
        return output_bam

    prefix = output_bam.parent / output_bam.stem
    trimmed_r1 = prefix.with_suffix(".trimmed_R1.fq.gz")
    trimmed_r2 = prefix.with_suffix(".trimmed_R2.fq.gz") if fastq_r2 else None

    # 1. fastp trimming
    if not trimmed_r1.exists():
        fastp_cmd = ["fastp", "-i", str(fastq_r1), "-o", str(trimmed_r1), "-w", str(min(threads, 4))]
        if fastq_r2:
            fastp_cmd.extend(["-I", str(fastq_r2), "-O", str(trimmed_r2)])
        run_cmd(fastp_cmd, desc=f"Running fastp quality & adapter trimming on {fastq_r1.name}")

    # 2. Bowtie2 index check
    idx_1 = Path(f"{bowtie2_index_prefix}.1.bt2")
    idx_1_l = Path(f"{bowtie2_index_prefix}.1.bt2l")
    if not (idx_1.exists() or idx_1_l.exists()):
        print(f"[*] Building Bowtie2 index for {genome_fa.name}...", flush=True)
        run_cmd(["bowtie2-build", "--threads", str(threads), str(genome_fa), bowtie2_index_prefix], desc="Building Bowtie2 index")

    # 3. Alignment
    raw_sam = prefix.with_suffix(".raw.sam")
    bt2_cmd = [
        "bowtie2", "-x", bowtie2_index_prefix,
        "-p", str(threads),
        "--very-sensitive",
        "-X", "2000"
    ]
    if trimmed_r2:
        bt2_cmd.extend(["-1", str(trimmed_r1), "-2", str(trimmed_r2)])
    else:
        bt2_cmd.extend(["-U", str(trimmed_r1)])
    bt2_cmd.extend(["-S", str(raw_sam)])

    run_cmd(bt2_cmd, desc=f"Bowtie2 alignment for {output_bam.stem}")

    # 4. Filter exact chrM/MT, sort, deduplicate (paired-end requires name-sort -> fixmate -m -> coord-sort -> markdup)
    markdup_bam = prefix.with_suffix(".markdup.bam")
    if trimmed_r2:
        pipe_cmd = (
            f"samtools view -h -q 30 -F 1804 {raw_sam} | "
            f"awk -F'\\t' '/^@/ || ($3 != \"chrM\" && $3 != \"MT\" && $3 != \"M\" && $3 != \"chrMT\")' | "
            f"samtools sort -n -@ {threads} - | "
            f"samtools fixmate -m -@ {threads} - - | "
            f"samtools sort -@ {threads} - | "
            f"samtools markdup -r -@ {threads} - {markdup_bam}"
        )
        run_cmd(pipe_cmd, desc=f"Filtering chrM, fixmate -m and markdup (paired-end) for {output_bam.stem}")
    else:
        sorted_bam = prefix.with_suffix(".sorted.bam")
        sort_cmd = (
            f"samtools view -h -q 30 -F 1804 {raw_sam} | "
            f"awk -F'\\t' '/^@/ || ($3 != \"chrM\" && $3 != \"MT\" && $3 != \"M\" && $3 != \"chrMT\")' | "
            f"samtools sort -@ {threads} -o {sorted_bam}"
        )
        run_cmd(sort_cmd, desc=f"Filtering chrM and coordinate sorting (single-end) for {output_bam.stem}")
        run_cmd(f"samtools markdup -r -@ {threads} {sorted_bam} {markdup_bam}", desc=f"Deduplication via samtools markdup for {output_bam.stem}")
        sorted_bam.unlink(missing_ok=True)

    raw_sam.unlink(missing_ok=True)

    shutil.move(str(markdup_bam), str(output_bam))
    run_cmd(f"samtools index -@ {threads} {output_bam}", desc=f"Indexing {output_bam.name}")

    # Cleanup trimmed FASTQs
    trimmed_r1.unlink(missing_ok=True)
    if trimmed_r2:
        trimmed_r2.unlink(missing_ok=True)

    print(f"[+] Successfully generated and indexed BAM: {output_bam}")
    return output_bam


def merge_bams(bam_list: List[Path], output_bam: Path, threads: int = 8) -> Path:
    """Merge multiple BAM files and index."""
    if len(bam_list) == 1:
        if bam_list[0].resolve() != output_bam.resolve():
            shutil.copy(bam_list[0], output_bam)
            run_cmd(f"samtools index -@ {threads} {output_bam}", desc=f"Indexing {output_bam.name}")
        return output_bam

    if output_bam.exists() and output_bam.stat().st_size > 10 * 1024 * 1024:
        print(f"[+] Merged BAM exists: {output_bam}")
        return output_bam

    cmd = ["samtools", "merge", "-@", str(threads), "-f", str(output_bam)] + [str(b) for b in bam_list]
    run_cmd(cmd, desc=f"Merging {len(bam_list)} BAMs into {output_bam.name}")
    run_cmd(f"samtools index -@ {threads} {output_bam}", desc=f"Indexing {output_bam.name}")
    return output_bam


def detect_mir430_cluster(gtf_path: Path | str) -> Tuple[str, int, int]:
    """Parse GTF to locate miR-430 cluster coordinates dynamically, grouped by chromosome."""
    gtf_path = Path(gtf_path)
    print(f"[*] Dynamically scanning GTF for miR-430 cluster ({gtf_path.name})...")
    mir_by_chrom: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    try:
        open_fn = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "rt")
        with open_fn as f:
            for line in f:
                if "mir-430" in line.lower() or "mir430" in line.lower():
                    p = line.split("\t")
                    if len(p) >= 5:
                        mir_by_chrom[p[0]].append((int(p[3]), int(p[4])))
    except Exception as e:
        print(f"[-] Notice reading GTF for miR-430: {e}")

    if mir_by_chrom:
        # Group mir-430 annotations by chromosome and select the dominant cluster chromosome (e.g. chr4 in GRCz11)
        dom_chrom = max(mir_by_chrom.keys(), key=lambda c: len(mir_by_chrom[c]))
        dom_loci = mir_by_chrom[dom_chrom]
        start = max(0, min(c[0] for c in dom_loci) - 50_000)
        end = max(c[1] for c in dom_loci) + 50_000
        print(f"[+] Detected dominant miR-430 cluster on {dom_chrom}:{start:,}-{end:,} ({len(dom_loci)} loci on {dom_chrom}, {sum(len(v) for v in mir_by_chrom.values())} total)")
        return dom_chrom, start, end

    # Fallback to canonical GRCz11 coordinates
    print("[*] Using canonical Danio rerio GRCz11 miR-430 cluster coordinates: chr4:27000000-30500000")
    return "4", 27000000, 30500000


def detect_held_out_chromosome(gtf_path: Path, target_gene_symbols: List[str]) -> Optional[str]:
    """Detect chromosome containing the target locus (e.g. notch2/slc35a3a) dynamically from GTF."""
    print(f"[*] Dynamically identifying held-out chromosome for target genes {target_gene_symbols} from {gtf_path.name}...")
    target_lower = set(s.lower() for s in target_gene_symbols)
    target_fallback = set()
    for s in target_lower:
        if len(s) > 3 and s[-1] in ('a', 'b'):
            target_fallback.add(s[:-1])

    chrom_counts = defaultdict(int)

    try:
        open_fn = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "rt")
        with open_fn as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split("\t")
                if len(p) >= 9 and p[2] == "gene":
                    attrs = p[8].lower()
                    matched = False
                    for tg in target_lower:
                        if f'gene_name "{tg}"' in attrs or f'gene_id "{tg}"' in attrs or f'"{tg}"' in attrs:
                            chrom_counts[p[0]] += 1
                            matched = True
                            break
                    if not matched:
                        for fb in target_fallback:
                            if f'gene_name "{fb}"' in attrs or f'gene_id "{fb}"' in attrs or f'"{fb}"' in attrs:
                                chrom_counts[p[0]] += 1
                                break
    except Exception as e:
        print(f"[-] Notice reading GTF for target genes: {e}")

    if chrom_counts:
        top_chrom = max(chrom_counts.items(), key=lambda x: x[1])[0]
        print(f"[+] Identified held-out chromosome for target locus: {top_chrom}")
        return top_chrom

    print("[-] Warning: Target genes not found in GTF; held-out chromosome could not be auto-detected.")
    return None


def call_peaks_macs3(
    late_bam: Path,
    output_dir: Path,
    species_name: str,
    control_bam: Optional[Path] = None,
    genome_size: Optional[str] = None
) -> Tuple[Path, Path]:
    """Call peaks on late blastula ATAC BAM with MACS3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{species_name}_late"
    out_narrow = output_dir / f"{prefix}_peaks.narrowPeak"
    out_summits = output_dir / f"{prefix}_summits.bed"

    if out_narrow.exists() and out_summits.exists() and out_narrow.stat().st_size > 1024:
        print(f"[+] MACS3 peaks already exist: {out_narrow}")
        return out_narrow, out_summits

    is_pe = is_bam_paired_end(late_bam)
    format_arg = "BAMPE" if is_pe else "BAM"

    g_size = genome_size
    if not g_size:
        sp_l = species_name.lower()
        if "danio" in sp_l or "rerio" in sp_l:
            g_size = "1.4e9"
        elif "oryzias" in sp_l or "latipes" in sp_l:
            g_size = "7.3e8"
        else:
            g_size = "1.0e9"

    cmd = [
        "macs3", "callpeak",
        "-t", str(late_bam),
        "-f", format_arg,
        "-g", str(g_size),
        "--outdir", str(output_dir),
        "-n", prefix,
        "-q", "0.01",
        "--keep-dup", "all",
        "--call-summits"
    ]
    if control_bam and control_bam.exists():
        cmd.extend(["-c", str(control_bam)])
    if not is_pe:
        cmd.extend(["--nomodel", "--shift", "-100", "--extsize", "200"])

    run_cmd(cmd, desc=f"MACS3 peak calling for {species_name} (Format: {format_arg}, gsize: {g_size})")

    assert out_narrow.exists() and out_narrow.stat().st_size > 1024, f"[FATAL] MACS3 failed to produce peaks at {out_narrow}"
    assert out_summits.exists() and out_summits.stat().st_size > 1024, f"[FATAL] MACS3 failed to produce summits at {out_summits}"
    print(f"[+] MACS3 peak calling complete: {out_narrow.name}, {out_summits.name}")
    return out_narrow, out_summits


def count_reads_in_summits(
    summits_bed: Path,
    early_bam: Path,
    late_bam: Path,
    window_bp: int = 500,
    threads: int = 8
) -> pd.DataFrame:
    """
    Quantify early and late blastula read counts in ±window_bp//2 windows around MACS3 summits.
    Computes continuous pre-MZT targets:
      y_abs = log2(cpm_early + 1.0)
      y_rel = log2((cpm_early + 0.01) / (cpm_late + 0.01))
    """
    half_w = window_bp // 2
    temp_windows_bed = summits_bed.parent / f"{summits_bed.stem}_windows_{window_bp}bp.bed"

    records = []
    with open(summits_bed, "r") as fin, open(temp_windows_bed, "w") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                chrom = parts[0]
                summit = int(parts[1])
                peak_id = parts[3] if len(parts) > 3 else f"{chrom}_{summit}"
                score = parts[4] if len(parts) > 4 else "0"
                start = max(0, summit - half_w)
                end = summit + half_w
                fout.write(f"{chrom}\t{start}\t{end}\t{peak_id}\t{score}\t{summit}\n")
                records.append((chrom, start, end, peak_id, score, summit))

    tot_early = get_bam_total_reads(early_bam)
    tot_late = get_bam_total_reads(late_bam)
    scale_early = 1e6 / max(tot_early, 1)
    scale_late = 1e6 / max(tot_late, 1)

    print(f"[*] Library sizes: Early={tot_early:,} reads, Late={tot_late:,} reads.")
    print(f"[*] Running bedtools multicov across {len(records):,} summit windows...")

    cmd = ["bedtools", "multicov", "-bams", str(early_bam), str(late_bam), "-bed", str(temp_windows_bed)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)

    data = []
    for line in res.stdout.strip().split("\n"):
        if not line:
            continue
        p = line.split("\t")
        chrom, start, end, peak_id, score, summit = p[0], int(p[1]), int(p[2]), p[3], p[4], int(p[5])
        cnt_early = int(p[6])
        cnt_late = int(p[7])

        cpm_early = cnt_early * scale_early
        cpm_late = cnt_late * scale_late

        # Continuous Pre-MZT Target Formulations
        y_abs = math.log2(cpm_early + 1.0)
        y_rel = math.log2((cpm_early + 0.01) / (cpm_late + 0.01))

        data.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "summit": summit,
            "peak_id": peak_id,
            "macs_score": float(score) if score.replace('.', '', 1).isdigit() else 0.0,
            "count_early": cnt_early,
            "count_late": cnt_late,
            "cpm_early": round(cpm_early, 4),
            "cpm_late": round(cpm_late, 4),
            "y_abs": round(y_abs, 4),
            "y_rel": round(y_rel, 4)
        })

    temp_windows_bed.unlink(missing_ok=True)
    df = pd.DataFrame(data)
    print(f"[+] Computed summit targets across {len(df):,} peaks.")
    return df


def parse_housekeeping_promoters(gtf_path: Path, flank_bp: int = 500) -> List[Tuple[str, int, int]]:
    """Extract promoter coordinates (TSS ± flank_bp) for canonical housekeeping genes from GTF."""
    hk_keywords = [
        "gapdh", "actb", "eif4", "rpl", "rps", "uba52", "tub", "hprt", "pgk1", "b2m", "tbp", "sdha", "polr2"
    ]
    regions = []
    try:
        open_fn = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "rt")
        with open_fn as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split("\t")
                if len(p) >= 9 and p[2] == "gene":
                    attrs = p[8].lower()
                    if any(k in attrs for k in hk_keywords):
                        chrom, s, e, strand = p[0], int(p[3]), int(p[4]), p[6]
                        tss = s if strand == "+" else e
                        regions.append((chrom, max(0, tss - flank_bp), tss + flank_bp))
    except Exception as e:
        print(f"[-] Notice parsing housekeeping promoters: {e}")
    return regions


def filter_primary_chromosomes(
    chrom_list: List[str],
    chrom_sizes: Optional[Dict[str, int]] = None,
    min_length_bp: int = 1_000_000
) -> List[str]:
    """
    Filter chromosome identifiers to assembled primary chromosomes only.
    - Requires length >= min_length_bp (default 1 Mb) when chrom_sizes is available.
    - Excludes unplaced contigs/scaffolds (containing 'un', 'random', 'scaffold', 'contig', etc.).
    - Excludes mitochondrial chromosomes ('chrM', 'MT').
    """
    exclude_patterns = ["chrm", "mt", "mitochondrion", "unplaced", "scaffold", "contig", "random"]
    primary = []

    for c in chrom_list:
        c_str = str(c)
        c_lower = c_str.lower()

        # Check exclusion keywords
        if any(pat in c_lower for pat in exclude_patterns):
            continue
        # Check underscore contigs that are not named like chr1_... unless primary
        if "_" in c_str and not (c_lower.startswith("chr") and c_lower.replace("chr", "").split("_")[0].isalnum()):
            # e.g. NW_0183921.1 or Scaffold_123
            continue

        # Check size if available
        if chrom_sizes is not None and c_str in chrom_sizes:
            if chrom_sizes[c_str] < min_length_bp:
                continue

        primary.append(c_str)

    return sorted(list(set(primary)))


def assign_leak_free_splits(
    df: pd.DataFrame,
    held_out_chr: Optional[str] = None,
    val_frac: float = 0.15,
    chrom_sizes: Optional[Dict[str, int]] = None,
    cv_folds: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assign chromosomes to train, val, test splits without data leakage.
    - Filters to assembled primary chromosomes (length >= 1 Mb, excluding unplaced scaffolds).
    - Ensures held_out_chr (containing target locus) is strictly in fold_0 test set.
    - Guarantees test set is NEVER empty and has at least 1-2 primary chromosomes.
    - Supports multi-fold cross-validation fold generation with clean disjoint test partitions.
    - Absolutely no circular references in returned dictionary.
    """
    all_chroms = [str(c) for c in df["chrom"].unique()]
    primary_chroms = filter_primary_chromosomes(all_chroms, chrom_sizes=chrom_sizes, min_length_bp=1_000_000)

    # Fallback if filtering was too aggressive
    if len(primary_chroms) < 3:
        print("[-] Notice: Primary chromosome filter returned < 3 chroms; using all non-MT chroms.")
        primary_chroms = [c for c in all_chroms if not (c.lower().startswith("chrm") or c.lower() == "mt")]

    df = df.copy()

    # Determine matched held_out_chr
    matched_held_out = None
    if held_out_chr:
        held_out_str = str(held_out_chr)
        for c in primary_chroms:
            if c == held_out_str or c.replace("chr", "") == held_out_str.replace("chr", ""):
                matched_held_out = c
                break
        if not matched_held_out and held_out_str in all_chroms:
            matched_held_out = held_out_str
            if held_out_str not in primary_chroms:
                primary_chroms.append(held_out_str)

    k = max(1, cv_folds)
    np.random.seed(42)

    other_chroms = [c for c in primary_chroms if c != matched_held_out]
    shuffled_others = np.random.permutation(other_chroms).tolist()

    # Partition primary chromosomes cleanly into k disjoint test folds
    test_buckets: List[List[str]] = [[] for _ in range(k)]
    if matched_held_out:
        test_buckets[0].append(matched_held_out)

    for i, c in enumerate(shuffled_others):
        b_idx = i % k
        test_buckets[b_idx].append(c)

    all_folds: Dict[str, Dict[str, List[str]]] = {}
    for fold_idx in range(k):
        t_chroms = test_buckets[fold_idx]
        rem = [c for c in primary_chroms if c not in t_chroms]
        n_val = max(1, int(len(primary_chroms) * val_frac))
        v_chroms = rem[:n_val]
        tr_chroms = rem[n_val:]
        if not tr_chroms and len(v_chroms) > 1:
            tr_chroms = [v_chroms.pop()]
        all_folds[f"fold_{fold_idx}"] = {
            "train": [str(c) for c in tr_chroms],
            "valid": [str(c) for c in v_chroms],
            "test": [str(c) for c in t_chroms]
        }

    # Fold 0 independent dict
    fold_0 = {
        "train": list(all_folds["fold_0"]["train"]),
        "valid": list(all_folds["fold_0"]["valid"]),
        "test": list(all_folds["fold_0"]["test"])
    }

    # Label rows in dataframe based on fold_0
    test_set = set(fold_0["test"])
    val_set = set(fold_0["valid"])
    split_col = []
    for c in df["chrom"]:
        c_str = str(c)
        if c_str in test_set:
            split_col.append("test")
        elif c_str in val_set:
            split_col.append("valid")
        else:
            split_col.append("train")
    df["split"] = split_col

    # Construct clean return dict without circular references
    chr_fold = {
        "train": fold_0["train"],
        "valid": fold_0["valid"],
        "test": fold_0["test"],
        "all_folds": all_folds
    }
    return df, chr_fold


def main():
    parser = argparse.ArgumentParser(description="Modular ATAC-seq Preprocessing and Summit Target Quantification")
    parser.add_argument("--species", required=True, help="Species name (e.g. Danio_rerio, Oryzias_latipes)")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory for study")
    parser.add_argument("--genome-fa", required=True, help="Path to reference genome FASTA")
    parser.add_argument("--gtf", default=None, help="Path to reference GTF")
    parser.add_argument("--fastq-manifest", default=None, help="Path to atac_fastq_manifest.tsv")
    parser.add_argument("--early-bam", default=None, help="BAM path for early blastula stage (e.g. 256-cell/st9)")
    parser.add_argument("--late-bam", default=None, help="BAM path for late blastula stage (e.g. dome/st11)")
    parser.add_argument("--control-bam", default=None, help="Optional naked gDNA control BAM")
    parser.add_argument("--held-out-chr", default=None, help="Chromosome containing target locus to hold out in test split")
    parser.add_argument("--target-genes", nargs="*", default=["notch2", "slc35a3a"], help="Target genes to detect held_out_chr")
    parser.add_argument("--summit-window-bp", type=int, default=500, help="Summit counting window width in bp (default 500, i.e. ±250 bp)")
    parser.add_argument("--genome-size", default=None, help="MACS3 effective genome size (e.g. 1.4e9 for zebrafish, 7.3e8 for medaka)")
    parser.add_argument("--exclude-housekeeping-promoters", action="store_true", help="Flag housekeeping promoters in summit targets")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default 5)")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads")
    parser.add_argument("--task-name", default=None, help="TaskManager task name for progress updates")
    args = parser.parse_args()

    task_name = args.task_name or f"prep_atac_{args.species.lower()}"
    work_dir = Path(args.work_dir).resolve()
    species_dir = work_dir / "atac" / args.species
    species_dir.mkdir(parents=True, exist_ok=True)
    bams_dir = species_dir / "bams"
    bams_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[*] Modular ATAC Preprocessing Pipeline for Species: {args.species}")
    print(f"    Work Directory: {species_dir}")
    print(f"    Threads:        {args.threads}")
    print(f"    CV Folds:       {args.cv_folds}")
    print("=" * 80)

    genome_fa = Path(args.genome_fa).resolve()
    assert genome_fa.exists() and genome_fa.stat().st_size > 100 * 1024 * 1024, (
        f"[FATAL] Genome FASTA not found or too small: {genome_fa}"
    )

    # 1. Resolve BAMs: either from manifest (FASTQs -> BAMs) or explicit arguments
    early_merged_bam = species_dir / f"{args.species}_early_merged.bam"
    late_merged_bam = species_dir / f"{args.species}_late_merged.bam"
    control_merged_bam = species_dir / f"{args.species}_control_merged.bam"

    if args.early_bam and args.late_bam:
        early_bam_path = Path(args.early_bam).resolve()
        late_bam_path = Path(args.late_bam).resolve()
        control_bam_path = Path(args.control_bam).resolve() if args.control_bam else None
    elif args.fastq_manifest and Path(args.fastq_manifest).exists():
        manifest_df = pd.read_csv(args.fastq_manifest, sep="\t")
        spec_df = manifest_df[manifest_df["species"].str.lower() == args.species.lower()]

        bt2_prefix = str(work_dir / "downloads" / f"{args.species}_bt2_idx")

        # Canonical accession mapping for robust stage assignment:
        # GSM3756599–600 = early, GSM3756603–605 = late, SRR9032640 = control
        # DRX143405–407 = early, DRX143412–413 = late
        early_accs = {"gsm3756599", "gsm3756600", "srr9032650", "srr9032651", "drx143405", "drx143406", "drx143407", "drr152711", "drr152712", "drr152713"}
        late_accs = {"gsm3756603", "gsm3756604", "gsm3756605", "srr9032658", "srr9032659", "srr9032660", "srr9032661", "drx143412", "drx143413", "drr152718", "drr152719"}
        ctrl_accs = {"srr9032640", "gsm3756588"}

        stage_bams: Dict[str, List[Path]] = defaultdict(list)
        total_runs = len(spec_df)
        for idx, (_, r) in enumerate(spec_df.iterrows(), 1):
            stage_raw = str(r.get("stage", "")).strip().lower()
            run_id = str(r.get("run_id", "")).strip()
            parent_acc = str(r.get("accession", "")).strip().lower()
            run_lower = run_id.lower()

            if stage_raw in ["early", "early_256cell", "early_st9"] or parent_acc in early_accs or run_lower in early_accs or "256" in stage_raw or "st9" in stage_raw:
                stage = "early"
            elif stage_raw in ["late", "late_dome", "late_st11"] or parent_acc in late_accs or run_lower in late_accs or "dome" in stage_raw or "st11" in stage_raw:
                stage = "late"
            elif stage_raw in ["control", "control_gdna"] or parent_acc in ctrl_accs or run_lower in ctrl_accs or "gdna" in stage_raw:
                stage = "control"
            else:
                stage = stage_raw

            fq1 = Path(r["fastq_r1"])
            fq2 = Path(r["fastq_r2"]) if pd.notna(r.get("fastq_r2")) and r.get("fastq_r2") else None

            set_task_progress(work_dir, task_name, f"Aligning {run_id} ({idx}/{total_runs})")
            out_b = bams_dir / f"{run_id}.sorted.bam"
            proc_b = align_and_process_fastq(
                fastq_r1=fq1,
                fastq_r2=fq2,
                genome_fa=genome_fa,
                bowtie2_index_prefix=bt2_prefix,
                output_bam=out_b,
                threads=args.threads
            )
            stage_bams[stage].append(proc_b)

        # Merge stages
        early_keys = [k for k in stage_bams if "early" in k]
        late_keys = [k for k in stage_bams if "late" in k]
        ctrl_keys = [k for k in stage_bams if "control" in k]

        early_bams = [b for k in early_keys for b in stage_bams[k]]
        late_bams = [b for k in late_keys for b in stage_bams[k]]
        ctrl_bams = [b for k in ctrl_keys for b in stage_bams[k]]

        assert early_bams, f"[FATAL] No early ATAC BAMs processed for {args.species}"
        assert late_bams, f"[FATAL] No late ATAC BAMs processed for {args.species}"

        early_bam_path = merge_bams(early_bams, early_merged_bam, threads=args.threads)
        late_bam_path = merge_bams(late_bams, late_merged_bam, threads=args.threads)
        control_bam_path = merge_bams(ctrl_bams, control_merged_bam, threads=args.threads) if ctrl_bams else None
    else:
        # Check default paths
        if early_merged_bam.exists() and late_merged_bam.exists():
            early_bam_path = early_merged_bam
            late_bam_path = late_merged_bam
            control_bam_path = control_merged_bam if control_merged_bam.exists() else None
        else:
            raise FileNotFoundError(
                f"[FATAL] Input ATAC-seq BAMs or FASTQ manifest not found for {args.species}. "
                f"Zero-mock policy strictly enforced: please supply --early-bam & --late-bam or a valid --fastq-manifest."
            )

    # Validate BAMs
    for p_bam, lbl in [(early_bam_path, "Early"), (late_bam_path, "Late")]:
        assert p_bam.exists() and p_bam.stat().st_size > 10 * 1024 * 1024, (
            f"[FATAL] {lbl} BAM missing or smaller than 10 MB: {p_bam}"
        )
        mapped = get_bam_total_reads(p_bam)
        assert mapped > 0, f"[FATAL] {lbl} BAM has 0 mapped reads: {p_bam}"
        print(f"[+] {lbl} BAM verified: {p_bam.name} ({mapped:,} mapped reads)")

    # Generate chrom.sizes early from genome_fa.fai (excluding only mitochondrial contigs)
    chrom_sizes = species_dir / "chrom.sizes"
    if not chrom_sizes.exists() or chrom_sizes.stat().st_size == 0:
        fai_path = Path(str(genome_fa) + ".fai")
        if not fai_path.exists():
            print(f"[*] Indexing genome FASTA with samtools faidx: {genome_fa.name}...")
            run_cmd(f"samtools faidx {genome_fa}", desc="Indexing genome FASTA")
        if fai_path.exists() and fai_path.stat().st_size > 0:
            run_cmd(f"awk '$1 !~ /^(chrM|MT|M|chrMT)$/ {{print $1\"\\t\"$2}}' {fai_path} > {chrom_sizes}")
        else:
            run_cmd(f"samtools idxstats {late_bam_path} | awk '$2 > 0 && $1 !~ /^(chrM|MT|M|chrMT)$/ {{print $1\"\\t\"$2}}' > {chrom_sizes}")

    chrom_sizes_dict: Dict[str, int] = {}
    with open(chrom_sizes, "r") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2 and p[1].isdigit():
                chrom_sizes_dict[p[0]] = int(p[1])

    # 2. Call MACS3 peaks on late stage
    set_task_progress(work_dir, task_name, "Calling MACS3 peaks on late blastula ATAC")
    peaks_dir = species_dir / "peaks"
    narrow_peaks, summits_bed = call_peaks_macs3(
        late_bam=late_bam_path,
        output_dir=peaks_dir,
        species_name=args.species,
        control_bam=control_bam_path,
        genome_size=args.genome_size
    )

    # 3. Quantify reads around summits (continuous y_abs, y_rel)
    set_task_progress(work_dir, task_name, "Quantifying summit counts & targets (y_abs, y_rel)")
    print(f"[*] Quantifying summit CPM and continuous targets (y_abs, y_rel) in ±{args.summit_window_bp//2} bp windows...")
    df_targets = count_reads_in_summits(
        summits_bed=summits_bed,
        early_bam=early_bam_path,
        late_bam=late_bam_path,
        window_bp=args.summit_window_bp,
        threads=args.threads
    )

    # 4. Dynamic miR-430 cluster downweighting / flagging
    gtf_path = Path(args.gtf).resolve() if args.gtf else None
    if args.species.lower() in ["danio_rerio", "zebrafish"] and gtf_path and gtf_path.exists():
        m_chrom, m_start, m_end = detect_mir430_cluster(gtf_path)
        m_chrom_norm = str(m_chrom).lower().replace("chr", "")
        df_chrom_norm = df_targets["chrom"].astype(str).str.lower().str.replace("chr", "")
        is_mir430 = (df_chrom_norm == m_chrom_norm) & (df_targets["start"] <= m_end) & (df_targets["end"] >= m_start)
        df_targets["is_repeat_downweight"] = is_mir430
        print(f"[+] Flagged {is_mir430.sum()} summits in miR-430 cluster ({m_chrom}:{m_start:,}-{m_end:,}) for down-weighting/exclusion.")
    else:
        df_targets["is_repeat_downweight"] = False

    # 5. Optional Housekeeping promoter flagging
    if args.exclude_housekeeping_promoters and gtf_path and gtf_path.exists():
        hk_regions = parse_housekeeping_promoters(gtf_path, flank_bp=500)
        is_hk = np.zeros(len(df_targets), dtype=bool)
        for h_chr, h_s, h_e in hk_regions:
            m = (df_targets["chrom"].astype(str) == str(h_chr)) & (df_targets["summit"] >= h_s) & (df_targets["summit"] <= h_e)
            is_hk = is_hk | m
        df_targets["is_housekeeping"] = is_hk
        print(f"[+] Flagged {is_hk.sum()} summits as housekeeping promoters.")
    else:
        df_targets["is_housekeeping"] = False

    # 6. Dynamically detect held-out chromosome for target locus
    held_out_chr = args.held_out_chr
    if not held_out_chr and gtf_path and gtf_path.exists():
        held_out_chr = detect_held_out_chromosome(gtf_path, args.target_genes)

    # 7. Assign leak-free chromosome splits (primary chroms only, guaranteed test split)
    df_splits, chr_fold = assign_leak_free_splits(
        df=df_targets,
        held_out_chr=held_out_chr,
        val_frac=0.15,
        chrom_sizes=chrom_sizes_dict,
        cv_folds=args.cv_folds
    )

    # Export canonical chr_fold.json (fold 0)
    chr_fold_json = species_dir / "chr_fold.json"
    fold_0_dict = {
        "train": chr_fold["train"],
        "valid": chr_fold["valid"],
        "test": chr_fold["test"]
    }
    with open(chr_fold_json, "w") as f:
        json.dump(fold_0_dict, f, indent=2)

    # Export all folds if multi-fold
    if "all_folds" in chr_fold:
        all_folds_json = species_dir / "chr_folds.json"
        with open(all_folds_json, "w") as f:
            json.dump(chr_fold["all_folds"], f, indent=2)

    print(f"[+] Exported leak-free chromosome fold: {chr_fold_json}")
    print(f"    Train: {chr_fold['train']}")
    print(f"    Valid: {chr_fold['valid']}")
    print(f"    Test:  {chr_fold['test']}")

    # Export summit targets TSV
    out_tsv = species_dir / f"{args.species}_summit_targets.tsv"
    df_splits.to_csv(out_tsv, sep="\t", index=False)
    print(f"[+] Exported {len(df_splits):,} summit targets to {out_tsv}")

    # Generate BigWigs with LC_ALL=C sorting and safety filtering against chrom.sizes
    for lbl, b_path in [("early", early_bam_path), ("late", late_bam_path)]:
        bw_out = species_dir / f"{args.species}_{lbl}_cpm.bw"
        if not bw_out.exists() and shutil.which("bedtools") and shutil.which("bedGraphToBigWig"):
            bg_tmp = species_dir / f"{lbl}.tmp.bg"
            bg_filtered = species_dir / f"{lbl}.filtered.bg"
            tot = get_bam_total_reads(b_path)
            scale = 1e6 / tot
            run_cmd(f"bedtools genomecov -ibam {b_path} -bg -scale {scale} | LC_ALL=C sort -k1,1 -k2,2n > {bg_tmp}")
            run_cmd(f"awk 'NR==FNR {{c[$1]=1; next}} ($1 in c)' {chrom_sizes} {bg_tmp} > {bg_filtered}", desc=f"Filtering {lbl} bedGraph to contigs present in chrom.sizes")
            run_cmd(f"bedGraphToBigWig {bg_filtered} {chrom_sizes} {bw_out}")
            bg_tmp.unlink(missing_ok=True)
            bg_filtered.unlink(missing_ok=True)
            print(f"[+] BigWig track generated: {bw_out}")

    set_task_progress(work_dir, task_name, "100% (Done)")
    print(f"[SUCCESS] Modular ATAC preprocessing completed for {args.species}.")


if __name__ == "__main__":
    main()
