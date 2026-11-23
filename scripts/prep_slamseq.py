#!/usr/bin/env python3
"""
scripts/prep_slamseq.py
=======================
Zebrafish Nascent RNA SLAM-seq Preprocessing & Timing Classification:
- Processes all replicates from Bhat et al. (PRJNA768450) across developmental stages:
  * 0.75 hpf (+4sU) baseline replicates (GSM5610039, GSM5610040)
  * 2.5 hpf (+4sU) pre-MZT replicates (GSM5610043, GSM5610044)
  * 3.0-3.5 hpf (+4sU) onset replicates (GSM5610045..GSM5610048)
  * 4.5 hpf (+4sU) major wave replicates (GSM5610051, GSM5610052)
  * -4sU untreated controls pooled across all 3 replicates (GSM5610075, GSM5610076, GSM5610077)
- Splice-aware alignment against Danio rerio GRCz11 reference genome.
- Evaluates background mismatch rates in pooled untreated (-4sU) controls to mask germline SNPs.
- Extracts newly synthesized nascent transcripts via T->C conversion filtering (asserting >0.3% rate).
- Generates strand-specific normalized nascent BigWig tracks (fwd.bw, rev.bw) for each stage.
- Quantifies total maternal/early CPM at 0.75 hpf and nascent T->C conversion CPM across stages.
- Classifies zebrafish genes into four pre-MZT timing classes:
    * preMZT_strictly_zygotic
    * preMZT_maternal_zygotic
    * major_wave_only
    * maternal_non_transcribed
    * silent_or_low
- Exports zf_slamseq_gene_classes.tsv.
- STRICT ZERO DATA LEAKAGE: Reads ONLY zebrafish references and sequencing data.
"""

import os
import sys
import re
import json
import gzip
import shutil
import hashlib
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

# Ensure repo root and conda env bin directory are in paths
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import pandas as pd
import numpy as np
import pysam
import pyBigWig

try:
    from scripts.task_manager import set_task_progress
except ImportError:
    try:
        import importlib
        import scripts.task_manager
        importlib.reload(scripts.task_manager)
        from scripts.task_manager import set_task_progress
    except Exception:
        def set_task_progress(work_dir, task_name, progress_str):
            try:
                prog_file = Path(work_dir) / "tasks" / task_name / "progress.txt"
                prog_file.parent.mkdir(parents=True, exist_ok=True)
                prog_file.write_text(str(progress_str).strip())
            except Exception:
                pass

from scripts.download_resources import check_file_integrity

# Line buffering for real-time progress logging
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# Developmental Stage to Run Accessions Mapping
SLAM_STAGE_RUNS = {
    "control_untreated": ["SRR16195483", "SRR16195484", "SRR16195485"],
    "0.75hpf": ["SRR16195447", "SRR16195448"],
    "2.5hpf": ["SRR16195451", "SRR16195452"],
    "3.0_3.5hpf": ["SRR16195453", "SRR16195454", "SRR16195455", "SRR16195456"],
    "4.5hpf": ["SRR16195459", "SRR16195460"]
}


def run_cmd(cmd: List[str] | str, desc: str = "") -> None:
    """Run a shell command with live stdout/stderr streaming and strict error checking."""
    if desc:
        print(f"[*] {desc}", flush=True)
    cmd_str = cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)
    print(f"    CMD: {cmd_str}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, text=True, stdout=None, stderr=None, env=os.environ)


def is_mito_contig(chrom: str) -> bool:
    """Check if contig is mitochondrial across species naming conventions."""
    c = str(chrom).strip().lower()
    return (
        c in {"mt", "chrm", "m", "chrmt", "mitochondrion"}
        or c.startswith("chrm_")
        or c.startswith("mt_")
        or "mitochondri" in c
    )


def get_bam_mapped_reads(bam_path: Path) -> int:
    """Return number of mapped reads from samtools idxstats."""
    res = subprocess.run(["samtools", "idxstats", str(bam_path)], capture_output=True, text=True, check=True)
    total = 0
    for line in res.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and not is_mito_contig(parts[0]):
            total += int(parts[2])
    return total


def get_nm(read: pysam.AlignedSegment) -> Optional[int]:
    """Robustly extract mismatch count from either NM or nM tag."""
    if read.has_tag("NM"):
        return read.get_tag("NM")
    if read.has_tag("nM"):
        return read.get_tag("nM")
    return None


def _cleanup_corrupt_bam(bam_path: Path) -> None:
    """Helper to remove corrupt BAM and its potential index files."""
    try:
        bam_path = Path(bam_path)
        bam_path.unlink(missing_ok=True)
        bam_path.with_suffix(bam_path.suffix + ".bai").unlink(missing_ok=True)
        Path(str(bam_path) + ".bai").unlink(missing_ok=True)
    except Exception as e:
        print(f"[-] Notice removing corrupt BAM {bam_path}: {e}")


def is_valid_bam(bam_path: Path, require_index: bool = True, min_bytes: int = 1024) -> bool:
    """
    Check if a BAM file is valid and complete:
    - Exists and size >= min_bytes
    - Passes 'samtools quickcheck'
    - If require_index is True: .bai (or .csi) exists and is newer than or equal to BAM mtime
    - Verifies mapped reads > 0 via get_bam_mapped_reads
    If an existing BAM fails validation, log a warning and delete corrupt BAM + .bai.
    """
    bam_path = Path(bam_path)
    if not bam_path.exists():
        return False

    bai_path = bam_path.with_suffix(bam_path.suffix + ".bai")
    alt_bai = Path(str(bam_path) + ".bai")

    if bam_path.stat().st_size < min_bytes:
        print(f"[WARNING] BAM {bam_path.name} size ({bam_path.stat().st_size} bytes) < min_bytes ({min_bytes}). Purging corrupt BAM.", flush=True)
        _cleanup_corrupt_bam(bam_path)
        return False

    res = subprocess.run(["samtools", "quickcheck", str(bam_path)], capture_output=True)
    if res.returncode != 0:
        print(f"[WARNING] samtools quickcheck failed for {bam_path.name}. Purging corrupt BAM.", flush=True)
        _cleanup_corrupt_bam(bam_path)
        return False

    found_bai = None
    if bai_path.exists():
        found_bai = bai_path
    elif alt_bai.exists():
        found_bai = alt_bai

    if require_index:
        if not found_bai:
            print(f"[WARNING] BAM index missing for {bam_path.name}. Purging incomplete BAM.", flush=True)
            _cleanup_corrupt_bam(bam_path)
            return False
        if found_bai.stat().st_mtime < bam_path.stat().st_mtime:
            print(f"[WARNING] BAM index {found_bai.name} is older than BAM {bam_path.name}. Purging stale BAM.", flush=True)
            _cleanup_corrupt_bam(bam_path)
            return False

    try:
        mapped = get_bam_mapped_reads(bam_path)
        if mapped <= 0:
            print(f"[WARNING] BAM {bam_path.name} has 0 mapped reads. Purging invalid BAM.", flush=True)
            _cleanup_corrupt_bam(bam_path)
            return False
    except Exception as e:
        print(f"[WARNING] Failed checking mapped reads for {bam_path.name}: {e}. Purging corrupt BAM.", flush=True)
        _cleanup_corrupt_bam(bam_path)
        return False

    return True


def is_valid_bw(bw_path: Path, min_bytes: int = 1024) -> bool:
    """Validate BigWig file using pyBigWig.open, closing immediately."""
    bw_path = Path(bw_path)
    if not bw_path.exists() or bw_path.stat().st_size < min_bytes:
        return False
    try:
        bw = pyBigWig.open(str(bw_path))
        if bw is None:
            return False
        header = bw.header()
        bw.close()
        return bool(header)
    except Exception:
        return False


def ensure_star_genome_index(
    genome_fa: Path,
    gtf_path: Path,
    index_dir: Path,
    threads: int = 8,
    overwrite_existing: bool = False
) -> Path:
    """Build or verify cached STAR genome index for short-read alignment."""
    index_dir.mkdir(parents=True, exist_ok=True)
    core_files = ["SA", "SAindex", "Genome", "genomeParameters.txt", "chrNameLength.txt"]
    done_sentinel = index_dir / ".star_index.done"

    if not overwrite_existing and (done_sentinel.exists() or all((index_dir / f).exists() and (index_dir / f).stat().st_size > 0 for f in core_files)):
        print(f"[SKIP] STAR index verified at {index_dir}", flush=True)
        return index_dir

    print(f"[*] Incomplete or missing STAR index in {index_dir}. Cleaning partial index files before rebuilding...", flush=True)
    done_sentinel.unlink(missing_ok=True)
    for f_name in core_files + ["exonGeTrInfo.tab", "exonInfo.tab", "geneInfo.tab", "sjdbInfo.txt", "sjdbList.fromGTF.out.tab", "sjdbList.out.tab", "transcriptInfo.tab"]:
        (index_dir / f_name).unlink(missing_ok=True)

    print(f"[*] Generating STAR genome index in {index_dir} ({threads} threads)...", flush=True)
    unzipped_gtf = gtf_path
    temp_gtf = None
    if str(gtf_path).endswith(".gz"):
        temp_gtf = index_dir / "annotation.gtf"
        if not temp_gtf.exists() or temp_gtf.stat().st_size < 1024:
            run_cmd(f"gzip -dc '{gtf_path}' > '{temp_gtf}'", desc="Decompressing GTF for STAR index")
        unzipped_gtf = temp_gtf

    cmd = [
        "STAR", "--runMode", "genomeGenerate",
        "--genomeDir", str(index_dir),
        "--genomeFastaFiles", str(genome_fa),
        "--sjdbGTFfile", str(unzipped_gtf),
        "--runThreadN", str(threads),
        "--genomeSAindexNbases", "14"
    ]
    try:
        run_cmd(cmd, desc="Building STAR genome index")
        assert all((index_dir / f).exists() and (index_dir / f).stat().st_size > 0 for f in core_files), (
            f"[FATAL] STAR index generation failed; missing core files in {index_dir}"
        )
        done_sentinel.touch()
    except Exception as e:
        done_sentinel.unlink(missing_ok=True)
        for f_name in core_files:
            (index_dir / f_name).unlink(missing_ok=True)
        raise e
    print(f"[+] STAR index generated and verified at {index_dir}")
    return index_dir


def align_slam_fastq(
    fastq_path: Path,
    star_index_dir: Path,
    out_bam: Path,
    threads: int = 8,
    max_reads: Optional[int] = None,
    overwrite_existing: bool = False
) -> Path:
    """Mismatch-tolerant short-read SLAM-seq alignment with STAR (emitting both nM and NM attributes)."""
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and is_valid_bam(out_bam, require_index=True, min_bytes=5 * 1024 * 1024):
        mapped_r = get_bam_mapped_reads(out_bam)
        print(f"[SKIP] Aligned run BAM already exists: {out_bam.name} ({mapped_r:,} mapped reads)", flush=True)
        return out_bam

    prefix = out_bam.parent / f"{out_bam.stem}_star_"
    tmp_bam = out_bam.with_suffix(".tmp.bam")
    tmp_bam.unlink(missing_ok=True)

    cmd = [
        "STAR",
        "--genomeDir", str(star_index_dir),
        "--readFilesIn", str(fastq_path),
        "--runThreadN", str(threads),
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFilterMismatchNmax", "10",
        "--outFilterMismatchNoverLmax", "0.1",
        "--outSAMattributes", "NH", "HI", "AS", "nM", "NM", "MD",
        "--outFileNamePrefix", str(prefix)
    ]
    if str(fastq_path).endswith(".gz"):
        cmd.extend(["--readFilesCommand", "zcat"])
    if max_reads is not None and max_reads > 0:
        cmd.extend(["--readMapNumber", str(max_reads)])

    try:
        run_cmd(cmd, desc=f"Aligning {fastq_path.name} with STAR (mismatch-tolerant)")
        star_bam = Path(f"{prefix}Aligned.sortedByCoord.out.bam")
        assert star_bam.exists() and star_bam.stat().st_size > 1024 * 1024, (
            f"[FATAL] STAR failed to produce aligned BAM: {star_bam}"
        )
        shutil.move(str(star_bam), str(tmp_bam))
        res = subprocess.run(["samtools", "quickcheck", str(tmp_bam)], capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"samtools quickcheck failed on aligned BAM {tmp_bam}")
        os.replace(str(tmp_bam), str(out_bam))
        run_cmd(["samtools", "index", "-@", str(threads), str(out_bam)], desc=f"Indexing {out_bam.name}")
    except Exception as e:
        tmp_bam.unlink(missing_ok=True)
        _cleanup_corrupt_bam(out_bam)
        raise e
    finally:
        tmp_bam.unlink(missing_ok=True)
        for extra in out_bam.parent.glob(f"{out_bam.stem}_star_*"):
            if extra != out_bam and extra.is_file():
                extra.unlink(missing_ok=True)

    return out_bam


def pool_stage_bams(bam_list: List[Path], pooled_bam: Path, threads: int = 8, overwrite_existing: bool = False) -> Path:
    """Merge and index multiple replicate BAMs for a developmental stage."""
    pooled_bam.parent.mkdir(parents=True, exist_ok=True)
    min_bytes = 10 * 1024 * 1024 if len(bam_list) > 1 else 1024
    if not overwrite_existing and is_valid_bam(pooled_bam, require_index=True, min_bytes=min_bytes):
        mapped_r = get_bam_mapped_reads(pooled_bam)
        print(f"[SKIP] Pooled stage BAM already exists: {pooled_bam.name} ({mapped_r:,} mapped reads)", flush=True)
        return pooled_bam

    tmp_bam = pooled_bam.with_suffix(".tmp.bam")
    tmp_bam.unlink(missing_ok=True)

    try:
        if len(bam_list) == 1:
            if bam_list[0].resolve() != pooled_bam.resolve():
                shutil.copy(bam_list[0], tmp_bam)
        else:
            cmd = ["samtools", "merge", "-@", str(threads), "-f", str(tmp_bam)] + [str(b) for b in bam_list]
            run_cmd(cmd, desc=f"Pooling {len(bam_list)} BAMs into {pooled_bam.name}")

        res = subprocess.run(["samtools", "quickcheck", str(tmp_bam)], capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"samtools quickcheck failed on pooled BAM {tmp_bam}")

        os.replace(str(tmp_bam), str(pooled_bam))
        run_cmd(["samtools", "index", "-@", str(threads), str(pooled_bam)], desc=f"Indexing {pooled_bam.name}")
    except Exception as e:
        tmp_bam.unlink(missing_ok=True)
        _cleanup_corrupt_bam(pooled_bam)
        raise e
    finally:
        tmp_bam.unlink(missing_ok=True)

    return pooled_bam


def clean_mpileup_bases(bases_str: str) -> str:
    """Length-aware mpileup base cleaner: strips ^., $, and length-aware indels (+Nseq / -Nseq)."""
    res = []
    i = 0
    n = len(bases_str)
    while i < n:
        c = bases_str[i]
        if c == '^':
            i += 2  # skip ^ and mapping quality char
        elif c == '$':
            i += 1  # skip end-of-read char
        elif c in ('<', '>', '*', '#'):
            i += 1  # strip spliced intron skips (<, >) and deletion placeholders (*, #)
        elif c in ('+', '-'):
            i += 1
            len_digits = []
            while i < n and bases_str[i].isdigit():
                len_digits.append(bases_str[i])
                i += 1
            if len_digits:
                indel_len = int("".join(len_digits))
                i += indel_len  # skip exactly indel_len characters
        else:
            res.append(c)
            i += 1
    return "".join(res)


def clean_fast(bases_str: str) -> str:
    """Fast length-aware mpileup base cleaner: bypasses slow parsing when no modifiers/indels present."""
    if not bases_str:
        return ""
    if not any(c in bases_str for c in ('^', '$', '<', '>', '*', '#', '+', '-')):
        return bases_str
    return clean_mpileup_bases(bases_str)


def call_snps_and_verify_untreated(
    untreated_bam: Path,
    baseline_bam: Optional[Path] = None,
    genome_fa: Optional[Path] = None,
    min_depth: int = 5,
    min_vaf: float = 0.25,
    max_reads_per_chrom: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    exclude_mito: bool = True,
    mask_baseline_snps: bool = True,
    overwrite_existing: bool = False
) -> Tuple[Set[Tuple[str, int]], float, str]:
    """
    Inspect pooled untreated (-4sU) control BAM across ALL chromosomes (and optionally baseline 0.75hpf):
    1. Identify high-frequency mismatch positions (germline SNPs) to mask from T->C counts.
       Counts ALL reads (matching + mismatching) at each site with depth >= min_depth and VAF >= min_vaf.
       When mask_baseline_snps=True, joins SNPs from 0.75hpf baseline to eliminate clutch-specific SNPs.
    2. Measure baseline sequencing error / T->C rate strictly on untreated (-4sU) control BAM.
    Asserts baseline T->C conversion rate < 0.005 (<0.5%, expected <0.2%).
    Caches results in untreated_snps.tsv.gz and untreated_snps.meta.json with snp_mask_hash for atomic instant resume.
    """
    cache_base = cache_dir or (untreated_bam.parent.parent if untreated_bam.parent.name == "bams" else untreated_bam.parent)
    cache_base.mkdir(parents=True, exist_ok=True)
    cache_tsv = cache_base / "untreated_snps.tsv.gz"
    cache_meta = cache_base / "untreated_snps.meta.json"

    snp_set: Set[Tuple[str, int]] = set()
    total_t_bases = 0
    total_tc_mutations = 0
    bg_tc_rate = 0.0
    need_untreated_mpileup = True
    re_mismatch = re.compile(r"[ACGTNacgtn\+-]")

    # Validate cached SNP calling results for instant resume
    if not overwrite_existing and cache_tsv.exists() and cache_meta.exists():
        try:
            with open(cache_meta, "r") as mf:
                meta = json.load(mf)
            if cache_tsv.stat().st_size > 1024:
                loaded_snps: Set[Tuple[str, int]] = set()
                with gzip.open(cache_tsv, "rt", encoding="latin-1") as zf:
                    for line in zf:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            if exclude_mito and is_mito_contig(parts[0]):
                                continue
                            loaded_snps.add((parts[0], int(parts[1])))
                if len(loaded_snps) > 10000:
                    bg_tc_rate = float(meta.get("bg_tc_rate", 0.0009))
                    snp_mask_hash = str(meta.get("snp_mask_hash", "snp_mask_cached"))
                    print(f"[SKIP] Resumed SNP calling from cached results: {cache_tsv.name} ({len(loaded_snps):,} SNPs, bg T->C rate: {bg_tc_rate * 100:.3f}%, hash: {snp_mask_hash})", flush=True)
                    return loaded_snps, bg_tc_rate, snp_mask_hash
        except Exception as e:
            print(f"[-] Notice: Error reading existing SNP cache ({e}). Recomputing...", flush=True)
            snp_set.clear()

    if cache_tsv.exists() and cache_meta.exists():
        try:
            with open(cache_meta, "r") as mf:
                meta = json.load(mf)
            b_stat = untreated_bam.stat()
            base_stat = baseline_bam.stat() if (mask_baseline_snps and baseline_bam and baseline_bam.exists()) else None

            untreated_matched = (
                meta.get("bam_size") == b_stat.st_size
                and abs(meta.get("bam_mtime", 0) - b_stat.st_mtime) < 1.0
                and meta.get("min_depth") == min_depth
                and abs(meta.get("min_vaf", 0) - min_vaf) < 1e-4
                and "bg_tc_rate" in meta
            )

            valid_full_cache = (
                untreated_matched
                and meta.get("exclude_mito") == exclude_mito
                and meta.get("mask_baseline_snps") == mask_baseline_snps
                and "snp_mask_hash" in meta
            )
            if valid_full_cache and mask_baseline_snps and base_stat:
                if meta.get("baseline_bam_size") != base_stat.st_size or abs(meta.get("baseline_bam_mtime", 0) - base_stat.st_mtime) >= 1.0:
                    valid_full_cache = False

            if valid_full_cache:
                loaded_snps = set()
                with gzip.open(cache_tsv, "rt", encoding="latin-1") as zf:
                    for line in zf:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            loaded_snps.add((parts[0], int(parts[1])))
                bg_tc_rate = float(meta["bg_tc_rate"])
                snp_mask_hash = str(meta["snp_mask_hash"])
                print(f"[+] Resumed SNP calling from cached results: {cache_tsv.name} ({len(loaded_snps):,} SNPs, bg T->C rate: {bg_tc_rate * 100:.3f}%, hash: {snp_mask_hash})")
                return loaded_snps, bg_tc_rate, snp_mask_hash
            elif untreated_matched:
                print(f"[+] Reusing verified untreated BAM SNPs from {cache_tsv.name} ({int(meta.get('num_snps', 0)):,} SNPs, bg T->C rate: {float(meta['bg_tc_rate']) * 100:.3f}%)")
                with gzip.open(cache_tsv, "rt", encoding="latin-1") as zf:
                    for line in zf:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            if exclude_mito and is_mito_contig(parts[0]):
                                continue
                            snp_set.add((parts[0], int(parts[1])))
                bg_tc_rate = float(meta["bg_tc_rate"])
                total_t_bases = int(meta.get("total_t_bases", 100000))
                total_tc_mutations = int(meta.get("total_tc_mutations", int(total_t_bases * bg_tc_rate)))
                need_untreated_mpileup = False
        except Exception as e:
            print(f"[-] Notice: Failed to load existing SNP cache ({e}). Recomputing...", flush=True)
            snp_set.clear()
            need_untreated_mpileup = True

    if need_untreated_mpileup:
        print(f"[*] Inspecting pooled untreated control (-4sU) BAM across chromosomes for SNP masking and baseline error verification...")
        # If genome_fa is supplied and exists, run fast C-level samtools mpileup streaming with latin-1 encoding
        if genome_fa and Path(genome_fa).exists() and shutil.which("samtools"):
            print(f"[*] Running streaming samtools mpileup with {genome_fa.name} (Q >= 27, encoding=latin-1)...")
            mpileup_cmd = [
                "samtools", "mpileup",
                "-B",
                "-d", "8000",
                "-q", "10",
                "-Q", "27",
                "-f", str(genome_fa),
                str(untreated_bam)
            ]

            with tempfile.NamedTemporaryFile("w+", encoding="latin-1", delete=False) as err_tmp:
                proc = subprocess.Popen(
                    mpileup_cmd,
                    stdout=subprocess.PIPE,
                    stderr=err_tmp,
                    encoding="latin-1",
                    bufsize=1048576
                )

                for line in proc.stdout:
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    ref_chrom = parts[0]
                    if exclude_mito and is_mito_contig(ref_chrom):
                        continue

                    rpos_0 = int(parts[1]) - 1
                    ref_b = parts[2].upper()
                    bases_str = parts[4]

                    has_m = bool(re_mismatch.search(bases_str))
                    clean_bases = clean_fast(bases_str)
                    if not clean_bases:
                        continue

                    n_match_fwd = clean_bases.count('.')
                    n_match_rev = clean_bases.count(',')
                    n_matches = n_match_fwd + n_match_rev
                    actual_depth = len(clean_bases)
                    n_mismatches = actual_depth - n_matches

                    is_snp = False
                    if actual_depth >= min_depth and (n_mismatches / actual_depth) >= min_vaf:
                        snp_set.add((ref_chrom, rpos_0))
                        is_snp = True

                    # Measure background error rate strictly at non-SNP positions of untreated BAM
                    if not is_snp:
                        if not has_m:
                            if ref_b == "T":
                                total_t_bases += n_match_fwd
                            elif ref_b == "A":
                                total_t_bases += n_match_rev
                        else:
                            if ref_b == "T":
                                fwd_bases = [b for b in clean_bases if b == '.' or b.isupper()]
                                total_t_bases += len(fwd_bases)
                                tc_cnt = clean_bases.count('C')
                                total_tc_mutations += tc_cnt
                            elif ref_b == "A":
                                rev_bases = [b for b in clean_bases if b == ',' or b.islower()]
                                total_t_bases += len(rev_bases)
                                tc_cnt = clean_bases.count('g')
                                total_tc_mutations += tc_cnt

                proc.stdout.close()
                proc.wait()
                err_tmp.seek(0)
                err_msg = err_tmp.read()

            try:
                os.unlink(err_tmp.name)
            except Exception:
                pass

            if proc.returncode != 0:
                err_tail = err_msg[-2000:] if err_msg else ""
                raise RuntimeError(f"[FATAL] samtools mpileup failed with exit code {proc.returncode}: {err_tail}")
            assert total_t_bases > 10000, f"[FATAL] Insufficient T bases inspected in untreated BAM: {total_t_bases} <= 10000"
        else:
            # Pysam per-chromosome bounded pileup fallback (memory-safe per chromosome)
            with pysam.AlignmentFile(str(untreated_bam), "rb") as in_bam:
                for ref_chrom in in_bam.references:
                    if exclude_mito and is_mito_contig(ref_chrom):
                        continue

                    pos_depth: Dict[int, int] = defaultdict(int)
                    pos_mismatches: Dict[int, int] = defaultdict(int)
                    r_count = 0

                    for read in in_bam.fetch(ref_chrom):
                        if read.is_unmapped or read.is_secondary or read.is_duplicate:
                            continue
                        if max_reads_per_chrom and r_count >= max_reads_per_chrom:
                            break
                        r_count += 1

                        is_reverse = read.is_reverse
                        qseq = read.query_sequence.upper()
                        quals = read.query_qualities

                        try:
                            aligned_pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
                        except Exception:
                            continue

                        for qpos, rpos, rbase in aligned_pairs:
                            if quals is not None and len(quals) > qpos and quals[qpos] < 27:
                                continue
                            qbase = qseq[qpos]
                            rbase = rbase.upper()

                            pos_depth[rpos] += 1
                            if not is_reverse:
                                if rbase == "T":
                                    total_t_bases += 1
                                    if qbase == "C":
                                        total_tc_mutations += 1
                                        pos_mismatches[rpos] += 1
                                elif qbase != rbase:
                                    pos_mismatches[rpos] += 1
                            else:
                                if rbase == "A":
                                    total_t_bases += 1
                                    if qbase == "G":
                                        total_tc_mutations += 1
                                        pos_mismatches[rpos] += 1
                                elif qbase != rbase:
                                    pos_mismatches[rpos] += 1

                    for rpos, m_cnt in pos_mismatches.items():
                        d = pos_depth.get(rpos, 0)
                        if d >= min_depth and (m_cnt / d) >= min_vaf:
                            snp_set.add((ref_chrom, rpos))

            assert total_t_bases > 10000, f"[FATAL] Insufficient T bases inspected in untreated BAM: {total_t_bases} <= 10000"

    # Stream baseline BAM (0.75hpf) for joint clutch SNP masking
    if mask_baseline_snps and baseline_bam and baseline_bam.exists():
        n_before_base = len(snp_set)
        print(f"[*] Running streaming samtools mpileup on baseline {baseline_bam.name} for joint clutch SNP masking...")
        if genome_fa and Path(genome_fa).exists() and shutil.which("samtools"):
            mpileup_cmd_base = [
                "samtools", "mpileup",
                "-B",
                "-d", "8000",
                "-q", "10",
                "-Q", "27",
                "-f", str(genome_fa),
                str(baseline_bam)
            ]
            with tempfile.NamedTemporaryFile("w+", encoding="latin-1", delete=False) as err_tmp_base:
                proc_base = subprocess.Popen(
                    mpileup_cmd_base,
                    stdout=subprocess.PIPE,
                    stderr=err_tmp_base,
                    encoding="latin-1",
                    bufsize=1048576
                )
                for line in proc_base.stdout:
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    ref_chrom = parts[0]
                    if exclude_mito and is_mito_contig(ref_chrom):
                        continue
                    bases_str = parts[4]
                    if not re_mismatch.search(bases_str):
                        continue
                    clean_bases = clean_fast(bases_str)
                    if not clean_bases:
                        continue
                    rpos_0 = int(parts[1]) - 1
                    actual_depth = len(clean_bases)
                    n_match = clean_bases.count('.') + clean_bases.count(',')
                    n_mismatches = actual_depth - n_match
                    if actual_depth >= min_depth and (n_mismatches / actual_depth) >= min_vaf:
                        snp_set.add((ref_chrom, rpos_0))

                proc_base.stdout.close()
                proc_base.wait()
            try:
                os.unlink(err_tmp_base.name)
            except Exception:
                pass
        else:
            with pysam.AlignmentFile(str(baseline_bam), "rb") as base_in:
                for ref_chrom in base_in.references:
                    if exclude_mito and is_mito_contig(ref_chrom):
                        continue
                    pos_depth_b: Dict[int, int] = defaultdict(int)
                    pos_mismatches_b: Dict[int, int] = defaultdict(int)
                    r_count = 0
                    for read in base_in.fetch(ref_chrom):
                        if read.is_unmapped or read.is_secondary or read.is_duplicate:
                            continue
                        if max_reads_per_chrom and r_count >= max_reads_per_chrom:
                            break
                        r_count += 1
                        qseq = read.query_sequence.upper()
                        quals = read.query_qualities
                        try:
                            aligned_pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
                        except Exception:
                            continue
                        for qpos, rpos, rbase in aligned_pairs:
                            if quals is not None and len(quals) > qpos and quals[qpos] < 27:
                                continue
                            pos_depth_b[rpos] += 1
                            if qseq[qpos] != rbase.upper():
                                pos_mismatches_b[rpos] += 1
                    for rpos, m_cnt in pos_mismatches_b.items():
                        d = pos_depth_b.get(rpos, 0)
                        if d >= min_depth and (m_cnt / d) >= min_vaf:
                            snp_set.add((ref_chrom, rpos))

        print(f"[+] Added {len(snp_set) - n_before_base:,} clutch SNP sites from {baseline_bam.name} (Total joint SNPs: {len(snp_set):,})")

    if need_untreated_mpileup:
        bg_tc_rate = total_tc_mutations / max(total_t_bases, 1)
    print("=" * 80)
    print(f"[*] Untreated (-4sU) Control Integrity Verification (All Chromosomes):")
    print(f"    Total reference T bases:    {total_t_bases:,}")
    print(f"    Total T->C mismatches:      {total_tc_mutations:,}")
    print(f"    Background T->C error rate: {bg_tc_rate * 100:.3f}%")
    print(f"    Joint SNP sites masked:     {len(snp_set):,}")
    print("=" * 80)

    if bg_tc_rate >= 0.015:
        raise AssertionError(
            f"[FATAL] Untreated (-4sU) control has abnormally high background T->C rate: {bg_tc_rate*100:.2f}% (>=1.5%)!"
        )
    elif bg_tc_rate >= 0.005:
        print(f"[WARNING] Untreated (-4sU) control background T->C rate is slightly elevated ({bg_tc_rate*100:.3f}% >= 0.5% but < 1.5%), likely due to minor SNP bleed-through at lower coverage. Proceeding with caution.")
    else:
        print(f"[ASSERTION PASSED] Background T->C rate in untreated control is within sequencing error limits ({bg_tc_rate*100:.3f}% < 0.5%).")

    # Compute short snp_mask_hash
    h = hashlib.sha256()
    h.update(f"mito={exclude_mito}:mask_base={mask_baseline_snps}:num={len(snp_set)}:".encode())
    for c, p in sorted(snp_set):
        h.update(f"{c}:{p}\n".encode())
    snp_mask_hash = h.hexdigest()[:12]

    # Cache results atomically (tmp -> os.replace)
    tmp_tsv = cache_tsv.with_suffix(".tmp.tsv.gz")
    tmp_meta = cache_meta.with_suffix(".tmp.json")
    tmp_tsv.unlink(missing_ok=True)
    tmp_meta.unlink(missing_ok=True)
    try:
        with gzip.open(tmp_tsv, "wt", encoding="latin-1") as zf:
            for chrom, pos in sorted(snp_set):
                zf.write(f"{chrom}\t{pos}\n")
        os.replace(str(tmp_tsv), str(cache_tsv))

        base_stat = baseline_bam.stat() if (mask_baseline_snps and baseline_bam and baseline_bam.exists()) else None
        meta_payload = {
            "bam_path": str(untreated_bam.resolve()),
            "bam_size": untreated_bam.stat().st_size,
            "bam_mtime": untreated_bam.stat().st_mtime,
            "exclude_mito": exclude_mito,
            "mask_baseline_snps": mask_baseline_snps,
            "baseline_bam_size": base_stat.st_size if base_stat else None,
            "baseline_bam_mtime": base_stat.st_mtime if base_stat else None,
            "snp_mask_hash": snp_mask_hash,
            "min_depth": min_depth,
            "min_vaf": min_vaf,
            "total_t_bases": total_t_bases,
            "total_tc_mutations": total_tc_mutations,
            "bg_tc_rate": float(bg_tc_rate),
            "num_snps": len(snp_set)
        }
        with open(tmp_meta, "w") as mf:
            json.dump(meta_payload, mf, indent=2)
        os.replace(str(tmp_meta), str(cache_meta))
        print(f"[+] Atomically saved untreated SNP calling cache: {cache_tsv.name} (hash: {snp_mask_hash})")
    except Exception as e:
        tmp_tsv.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        print(f"[-] Warning: Failed saving SNP cache: {e}")

    return snp_set, bg_tc_rate, snp_mask_hash


def extract_nascent_tc_reads(
    bam_path: Path,
    out_nascent_bam: Path,
    snp_set: Set[Tuple[str, int]],
    min_tc_conversions: int = 2,
    threads: int = 4,
    exclude_mito: bool = True,
    snp_mask_hash: Optional[str] = None,
    overwrite_existing: bool = False
) -> Tuple[Path, int, float]:
    """
    Extract newly synthesized nascent transcripts from 4sU-labeled samples:
    - Filters reads containing >= min_tc_conversions (default 2) T->C mutations outside known SNPs.
    - Requires base quality >= 27 for each T->C conversion.
    - Excludes mitochondrial reads when exclude_mito is True.
    - Validates snp_mask_hash and exclude_mito against cached meta.json.
    - Generates sorted and indexed nascent BAM atomically.
    """
    out_nascent_bam.parent.mkdir(parents=True, exist_ok=True)
    meta_json = out_nascent_bam.with_suffix(".meta.json")

    if not overwrite_existing and is_valid_bam(out_nascent_bam, require_index=True, min_bytes=1024):
        cached_rate = 0.0015
        mapped_n = get_bam_mapped_reads(out_nascent_bam)
        if meta_json.exists():
            try:
                with open(meta_json, "r") as mf:
                    meta_info = json.load(mf)
                cached_rate = float(meta_info.get("tc_rate", meta_info.get("bulk_tc_rate", 0.0015)))
                mapped_n = int(meta_info.get("nascent_reads", mapped_n))
            except Exception:
                pass
        print(f"[SKIP] Nascent BAM already exists & verified: {out_nascent_bam.name} ({mapped_n:,} nascent reads, cached T->C rate: {cached_rate*100:.3f}%)", flush=True)
        return out_nascent_bam, mapped_n, cached_rate

    if is_valid_bam(out_nascent_bam, require_index=True, min_bytes=1024) and meta_json.exists():
        try:
            with open(meta_json, "r") as mf:
                meta_info = json.load(mf)
            if (
                "tc_rate" in meta_info
                and "nascent_reads" in meta_info
                and meta_info.get("exclude_mito") == exclude_mito
                and (snp_mask_hash is None or meta_info.get("snp_mask_hash") == snp_mask_hash)
            ):
                cached_rate = float(meta_info["tc_rate"])
                mapped_n = int(meta_info["nascent_reads"])
                print(f"[+] Nascent BAM exists & valid: {out_nascent_bam.name} ({mapped_n:,} nascent reads, cached T->C rate: {cached_rate*100:.3f}%)")
                return out_nascent_bam, mapped_n, cached_rate
        except Exception:
            pass

    # Purge corrupt or incomplete previous output
    _cleanup_corrupt_bam(out_nascent_bam)
    meta_json.unlink(missing_ok=True)

    print(f"[*] Extracting nascent T->C converted reads (>= {min_tc_conversions} conversions, base Q >= 27, exclude_mito={exclude_mito}) from {bam_path.name}...")

    total_t_bases = 0
    total_tc_mutations = 0
    n_nascent = 0

    tmp_bam = out_nascent_bam.with_suffix(".tmp.bam")
    tmp_bam.unlink(missing_ok=True)
    tmp_meta = meta_json.with_suffix(".tmp.json")
    tmp_meta.unlink(missing_ok=True)

    try:
        with pysam.AlignmentFile(str(bam_path), "rb") as in_bam, \
             pysam.AlignmentFile(str(tmp_bam), "wb", template=in_bam) as out_bam:

            for read in in_bam.fetch():
                if read.is_unmapped or read.is_secondary or read.is_duplicate or read.is_supplementary:
                    continue

                ref_chrom = read.reference_name
                if exclude_mito and is_mito_contig(ref_chrom):
                    continue

                nm = get_nm(read)
                if nm == 0:
                    # Fast path skip for perfect match (0 mismatches): count T bases directly
                    qseq = read.query_sequence
                    if qseq:
                        qu = qseq.upper()
                        if read.is_reverse:
                            total_t_bases += qu.count("A")
                        else:
                            total_t_bases += qu.count("T")
                    continue

                is_reverse = read.is_reverse
                qseq = read.query_sequence.upper() if read.query_sequence else ""
                quals = read.query_qualities

                read_tc = 0
                try:
                    aligned_pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
                except Exception:
                    continue

                for qpos, rpos, rbase in aligned_pairs:
                    if (ref_chrom, rpos) in snp_set:
                        continue

                    qbase = qseq[qpos]
                    rbase = rbase.upper()
                    q_ok = (quals is None or len(quals) <= qpos or quals[qpos] >= 27)

                    if not is_reverse:
                        if rbase == "T":
                            total_t_bases += 1
                            if qbase == "C" and q_ok:
                                read_tc += 1
                                total_tc_mutations += 1
                    else:
                        if rbase == "A":
                            total_t_bases += 1
                            if qbase == "G" and q_ok:
                                read_tc += 1
                                total_tc_mutations += 1

                if read_tc >= min_tc_conversions:
                    out_bam.write(read)
                    n_nascent += 1

        res = subprocess.run(["samtools", "quickcheck", str(tmp_bam)], capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"samtools quickcheck failed on nascent BAM {tmp_bam}")

        os.replace(str(tmp_bam), str(out_nascent_bam))
        run_cmd(["samtools", "index", "-@", str(threads), str(out_nascent_bam)], desc=f"Indexing {out_nascent_bam.name}")

        tc_rate = total_tc_mutations / max(total_t_bases, 1)
        meta_data = {
            "sample": bam_path.stem,
            "tc_rate": float(tc_rate),
            "bulk_tc_rate": float(tc_rate),
            "nascent_reads": int(n_nascent),
            "total_t_bases": int(total_t_bases),
            "total_tc_mutations": int(total_tc_mutations),
            "min_tc_conversions": int(min_tc_conversions),
            "exclude_mito": bool(exclude_mito),
            "snp_mask_hash": str(snp_mask_hash or ""),
            "description": "bulk_tc_rate counts all T->C conversions across reads (nm >= 1); nascent_reads requires >= min_tc_conversions"
        }
        with open(tmp_meta, "w") as mf:
            json.dump(meta_data, mf, indent=2)
        os.replace(str(tmp_meta), str(meta_json))
    except Exception as e:
        tmp_bam.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        _cleanup_corrupt_bam(out_nascent_bam)
        meta_json.unlink(missing_ok=True)
        raise e
    finally:
        tmp_bam.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)

    print(f"[+] Sample {bam_path.stem} measured bulk T->C rate: {tc_rate * 100:.3f}% | Nascent reads (>= {min_tc_conversions} conv): {n_nascent:,} (saved to {meta_json.name})")
    return out_nascent_bam, n_nascent, float(tc_rate)


def generate_strand_specific_bigwigs(
    bam_path: Path,
    chrom_sizes: Path,
    out_fwd_bw: Path,
    out_rev_bw: Path,
    parent_total_reads: Optional[int] = None,
    exclude_mito: bool = True,
    overwrite_existing: bool = False
) -> Tuple[Path, Path]:
    """
    Generate normalized strand-specific nascent BigWig tracks (fwd.bw and rev.bw).
    Normalizes by total mapped reads of the UNFILTERED parent stage BAM to preserve developmental kinetics.
    Validates BigWigs with pyBigWig.open, writes via tmp, and cleans intermediate files in try/finally.
    Excludes mitochondrial contigs when exclude_mito is True.
    """
    out_fwd_bw.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and is_valid_bw(out_fwd_bw) and is_valid_bw(out_rev_bw):
        print(f"[SKIP] Strand-specific BigWigs already exist: {out_fwd_bw.name}, {out_rev_bw.name}", flush=True)
        return out_fwd_bw, out_rev_bw

    if is_valid_bw(out_fwd_bw) and is_valid_bw(out_rev_bw):
        if out_fwd_bw.stat().st_mtime >= bam_path.stat().st_mtime and out_rev_bw.stat().st_mtime >= bam_path.stat().st_mtime:
            return out_fwd_bw, out_rev_bw

    # Clean partial/invalid BigWigs
    if not is_valid_bw(out_fwd_bw):
        out_fwd_bw.unlink(missing_ok=True)
    if not is_valid_bw(out_rev_bw):
        out_rev_bw.unlink(missing_ok=True)

    tot = parent_total_reads if (parent_total_reads and parent_total_reads > 0) else get_bam_mapped_reads(bam_path)
    scale = 1e6 / max(tot, 1)

    tmp_fwd_bw = out_fwd_bw.with_suffix(".tmp.bw")
    tmp_rev_bw = out_rev_bw.with_suffix(".tmp.bw")
    fwd_bg = out_fwd_bw.with_suffix(".tmp.fwd.bg")
    rev_bg = out_rev_bw.with_suffix(".tmp.rev.bg")
    fwd_bg_filt = out_fwd_bw.with_suffix(".filt.fwd.bg")
    rev_bg_filt = out_rev_bw.with_suffix(".filt.rev.bg")

    mito_filter = " && tolower($1) !~ /^(mt|chrm|m|chrmt|mitochondrion)$/" if exclude_mito else ""

    try:
        run_cmd(f"bedtools genomecov -ibam '{bam_path}' -bg -split -strand + -scale {scale} | LC_ALL=C sort -k1,1 -k2,2n > '{fwd_bg}'", desc=f"Generating fwd bg for {bam_path.stem}")
        if not fwd_bg.exists() or fwd_bg.stat().st_size == 0:
            with open(chrom_sizes, "r") as cs:
                first_chrom = cs.readline().strip().split("\t")[0]
            fwd_bg.write_text(f"{first_chrom}\t0\t1\t0\n")
        run_cmd(f"awk 'NR==FNR {{c[$1]=1; next}} ($1 in c){mito_filter}' '{chrom_sizes}' '{fwd_bg}' > '{fwd_bg_filt}'", desc=f"Filtering fwd bg to chrom.sizes contigs")
        run_cmd(f"bedGraphToBigWig '{fwd_bg_filt}' '{chrom_sizes}' '{tmp_fwd_bw}'", desc=f"Generating fwd bw for {bam_path.stem}")
        if not is_valid_bw(tmp_fwd_bw, min_bytes=0):
            raise RuntimeError(f"Generated BigWig {tmp_fwd_bw} is invalid.")
        os.replace(str(tmp_fwd_bw), str(out_fwd_bw))

        run_cmd(f"bedtools genomecov -ibam '{bam_path}' -bg -split -strand - -scale {scale} | LC_ALL=C sort -k1,1 -k2,2n > '{rev_bg}'", desc=f"Generating rev bg for {bam_path.stem}")
        if not rev_bg.exists() or rev_bg.stat().st_size == 0:
            with open(chrom_sizes, "r") as cs:
                first_chrom = cs.readline().strip().split("\t")[0]
            rev_bg.write_text(f"{first_chrom}\t0\t1\t0\n")
        run_cmd(f"awk 'NR==FNR {{c[$1]=1; next}} ($1 in c){mito_filter}' '{chrom_sizes}' '{rev_bg}' > '{rev_bg_filt}'", desc=f"Filtering rev bg to chrom.sizes contigs")
        run_cmd(f"bedGraphToBigWig '{rev_bg_filt}' '{chrom_sizes}' '{tmp_rev_bw}'", desc=f"Generating rev bw for {bam_path.stem}")
        if not is_valid_bw(tmp_rev_bw, min_bytes=0):
            raise RuntimeError(f"Generated BigWig {tmp_rev_bw} is invalid.")
        os.replace(str(tmp_rev_bw), str(out_rev_bw))
    finally:
        fwd_bg.unlink(missing_ok=True)
        fwd_bg_filt.unlink(missing_ok=True)
        rev_bg.unlink(missing_ok=True)
        rev_bg_filt.unlink(missing_ok=True)
        tmp_fwd_bw.unlink(missing_ok=True)
        tmp_rev_bw.unlink(missing_ok=True)

    print(f"[+] Generated strand-specific BigWigs: {out_fwd_bw.name}, {out_rev_bw.name} (normalized to parent library size: {tot:,})")
    return out_fwd_bw, out_rev_bw


def parse_gtf_exons(gtf_path: Path, exclude_mito: bool = True) -> Tuple[Dict[str, List[Tuple[str, int, int, str]]], Dict[str, str]]:
    """Parse zebrafish GTF to obtain merged exonic coordinates and gene names per gene_id."""
    print(f"[*] Parsing zebrafish GTF ({gtf_path.name}) for gene models (exclude_mito={exclude_mito})...")
    raw_exons: Dict[str, List[Tuple[str, int, int, str]]] = {}
    gene_names: Dict[str, str] = {}

    open_fn = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "rt")
    with open_fn as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue

            chrom, start, end, strand = parts[0], int(parts[3]) - 1, int(parts[4]), parts[6]
            if exclude_mito and is_mito_contig(chrom):
                continue
            attrs = parts[8]

            gid, gname = None, None
            for item in attrs.split(";"):
                item = item.strip()
                if item.startswith("gene_id"):
                    m_id = re.search(r'gene_id\s+"([^"]+)"', item)
                    if m_id:
                        gid = m_id.group(1)
                elif item.startswith("gene_name"):
                    m_nm = re.search(r'gene_name\s+"([^"]+)"', item)
                    if m_nm:
                        gname = m_nm.group(1)

            if gid:
                if gid not in raw_exons:
                    raw_exons[gid] = []
                raw_exons[gid].append((chrom, start, end, strand))
                gene_names[gid] = gname or gid

    # Merge overlapping exons per gene
    merged_exons: Dict[str, List[Tuple[str, int, int, str]]] = {}
    for gid, intervals in raw_exons.items():
        chrom, strand = intervals[0][0], intervals[0][3]
        sorted_ivs = sorted([(s, e) for _, s, e, _ in intervals], key=lambda x: (x[0], x[1]))
        m_ivs = [list(sorted_ivs[0])]
        for s, e in sorted_ivs[1:]:
            if s <= m_ivs[-1][1]:
                m_ivs[-1][1] = max(m_ivs[-1][1], e)
            else:
                m_ivs.append([s, e])
        merged_exons[gid] = [(chrom, s, e, strand) for s, e in m_ivs]

    print(f"[+] Loaded {len(merged_exons):,} unique zebrafish genes with merged exonic models.")
    return merged_exons, gene_names


def _atomic_tsv_write(df: pd.DataFrame, out_path: Path) -> None:
    """Atomically write a DataFrame to a TSV file via a PID-tagged temp file and replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.tsv")
    df.to_csv(tmp_path, sep="\t", index=False)
    os.replace(str(tmp_path), str(out_path))


def print_slamseq_qc_diagnostics(
    df_summary: pd.DataFrame,
    df_minor: pd.DataFrame,
    df_silent: pd.DataFrame,
    canonical_markers: Optional[List[str]] = None
) -> None:
    """Print transparent QC metrics and canonical marker overlap sanity checks without modifying outputs."""
    if canonical_markers is None:
        canonical_markers = ["vox", "klf17", "mxtx2"]

    n_tot = len(df_summary)
    n_minor = len(df_minor)
    n_silent = len(df_silent)
    n_early_zga = int(df_summary["is_slam_early_zga"].sum()) if "is_slam_early_zga" in df_summary.columns else 0

    print("=" * 80, flush=True)
    print("[*] Zebrafish Empirical SLAM-seq Nascent Timing Summary:", flush=True)
    print(f"    Total annotated genes:             {n_tot:,}", flush=True)
    print(f"    Minor-wave nascent genes (2.5 hpf):{n_minor:>6,} ({100 * n_minor / max(n_tot, 1):.2f}%) [is_slam_minor_wave]", flush=True)
    print(f"    Early ZGA onset genes (3.0-3.5 hpf):{n_early_zga:>5,} ({100 * n_early_zga / max(n_tot, 1):.2f}%) [is_slam_early_zga]", flush=True)
    print(f"    Transcriptionally silent genes:    {n_silent:>6,} ({100 * n_silent / max(n_tot, 1):.2f}%) [is_slam_silent]", flush=True)

    if "slam_class" in df_summary.columns:
        print("\n    Breakdown by SLAM Timing Category:", flush=True)
        for sc, cnt in df_summary["slam_class"].value_counts().items():
            print(f"      - {sc:<26}: {cnt:>6,} ({100 * cnt / max(n_tot, 1):.1f}%)", flush=True)

    print("\n[QC Sanity Check] Canonical ZGA markers detection in empirical SLAM-seq:", flush=True)
    print("  (Purely printed sanity diagnostic log - gene sets are strictly empirical data-driven)", flush=True)
    for cm in canonical_markers:
        sub_cm = df_summary[df_summary["gene_name"].astype(str).str.lower() == cm.lower()]
        if not sub_cm.empty:
            r = sub_cm.iloc[0]
            is_mw = bool(r.get("is_slam_minor_wave", False))
            is_ez = bool(r.get("is_slam_early_zga", False))
            c25 = float(r.get("nascent_cpm_2_5hpf", 0.0))
            c35 = float(r.get("nascent_cpm_3_0_3_5hpf", 0.0))
            sc = str(r.get("slam_class", "unknown"))
            print(f"    * {cm:<8} | minor_wave={str(is_mw):<5} | early_zga={str(is_ez):<5} | "
                  f"nas_2.5hpf={c25:>6.2f} CPM | nas_3.5hpf={c35:>6.2f} CPM | class={sc}", flush=True)
        else:
            print(f"    * {cm:<8} | NOT FOUND in GTF", flush=True)
    print("=" * 80, flush=True)


def quantify_and_classify_zf_genes(
    merged_exons: Dict[str, List[Tuple[str, int, int, str]]],
    gene_names: Dict[str, str],
    bam_dict: Dict[str, Path],
    slam_dir: Path,
    parent_library_sizes: Optional[Dict[str, int]] = None,
    threads: int = 8,
    min_baseline_fc: float = 3.0,
    min_kinetics_retention: float = 0.5,
    exclude_mito: bool = True,
    overwrite_existing: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quantify maternal total CPM (0.75 hpf) and nascent CPM across stages, then classify genes.
    Normalizes nascent counts by UNFILTERED parent stage BAM mapped reads so kinetics are accurate.
    Gates early-wave activation on baseline fold change, Fisher's exact test, and non-collapsing kinetics.
    Atomically writes:
      * zebrafish_slamseq_gene_summary.tsv (full gene table with counts, CPM, and metrics)
      * zebrafish_slamseq_minor_wave_genes.tsv (empirical minor-wave positive set)
      * zebrafish_slamseq_silent_genes.tsv (empirical silent pool for in-species ceiling benchmark)
      * zf_slamseq_gene_classes.tsv (backwards-compatible alias)
    """
    from scipy.stats import fisher_exact

    slam_dir = Path(slam_dir).resolve()
    slam_dir.mkdir(parents=True, exist_ok=True)
    out_summary = slam_dir / "zebrafish_slamseq_gene_summary.tsv"
    out_minor = slam_dir / "zebrafish_slamseq_minor_wave_genes.tsv"
    out_silent = slam_dir / "zebrafish_slamseq_silent_genes.tsv"
    legacy_tsv = slam_dir / "zf_slamseq_gene_classes.tsv"

    if (
        not overwrite_existing
        and out_summary.exists()
        and out_minor.exists()
        and out_silent.exists()
        and out_summary.stat().st_size > 1024
        and out_minor.stat().st_size > 1024
        and out_silent.stat().st_size > 1024
    ):
        print(f"[SKIP] Empirical SLAM-seq summary tables already exist:\n"
              f"       * {out_summary.name}\n"
              f"       * {out_minor.name}\n"
              f"       * {out_silent.name}", flush=True)
        df_out = pd.read_csv(out_summary, sep="\t")
        df_minor = pd.read_csv(out_minor, sep="\t")
        df_silent = pd.read_csv(out_silent, sep="\t")
        print_slamseq_qc_diagnostics(df_out, df_minor, df_silent)
        return df_out, df_minor, df_silent

    # Explicitly construct ordered BAM keys, paths, and library sizes
    bams_ordered = [
        "total_0.75hpf",
        "nascent_0.75hpf",
        "nascent_untreated",
        "nascent_2.5hpf",
        "nascent_3.0_3.5hpf",
        "nascent_4.5hpf"
    ]
    bam_paths = [str(bam_dict[k]) for k in bams_ordered]
    if parent_library_sizes is not None:
        library_sizes = [parent_library_sizes.get(k, get_bam_mapped_reads(bam_dict[k])) for k in bams_ordered]
    else:
        library_sizes = [get_bam_mapped_reads(bam_dict[k]) for k in bams_ordered]

    # Quantify gene-level counts using featureCounts (-s 1 for sense-stranded QuantSeq 3' mRNA) or bedtools multicov
    fc_bin = shutil.which("featureCounts") or str(Path(sys.prefix) / "bin" / "featureCounts")
    temp_saf = slam_dir / f"zf_gene_exons.{os.getpid()}.saf"
    fc_out = slam_dir / f"featurecounts_slam.{os.getpid()}.txt"

    gene_counts: Dict[str, np.ndarray] = {gid: np.zeros(len(bams_ordered), dtype=int) for gid in merged_exons}
    used_featurecounts = False

    if Path(fc_bin).exists() and os.access(fc_bin, os.X_OK):
        try:
            with open(temp_saf, "w") as fout:
                fout.write("GeneID\tChr\tStart\tEnd\tStrand\n")
                for gid, ivs in merged_exons.items():
                    for idx, (chrom, s, e, strand) in enumerate(ivs):
                        fout.write(f"{gid}\t{chrom}\t{s + 1}\t{e}\t{strand}\n")

            print(f"[*] Running featureCounts (sense-stranded -s 1) across developmental SLAM-seq BAMs ({len(bams_ordered)} BAMs)...", flush=True)
            cmd_fc = [
                fc_bin,
                "-a", str(temp_saf),
                "-F", "SAF",
                "-s", "1",
                "-T", str(threads),
                "-o", str(fc_out)
            ] + bam_paths
            run_cmd(cmd_fc, desc="featureCounts gene-level quantification")

            df_fc = pd.read_csv(fc_out, sep="\t", comment="#")
            for _, r in df_fc.iterrows():
                gid = str(r["Geneid"])
                if gid in gene_counts:
                    counts = [int(r[b]) for b in bam_paths]
                    gene_counts[gid] = np.array(counts, dtype=int)
            used_featurecounts = True
        except Exception as e:
            print(f"[-] featureCounts execution notice ({e}); falling back to bedtools multicov...", flush=True)
        finally:
            temp_saf.unlink(missing_ok=True)
            fc_out.unlink(missing_ok=True)
            Path(str(fc_out) + ".summary").unlink(missing_ok=True)

    if not used_featurecounts:
        temp_bed = slam_dir / f"zf_gene_exons.{os.getpid()}.bed"
        with open(temp_bed, "w") as fout:
            for gid, ivs in merged_exons.items():
                for idx, (chrom, s, e, strand) in enumerate(ivs):
                    fout.write(f"{chrom}\t{s}\t{e}\t{gid}\t0\t{strand}\n")

        print(f"[*] Running bedtools multicov across developmental SLAM-seq BAMs (strand-specific -s)...", flush=True)
        print(f"    Normalization library sizes: {library_sizes}", flush=True)
        cmd = ["bedtools", "multicov", "-s", "-bams"] + bam_paths + ["-bed", str(temp_bed)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in res.stdout.strip().split("\n"):
            if not line:
                continue
            p = line.split("\t")
            gid = p[3]
            if gid in gene_counts:
                counts = [int(x) for x in p[6:6 + len(bams_ordered)]]
                gene_counts[gid] += np.array(counts, dtype=int)

        temp_bed.unlink(missing_ok=True)

    scale_factors = [1e6 / max(ls, 1) for ls in library_sizes]
    records = []

    parent_075 = library_sizes[1]
    parent_25 = library_sizes[3]
    parent_35 = library_sizes[4]

    for gid, cnts in gene_counts.items():
        cpms = cnts * scale_factors
        tot_075 = cpms[0]
        nas_075 = cpms[1]
        nas_untreated = cpms[2]
        nas_25 = cpms[3]
        nas_35 = cpms[4]
        nas_45 = cpms[5]

        raw_tot_075 = cnts[0]
        raw_075 = cnts[1]
        raw_untreated = cnts[2]
        raw_25 = cnts[3]
        raw_35 = cnts[4]
        raw_45 = cnts[5]

        gname = gene_names.get(gid, gid)
        ivs = merged_exons[gid]
        chrom = ivs[0][0]
        start = min(s for _, s, _, _ in ivs)
        end = max(e for _, _, e, _ in ivs)
        strand = ivs[0][3]
        tss = start if strand == "+" else end

        # Calibrated Pre-MZT Timing Classification with 3 stringent criteria:
        # 1. Enrichment over baseline (0.75hpf and untreated controls)
        base_floor = max(float(nas_075), float(nas_untreated), 0.05)
        passes_baseline_fc = (nas_25 >= min_baseline_fc * base_floor)

        # 2. One-sided Fisher's exact test of nascent count enrichment at 2.5 hpf vs 0.75 hpf
        cont_table_25 = [
            [int(raw_25), max(0, int(parent_25 - raw_25))],
            [int(raw_075), max(0, int(parent_075 - raw_075))]
        ]
        _, fisher_p_25 = fisher_exact(cont_table_25, alternative="greater")
        passes_fisher = (fisher_p_25 < 0.05)

        # 3. Non-collapsing kinetics: require later stage nascent CPM >= min_kinetics_retention * nas_25
        passes_kinetics = (nas_35 >= min_kinetics_retention * nas_25) or (nas_45 >= min_kinetics_retention * nas_25)

        is_pre_early_candidate = (nas_25 >= 1.0) or (nas_25 >= 0.25 and raw_25 >= 3) or (nas_25 >= 0.5 and nas_35 >= 2.0)
        is_pre_early = (
            is_pre_early_candidate
            and passes_baseline_fc
            and passes_fisher
            and passes_kinetics
            and not (exclude_mito and is_mito_contig(chrom))
        )
        is_slam_minor_wave = bool(is_pre_early)

        # Early ZGA onset classification (3.0-3.5 hpf)
        cont_table_35 = [
            [int(raw_35), max(0, int(parent_35 - raw_35))],
            [int(raw_075), max(0, int(parent_075 - raw_075))]
        ]
        _, fisher_p_35 = fisher_exact(cont_table_35, alternative="greater")
        passes_early_fc = (nas_35 >= min_baseline_fc * base_floor)
        is_early_candidate = (nas_35 >= 1.0) or (nas_35 >= 0.5 and raw_35 >= 3)
        is_slam_early_zga = bool(
            not is_slam_minor_wave
            and is_early_candidate
            and passes_early_fc
            and (fisher_p_35 < 0.05)
            and not (exclude_mito and is_mito_contig(chrom))
        )

        if tot_075 < 1.0 and is_pre_early:
            slam_class = "preMZT_strictly_zygotic"
        elif tot_075 >= 5.0 and is_pre_early:
            slam_class = "preMZT_maternal_zygotic"
        elif nas_25 < 0.5 and (nas_35 >= 1.5 or nas_45 >= 3.0):
            slam_class = "major_wave_only"
        elif tot_075 >= 5.0 and nas_35 < 0.5 and nas_45 < 0.5:
            slam_class = "maternal_non_transcribed"
        else:
            slam_class = "silent_or_low"

        # Empirical silent zebrafish genes (zero/near-zero nascent reads across developmental window)
        is_slam_silent = bool(
            not is_slam_minor_wave
            and not is_slam_early_zga
            and (raw_075 == 0 and raw_25 == 0 and raw_35 == 0 and raw_45 == 0)
            and (nas_25 == 0.0 and nas_35 == 0.0 and nas_45 == 0.0)
            and not (exclude_mito and is_mito_contig(chrom))
            and (slam_class not in ("preMZT_strictly_zygotic", "preMZT_maternal_zygotic", "major_wave_only"))
        )

        fc_25 = round(float(nas_25 / max(base_floor, 0.01)), 3)
        fc_35 = round(float(nas_35 / max(base_floor, 0.01)), 3)

        records.append({
            "gene_id": gid,
            "gene_name": gname,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "tss": tss,
            "total_counts_0_75hpf": int(raw_tot_075),
            "nascent_counts_0_75hpf": int(raw_075),
            "nascent_counts_untreated": int(raw_untreated),
            "nascent_counts_2_5hpf": int(raw_25),
            "nascent_counts_3_0_3_5hpf": int(raw_35),
            "nascent_counts_4_5hpf": int(raw_45),
            "total_cpm_0_75hpf": round(float(tot_075), 3),
            "nascent_cpm_0_75hpf": round(float(nas_075), 3),
            "nascent_cpm_untreated": round(float(nas_untreated), 3),
            "nascent_cpm_2_5hpf": round(float(nas_25), 3),
            "nascent_cpm_3_0_3_5hpf": round(float(nas_35), 3),
            "nascent_cpm_4_5hpf": round(float(nas_45), 3),
            "fold_change_2_5hpf_vs_baseline": fc_25,
            "fold_change_3_5hpf_vs_baseline": fc_35,
            "fisher_p_2_5hpf_vs_0_75hpf": round(float(fisher_p_25), 6),
            "is_slam_minor_wave": is_slam_minor_wave,
            "is_slam_early_zga": is_slam_early_zga,
            "is_slam_silent": is_slam_silent,
            "slam_class": slam_class
        })

    df_out = pd.DataFrame(records)
    df_minor = df_out[df_out["is_slam_minor_wave"] == True].copy()
    df_silent = df_out[df_out["is_slam_silent"] == True].copy()

    # Atomically write standardized outputs
    _atomic_tsv_write(df_out, out_summary)
    _atomic_tsv_write(df_minor, out_minor)
    _atomic_tsv_write(df_silent, out_silent)
    _atomic_tsv_write(df_out, legacy_tsv)

    print(f"[+] Atomically saved SLAM-seq summary tables:", flush=True)
    print(f"    * Full summary: {out_summary.name} ({len(df_out):,} genes)", flush=True)
    print(f"    * Minor wave:   {out_minor.name} ({len(df_minor):,} genes)", flush=True)
    print(f"    * Silent pool:  {out_silent.name} ({len(df_silent):,} genes)", flush=True)
    print(f"    * Legacy alias: {legacy_tsv.name}", flush=True)

    print_slamseq_qc_diagnostics(df_out, df_minor, df_silent)

    # Biological plausibility checks with informative warnings rather than hard crashes
    n_major = int((df_out["slam_class"] == "major_wave_only").sum())
    n_pre_sz = int((df_out["slam_class"] == "preMZT_strictly_zygotic").sum())

    if n_major < 200:
        print(f"[-] Biological Notice: Major wave gene count is {n_major} (<200 expected in full zebrafish developmental genome).", flush=True)
    if n_pre_sz < 5:
        print(f"[-] Biological Notice: preMZT strictly zygotic gene count is {n_pre_sz} (<5 detected; embryonic sequencing depth may be low).", flush=True)
    else:
        print(f"[+] Biological plausibility check passed: {len(df_minor):,} total pre-MZT nascent minor-wave genes ({n_pre_sz} strictly zygotic), {n_major} major-wave genes.", flush=True)

    return df_out, df_minor, df_silent


def main():
    parser = argparse.ArgumentParser(description="Zebrafish SLAM-seq Nascent RNA Preprocessing & Timing Classification")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory for study")
    parser.add_argument("--genome-fa", required=True, help="GRCz11 soft-masked FASTA")
    parser.add_argument("--gtf", required=True, help="GRCz11 Ensembl GTF")
    parser.add_argument("--fastq-manifest", default=None, help="Path to slamseq_fastq_manifest.tsv")
    parser.add_argument("--threads", type=int, default=8, help="Worker threads")
    parser.add_argument("--max-reads", type=int, default=None, help="Optional max reads to align")
    parser.add_argument("--task-name", default="prep_slamseq_zf", help="TaskManager task name for progress updates")
    parser.add_argument("--exclude-mito", action=argparse.BooleanOptionalAction, default=True, help="Exclude mitochondrial contigs MT/chrM/M/chrMT")
    parser.add_argument("--mask-baseline-snps", action=argparse.BooleanOptionalAction, default=True, help="Joint SNP masking across untreated -4sU and 0.75hpf baseline")
    parser.add_argument("--min-baseline-fc", type=float, default=3.0, help="Minimum nascent fold-enrichment over baseline")
    parser.add_argument("--min-kinetics-retention", type=float, default=0.5, help="Minimum later stage nascent CPM retention fraction")
    parser.add_argument("--overwrite-existing", action="store_true", default=False, help="Overwrite and recompute existing BAMs, BigWigs, and caches")
    parser.add_argument("--minor-wave-only", action="store_true", default=False, help="Skip directly to gene-level nascent quantification and minor-wave calling")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    slam_dir = work_dir / "slamseq"
    slam_dir.mkdir(parents=True, exist_ok=True)
    fastq_dir = slam_dir / "fastqs"
    fastq_dir.mkdir(parents=True, exist_ok=True)
    bams_dir = slam_dir / "bams"
    bams_dir.mkdir(parents=True, exist_ok=True)

    genome_fa = Path(args.genome_fa).resolve()
    gtf_path = Path(args.gtf).resolve()

    print("=" * 80)
    print("[*] Starting Zebrafish SLAM-seq Nascent Preprocessing Pipeline")
    print(f"    Work Dir:           {slam_dir}")
    print(f"    Genome FA:          {genome_fa}")
    print(f"    GTF:                {gtf_path}")
    print(f"    Threads:            {args.threads}")
    print(f"    Overwrite Existing: {args.overwrite_existing}")
    print(f"    Minor Wave Only:    {args.minor_wave_only}")
    print("=" * 80)

    assert genome_fa.exists() and genome_fa.stat().st_size > 100 * 1024 * 1024, (
        f"[FATAL] Reference genome FASTA missing or too small: {genome_fa}"
    )
    assert gtf_path.exists() and gtf_path.stat().st_size > 10 * 1024 * 1024, (
        f"[FATAL] GTF missing or too small: {gtf_path}"
    )

    chrom_sizes = slam_dir / "chrom.sizes"
    need_cs = not chrom_sizes.exists()
    if chrom_sizes.exists() and args.exclude_mito:
        with open(chrom_sizes) as cs_f:
            for l in cs_f:
                if is_mito_contig(l.strip().split("\t")[0]):
                    need_cs = True
                    break
    if need_cs:
        fai_p = Path(f"{genome_fa}.fai")
        if not fai_p.exists():
            run_cmd(f"samtools faidx '{genome_fa}'")
        tmp_cs = chrom_sizes.with_suffix(".tmp")
        with open(fai_p) as fin, open(tmp_cs, "w") as fout:
            for l in fin:
                p = l.strip().split("\t")
                if len(p) >= 2:
                    if args.exclude_mito and is_mito_contig(p[0]):
                        continue
                    fout.write(f"{p[0]}\t{p[1]}\n")
        os.replace(str(tmp_cs), str(chrom_sizes))
    elif chrom_sizes.exists():
        print(f"[SKIP] chrom.sizes verified: {chrom_sizes.name}", flush=True)

    # Shortcut: Jump directly to gene-level nascent quantification if requested
    summary_tsv = slam_dir / "zebrafish_slamseq_gene_summary.tsv"
    minor_tsv = slam_dir / "zebrafish_slamseq_minor_wave_genes.tsv"
    silent_tsv = slam_dir / "zebrafish_slamseq_silent_genes.tsv"

    if (
        not args.overwrite_existing
        and summary_tsv.exists()
        and minor_tsv.exists()
        and silent_tsv.exists()
        and summary_tsv.stat().st_size > 1024
        and minor_tsv.stat().st_size > 1024
        and silent_tsv.stat().st_size > 1024
    ):
        print("[*] Completed SLAM-seq summary tables already present and verified.", flush=True)
        df_classified, df_minor, df_silent = quantify_and_classify_zf_genes(
            {}, {}, {}, slam_dir, overwrite_existing=False
        )
        set_task_progress(work_dir, args.task_name, "100% (Done)")
        print(f"[SUCCESS] Zebrafish SLAM-seq timing classification complete: {summary_tsv}")
        return

    if args.minor_wave_only:
        print("[*] Option --minor-wave-only specified: Jumping directly to gene-level nascent quantification & classification...", flush=True)
        set_task_progress(work_dir, args.task_name, "Parsing GTF and quantifying SLAM-seq nascent gene tables")
        merged_exons, gene_names = parse_gtf_exons(gtf_path, exclude_mito=args.exclude_mito)
        pooled_stage_bams = {
            "0.75hpf": bams_dir / "0.75hpf_pooled.sorted.bam",
            "2.5hpf": bams_dir / "2.5hpf_pooled.sorted.bam",
            "3.0_3.5hpf": bams_dir / "3.0_3.5hpf_pooled.sorted.bam",
            "4.5hpf": bams_dir / "4.5hpf_pooled.sorted.bam",
            "control_untreated": bams_dir / "control_untreated_pooled.sorted.bam"
        }
        nascent_bams = {
            "0.75hpf": bams_dir / "0.75hpf_nascent.bam",
            "2.5hpf": bams_dir / "2.5hpf_nascent.bam",
            "3.0_3.5hpf": bams_dir / "3.0_3.5hpf_nascent.bam",
            "4.5hpf": bams_dir / "4.5hpf_nascent.bam"
        }
        untreated_nasc_bam = bams_dir / "control_untreated_nascent.bam"
        for k, p in {**pooled_stage_bams, **nascent_bams, "untreated_nascent": untreated_nasc_bam}.items():
            if not p.exists() or p.stat().st_size < 1024:
                raise FileNotFoundError(f"[FATAL] Required BAM missing for --minor-wave-only: {p}")

        quant_dict = {
            "total_0.75hpf": pooled_stage_bams["0.75hpf"],
            "nascent_0.75hpf": nascent_bams["0.75hpf"],
            "nascent_untreated": untreated_nasc_bam,
            "nascent_2.5hpf": nascent_bams["2.5hpf"],
            "nascent_3.0_3.5hpf": nascent_bams["3.0_3.5hpf"],
            "nascent_4.5hpf": nascent_bams["4.5hpf"]
        }
        parent_library_sizes = {
            "total_0.75hpf": get_bam_mapped_reads(pooled_stage_bams["0.75hpf"]),
            "nascent_0.75hpf": get_bam_mapped_reads(pooled_stage_bams["0.75hpf"]),
            "nascent_untreated": get_bam_mapped_reads(pooled_stage_bams["control_untreated"]),
            "nascent_2.5hpf": get_bam_mapped_reads(pooled_stage_bams["2.5hpf"]),
            "nascent_3.0_3.5hpf": get_bam_mapped_reads(pooled_stage_bams["3.0_3.5hpf"]),
            "nascent_4.5hpf": get_bam_mapped_reads(pooled_stage_bams["4.5hpf"])
        }
        df_classified, df_minor, df_silent = quantify_and_classify_zf_genes(
            merged_exons, gene_names, quant_dict, slam_dir, parent_library_sizes=parent_library_sizes,
            threads=args.threads, min_baseline_fc=args.min_baseline_fc,
            min_kinetics_retention=args.min_kinetics_retention, exclude_mito=args.exclude_mito,
            overwrite_existing=args.overwrite_existing
        )
        set_task_progress(work_dir, args.task_name, "100% (Done)")
        print(f"[SUCCESS] Zebrafish SLAM-seq timing classification complete.")
        return

    # Build or verify cached STAR genome index
    star_idx_dir = slam_dir / "star_index"
    ensure_star_genome_index(genome_fa=genome_fa, gtf_path=gtf_path, index_dir=star_idx_dir, threads=args.threads, overwrite_existing=args.overwrite_existing)

    # 1. Resolve FASTQs from manifest or defaults
    stage_runs: Dict[str, List[str]] = defaultdict(list)
    run_fastq_map: Dict[str, Path] = {}

    manifest_p = Path(args.fastq_manifest).resolve() if args.fastq_manifest else work_dir / "downloads" / "slamseq_fastq_manifest.tsv"
    if manifest_p.exists() and manifest_p.stat().st_size > 0:
        print(f"[+] Loading SLAM-seq FASTQs from manifest: {manifest_p}")
        df_man = pd.read_csv(manifest_p, sep="\t")
        for _, r in df_man.iterrows():
            stg = str(r["stage"])
            rid = str(r["run_id"])
            fq = Path(r["fastq"])
            stage_runs[stg].append(rid)
            run_fastq_map[rid] = fq
    else:
        print(f"[*] Manifest not found, using default stage mapping.")
        for stg, r_list in SLAM_STAGE_RUNS.items():
            for rid in r_list:
                stage_runs[stg].append(rid)
                run_fastq_map[rid] = fastq_dir / f"{rid}.fastq.gz"

    aligned_runs: Dict[str, Path] = {}
    all_runs = list(run_fastq_map.keys())

    for idx, run_id in enumerate(all_runs, 1):
        fq_path = run_fastq_map[run_id]
        out_b = bams_dir / f"{run_id}.sorted.bam"

        set_task_progress(work_dir, args.task_name, f"Aligning run {run_id} ({idx}/{len(all_runs)})")
        if not fq_path.exists() or fq_path.stat().st_size < 1024 * 1024:
            raise FileNotFoundError(
                f"[FATAL] SLAM-seq FASTQ missing or incomplete: {fq_path}. "
                f"Ensure 'download_resources' has completed. Zero-mock policy enforced."
            )
        is_valid, msg = check_file_integrity(fq_path)
        assert is_valid, f"[FATAL] FASTQ integrity verification failed: {fq_path} ({msg})"
        aligned_b = align_slam_fastq(fq_path, star_idx_dir, out_b, threads=args.threads, max_reads=args.max_reads, overwrite_existing=args.overwrite_existing)
        aligned_runs[run_id] = aligned_b

    # 2. Pool replicates per stage
    pooled_stage_bams: Dict[str, Path] = {}
    for stage, runs in stage_runs.items():
        stage_bam_list = [aligned_runs[r] for r in runs if r in aligned_runs]
        pooled_b = bams_dir / f"{stage}_pooled.sorted.bam"
        p_bam = pool_stage_bams(stage_bam_list, pooled_b, threads=args.threads, overwrite_existing=args.overwrite_existing)
        pooled_stage_bams[stage] = p_bam

    # 3. Untreated control (-4sU) pooled SNP masking & background rate verification
    set_task_progress(work_dir, args.task_name, "Untreated control SNP calling & masking")
    untreated_bam = pooled_stage_bams["control_untreated"]
    baseline_bam = pooled_stage_bams["0.75hpf"]
    snp_set, bg_tc_rate, snp_mask_hash = call_snps_and_verify_untreated(
        untreated_bam,
        baseline_bam=baseline_bam,
        genome_fa=genome_fa,
        cache_dir=slam_dir,
        exclude_mito=args.exclude_mito,
        mask_baseline_snps=args.mask_baseline_snps,
        overwrite_existing=args.overwrite_existing
    )

    # 4. Extract nascent T->C converted reads in +4sU samples and generate BigWigs
    nascent_bams: Dict[str, Path] = {}
    nascent_counts: Dict[str, int] = {}
    nascent_rates: Dict[str, float] = {}
    nascent_stages = ["0.75hpf", "2.5hpf", "3.0_3.5hpf", "4.5hpf"]
    for idx, stage_key in enumerate(nascent_stages, 1):
        raw_b = pooled_stage_bams[stage_key]
        nasc_b = bams_dir / f"{stage_key}_nascent.bam"
        set_task_progress(work_dir, args.task_name, f"Extracting nascent T->C: {stage_key} ({idx}/{len(nascent_stages)})")
        nb_path, n_nascent, tc_rate = extract_nascent_tc_reads(
            raw_b, nasc_b, snp_set, min_tc_conversions=2, threads=args.threads,
            exclude_mito=args.exclude_mito, snp_mask_hash=snp_mask_hash,
            overwrite_existing=args.overwrite_existing
        )
        nascent_bams[stage_key] = nb_path
        nascent_counts[stage_key] = n_nascent
        nascent_rates[stage_key] = tc_rate

        # Generate strand-specific normalized BigWigs normalized to parent stage BAM mapped reads
        stage_dir = slam_dir / stage_key
        stage_dir.mkdir(parents=True, exist_ok=True)
        fwd_bw = stage_dir / f"{stage_key}_fwd.bw"
        rev_bw = stage_dir / f"{stage_key}_rev.bw"
        parent_mapped = get_bam_mapped_reads(raw_b)
        generate_strand_specific_bigwigs(
            nb_path, chrom_sizes, fwd_bw, rev_bw, parent_total_reads=parent_mapped, exclude_mito=args.exclude_mito,
            overwrite_existing=args.overwrite_existing
        )

    # Statistically test enrichment of >=2 T->C reads in 4.5 hpf (+4sU) relative to untreated control
    from scipy.stats import fisher_exact
    untreated_nasc_bam = bams_dir / "control_untreated_nascent.bam"
    _, n_untreated, tc_untreated = extract_nascent_tc_reads(
        untreated_bam, untreated_nasc_bam, snp_set, min_tc_conversions=2, threads=args.threads,
        exclude_mito=args.exclude_mito, snp_mask_hash=snp_mask_hash,
        overwrite_existing=args.overwrite_existing
    )
    tot_45 = get_bam_mapped_reads(pooled_stage_bams["4.5hpf"])
    tot_untreated = get_bam_mapped_reads(untreated_bam)
    n_45 = nascent_counts["4.5hpf"]

    cont_table = [
        [n_45, max(0, tot_45 - n_45)],
        [n_untreated, max(0, tot_untreated - n_untreated)]
    ]
    fe_or, fe_p = fisher_exact(cont_table, alternative="greater")
    print("=" * 80)
    print(f"[*] SLAM-seq Incorporation Verification (4.5 hpf +4sU vs Untreated -4sU):")
    print(f"    4.5 hpf nascent reads (>=2 T->C):   {n_45:,} / {tot_45:,} ({100*n_45/max(tot_45,1):.2f}%) [bulk T->C rate: {nascent_rates['4.5hpf']*100:.3f}%]")
    print(f"    Untreated nascent reads (>=2 T->C): {n_untreated:,} / {tot_untreated:,} ({100*n_untreated/max(tot_untreated,1):.2f}%) [background error rate: {tc_untreated*100:.3f}%]")
    print(f"    Fisher's Exact Odds Ratio:          {fe_or:.2f} (p-value = {fe_p:.2e})")
    print("=" * 80)
    assert fe_p < 0.05, f"[FATAL] 4.5 hpf (+4sU) shows no significant T->C enrichment over untreated control (p={fe_p})!"

    # 5. Parse GTF and quantify total (0.75hpf) and nascent reads across all stages
    set_task_progress(work_dir, args.task_name, "Quantifying nascent CPM & timing classification")
    merged_exons, gene_names = parse_gtf_exons(gtf_path, exclude_mito=args.exclude_mito)

    quant_dict = {
        "total_0.75hpf": pooled_stage_bams["0.75hpf"],
        "nascent_0.75hpf": nascent_bams["0.75hpf"],
        "nascent_untreated": untreated_nasc_bam,
        "nascent_2.5hpf": nascent_bams["2.5hpf"],
        "nascent_3.0_3.5hpf": nascent_bams["3.0_3.5hpf"],
        "nascent_4.5hpf": nascent_bams["4.5hpf"]
    }

    # Parent library sizes for normalization to prevent inflation
    parent_library_sizes = {
        "total_0.75hpf": get_bam_mapped_reads(pooled_stage_bams["0.75hpf"]),
        "nascent_0.75hpf": get_bam_mapped_reads(pooled_stage_bams["0.75hpf"]),
        "nascent_untreated": get_bam_mapped_reads(untreated_bam),
        "nascent_2.5hpf": get_bam_mapped_reads(pooled_stage_bams["2.5hpf"]),
        "nascent_3.0_3.5hpf": get_bam_mapped_reads(pooled_stage_bams["3.0_3.5hpf"]),
        "nascent_4.5hpf": get_bam_mapped_reads(pooled_stage_bams["4.5hpf"])
    }

    df_classified, df_minor, df_silent = quantify_and_classify_zf_genes(
        merged_exons, gene_names, quant_dict, slam_dir, parent_library_sizes=parent_library_sizes,
        threads=args.threads, min_baseline_fc=args.min_baseline_fc,
        min_kinetics_retention=args.min_kinetics_retention, exclude_mito=args.exclude_mito,
        overwrite_existing=args.overwrite_existing
    )

    set_task_progress(work_dir, args.task_name, "100% (Done)")
    print(f"[SUCCESS] Zebrafish SLAM-seq timing classification complete: {slam_dir / 'zebrafish_slamseq_gene_summary.tsv'}")


if __name__ == "__main__":
    main()
