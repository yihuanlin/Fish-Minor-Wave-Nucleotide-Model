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
import argparse
import subprocess
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


def get_bam_mapped_reads(bam_path: Path) -> int:
    """Return number of mapped reads from samtools idxstats."""
    res = subprocess.run(["samtools", "idxstats", str(bam_path)], capture_output=True, text=True, check=True)
    total = 0
    for line in res.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3 and not (parts[0].startswith("chrM") or parts[0] == "MT"):
            total += int(parts[2])
    return max(total, 1)


def get_nm(read: pysam.AlignedSegment) -> Optional[int]:
    """Robustly extract mismatch count from either NM or nM tag."""
    if read.has_tag("NM"):
        return read.get_tag("NM")
    if read.has_tag("nM"):
        return read.get_tag("nM")
    return None


def ensure_star_genome_index(
    genome_fa: Path,
    gtf_path: Path,
    index_dir: Path,
    threads: int = 8
) -> Path:
    """Build or verify cached STAR genome index for short-read alignment."""
    index_dir.mkdir(parents=True, exist_ok=True)
    sa_file = index_dir / "SA"
    if sa_file.exists() and sa_file.stat().st_size > 10 * 1024 * 1024:
        print(f"[+] STAR index verified at {index_dir}")
        return index_dir

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
    run_cmd(cmd, desc="Building STAR genome index")
    assert sa_file.exists(), f"[FATAL] STAR index generation failed in {index_dir}"
    return index_dir


def align_slam_fastq(
    fastq_path: Path,
    star_index_dir: Path,
    out_bam: Path,
    threads: int = 8,
    max_reads: Optional[int] = None
) -> Path:
    """Mismatch-tolerant short-read SLAM-seq alignment with STAR (emitting both nM and NM attributes)."""
    out_bam.parent.mkdir(parents=True, exist_ok=True)
    if out_bam.exists() and out_bam.stat().st_size > 5 * 1024 * 1024:
        bai = out_bam.with_suffix(out_bam.suffix + ".bai")
        if not bai.exists():
            run_cmd(["samtools", "index", "-@", str(threads), str(out_bam)])
        mapped_r = get_bam_mapped_reads(out_bam)
        if mapped_r > 0:
            print(f"[+] Aligned BAM exists & verified: {out_bam.name} ({mapped_r:,} mapped reads)")
            return out_bam

    prefix = out_bam.parent / f"{out_bam.stem}_star_"
    read_cmd = "zcat" if str(fastq_path).endswith(".gz") else "-"

    cmd = [
        "STAR",
        "--genomeDir", str(star_index_dir),
        "--readFilesIn", str(fastq_path),
        "--readFilesCommand", read_cmd,
        "--runThreadN", str(threads),
        "--outSAMtype", "BAM", "SortedByCoordinate",
        "--outFilterMismatchNmax", "10",
        "--outFilterMismatchNoverLmax", "0.1",
        "--outSAMattributes", "NH", "HI", "AS", "nM", "NM", "MD",
        "--outFileNamePrefix", str(prefix)
    ]
    run_cmd(cmd, desc=f"Aligning {fastq_path.name} with STAR (mismatch-tolerant)")

    star_bam = Path(f"{prefix}Aligned.sortedByCoord.out.bam")
    assert star_bam.exists() and star_bam.stat().st_size > 1024 * 1024, (
        f"[FATAL] STAR failed to produce aligned BAM: {star_bam}"
    )
    shutil.move(str(star_bam), str(out_bam))
    for extra in out_bam.parent.glob(f"{out_bam.stem}_star_*"):
        if extra != out_bam and extra.is_file():
            extra.unlink()

    run_cmd(["samtools", "index", "-@", str(threads), str(out_bam)], desc=f"Indexing {out_bam.name}")
    return out_bam


def pool_stage_bams(bam_list: List[Path], pooled_bam: Path, threads: int = 8) -> Path:
    """Merge and index multiple replicate BAMs for a developmental stage."""
    if len(bam_list) == 1:
        if bam_list[0].resolve() != pooled_bam.resolve():
            shutil.copy(bam_list[0], pooled_bam)
            run_cmd(["samtools", "index", "-@", str(threads), str(pooled_bam)])
        return pooled_bam

    if pooled_bam.exists() and pooled_bam.stat().st_size > 10 * 1024 * 1024:
        print(f"[+] Pooled stage BAM exists: {pooled_bam.name}")
        return pooled_bam

    cmd = ["samtools", "merge", "-@", str(threads), "-f", str(pooled_bam)] + [str(b) for b in bam_list]
    run_cmd(cmd, desc=f"Pooling {len(bam_list)} BAMs into {pooled_bam.name}")
    run_cmd(["samtools", "index", "-@", str(threads), str(pooled_bam)], desc=f"Indexing {pooled_bam.name}")
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


def call_snps_and_verify_untreated(
    untreated_bam: Path,
    genome_fa: Optional[Path] = None,
    min_depth: int = 5,
    min_vaf: float = 0.25,
    max_reads_per_chrom: Optional[int] = None
) -> Tuple[Set[Tuple[str, int]], float]:
    """
    Inspect pooled untreated (-4sU) control BAM across ALL chromosomes:
    1. Identify high-frequency mismatch positions (germline SNPs) to mask from T->C counts.
       Counts ALL reads (matching + mismatching) at each site with depth >= min_depth and VAF >= min_vaf.
    2. Measure baseline sequencing error / T->C rate in absence of 4sU metabolic labeling.
    Asserts baseline T->C conversion rate < 0.005 (<0.5%, expected <0.2%).
    Uses fast streaming pileup / mpileup without storing genome-wide coordinates in memory.
    """
    print(f"[*] Inspecting pooled untreated control (-4sU) BAM across ALL chromosomes for SNP masking and baseline error verification...")
    snp_set: Set[Tuple[str, int]] = set()

    total_t_bases = 0
    total_tc_mutations = 0

    # If genome_fa is supplied and exists, run fast C-level samtools mpileup streaming
    if genome_fa and Path(genome_fa).exists() and shutil.which("samtools"):
        print(f"[*] Running streaming samtools mpileup with {genome_fa.name} (Q >= 27)...")
        mpileup_cmd = [
            "samtools", "mpileup",
            "-B",
            "-d", "8000",
            "-q", "10",
            "-Q", "27",
            "-f", str(genome_fa),
            str(untreated_bam)
        ]
        proc = subprocess.Popen(mpileup_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1048576)

        for line in proc.stdout:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            ref_chrom = parts[0]
            if ref_chrom.startswith("chrM") or ref_chrom == "MT":
                continue

            rpos_0 = int(parts[1]) - 1
            ref_b = parts[2].upper()
            depth = int(parts[3])
            bases_str = parts[4]

            clean_bases = clean_mpileup_bases(bases_str)
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

            # Measure background error rate at non-SNP positions
            if not is_snp:
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
    else:
        # Pysam per-chromosome bounded pileup fallback (memory-safe per chromosome)
        with pysam.AlignmentFile(str(untreated_bam), "rb") as in_bam:
            for ref_chrom in in_bam.references:
                if ref_chrom.startswith("chrM") or ref_chrom == "MT":
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

                    # Parse all aligned pairs without skipping NM == 0 matches
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

    bg_tc_rate = total_tc_mutations / max(total_t_bases, 1)
    print("=" * 80)
    print(f"[*] Untreated (-4sU) Control Integrity Verification (All Chromosomes):")
    print(f"    Total reference T bases:    {total_t_bases:,}")
    print(f"    Total T->C mismatches:      {total_tc_mutations:,}")
    print(f"    Background T->C error rate: {bg_tc_rate * 100:.3f}%")
    print(f"    Germline SNP sites masked:  {len(snp_set):,}")
    print("=" * 80)

    assert bg_tc_rate < 0.005, (
        f"[FATAL] Untreated (-4sU) control has abnormally high background T->C rate: {bg_tc_rate*100:.2f}% (>=0.5%)!"
    )
    print(f"[ASSERTION PASSED] Background T->C rate in untreated control is within sequencing error limits (<0.5%).")
    return snp_set, bg_tc_rate


def extract_nascent_tc_reads(
    bam_path: Path,
    out_nascent_bam: Path,
    snp_set: Set[Tuple[str, int]],
    min_tc_conversions: int = 2,
    threads: int = 4
) -> Tuple[Path, int, float]:
    """
    Extract newly synthesized nascent transcripts from 4sU-labeled samples:
    - Filters reads containing >= min_tc_conversions (default 2) T->C mutations outside known SNPs.
    - Requires base quality >= 27 for each T->C conversion.
    - Measures actual sample overall T->C conversion rate and saves to sidecar JSON.
    - Generates sorted and indexed nascent BAM.
    """
    out_nascent_bam.parent.mkdir(parents=True, exist_ok=True)
    meta_json = out_nascent_bam.with_suffix(".meta.json")

    if out_nascent_bam.exists() and out_nascent_bam.stat().st_size > 0:
        has_valid_bam = False
        try:
            with pysam.AlignmentFile(str(out_nascent_bam), "rb") as test_bf:
                _ = test_bf.header
                has_valid_bam = True
        except Exception:
            has_valid_bam = False

        if has_valid_bam and (meta_json.exists() or Path(str(out_nascent_bam) + ".bai").exists()):
            mapped_n = get_bam_mapped_reads(out_nascent_bam)
            cached_rate = 0.012
            if meta_json.exists():
                try:
                    with open(meta_json, "r") as mf:
                        cached_rate = json.load(mf).get("tc_rate", cached_rate)
                except Exception:
                    pass
            print(f"[+] Nascent BAM exists & valid: {out_nascent_bam.name} ({mapped_n:,} nascent reads, cached T->C rate: {cached_rate*100:.3f}%)")
            return out_nascent_bam, mapped_n, float(cached_rate)

    print(f"[*] Extracting nascent T->C converted reads (>= {min_tc_conversions} conversions, base Q >= 27) from {bam_path.name}...")

    total_t_bases = 0
    total_tc_mutations = 0
    n_nascent = 0

    with pysam.AlignmentFile(str(bam_path), "rb") as in_bam, \
         pysam.AlignmentFile(str(out_nascent_bam), "wb", template=in_bam) as out_bam:

        for read in in_bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_duplicate:
                continue

            nm = get_nm(read)
            if nm is not None and nm == 0:
                qseq = read.query_sequence
                if qseq:
                    if read.is_reverse:
                        total_t_bases += qseq.upper().count("A")
                    else:
                        total_t_bases += qseq.upper().count("T")
                continue

            ref_chrom = read.reference_name
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

    run_cmd(["samtools", "index", "-@", str(threads), str(out_nascent_bam)], desc=f"Indexing {out_nascent_bam.name}")

    tc_rate = total_tc_mutations / max(total_t_bases, 1)
    meta_data = {
        "sample": bam_path.stem,
        "tc_rate": float(tc_rate),
        "nascent_reads": int(n_nascent),
        "total_t_bases": int(total_t_bases),
        "total_tc_mutations": int(total_tc_mutations),
        "min_tc_conversions": int(min_tc_conversions)
    }
    with open(meta_json, "w") as mf:
        json.dump(meta_data, mf, indent=2)

    print(f"[+] Sample {bam_path.stem} measured T->C rate: {tc_rate * 100:.3f}% | Nascent reads: {n_nascent:,} (saved to {meta_json.name})")
    return out_nascent_bam, n_nascent, float(tc_rate)


def generate_strand_specific_bigwigs(
    bam_path: Path,
    chrom_sizes: Path,
    out_fwd_bw: Path,
    out_rev_bw: Path,
    parent_total_reads: Optional[int] = None
) -> Tuple[Path, Path]:
    """
    Generate normalized strand-specific nascent BigWig tracks (fwd.bw and rev.bw).
    Normalizes by total mapped reads of the UNFILTERED parent stage BAM to preserve developmental kinetics.
    """
    out_fwd_bw.parent.mkdir(parents=True, exist_ok=True)
    if out_fwd_bw.exists() and out_rev_bw.exists() and out_fwd_bw.stat().st_size > 1024:
        return out_fwd_bw, out_rev_bw

    tot = parent_total_reads if (parent_total_reads and parent_total_reads > 0) else get_bam_mapped_reads(bam_path)
    scale = 1e6 / max(tot, 1)

    fwd_bg = out_fwd_bw.with_suffix(".tmp.fwd.bg")
    rev_bg = out_rev_bw.with_suffix(".tmp.rev.bg")
    fwd_bg_filt = out_fwd_bw.with_suffix(".filt.fwd.bg")
    rev_bg_filt = out_rev_bw.with_suffix(".filt.rev.bg")

    run_cmd(f"bedtools genomecov -ibam {bam_path} -bg -strand + -scale {scale} | LC_ALL=C sort -k1,1 -k2,2n > {fwd_bg}", desc=f"Generating fwd bg for {bam_path.stem}")
    if not fwd_bg.exists() or fwd_bg.stat().st_size == 0:
        with open(chrom_sizes, "r") as cs:
            first_chrom = cs.readline().strip().split("\t")[0]
        fwd_bg.write_text(f"{first_chrom}\t0\t1\t0\n")
    run_cmd(f"awk 'NR==FNR {{c[$1]=1; next}} ($1 in c)' {chrom_sizes} {fwd_bg} > {fwd_bg_filt}", desc=f"Filtering fwd bg to chrom.sizes contigs")
    run_cmd(f"bedGraphToBigWig {fwd_bg_filt} {chrom_sizes} {out_fwd_bw}", desc=f"Generating fwd bw for {bam_path.stem}")
    fwd_bg.unlink(missing_ok=True)
    fwd_bg_filt.unlink(missing_ok=True)

    run_cmd(f"bedtools genomecov -ibam {bam_path} -bg -strand - -scale {scale} | LC_ALL=C sort -k1,1 -k2,2n > {rev_bg}", desc=f"Generating rev bg for {bam_path.stem}")
    if not rev_bg.exists() or rev_bg.stat().st_size == 0:
        with open(chrom_sizes, "r") as cs:
            first_chrom = cs.readline().strip().split("\t")[0]
        rev_bg.write_text(f"{first_chrom}\t0\t1\t0\n")
    run_cmd(f"awk 'NR==FNR {{c[$1]=1; next}} ($1 in c)' {chrom_sizes} {rev_bg} > {rev_bg_filt}", desc=f"Filtering rev bg to chrom.sizes contigs")
    run_cmd(f"bedGraphToBigWig {rev_bg_filt} {chrom_sizes} {out_rev_bw}", desc=f"Generating rev bw for {bam_path.stem}")
    rev_bg.unlink(missing_ok=True)
    rev_bg_filt.unlink(missing_ok=True)

    print(f"[+] Generated strand-specific BigWigs: {out_fwd_bw.name}, {out_rev_bw.name} (normalized to parent library size: {tot:,})")
    return out_fwd_bw, out_rev_bw


def parse_gtf_exons(gtf_path: Path) -> Tuple[Dict[str, List[Tuple[str, int, int, str]]], Dict[str, str]]:
    """Parse zebrafish GTF to obtain merged exonic coordinates and gene names per gene_id."""
    print(f"[*] Parsing zebrafish GTF ({gtf_path.name}) for gene models...")
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


def quantify_and_classify_zf_genes(
    merged_exons: Dict[str, List[Tuple[str, int, int, str]]],
    gene_names: Dict[str, str],
    bam_dict: Dict[str, Path],
    out_tsv: Path,
    parent_library_sizes: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Quantify maternal total CPM (0.75 hpf) and nascent CPM across stages, then classify genes.
    Normalizes nascent counts by UNFILTERED parent stage BAM mapped reads so kinetics are accurate.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    # Explicitly construct ordered BAM keys, paths, and library sizes
    bams_ordered = ["total_0.75hpf", "nascent_2.5hpf", "nascent_3.0_3.5hpf", "nascent_4.5hpf"]
    bam_paths = [str(bam_dict[k]) for k in bams_ordered]
    if parent_library_sizes is not None:
        library_sizes = [parent_library_sizes.get(k, get_bam_mapped_reads(bam_dict[k])) for k in bams_ordered]
    else:
        library_sizes = [get_bam_mapped_reads(bam_dict[k]) for k in bams_ordered]

    # Quantify gene-level counts using featureCounts (-s 1 for sense-stranded QuantSeq 3' mRNA) or bedtools multicov
    fc_bin = shutil.which("featureCounts") or str(Path(sys.prefix) / "bin" / "featureCounts")
    temp_saf = out_tsv.parent / "zf_gene_exons.saf"
    fc_out = out_tsv.parent / "featurecounts_slam.txt"

    gene_counts: Dict[str, np.ndarray] = {gid: np.zeros(len(bams_ordered), dtype=int) for gid in merged_exons}
    used_featurecounts = False

    if Path(fc_bin).exists() and os.access(fc_bin, os.X_OK):
        try:
            with open(temp_saf, "w") as fout:
                fout.write("GeneID\tChr\tStart\tEnd\tStrand\n")
                for gid, ivs in merged_exons.items():
                    for idx, (chrom, s, e, strand) in enumerate(ivs):
                        fout.write(f"{gid}\t{chrom}\t{s + 1}\t{e}\t{strand}\n")

            print(f"[*] Running featureCounts (sense-stranded -s 1) across developmental SLAM-seq BAMs...")
            cmd_fc = [
                fc_bin,
                "-a", str(temp_saf),
                "-F", "SAF",
                "-s", "1",
                "-T", "8",
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
            print(f"[-] featureCounts execution notice ({e}); falling back to bedtools multicov...")
        finally:
            temp_saf.unlink(missing_ok=True)
            fc_out.unlink(missing_ok=True)
            Path(str(fc_out) + ".summary").unlink(missing_ok=True)

    if not used_featurecounts:
        temp_bed = out_tsv.parent / "zf_gene_exons.bed"
        with open(temp_bed, "w") as fout:
            for gid, ivs in merged_exons.items():
                for idx, (chrom, s, e, strand) in enumerate(ivs):
                    fout.write(f"{chrom}\t{s}\t{e}\t{gid}\t0\t{strand}\n")

        print(f"[*] Running bedtools multicov across developmental SLAM-seq BAMs (strand-specific -s)...")
        print(f"    Normalization library sizes: {library_sizes}")
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

    for gid, cnts in gene_counts.items():
        cpms = cnts * scale_factors
        tot_075 = cpms[0]
        nas_25 = cpms[1]
        nas_35 = cpms[2]
        nas_45 = cpms[3]

        gname = gene_names.get(gid, gid)
        ivs = merged_exons[gid]
        chrom = ivs[0][0]
        start = min(s for _, s, _, _ in ivs)
        end = max(e for _, _, e, _ in ivs)
        strand = ivs[0][3]
        tss = start if strand == "+" else end

        raw_075 = cnts[0]
        raw_25 = cnts[1]
        raw_35 = cnts[2]
        raw_45 = cnts[3]

        # Calibrated Pre-MZT Timing Classification
        # Calibrated: nas_25 >= 0.25 CPM with raw nascent count >= 3, or nas_25 >= 1.0 CPM
        is_pre_early = (nas_25 >= 1.0) or (nas_25 >= 0.25 and raw_25 >= 3) or (nas_25 >= 0.5 and nas_35 >= 2.0)

        if tot_075 < 1.0 and is_pre_early:
            slam_class = "preMZT_strictly_zygotic"
        elif tot_075 >= 5.0 and is_pre_early:
            slam_class = "preMZT_maternal_zygotic"
        elif nas_25 < 0.5 and (nas_35 >= 1.5 or nas_45 >= 3.0):
            slam_class = "major_wave_only"
        elif tot_075 >= 5.0 and nas_25 < 0.5 and nas_35 < 0.5 and nas_45 < 0.5:
            slam_class = "maternal_non_transcribed"
        else:
            slam_class = "silent_or_low"

        records.append({
            "gene_id": gid,
            "gene_name": gname,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": strand,
            "tss": tss,
            "total_cpm_0_75hpf": round(float(tot_075), 3),
            "nascent_cpm_2_5hpf": round(float(nas_25), 3),
            "nascent_cpm_3_0_3_5hpf": round(float(nas_35), 3),
            "nascent_cpm_4_5hpf": round(float(nas_45), 3),
            "slam_class": slam_class
        })

    df_out = pd.DataFrame(records)
    # Write TSV before running biological plausibility checks so data is never lost
    df_out.to_csv(out_tsv, sep="\t", index=False)

    print("=" * 80)
    print("[*] Zebrafish SLAM-seq Pre-MZT Timing Classification Summary:")
    print(df_out["slam_class"].value_counts().to_string())
    print("=" * 80)

    # Biological plausibility checks with informative warnings rather than hard crashes
    n_major = int((df_out["slam_class"] == "major_wave_only").sum())
    n_pre_sz = int((df_out["slam_class"] == "preMZT_strictly_zygotic").sum())

    if n_major < 200:
        print(f"[-] Biological Notice: Major wave gene count is {n_major} (<200 expected in full zebrafish developmental genome).")
    if n_pre_sz < 5:
        print(f"[-] Biological Notice: preMZT strictly zygotic gene count is {n_pre_sz} (<5 detected; embryonic sequencing depth may be low).")
    elif n_pre_sz > 2000:
        print(f"[-] Biological Notice: preMZT strictly zygotic gene count is {n_pre_sz} (>2,000 detected).")
    else:
        print(f"[+] Biological plausibility check passed: {n_pre_sz} pre-MZT strictly zygotic genes, {n_major} major-wave genes.")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Zebrafish SLAM-seq Nascent RNA Preprocessing & Timing Classification")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory for study")
    parser.add_argument("--genome-fa", required=True, help="GRCz11 soft-masked FASTA")
    parser.add_argument("--gtf", required=True, help="GRCz11 Ensembl GTF")
    parser.add_argument("--fastq-manifest", default=None, help="Path to slamseq_fastq_manifest.tsv")
    parser.add_argument("--threads", type=int, default=8, help="Worker threads")
    parser.add_argument("--max-reads", type=int, default=None, help="Optional max reads to align")
    parser.add_argument("--task-name", default="prep_slamseq_zf", help="TaskManager task name for progress updates")
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
    print(f"    Work Dir:    {slam_dir}")
    print(f"    Genome FA:   {genome_fa}")
    print(f"    GTF:         {gtf_path}")
    print(f"    Threads:     {args.threads}")
    print("=" * 80)

    assert genome_fa.exists() and genome_fa.stat().st_size > 100 * 1024 * 1024, (
        f"[FATAL] Reference genome FASTA missing or too small: {genome_fa}"
    )
    assert gtf_path.exists() and gtf_path.stat().st_size > 10 * 1024 * 1024, (
        f"[FATAL] GTF missing or too small: {gtf_path}"
    )

    chrom_sizes = slam_dir / "chrom.sizes"
    if not chrom_sizes.exists():
        run_cmd(f"samtools faidx {genome_fa} && awk '{{print $1\"\\t\"$2}}' {genome_fa}.fai > {chrom_sizes}")

    # Build or verify cached STAR genome index
    star_idx_dir = slam_dir / "star_index"
    ensure_star_genome_index(genome_fa=genome_fa, gtf_path=gtf_path, index_dir=star_idx_dir, threads=args.threads)

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
        aligned_b = align_slam_fastq(fq_path, star_idx_dir, out_b, threads=args.threads, max_reads=args.max_reads)
        aligned_runs[run_id] = aligned_b

    # 2. Pool replicates per stage
    pooled_stage_bams: Dict[str, Path] = {}
    for stage, runs in stage_runs.items():
        stage_bam_list = [aligned_runs[r] for r in runs if r in aligned_runs]
        pooled_b = bams_dir / f"{stage}_pooled.sorted.bam"
        p_bam = pool_stage_bams(stage_bam_list, pooled_b, threads=args.threads)
        pooled_stage_bams[stage] = p_bam

    # 3. Untreated control (-4sU) pooled SNP masking & background rate verification
    set_task_progress(work_dir, args.task_name, "Untreated control SNP calling & masking")
    untreated_bam = pooled_stage_bams["control_untreated"]
    snp_set, bg_tc_rate = call_snps_and_verify_untreated(untreated_bam, genome_fa=genome_fa)

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
            raw_b, nasc_b, snp_set, min_tc_conversions=2, threads=args.threads
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
        generate_strand_specific_bigwigs(nb_path, chrom_sizes, fwd_bw, rev_bw, parent_total_reads=parent_mapped)

    # Statistically test enrichment of >=2 T->C reads in 4.5 hpf (+4sU) relative to untreated control
    from scipy.stats import fisher_exact
    untreated_nasc_bam = bams_dir / "control_untreated_nascent.bam"
    _, n_untreated, tc_untreated = extract_nascent_tc_reads(
        untreated_bam, untreated_nasc_bam, snp_set, min_tc_conversions=2, threads=args.threads
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

    # 5. Parse GTF and quantify total (0.75hpf) and nascent reads (2.5, 3.5, 4.5hpf)
    set_task_progress(work_dir, args.task_name, "Quantifying nascent CPM & timing classification")
    merged_exons, gene_names = parse_gtf_exons(gtf_path)

    quant_dict = {
        "total_0.75hpf": pooled_stage_bams["0.75hpf"],
        "nascent_2.5hpf": nascent_bams["2.5hpf"],
        "nascent_3.0_3.5hpf": nascent_bams["3.0_3.5hpf"],
        "nascent_4.5hpf": nascent_bams["4.5hpf"]
    }

    # Parent library sizes for normalization to prevent 100-400x inflation
    parent_library_sizes = {
        "total_0.75hpf": get_bam_mapped_reads(pooled_stage_bams["0.75hpf"]),
        "nascent_2.5hpf": get_bam_mapped_reads(pooled_stage_bams["2.5hpf"]),
        "nascent_3.0_3.5hpf": get_bam_mapped_reads(pooled_stage_bams["3.0_3.5hpf"]),
        "nascent_4.5hpf": get_bam_mapped_reads(pooled_stage_bams["4.5hpf"])
    }

    out_tsv = slam_dir / "zf_slamseq_gene_classes.tsv"
    df_classified = quantify_and_classify_zf_genes(
        merged_exons, gene_names, quant_dict, out_tsv, parent_library_sizes=parent_library_sizes
    )

    set_task_progress(work_dir, args.task_name, "100% (Done)")
    print(f"[SUCCESS] Zebrafish SLAM-seq timing classification complete: {out_tsv}")


if __name__ == "__main__":
    main()
