#!/usr/bin/env python3
"""
scripts/download_resources.py
=============================
Consolidated, Background-Ready Resource Downloader & Integrity Verifier:
- Gathers all external resources required by the pipeline into a single entry point.
- Driven strictly by study configuration and Block 0 settings input (only downloads
  what is actually needed according to the settings).
- Fast pre-download integrity checks: skips download if valid file already exists.
- Host-aware parallel downloading: downloads from up to 3 distinct host websites
  in parallel, with serial, single-connection downloading within each host to prevent
  server throttling or connection drops.
- Supports ENA Portal API resolution and deterministic EBI FTP URL construction for
  Zebrafish ATAC-seq (GSE130944), Medaka ATAC-seq (PRJDB7492), and SLAM-seq (PRJNA768450).
- Real-time progress logging to stdout and persistent TaskManager progress.txt.
- Rigorous post-download integrity verification (size, gzip stream, format checks).
- Automated cleanup of partial (.part, .aria2) or corrupted files.
- Generates atac_fastq_manifest.tsv, slamseq_fastq_manifest.tsv, and resource_manifest.tsv.
"""

import os
import sys
import re
import json
import gzip
import shutil
import argparse
import subprocess
import threading
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Ensure repo root and conda bin are in paths
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import pandas as pd

# Line buffering for real-time progress logging (guarded for Jupyter OutStream compatibility)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

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


# ==============================================================================
# Public Resource URLs & Definitions
# ==============================================================================
JASPAR_2024_CORE_VERTEBRATES_URL = (
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
)

# Zebrafish GRCz11 (Ensembl Release 112)
ZEBRAFISH_ENSEMBL_FASTA_URL = (
    "https://ftp.ensembl.org/pub/release-112/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna_sm.primary_assembly.fa.gz"
)
ZEBRAFISH_ENSEMBL_GTF_URL = (
    "https://ftp.ensembl.org/pub/release-112/gtf/danio_rerio/Danio_rerio.GRCz11.112.gtf.gz"
)

# Medaka ASM223467v1 (Ensembl Release 112)
MEDAKA_ENSEMBL_FASTA_URL = (
    "https://ftp.ensembl.org/pub/release-112/fasta/oryzias_latipes/dna/Oryzias_latipes.ASM223467v1.dna_sm.toplevel.fa.gz"
)
MEDAKA_ENSEMBL_GTF_URL = (
    "https://ftp.ensembl.org/pub/release-112/gtf/oryzias_latipes/Oryzias_latipes.ASM223467v1.112.gtf.gz"
)

# Astatotilapia calliptera fAstCal1.3 (Ensembl Release 112)
ASTATOTILAPIA_112_FASTA_URL = (
    "https://ftp.ensembl.org/pub/release-112/fasta/astatotilapia_calliptera/dna/Astatotilapia_calliptera.fAstCal1.3.dna_sm.toplevel.fa.gz"
)
ASTATOTILAPIA_112_GTF_URL = (
    "https://ftp.ensembl.org/pub/release-112/gtf/astatotilapia_calliptera/Astatotilapia_calliptera.fAstCal1.3.112.gtf.gz"
)


# ==============================================================================
# SRA / ENA Deterministic URL Generator & Accession Mappings
# ==============================================================================
# Known GSM -> SRR mapping table
GEO_TO_SRR = {
    # Zebrafish ATAC-seq (GSE130944, Palfy et al. 2020)
    "GSM3756599": ["SRR9032650"],  # 256-cell rep 1
    "GSM3756600": ["SRR9032651"],  # 256-cell rep 2
    "GSM3756603": ["SRR9032658"],  # dome rep 1
    "GSM3756604": ["SRR9032659"],  # dome rep 2
    "GSM3756605": ["SRR9032660", "SRR9032661"],  # dome rep 3
    "SRR9032640": ["SRR9032640"],  # naked gDNA Tn5 control
    # Medaka ATAC-seq (PRJDB7492, Nakamura et al. 2021)
    "DRX143405": ["DRR152711"],  # st9 rep 1
    "DRX143406": ["DRR152712"],  # st9 rep 2
    "DRX143407": ["DRR152713"],  # st9 rep 3
    "DRX143412": ["DRR152718"],  # st11 rep 1
    "DRX143413": ["DRR152719"],  # st11 rep 2
    # Zebrafish SLAM-seq (PRJNA768450, Bhat et al. 2023)
    "GSM5610039": ["SRR16195447"],  # 0.75hpf rep 1
    "GSM5610040": ["SRR16195448"],  # 0.75hpf rep 2
    "GSM5610043": ["SRR16195451"],  # 2.5hpf pre-MZT rep 1
    "GSM5610044": ["SRR16195452"],  # 2.5hpf pre-MZT rep 2
    "GSM5610045": ["SRR16195453"],  # 3.0-3.5hpf onset rep 1
    "GSM5610046": ["SRR16195454"],  # 3.0-3.5hpf onset rep 2
    "GSM5610047": ["SRR16195455"],  # 3.0-3.5hpf onset rep 3
    "GSM5610048": ["SRR16195456"],  # 3.0-3.5hpf onset rep 4
    "GSM5610051": ["SRR16195459"],  # 4.5hpf major wave rep 1
    "GSM5610052": ["SRR16195460"],  # 4.5hpf major wave rep 2
    "GSM5610075": ["SRR16195483"],  # -4sU control rep 1
    "GSM5610076": ["SRR16195484"],  # -4sU control rep 2
    "GSM5610077": ["SRR16195485"],  # -4sU control rep 3
}

STAGE_BY_PARENT_ACCESSION = {
    # Zebrafish ATAC-seq (GSE130944)
    "GSM3756599": "early_256cell",
    "GSM3756600": "early_256cell",
    "GSM3756603": "late_dome",
    "GSM3756604": "late_dome",
    "GSM3756605": "late_dome",
    "SRR9032640": "control_gdna",
    # Medaka ATAC-seq (PRJDB7492)
    "DRX143405": "early_st9",
    "DRX143406": "early_st9",
    "DRX143407": "early_st9",
    "DRX143412": "late_st11",
    "DRX143413": "late_st11",
    # Zebrafish SLAM-seq (PRJNA768450)
    "GSM5610039": "0.75hpf",
    "GSM5610040": "0.75hpf",
    "GSM5610043": "2.5hpf",
    "GSM5610044": "2.5hpf",
    "GSM5610045": "3.0_3.5hpf",
    "GSM5610046": "3.0_3.5hpf",
    "GSM5610047": "3.0_3.5hpf",
    "GSM5610048": "3.0_3.5hpf",
    "GSM5610051": "4.5hpf",
    "GSM5610052": "4.5hpf",
    "GSM5610075": "control_untreated",
    "GSM5610076": "control_untreated",
    "GSM5610077": "control_untreated",
}


def sra_run_to_urls(run_id: str, is_paired: bool = True) -> List[str]:
    """
    Construct deterministic ENA / SRA HTTP URLs from run accession.
    Standard EBI directory tree:
    - 6-digit number: /vol1/fastq/<prefix6>/<run_id>/
    - 7-digit number: /vol1/fastq/<prefix6>/00<last_1>/<run_id>/
    - 8-digit number: /vol1/fastq/<prefix6>/0<last_2>/<run_id>/
    - 9-digit number: /vol1/fastq/<prefix6>/<last_3>/<run_id>/
    """
    run_id = run_id.strip()
    prefix = run_id[:6]
    digits_len = len(run_id) - 3  # length of digits part
    if digits_len == 6:
        subdir = ""
    elif digits_len == 7:
        subdir = f"00{run_id[-1]}/"
    elif digits_len == 8:
        subdir = f"0{run_id[-2:]}/"
    elif digits_len == 9:
        subdir = f"{run_id[-3:]}/"
    else:
        subdir = ""

    base = f"http://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/{subdir}{run_id}/{run_id}"
    if is_paired:
        return [f"{base}_1.fastq.gz", f"{base}_2.fastq.gz"]
    else:
        return [f"{base}.fastq.gz"]


def query_ena_portal_api(acc: str, timeout: int = 6) -> List[Dict[str, Any]]:
    """
    Query ENA Portal API (filereport or search) for run_accession, library_layout, and fastq_ftp.
    Returns list of dicts: [{'run_accession': ..., 'library_layout': 'PAIRED'|'SINGLE', 'urls': [...]}]
    """
    import urllib.request
    import urllib.parse

    acc = acc.strip()
    results = []

    # 1. Try filereport directly
    url_fr = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={urllib.parse.quote(acc)}&result=read_run&fields=run_accession,library_layout,fastq_ftp&format=tsv"
    try:
        req = urllib.request.Request(url_fr, headers={"User-Agent": "Mozilla/5.0 (PreMZT-Pipeline)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8").strip()
            lines = text.split("\n")
            if len(lines) > 1 and "run_accession" in lines[0]:
                headers = lines[0].split("\t")
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    row = dict(zip(headers, parts))
                    run_id = row.get("run_accession")
                    if run_id:
                        layout = row.get("library_layout", "").upper()
                        fastq_ftp = row.get("fastq_ftp", "")
                        urls = []
                        if fastq_ftp:
                            for u in fastq_ftp.split(";"):
                                u = u.strip()
                                if u:
                                    if not u.startswith("http://") and not u.startswith("https://"):
                                        u = "https://" + u
                                    urls.append(u)
                        is_paired = (layout == "PAIRED") or len(urls) > 1
                        results.append({
                            "run_accession": run_id,
                            "library_layout": "PAIRED" if is_paired else "SINGLE",
                            "is_paired": is_paired,
                            "urls": urls
                        })
                if results:
                    return results
    except Exception:
        pass

    # 2. Try search endpoint (e.g. for GSM accessions)
    query_str = f'accession="{acc}" OR sample_accession="{acc}" OR experiment_accession="{acc}" OR secondary_sample_accession="{acc}" OR sample_alias="{acc}"'
    url_sr = f"https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query={urllib.parse.quote(query_str)}&fields=run_accession,library_layout,fastq_ftp&format=tsv"
    try:
        req = urllib.request.Request(url_sr, headers={"User-Agent": "Mozilla/5.0 (PreMZT-Pipeline)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8").strip()
            lines = text.split("\n")
            if len(lines) > 1 and "run_accession" in lines[0]:
                headers = lines[0].split("\t")
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    row = dict(zip(headers, parts))
                    run_id = row.get("run_accession")
                    if run_id:
                        layout = row.get("library_layout", "").upper()
                        fastq_ftp = row.get("fastq_ftp", "")
                        urls = []
                        if fastq_ftp:
                            for u in fastq_ftp.split(";"):
                                u = u.strip()
                                if u:
                                    if not u.startswith("http://") and not u.startswith("https://"):
                                        u = "https://" + u
                                    urls.append(u)
                        is_paired = (layout == "PAIRED") or len(urls) > 1
                        results.append({
                            "run_accession": run_id,
                            "library_layout": "PAIRED" if is_paired else "SINGLE",
                            "is_paired": is_paired,
                            "urls": urls
                        })
                if results:
                    return results
    except Exception:
        pass

    return []


def resolve_ena_fastqs(
    accessions: Union[List[str], Dict[str, List[str]]],
    default_paired: bool = True,
    cache_path: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve accessions (GSM, SRR, DRR, DRX, SRX) into run accessions and FASTQ URLs.
    Performs dynamic lookup via the ENA Portal API (filereport / search) to fetch true
    run_accession, library_layout (PAIRED vs SINGLE), and fastq_ftp URLs,
    keeping GEO_TO_SRR only as an offline fallback.
    Caches lookups in ena_resolution_cache.json to avoid redundant serial HTTP queries.
    Never blindly assumes DR* is single-end—uses ENA's actual library_layout / file count.
    Supports input as a list of accessions or a stage-mapped dict {'early': [...], 'late': [...]}.
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    stage_assigned: Dict[str, str] = {}

    cache: Dict[str, List[Dict[str, Any]]] = {}
    c_path = Path(cache_path).resolve() if cache_path else Path("./work_astatotilapia_premzt/downloads/ena_resolution_cache.json")
    if c_path.exists():
        try:
            with open(c_path, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    acc_list = []
    if isinstance(accessions, dict):
        for stage, accs in accessions.items():
            if isinstance(accs, str):
                accs = [accs]
            for a in accs:
                acc_list.append(a)
                stage_assigned[a] = stage
    else:
        acc_list = list(accessions)

    cache_dirty = False
    for acc in acc_list:
        acc = acc.strip()
        if not acc:
            continue
        stage = stage_assigned.get(acc)

        ena_runs = None
        if acc in cache:
            ena_runs = cache[acc]
        else:
            ena_runs = query_ena_portal_api(acc)
            if ena_runs:
                cache[acc] = ena_runs
                cache_dirty = True

        if ena_runs:
            for r_item in ena_runs:
                run_id = r_item["run_accession"]
                is_paired = r_item["is_paired"]
                urls = r_item["urls"]
                if not urls:
                    urls = sra_run_to_urls(run_id, is_paired=is_paired)
                resolved[run_id] = {
                    "parent_accession": acc,
                    "run_accession": run_id,
                    "is_paired": is_paired,
                    "library_layout": "PAIRED" if is_paired else "SINGLE",
                    "urls": urls,
                    "stage": stage
                }
        else:
            # Offline fallback via GEO_TO_SRR
            runs = GEO_TO_SRR.get(acc, [acc])
            is_paired = default_paired
            if "16195" in str(runs):
                is_paired = False
            for run in runs:
                urls = sra_run_to_urls(run, is_paired=is_paired)
                resolved[run] = {
                    "parent_accession": acc,
                    "run_accession": run,
                    "is_paired": is_paired,
                    "library_layout": "PAIRED" if is_paired else "SINGLE",
                    "urls": urls,
                    "stage": stage
                }

    if cache_dirty:
        try:
            c_path.parent.mkdir(parents=True, exist_ok=True)
            with open(c_path, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    return resolved


@dataclass
class DownloadTask:
    """Represents a discrete resource download item."""
    key: str
    label: str
    url: str
    dest_path: Path
    min_size_bytes: int
    domain: str
    post_action: Optional[str] = None  # e.g. 'decompress_fa'


# ==============================================================================
# File Integrity Verification
# ==============================================================================
def check_file_integrity(
    file_path: Path | str,
    min_size_bytes: int = 1024,
    fast: bool = True
) -> Tuple[bool, str]:
    """
    Rapidly verify the physical and structural integrity of a file.

    Args:
        file_path: Path to the target file.
        min_size_bytes: Minimum expected size in bytes.
        fast: If True, tests file headers and sample records; if False, performs
              deeper decompression/checksum verification.

    Returns:
        (is_valid: bool, status_message: str)
    """
    p = Path(file_path)
    if not p.exists():
        return False, "File does not exist"

    sz = p.stat().st_size
    if sz < min_size_bytes:
        return False, f"File too small ({sz} < {min_size_bytes} bytes)"

    # Gzip archive verification
    if p.name.endswith(".gz"):
        try:
            with open(p, "rb") as f:
                magic = f.read(2)
                if magic != b"\x1f\x8b":
                    return False, f"Invalid gzip magic bytes: {magic.hex()}"
        except Exception as e:
            return False, f"Cannot read file header: {e}"

        # Decompression check
        try:
            with gzip.open(p, "rb") as gz:
                buffer = gz.read(64 * 1024)
                if len(buffer) == 0:
                    return False, "Gzip archive decompressed to empty stream"
        except Exception as e:
            return False, f"Gzip stream decompression failed: {e}"

        # Format-specific payload checks
        if p.name.endswith((".fastq.gz", ".fq.gz")):
            try:
                with gzip.open(p, "rt", errors="replace") as gz:
                    for rec_idx in range(5):
                        h = gz.readline()
                        s = gz.readline()
                        p_ = gz.readline()
                        q = gz.readline()
                        if not h:
                            break
                        if not h.startswith("@"):
                            return False, f"FASTQ record {rec_idx} header missing '@'"
                        if not p_.startswith("+"):
                            return False, f"FASTQ record {rec_idx} separator missing '+'"
                        if len(s.strip()) != len(q.strip()):
                            return False, f"FASTQ record {rec_idx} sequence/quality length mismatch"
            except Exception as e:
                return False, f"FASTQ record parse error: {e}"

            if not fast:
                res = subprocess.run(["gzip", "-t", str(p)], capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    return False, f"gzip -t check failed: {res.stderr.strip()}"

        elif p.name.endswith(".gtf.gz"):
            try:
                with gzip.open(p, "rt", errors="replace") as gz:
                    valid_rows = 0
                    for line in gz:
                        if line.startswith("#"):
                            continue
                        fields = line.strip().split("\t")
                        if len(fields) < 8:
                            return False, f"Invalid GTF line with {len(fields)} columns"
                        valid_rows += 1
                        if valid_rows >= 10:
                            break
                    if valid_rows == 0:
                        return False, "No valid GTF feature lines found"
            except Exception as e:
                return False, f"GTF parse error: {e}"

    # Uncompressed FASTA verification
    elif p.name.endswith((".fa", ".fasta")):
        try:
            with open(p, "r", errors="replace") as f:
                first_char = ""
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s:
                        first_char = s[0]
                        break
                if first_char != ">":
                    return False, f"FASTA missing initial '>' identifier (found: '{first_char}')"
        except Exception as e:
            return False, f"FASTA read error: {e}"

    # MEME motif database verification
    elif p.name.endswith(".meme"):
        try:
            with open(p, "r", errors="replace") as f:
                content = f.read(2048)
                if "MEME version" not in content:
                    return False, "File missing 'MEME version' declaration"
                if "ALPHABET=" not in content and "MOTIF" not in content:
                    return False, "File missing MEME alphabet or motif records"
        except Exception as e:
            return False, f"MEME read error: {e}"

    elif p.name.endswith((".tsv", ".csv")):
        try:
            sep = "\t" if p.name.endswith(".tsv") else ","
            with open(p, "r", errors="replace") as f:
                header = f.readline().strip()
                if not header or sep not in header:
                    return False, "Invalid or missing tabular header"
        except Exception as e:
            return False, f"Tabular read error: {e}"

    return True, "OK"


def cleanup_failed_downloads(target_path: Path) -> None:
    """Safely remove leftover partial or corrupted download files."""
    part_file = target_path.with_suffix(target_path.suffix + ".part")
    aria2_file = Path(str(target_path) + ".aria2")
    part_aria2 = Path(str(part_file) + ".aria2")

    for f in [part_file, aria2_file, part_aria2]:
        if f.exists():
            try:
                f.unlink()
                print(f"[*] Cleaned up leftover temporary file: {f.name}", flush=True)
            except Exception:
                pass


# ==============================================================================
# Multi-Domain Progress Tracker
# ==============================================================================
class MultiDomainProgressTracker:
    """Thread-safe coordinator for tracking download progress across concurrent host domains."""

    def __init__(self, total_tasks: int, work_dir: Path, task_name: str, task_manager: Optional[Any] = None):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.work_dir = work_dir
        self.task_name = task_name
        self.task_manager = task_manager
        self.lock = threading.Lock()
        self.active_domain_status: Dict[str, str] = {}
        self.errors: List[str] = []

    def update_domain_progress(self, domain: str, status_str: str) -> None:
        with self.lock:
            self.active_domain_status[domain] = status_str
            active_items = [f"{d}: {s}" for d, s in sorted(self.active_domain_status.items()) if s]
            active_summary = " | ".join(active_items) if active_items else "Idle"
            prog_str = f"[{self.completed_tasks}/{self.total_tasks} done] {active_summary}"
            set_task_progress(self.work_dir, self.task_name, prog_str)
            if self.task_manager:
                try:
                    self.task_manager.update_progress(self.task_name, prog_str)
                except Exception:
                    pass

    def mark_task_complete(self, domain: str, label: str) -> None:
        with self.lock:
            self.completed_tasks += 1
            self.active_domain_status[domain] = f"{label} (Done)"
            active_items = [f"{d}: {s}" for d, s in sorted(self.active_domain_status.items()) if s]
            active_summary = " | ".join(active_items) if active_items else "All completed"
            prog_str = f"[{self.completed_tasks}/{self.total_tasks} done] {active_summary}"
            set_task_progress(self.work_dir, self.task_name, prog_str)
            if self.task_manager:
                try:
                    self.task_manager.update_progress(self.task_name, prog_str)
                except Exception:
                    pass

    def record_error(self, domain: str, err: str) -> None:
        with self.lock:
            self.errors.append(f"[{domain}] {err}")


# ==============================================================================
# Single-Connection Resilient Downloader
# ==============================================================================
def download_file_with_progress(
    url: str,
    dest_path: Path | str,
    label: str = "",
    min_size_bytes: int = 1024,
    progress_callback: Optional[Any] = None,
    retries: Optional[int] = None,
    use_aria2: bool = True
) -> Path:
    """
    Download a file with checkpointing, real-time progress logging, and integrity check.
    Supports indefinite retries (retries=None or <= 0), exponential backoff, true resuming
    across aria2c and curl without deleting partial files on transient errors, and stall recovery.
    """
    import time
    dest = Path(dest_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_label = label or dest.name

    # 1. Pre-download check: Skip if already valid
    is_valid, msg = check_file_integrity(dest, min_size_bytes=min_size_bytes, fast=True)
    if is_valid:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[SKIP] Resource already exists & verified intact: {dest.name} ({size_mb:.1f} MB)", flush=True)
        return dest

    if dest.exists() and not is_valid:
        existing_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[*] Partial or unverified file {dest.name} ({existing_mb:.1f} MB) found. Resuming download...", flush=True)

    aria2c_path = shutil.which("aria2c")
    conda_aria2c = Path(sys.prefix) / "bin" / "aria2c"
    if not aria2c_path and conda_aria2c.exists():
        aria2c_path = str(conda_aria2c)

    aria2_re = re.compile(
        r"\[#[0-9a-fA-F]+\s+([0-9\.]+[A-Za-z]+)/([0-9\.]+[A-Za-z]+)\(([0-9]+)%\)\s+.*?DL:([0-9\.]+[A-Za-z/]+).*?ETA:([^\s\]]+)"
    )

    url_https = url if url.startswith("https://") else url.replace("http://", "https://")
    url_http = url if url.startswith("http://") else url.replace("https://", "http://")

    attempt = 0
    while True:
        attempt += 1
        if retries is not None and retries > 0 and attempt > retries:
            raise RuntimeError(f"Failed to download and verify {file_label} after {retries} attempts from: {url}")

        retries_str = str(retries) if retries is not None and retries > 0 else "∞"
        # Alternate protocol between attempts on repeated failures for EBI/SRA hosts
        if attempt % 2 == 0 and ("ebi.ac.uk" in url or "sra" in url):
            active_url = url_http
        else:
            active_url = url_https

        print(f"[*] Downloading {file_label} (Attempt {attempt}/{retries_str}) from: {active_url}", flush=True)
        if progress_callback:
            progress_callback(f"{file_label}: Starting attempt {attempt}/{retries_str}")

        is_transient_error = True
        is_range_error = False

        try:
            # Prefer aria2c with resilient flags; fall back to curl on alternating attempts or if aria2c unavailable
            use_curl_this_attempt = (not use_aria2) or (not aria2c_path) or (attempt >= 2 and attempt % 2 == 0)

            if not use_curl_this_attempt and aria2c_path:
                cmd = [
                    aria2c_path,
                    "-U", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "-s", "1", "-x", "1", "-j", "1", "-c",
                    "--auto-file-renaming=false",
                    "--allow-overwrite=true",
                    "--lowest-speed-limit=20K",
                    "--timeout=30", "--connect-timeout=15",
                    "--summary-interval=3",
                    "--console-log-level=warn",
                    "--file-allocation=none",
                    "-o", dest.name,
                    "-d", str(dest.parent),
                    active_url
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ
                )

                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    m = aria2_re.search(line)
                    if m:
                        downloaded, total_size, pct, speed, eta = m.groups()
                        prog_str = f"{file_label}: {pct}% ({downloaded}/{total_size}, {speed}, ETA {eta})"
                        print(f"    {prog_str}", flush=True)
                        if progress_callback:
                            progress_callback(prog_str)
                    elif "Download complete" in line:
                        print(f"    [{file_label}] {line}", flush=True)

                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)

            else:
                # Direct curl download resuming directly to dest (-C -) so bytes are preserved across tools
                cmd = [
                    "curl",
                    "-sS",
                    "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "-L", "-C", "-",
                    "--speed-limit", "20000",
                    "--speed-time", "30",
                    "--connect-timeout", "15",
                    "-o", str(dest),
                    active_url
                ]
                proc = subprocess.Popen(
                    cmd,
                    env=os.environ,
                    stderr=subprocess.PIPE,
                    text=True
                )
                last_size = dest.stat().st_size if dest.exists() else 0
                last_time = time.time()
                while proc.poll() is None:
                    time.sleep(3)
                    curr_size = dest.stat().st_size if dest.exists() else 0
                    now = time.time()
                    elapsed = max(0.1, now - last_time)
                    speed_mb_s = max(0.0, (curr_size - last_size) / (1024 * 1024) / elapsed)
                    curr_mb = curr_size / (1024 * 1024)
                    prog_str = f"{file_label}: curl {curr_mb:.1f} MB downloaded ({speed_mb_s:.2f} MB/s)"
                    print(f"    {prog_str}", flush=True)
                    if progress_callback:
                        progress_callback(prog_str)
                    last_size = curr_size
                    last_time = now

                _, stderr_out = proc.communicate()
                if proc.returncode != 0:
                    err_out = stderr_out or ""
                    if "416" in err_out or "Requested Range Not Satisfiable" in err_out:
                        is_range_error = True
                    raise subprocess.CalledProcessError(proc.returncode, cmd, output=None, stderr=err_out)

            # 2. Post-download integrity verification
            post_valid, post_msg = check_file_integrity(dest, min_size_bytes=min_size_bytes, fast=False)
            if post_valid:
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"[VERIFIED] Successfully downloaded & verified: {dest.name} ({size_mb:.1f} MB)", flush=True)
                cleanup_failed_downloads(dest)
                return dest
            else:
                print(f"[!] Post-download integrity verification failed: {post_msg}", flush=True)
                # Download exited cleanly but file is corrupt: purge corrupted file to start fresh
                dest.unlink(missing_ok=True)
                cleanup_failed_downloads(dest)

        except Exception as e:
            err_str = str(e)
            if "416" in err_str:
                is_range_error = True

            if is_range_error:
                print(f"[!] HTTP 416 (Range Not Satisfiable) detected for {dest.name}. Resetting partial file...", flush=True)
                dest.unlink(missing_ok=True)
                cleanup_failed_downloads(dest)
            else:
                # CRITICAL: Preserve partial files across transient network errors (exit code 22, timeouts, resets)
                curr_sz = f"{dest.stat().st_size / 1e6:.1f} MB" if dest.exists() else "0 MB"
                print(f"[-] Download attempt {attempt} failed ({e}). Preserving partial file ({curr_sz}) for resume.", flush=True)

            backoff = min(30, 2 * (2 ** (attempt - 1)))  # 2s -> 4s -> 8s -> 16s -> 30s max
            wait_msg = f"{file_label}: Waiting {backoff}s before retry #{attempt + 1}..."
            print(f"[*] {wait_msg}", flush=True)
            if progress_callback:
                progress_callback(wait_msg)
            time.sleep(backoff)


def download_domain_worker(
    domain: str,
    tasks: List[DownloadTask],
    tracker: MultiDomainProgressTracker,
    retries: Optional[int] = None
) -> None:
    """Serial worker dedicated to a single host domain."""
    print(f"[DOMAIN WORKER START] Host: {domain} ({len(tasks)} file(s))", flush=True)
    for idx, t in enumerate(tasks, 1):
        def cb(status):
            tracker.update_domain_progress(domain, status)

        print(f"[{domain}] ({idx}/{len(tasks)}) Beginning: {t.label}", flush=True)
        try:
            download_file_with_progress(
                url=t.url,
                dest_path=t.dest_path,
                label=t.label,
                min_size_bytes=t.min_size_bytes,
                progress_callback=cb,
                retries=retries
            )
            tracker.mark_task_complete(domain, t.label)
        except Exception as e:
            tracker.record_error(domain, f"{t.label}: {e}")
            raise
    print(f"[DOMAIN WORKER DONE] Host: {domain}", flush=True)


# ==============================================================================
# Audit & Manifest Generation
# ==============================================================================
def check_resource_status(
    work_dir: Path,
    target_species: Optional[str] = None,
    target_fa: Optional[str] = None,
    target_gtf: Optional[str] = None,
    zf_fa: Optional[str] = None,
    zf_gtf: Optional[str] = None,
    enable_medaka: bool = False,
    medaka_fa: Optional[str] = None,
    medaka_gtf: Optional[str] = None,
    jaspar_meme_path: Optional[str] = None,
    premzt_tf_list: Optional[str] = None,
    slamseq_gene_table: Optional[str] = None,
    ortholog_map_path: Optional[str] = None,
    skip_slamseq: bool = False,
    atac_accessions_zf: Optional[List[str]] = None,
    atac_accessions_medaka: Optional[List[str]] = None,
    slam_accessions: Optional[List[str]] = None,
    use_pretrained_bias: bool = False,
    write_manifests: bool = True
) -> Tuple[pd.DataFrame, bool, List[DownloadTask]]:
    """Audit all study resources and compile pending download tasks."""
    work_dir = Path(work_dir).resolve()
    downloads_dir = work_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    slam_dir = work_dir / "slamseq"
    slam_dir.mkdir(parents=True, exist_ok=True)
    fastqs_dir = slam_dir / "fastqs"
    fastqs_dir.mkdir(parents=True, exist_ok=True)
    atac_zf_dir = work_dir / "atac" / "Danio_rerio" / "fastqs"
    atac_zf_dir.mkdir(parents=True, exist_ok=True)
    atac_med_dir = work_dir / "atac" / "Oryzias_latipes" / "fastqs"
    atac_med_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    pending_tasks: List[DownloadTask] = []

    # 1. Target Species FASTA & GTF
    if target_fa:
        t_fa_p = Path(target_fa).resolve()
        valid, msg = check_file_integrity(t_fa_p, min_size_bytes=100 * 1024 * 1024)
        if valid:
            records.append({
                "Resource": f"{target_species or 'Target'} Genome FASTA",
                "Path": str(t_fa_p),
                "Size_MB": f"{t_fa_p.stat().st_size / 1e6:.1f}",
                "Status": "VERIFIED"
            })
        else:
            t_fa_gz = t_fa_p.with_suffix(t_fa_p.suffix + ".gz") if not str(t_fa_p).endswith(".gz") else t_fa_p
            v_gz, _ = check_file_integrity(t_fa_gz, min_size_bytes=100 * 1024 * 1024)
            if v_gz:
                records.append({
                    "Resource": f"{target_species or 'Target'} Genome FASTA",
                    "Path": str(t_fa_p),
                    "Size_MB": f"{t_fa_gz.stat().st_size / 1e6:.1f}",
                    "Status": "PENDING_DECOMPRESS"
                })
            elif target_species and "astatotilapia" in target_species.lower():
                records.append({
                    "Resource": f"{target_species or 'Target'} Genome FASTA",
                    "Path": str(t_fa_p),
                    "Size_MB": "-",
                    "Status": "PENDING_DOWNLOAD"
                })
                pending_tasks.append(DownloadTask(
                    key="target_fa",
                    label=f"{target_species} Genome FASTA",
                    url=ASTATOTILAPIA_112_FASTA_URL,
                    dest_path=t_fa_gz,
                    min_size_bytes=100 * 1024 * 1024,
                    domain=urlparse(ASTATOTILAPIA_112_FASTA_URL).netloc,
                    post_action="decompress_fa"
                ))
            else:
                records.append({
                    "Resource": f"{target_species or 'Target'} Genome FASTA",
                    "Path": str(t_fa_p),
                    "Size_MB": "-",
                    "Status": f"MISSING ({msg})"
                })

    if target_gtf:
        t_gtf_p = Path(target_gtf).resolve()
        valid, msg = check_file_integrity(t_gtf_p, min_size_bytes=10 * 1024 * 1024)
        if valid:
            records.append({
                "Resource": f"{target_species or 'Target'} GTF",
                "Path": str(t_gtf_p),
                "Size_MB": f"{t_gtf_p.stat().st_size / 1e6:.1f}",
                "Status": "VERIFIED"
            })
        elif target_species and "astatotilapia" in target_species.lower():
            records.append({
                "Resource": f"{target_species or 'Target'} GTF",
                "Path": str(t_gtf_p),
                "Size_MB": "-",
                "Status": "PENDING_DOWNLOAD"
            })
            pending_tasks.append(DownloadTask(
                key="target_gtf",
                label=f"{target_species} GTF",
                url=ASTATOTILAPIA_112_GTF_URL,
                dest_path=t_gtf_p,
                min_size_bytes=10 * 1024 * 1024,
                domain=urlparse(ASTATOTILAPIA_112_GTF_URL).netloc
            ))
        else:
            records.append({
                "Resource": f"{target_species or 'Target'} GTF",
                "Path": str(t_gtf_p),
                "Size_MB": "-",
                "Status": f"MISSING ({msg})"
            })

    # 2. Zebrafish GRCz11 References
    zf_fa_dest = Path(zf_fa).resolve() if zf_fa else downloads_dir / "Danio_rerio.GRCz11.dna_sm.primary_assembly.fa"
    zf_fa_gz = zf_fa_dest.with_suffix(zf_fa_dest.suffix + ".gz")

    v_fa, _ = check_file_integrity(zf_fa_dest, min_size_bytes=800 * 1024 * 1024)
    if v_fa:
        records.append({
            "Resource": "Zebrafish Genome FASTA",
            "Path": str(zf_fa_dest),
            "Size_MB": f"{zf_fa_dest.stat().st_size / 1e6:.1f}",
            "Status": "VERIFIED"
        })
    else:
        v_gz, _ = check_file_integrity(zf_fa_gz, min_size_bytes=350 * 1024 * 1024)
        if v_gz:
            records.append({
                "Resource": "Zebrafish Genome FASTA",
                "Path": str(zf_fa_dest),
                "Size_MB": f"{zf_fa_gz.stat().st_size / 1e6:.1f}",
                "Status": "PENDING_DECOMPRESS"
            })
        else:
            records.append({
                "Resource": "Zebrafish Genome FASTA",
                "Path": str(zf_fa_dest),
                "Size_MB": "-",
                "Status": "PENDING_DOWNLOAD"
            })
            pending_tasks.append(DownloadTask(
                key="zf_fa",
                label="Zebrafish Genome FASTA",
                url=ZEBRAFISH_ENSEMBL_FASTA_URL,
                dest_path=zf_fa_gz,
                min_size_bytes=350 * 1024 * 1024,
                domain=urlparse(ZEBRAFISH_ENSEMBL_FASTA_URL).netloc,
                post_action="decompress_fa"
            ))

    zf_gtf_dest = Path(zf_gtf).resolve() if zf_gtf else downloads_dir / "Danio_rerio.GRCz11.112.gtf.gz"
    v_gtf, _ = check_file_integrity(zf_gtf_dest, min_size_bytes=10 * 1024 * 1024)
    if v_gtf:
        records.append({
            "Resource": "Zebrafish GTF",
            "Path": str(zf_gtf_dest),
            "Size_MB": f"{zf_gtf_dest.stat().st_size / 1e6:.1f}",
            "Status": "VERIFIED"
        })
    else:
        records.append({
            "Resource": "Zebrafish GTF",
            "Path": str(zf_gtf_dest),
            "Size_MB": "-",
            "Status": "PENDING_DOWNLOAD"
        })
        pending_tasks.append(DownloadTask(
            key="zf_gtf",
            label="Zebrafish GTF",
            url=ZEBRAFISH_ENSEMBL_GTF_URL,
            dest_path=zf_gtf_dest,
            min_size_bytes=10 * 1024 * 1024,
            domain=urlparse(ZEBRAFISH_ENSEMBL_GTF_URL).netloc
        ))

    # 3. Medaka References (if enable_medaka=True)
    if enable_medaka:
        med_fa_dest = Path(medaka_fa).resolve() if medaka_fa else downloads_dir / "Oryzias_latipes.ASM223467v1.dna_sm.toplevel.fa"
        med_fa_gz = med_fa_dest.with_suffix(med_fa_dest.suffix + ".gz")

        v_mfa, _ = check_file_integrity(med_fa_dest, min_size_bytes=300 * 1024 * 1024)
        if v_mfa:
            records.append({
                "Resource": "Medaka Genome FASTA",
                "Path": str(med_fa_dest),
                "Size_MB": f"{med_fa_dest.stat().st_size / 1e6:.1f}",
                "Status": "VERIFIED"
            })
        else:
            v_mgz, _ = check_file_integrity(med_fa_gz, min_size_bytes=150 * 1024 * 1024)
            if v_mgz:
                records.append({
                    "Resource": "Medaka Genome FASTA",
                    "Path": str(med_fa_dest),
                    "Size_MB": f"{med_fa_gz.stat().st_size / 1e6:.1f}",
                    "Status": "PENDING_DECOMPRESS"
                })
            else:
                records.append({
                    "Resource": "Medaka Genome FASTA",
                    "Path": str(med_fa_dest),
                    "Size_MB": "-",
                    "Status": "PENDING_DOWNLOAD"
                })
                pending_tasks.append(DownloadTask(
                    key="medaka_fa",
                    label="Medaka Genome FASTA",
                    url=MEDAKA_ENSEMBL_FASTA_URL,
                    dest_path=med_fa_gz,
                    min_size_bytes=150 * 1024 * 1024,
                    domain=urlparse(MEDAKA_ENSEMBL_FASTA_URL).netloc,
                    post_action="decompress_fa"
                ))

        med_gtf_dest = Path(medaka_gtf).resolve() if medaka_gtf else downloads_dir / "Oryzias_latipes.ASM223467v1.112.gtf.gz"
        v_mgtf, _ = check_file_integrity(med_gtf_dest, min_size_bytes=10 * 1024 * 1024)
        if v_mgtf:
            records.append({
                "Resource": "Medaka GTF",
                "Path": str(med_gtf_dest),
                "Size_MB": f"{med_gtf_dest.stat().st_size / 1e6:.1f}",
                "Status": "VERIFIED"
            })
        else:
            records.append({
                "Resource": "Medaka GTF",
                "Path": str(med_gtf_dest),
                "Size_MB": "-",
                "Status": "PENDING_DOWNLOAD"
            })
            pending_tasks.append(DownloadTask(
                key="medaka_gtf",
                label="Medaka GTF",
                url=MEDAKA_ENSEMBL_GTF_URL,
                dest_path=med_gtf_dest,
                min_size_bytes=10 * 1024 * 1024,
                domain=urlparse(MEDAKA_ENSEMBL_GTF_URL).netloc
            ))

    # 4. JASPAR 2024 CORE Database
    jaspar_dest = Path(jaspar_meme_path).resolve() if jaspar_meme_path else downloads_dir / "JASPAR2024_CORE_vertebrates_non-redundant.meme"
    v_j, _ = check_file_integrity(jaspar_dest, min_size_bytes=100 * 1024)
    if v_j:
        records.append({
            "Resource": "JASPAR 2024 CORE Database",
            "Path": str(jaspar_dest),
            "Size_MB": f"{jaspar_dest.stat().st_size / 1e6:.2f}",
            "Status": "VERIFIED"
        })
    else:
        records.append({
            "Resource": "JASPAR 2024 CORE Database",
            "Path": str(jaspar_dest),
            "Size_MB": "-",
            "Status": "PENDING_DOWNLOAD"
        })
        pending_tasks.append(DownloadTask(
            key="jaspar",
            label="JASPAR Motif Database",
            url=JASPAR_2024_CORE_VERTEBRATES_URL,
            dest_path=jaspar_dest,
            min_size_bytes=100 * 1024,
            domain=urlparse(JASPAR_2024_CORE_VERTEBRATES_URL).netloc
        ))

    # Pre-MZT TF MEME Subset
    filtered_meme_dest = downloads_dir / "premzt_tfs.meme"
    v_filt, _ = check_file_integrity(filtered_meme_dest, min_size_bytes=1024)
    records.append({
        "Resource": "Pre-MZT TF Motifs",
        "Path": str(filtered_meme_dest),
        "Size_MB": f"{filtered_meme_dest.stat().st_size / 1e6:.2f}" if filtered_meme_dest.exists() else "-",
        "Status": "VERIFIED" if v_filt else "PENDING_PREP"
    })

    # Target <-> Zebrafish Ortholog Map
    ortho_dest = Path(ortholog_map_path).resolve() if ortholog_map_path else downloads_dir / "orthologs_target_to_zf.tsv"
    v_ortho, _ = check_file_integrity(ortho_dest, min_size_bytes=10 * 1024)
    records.append({
        "Resource": "Target <-> Zebrafish Ortholog Map",
        "Path": str(ortho_dest),
        "Size_MB": f"{ortho_dest.stat().st_size / 1e6:.2f}" if ortho_dest.exists() else "-",
        "Status": "VERIFIED" if v_ortho else "PENDING_PREP"
    })

    # 5. Zebrafish ATAC-seq FASTQs (GSE130944)
    zf_atac_accs = atac_accessions_zf or [
        "GSM3756599", "GSM3756600", "GSM3756603", "GSM3756604", "GSM3756605", "SRR9032640"
    ]
    resolved_zf_atac = resolve_ena_fastqs(zf_atac_accs, default_paired=True, cache_path=downloads_dir / "ena_resolution_cache.json")
    for run_id, r_info in resolved_zf_atac.items():
        for url in r_info["urls"]:
            fn = Path(url).name
            dest_fq = atac_zf_dir / fn
            v_fq, _ = check_file_integrity(dest_fq, min_size_bytes=10 * 1024 * 1024)
            if v_fq:
                records.append({
                    "Resource": f"Zebrafish ATAC FASTQ ({run_id})",
                    "Path": str(dest_fq),
                    "Size_MB": f"{dest_fq.stat().st_size / 1e6:.1f}",
                    "Status": "VERIFIED"
                })
            else:
                records.append({
                    "Resource": f"Zebrafish ATAC FASTQ ({run_id})",
                    "Path": str(dest_fq),
                    "Size_MB": "-",
                    "Status": "PENDING_DOWNLOAD"
                })
                pending_tasks.append(DownloadTask(
                    key=f"atac_zf_{run_id}_{dest_fq.name}",
                    label=f"Zebrafish ATAC {dest_fq.name}",
                    url=url,
                    dest_path=dest_fq,
                    min_size_bytes=10 * 1024 * 1024,
                    domain=urlparse(url).netloc
                ))

    # 6. Medaka ATAC-seq FASTQs (PRJDB7492)
    if enable_medaka:
        med_atac_accs = atac_accessions_medaka or [
            "DRX143405", "DRX143406", "DRX143407", "DRX143412", "DRX143413"
        ]
        resolved_med_atac = resolve_ena_fastqs(med_atac_accs, default_paired=False, cache_path=downloads_dir / "ena_resolution_cache.json")
        for run_id, r_info in resolved_med_atac.items():
            for url in r_info["urls"]:
                fn = Path(url).name
                dest_fq = atac_med_dir / fn
                v_fq, _ = check_file_integrity(dest_fq, min_size_bytes=10 * 1024 * 1024)
                if v_fq:
                    records.append({
                        "Resource": f"Medaka ATAC FASTQ ({run_id})",
                        "Path": str(dest_fq),
                        "Size_MB": f"{dest_fq.stat().st_size / 1e6:.1f}",
                        "Status": "VERIFIED"
                    })
                else:
                    records.append({
                        "Resource": f"Medaka ATAC FASTQ ({run_id})",
                        "Path": str(dest_fq),
                        "Size_MB": "-",
                        "Status": "PENDING_DOWNLOAD"
                    })
                    med_domain = "ddbj.nig.ac.jp" if run_id.startswith("DRR") else urlparse(url).netloc
                    pending_tasks.append(DownloadTask(
                        key=f"atac_med_{run_id}_{dest_fq.name}",
                        label=f"Medaka ATAC {dest_fq.name}",
                        url=url,
                        dest_path=dest_fq,
                        min_size_bytes=10 * 1024 * 1024,
                        domain=med_domain
                    ))

    # 7. Zebrafish SLAM-seq FASTQs (PRJNA768450)
    zf_slam_table = slam_dir / "zf_slamseq_gene_classes.tsv"
    user_slam_table = Path(slamseq_gene_table).resolve() if slamseq_gene_table else None

    need_slamseq = not skip_slamseq
    if need_slamseq:
        slam_acc_list = slam_accessions or [
            "GSM5610039", "GSM5610040", "GSM5610043", "GSM5610044",
            "GSM5610045", "GSM5610046", "GSM5610047", "GSM5610048",
            "GSM5610051", "GSM5610052", "GSM5610075", "GSM5610076", "GSM5610077"
        ]
        resolved_slam = resolve_ena_fastqs(slam_acc_list, default_paired=False, cache_path=downloads_dir / "ena_resolution_cache.json")
        for run_id, r_info in resolved_slam.items():
            for url in r_info["urls"]:
                fn = Path(url).name
                dest_fq = fastqs_dir / fn
                v_fq, _ = check_file_integrity(dest_fq, min_size_bytes=10 * 1024 * 1024)
                if v_fq:
                    records.append({
                        "Resource": f"SLAM-seq FASTQ ({run_id})",
                        "Path": str(dest_fq),
                        "Size_MB": f"{dest_fq.stat().st_size / 1e6:.1f}",
                        "Status": "VERIFIED"
                    })
                else:
                    records.append({
                        "Resource": f"SLAM-seq FASTQ ({run_id})",
                        "Path": str(dest_fq),
                        "Size_MB": "-",
                        "Status": "PENDING_DOWNLOAD"
                    })
                    pending_tasks.append(DownloadTask(
                        key=f"slam_{run_id}_{dest_fq.name}",
                        label=f"SLAM-seq {dest_fq.name}",
                        url=url,
                        dest_path=dest_fq,
                        min_size_bytes=10 * 1024 * 1024,
                        domain=urlparse(url).netloc
                    ))

    # 8. Pretrained Bias Model (if requested)
    if use_pretrained_bias:
        bias_models_dir = downloads_dir / "bias_models"
        bias_dest = bias_models_dir / "atac_bias.h5"
        bias_zip = downloads_dir / "bias_models.zip"
        v_b, _ = check_file_integrity(bias_dest, min_size_bytes=100 * 1024)
        if v_b:
            records.append({
                "Resource": "Pretrained Tn5 Bias Model",
                "Path": str(bias_dest),
                "Size_MB": f"{bias_dest.stat().st_size / 1e6:.1f}",
                "Status": "VERIFIED"
            })
        else:
            records.append({
                "Resource": "Pretrained Tn5 Bias Model",
                "Path": str(bias_dest),
                "Size_MB": "-",
                "Status": "PENDING_DOWNLOAD"
            })
            pending_tasks.append(DownloadTask(
                key="bias_models_zenodo",
                label="Pretrained Bias Models (Zenodo 7443683)",
                url="https://zenodo.org/records/7443683/files/bias_models.zip?download=1",
                dest_path=bias_zip,
                min_size_bytes=100 * 1024,
                domain="zenodo.org",
                post_action="extract_bias"
            ))

    df = pd.DataFrame(records)
    has_pending = any(r.get("Status", "").startswith("PENDING") for r in records)
    all_ready = (len(pending_tasks) == 0) and not has_pending

    # Write manifests immediately to downloads_dir so they are available to downstream tasks
    if write_manifests:
        is_task_running = False
        status_file = work_dir / "tasks" / "download_resources" / "status.json"
        if status_file.exists():
            try:
                with open(status_file) as f:
                    st = json.load(f)
                if st.get("status") == "RUNNING":
                    is_task_running = True
            except Exception:
                pass

        if not is_task_running:
            try:
                downloads_dir.mkdir(parents=True, exist_ok=True)
                # 1. Resource audit manifest
                df.to_csv(downloads_dir / "resource_manifest.tsv", sep="\t", index=False)

                # 2. ATAC-seq FASTQ manifest
                atac_manifest_records = []
                for run_id, r_info in resolved_zf_atac.items():
                    parent_acc = r_info.get("parent_accession", "")
                    stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc)
                    if not stage:
                        stage = "control_gdna" if run_id == "SRR9032640" else "late_dome"
                    fqs = [str(atac_zf_dir / Path(u).name) for u in r_info["urls"]]
                    atac_manifest_records.append({
                        "species": "Danio_rerio",
                        "stage": stage,
                        "accession": parent_acc,
                        "run_id": run_id,
                        "fastq_r1": fqs[0],
                        "fastq_r2": fqs[1] if len(fqs) > 1 else "",
                        "layout": r_info.get("library_layout", "PAIRED" if len(fqs) > 1 else "SINGLE")
                    })

                if enable_medaka and 'resolved_med_atac' in locals():
                    for run_id, r_info in resolved_med_atac.items():
                        parent_acc = r_info.get("parent_accession", "")
                        stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc)
                        if not stage:
                            stage = "early_st9" if run_id in ["DRR152711", "DRR152712", "DRR152713"] else "late_st11"
                        fqs = [str(atac_med_dir / Path(u).name) for u in r_info["urls"]]
                        atac_manifest_records.append({
                            "species": "Oryzias_latipes",
                            "stage": stage,
                            "accession": parent_acc,
                            "run_id": run_id,
                            "fastq_r1": fqs[0],
                            "fastq_r2": fqs[1] if len(fqs) > 1 else "",
                            "layout": r_info.get("library_layout", "PAIRED" if len(fqs) > 1 else "SINGLE")
                        })

                pd.DataFrame(atac_manifest_records).to_csv(downloads_dir / "atac_fastq_manifest.tsv", sep="\t", index=False)

                # 3. SLAM-seq FASTQ manifest
                slam_manifest_records = []
                if need_slamseq and 'resolved_slam' in locals():
                    default_stage_map = {
                        "SRR16195447": "0.75hpf", "SRR16195448": "0.75hpf",
                        "SRR16195451": "2.5hpf", "SRR16195452": "2.5hpf",
                        "SRR16195453": "3.0_3.5hpf", "SRR16195454": "3.0_3.5hpf",
                        "SRR16195455": "3.0_3.5hpf", "SRR16195456": "3.0_3.5hpf",
                        "SRR16195459": "4.5hpf", "SRR16195460": "4.5hpf",
                        "SRR16195483": "control_untreated", "SRR16195484": "control_untreated", "SRR16195485": "control_untreated"
                    }
                    for run_id, r_info in resolved_slam.items():
                        parent_acc = r_info.get("parent_accession", "")
                        stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc) or default_stage_map.get(run_id, "unknown")
                        fqs = [str(fastqs_dir / Path(u).name) for u in r_info["urls"]]
                        slam_manifest_records.append({
                            "stage": stage,
                            "accession": parent_acc,
                            "run_id": run_id,
                            "fastq": fqs[0]
                        })
                    pd.DataFrame(slam_manifest_records).to_csv(downloads_dir / "slamseq_fastq_manifest.tsv", sep="\t", index=False)
            except Exception as e:
                print(f"Warning: could not write manifests in check_resource_status: {e}")

    return df, all_ready, pending_tasks


# ==============================================================================
# Consolidated Download Manager
# ==============================================================================
def download_all_resources(
    work_dir: Path | str,
    target_species: Optional[str] = None,
    target_fa: Optional[str] = None,
    target_gtf: Optional[str] = None,
    zf_fa: Optional[str] = None,
    zf_gtf: Optional[str] = None,
    enable_medaka: bool = False,
    medaka_fa: Optional[str] = None,
    medaka_gtf: Optional[str] = None,
    jaspar_meme_path: Optional[str] = None,
    premzt_tf_list: Optional[str] = None,
    slamseq_gene_table: Optional[str] = None,
    ortholog_map_path: Optional[str] = None,
    skip_slamseq: bool = False,
    atac_accessions_zf: Optional[List[str]] = None,
    atac_accessions_medaka: Optional[List[str]] = None,
    slam_accessions: Optional[List[str]] = None,
    use_pretrained_bias: bool = False,
    max_host_workers: int = 3,
    max_per_host_workers: int = 2,
    max_retries: Optional[int] = None,
    task_name: str = "download_resources",
    task_manager: Optional[Any] = None
) -> pd.DataFrame:
    """
    Consolidated download manager:
    1. Audits required study resources.
    2. Groups pending downloads by host domain.
    3. Concurrently downloads from up to max_host_workers (default 3) distinct domains in parallel,
       with sub-sharding up to max_per_host_workers for high-volume hosts like EBI.
    4. Executes post-download decompression and preparatory derivative steps.
    5. Writes manifests: atac_fastq_manifest.tsv, slamseq_fastq_manifest.tsv, resource_manifest.tsv.
    6. Updates TaskManager progress in real time.
    """
    work_dir_p = Path(work_dir).resolve()
    downloads_dir = work_dir_p / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    slam_dir = work_dir_p / "slamseq"
    slam_dir.mkdir(parents=True, exist_ok=True)
    fastqs_dir = slam_dir / "fastqs"
    fastqs_dir.mkdir(parents=True, exist_ok=True)
    atac_zf_dir = work_dir_p / "atac" / "Danio_rerio" / "fastqs"
    atac_zf_dir.mkdir(parents=True, exist_ok=True)
    atac_med_dir = work_dir_p / "atac" / "Oryzias_latipes" / "fastqs"
    atac_med_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("[*] Consolidated Resource Download & Multi-Domain Pipeline", flush=True)
    print(f"    Work Directory:          {work_dir_p}", flush=True)
    print(f"    Target Species:          {target_species or 'Unspecified'}", flush=True)
    print(f"    Enable Medaka:           {enable_medaka}", flush=True)
    print(f"    Max Host Workers:        {max_host_workers}", flush=True)
    print(f"    Max Per-Host Workers:    {max_per_host_workers}", flush=True)
    print("=" * 80, flush=True)

    # 1. Audit status (do not write manifests yet if tasks are pending)
    df_manifest, all_ready, pending_tasks = check_resource_status(
        work_dir=work_dir_p,
        target_species=target_species,
        target_fa=target_fa,
        target_gtf=target_gtf,
        zf_fa=zf_fa,
        zf_gtf=zf_gtf,
        enable_medaka=enable_medaka,
        medaka_fa=medaka_fa,
        medaka_gtf=medaka_gtf,
        jaspar_meme_path=jaspar_meme_path,
        premzt_tf_list=premzt_tf_list,
        slamseq_gene_table=slamseq_gene_table,
        ortholog_map_path=ortholog_map_path,
        skip_slamseq=skip_slamseq,
        atac_accessions_zf=atac_accessions_zf,
        atac_accessions_medaka=atac_accessions_medaka,
        slam_accessions=slam_accessions,
        use_pretrained_bias=use_pretrained_bias,
        write_manifests=False
    )

    if not pending_tasks:
        print("[+] All downloadable resources already exist and pass integrity checks.", flush=True)
    else:
        # Group tasks by host domain, sub-sharding heavy domains like ftp.sra.ebi.ac.uk
        domain_queues: Dict[str, List[DownloadTask]] = defaultdict(list)
        ebi_count = 0
        for t in pending_tasks:
            dom = t.domain
            if "ebi.ac.uk" in dom and max_per_host_workers > 1:
                sub_dom = f"{dom}#{(ebi_count % max_per_host_workers) + 1}"
                domain_queues[sub_dom].append(t)
                ebi_count += 1
            else:
                domain_queues[dom].append(t)

        print(f"[*] Queued {len(pending_tasks)} file(s) across {len(domain_queues)} distinct host domain(s):", flush=True)
        for dom, tasks in domain_queues.items():
            print(f"    - [{dom}]: {len(tasks)} file(s)", flush=True)

        n_workers = min(max_host_workers, len(domain_queues))
        print(f"[*] Launching ThreadPoolExecutor with {n_workers} concurrent domain worker(s)...", flush=True)

        tracker = MultiDomainProgressTracker(
            total_tasks=len(pending_tasks),
            work_dir=work_dir_p,
            task_name=task_name,
            task_manager=task_manager
        )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(download_domain_worker, dom, tasks, tracker, max_retries): dom
                for dom, tasks in domain_queues.items()
            }
            for fut in as_completed(futures):
                dom = futures[fut]
                try:
                    fut.result()
                    print(f"[+] Domain queue [{dom}] finished successfully.", flush=True)
                except Exception as e:
                    print(f"[-] Domain queue [{dom}] encountered error: {e}", flush=True)
                    tracker.record_error(dom, str(e))
                    raise

    # --------------------------------------------------------------------------
    # 2. Post-Download Derivative Extractions & Preparations
    # --------------------------------------------------------------------------
    # Target FASTA decompression & indexing if needed
    if target_fa:
        t_fa_dest = Path(target_fa).resolve()
        t_fa_gz = t_fa_dest.with_suffix(t_fa_dest.suffix + ".gz")
        if not check_file_integrity(t_fa_dest, 100 * 1024 * 1024)[0] and t_fa_gz.exists():
            print(f"[*] Decompressing {t_fa_gz.name} -> {t_fa_dest.name}...", flush=True)
            set_task_progress(work_dir_p, task_name, f"Decompressing {t_fa_dest.name}")
            subprocess.run(f"gzip -dc '{t_fa_gz}' > '{t_fa_dest}'", shell=True, check=True)
            subprocess.run(["samtools", "faidx", str(t_fa_dest)], check=True)

    # Zebrafish FASTA decompression & indexing if needed
    zf_fa_dest = Path(zf_fa).resolve() if zf_fa else downloads_dir / "Danio_rerio.GRCz11.dna_sm.primary_assembly.fa"
    zf_fa_gz = zf_fa_dest.with_suffix(zf_fa_dest.suffix + ".gz")
    if not check_file_integrity(zf_fa_dest, 500 * 1024 * 1024)[0] and zf_fa_gz.exists():
        print(f"[*] Decompressing {zf_fa_gz.name} -> {zf_fa_dest.name}...", flush=True)
        set_task_progress(work_dir_p, task_name, "Decompressing Zebrafish FASTA")
        subprocess.run(f"gzip -dc '{zf_fa_gz}' > '{zf_fa_dest}'", shell=True, check=True)
        subprocess.run(["samtools", "faidx", str(zf_fa_dest)], check=True)

    # Medaka FASTA decompression & indexing if needed
    if enable_medaka:
        med_fa_dest = Path(medaka_fa).resolve() if medaka_fa else downloads_dir / "Oryzias_latipes.ASM223467v1.dna_sm.toplevel.fa"
        med_fa_gz = med_fa_dest.with_suffix(med_fa_dest.suffix + ".gz")
        if not check_file_integrity(med_fa_dest, 300 * 1024 * 1024)[0] and med_fa_gz.exists():
            print(f"[*] Decompressing {med_fa_gz.name} -> {med_fa_dest.name}...", flush=True)
            set_task_progress(work_dir_p, task_name, "Decompressing Medaka FASTA")
            subprocess.run(f"gzip -dc '{med_fa_gz}' > '{med_fa_dest}'", shell=True, check=True)
            subprocess.run(["samtools", "faidx", str(med_fa_dest)], check=True)

    # Pre-MZT TF MEME Subset filter
    jaspar_dest = Path(jaspar_meme_path).resolve() if jaspar_meme_path else downloads_dir / "JASPAR2024_CORE_vertebrates_non-redundant.meme"
    filtered_meme_dest = downloads_dir / "premzt_tfs.meme"
    if not check_file_integrity(filtered_meme_dest, 1024)[0] and jaspar_dest.exists() and premzt_tf_list and Path(premzt_tf_list).exists():
        from scripts.infer_and_attribute import filter_jaspar_meme
        print(f"[*] Filtering JASPAR database for pre-MZT TFs ({premzt_tf_list})...", flush=True)
        filter_jaspar_meme(
            jaspar_meme_path=str(jaspar_dest),
            tf_list_path=premzt_tf_list,
            out_filtered_meme=str(filtered_meme_dest)
        )

    # Target <-> Zebrafish Ortholog Map
    zf_gtf_dest = Path(zf_gtf).resolve() if zf_gtf else downloads_dir / "Danio_rerio.GRCz11.112.gtf.gz"
    ortho_dest = Path(ortholog_map_path).resolve() if ortholog_map_path else downloads_dir / "orthologs_target_to_zf.tsv"
    if not check_file_integrity(ortho_dest, 10 * 1024)[0] and target_gtf and Path(target_gtf).exists() and zf_gtf_dest.exists():
        from scripts.infer_and_attribute import get_or_create_ortholog_map
        print(f"[*] Generating target <-> zebrafish ortholog map...", flush=True)
        get_or_create_ortholog_map(
            target_gtf_path=target_gtf,
            zf_gtf_path=str(zf_gtf_dest),
            out_tsv=str(ortho_dest)
        )

    # Pretrained Tn5 Bias Model Extraction
    if use_pretrained_bias:
        bias_models_dir = downloads_dir / "bias_models"
        bias_dest = bias_models_dir / "atac_bias.h5"
        bias_zip = downloads_dir / "bias_models.zip"
        if not check_file_integrity(bias_dest, 100 * 1024)[0] and bias_zip.exists():
            import zipfile
            print(f"[*] Extracting pretrained ATAC bias model from {bias_zip.name}...", flush=True)
            bias_models_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bias_zip, "r") as z:
                for zinfo in z.namelist():
                    if "atac" in zinfo.lower() and zinfo.endswith(".h5"):
                        with z.open(zinfo) as zf_in, open(bias_dest, "wb") as bf_out:
                            shutil.copyfileobj(zf_in, bf_out)
                        print(f"[+] Extracted ATAC bias model: {bias_dest}", flush=True)
                        break

    # --------------------------------------------------------------------------
    # 3. Export Manifests
    # --------------------------------------------------------------------------
    # ATAC FASTQ manifest
    atac_manifest_records = []
    zf_atac_accs = atac_accessions_zf or [
        "GSM3756599", "GSM3756600", "GSM3756603", "GSM3756604", "GSM3756605", "SRR9032640"
    ]
    resolved_zf_atac = resolve_ena_fastqs(zf_atac_accs, default_paired=True, cache_path=downloads_dir / "ena_resolution_cache.json")
    for run_id, r_info in resolved_zf_atac.items():
        parent_acc = r_info.get("parent_accession", "")
        stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc)
        if not stage:
            stage = "control_gdna" if run_id == "SRR9032640" else "late_dome"
        fqs = [str(atac_zf_dir / Path(u).name) for u in r_info["urls"]]
        atac_manifest_records.append({
            "species": "Danio_rerio",
            "stage": stage,
            "accession": parent_acc,
            "run_id": run_id,
            "fastq_r1": fqs[0],
            "fastq_r2": fqs[1] if len(fqs) > 1 else "",
            "layout": r_info.get("library_layout", "PAIRED" if len(fqs) > 1 else "SINGLE")
        })

    if enable_medaka:
        med_atac_accs = atac_accessions_medaka or [
            "DRX143405", "DRX143406", "DRX143407", "DRX143412", "DRX143413"
        ]
        resolved_med_atac = resolve_ena_fastqs(med_atac_accs, default_paired=False, cache_path=downloads_dir / "ena_resolution_cache.json")
        for run_id, r_info in resolved_med_atac.items():
            parent_acc = r_info.get("parent_accession", "")
            stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc)
            if not stage:
                stage = "early_st9" if run_id in ["DRR152711", "DRR152712", "DRR152713"] else "late_st11"
            fqs = [str(atac_med_dir / Path(u).name) for u in r_info["urls"]]
            atac_manifest_records.append({
                "species": "Oryzias_latipes",
                "stage": stage,
                "accession": parent_acc,
                "run_id": run_id,
                "fastq_r1": fqs[0],
                "fastq_r2": fqs[1] if len(fqs) > 1 else "",
                "layout": r_info.get("library_layout", "PAIRED" if len(fqs) > 1 else "SINGLE")
            })

    atac_manifest_tsv = downloads_dir / "atac_fastq_manifest.tsv"
    pd.DataFrame(atac_manifest_records).to_csv(atac_manifest_tsv, sep="\t", index=False)

    # SLAM-seq FASTQ manifest
    if not skip_slamseq:
        slam_manifest_records = []
        slam_acc_list = slam_accessions or [
            "GSM5610039", "GSM5610040", "GSM5610043", "GSM5610044",
            "GSM5610045", "GSM5610046", "GSM5610047", "GSM5610048",
            "GSM5610051", "GSM5610052", "GSM5610075", "GSM5610076", "GSM5610077"
        ]
        default_stage_map = {
            "SRR16195447": "0.75hpf", "SRR16195448": "0.75hpf",
            "SRR16195451": "2.5hpf", "SRR16195452": "2.5hpf",
            "SRR16195453": "3.0_3.5hpf", "SRR16195454": "3.0_3.5hpf",
            "SRR16195455": "3.0_3.5hpf", "SRR16195456": "3.0_3.5hpf",
            "SRR16195459": "4.5hpf", "SRR16195460": "4.5hpf",
            "SRR16195483": "control_untreated", "SRR16195484": "control_untreated", "SRR16195485": "control_untreated"
        }
        resolved_slam = resolve_ena_fastqs(slam_acc_list, default_paired=False, cache_path=downloads_dir / "ena_resolution_cache.json")
        for run_id, r_info in resolved_slam.items():
            parent_acc = r_info.get("parent_accession", "")
            stage = r_info.get("stage") or STAGE_BY_PARENT_ACCESSION.get(parent_acc) or default_stage_map.get(run_id, "unknown")
            fqs = [str(fastqs_dir / Path(u).name) for u in r_info["urls"]]
            slam_manifest_records.append({
                "stage": stage,
                "accession": parent_acc,
                "run_id": run_id,
                "fastq": fqs[0]
            })
        slam_manifest_tsv = downloads_dir / "slamseq_fastq_manifest.tsv"
        pd.DataFrame(slam_manifest_records).to_csv(slam_manifest_tsv, sep="\t", index=False)

    # 4. Final Verification Audit
    df_manifest_final, all_ready_final, _ = check_resource_status(
        work_dir=work_dir_p,
        target_species=target_species,
        target_fa=target_fa,
        target_gtf=target_gtf,
        zf_fa=zf_fa,
        zf_gtf=zf_gtf,
        enable_medaka=enable_medaka,
        medaka_fa=medaka_fa,
        medaka_gtf=medaka_gtf,
        jaspar_meme_path=jaspar_meme_path,
        premzt_tf_list=premzt_tf_list,
        slamseq_gene_table=slamseq_gene_table,
        ortholog_map_path=ortholog_map_path,
        skip_slamseq=skip_slamseq,
        atac_accessions_zf=atac_accessions_zf,
        atac_accessions_medaka=atac_accessions_medaka,
        slam_accessions=slam_accessions,
        use_pretrained_bias=use_pretrained_bias,
        write_manifests=False
    )

    manifest_tsv = downloads_dir / "resource_manifest.tsv"
    df_manifest_final.to_csv(manifest_tsv, sep="\t", index=False)

    print("\n" + "=" * 80, flush=True)
    print("Resource Download & Verification Final Summary:", flush=True)
    print("=" * 80, flush=True)
    print(df_manifest_final.to_string(index=False), flush=True)
    print("=" * 80 + "\n", flush=True)

    set_task_progress(work_dir_p, task_name, "100% (Done)")
    if task_manager:
        task_manager.update_progress(task_name, "100% (Done)")
        task_dir = task_manager._get_task_dir(task_name)
        status_file = task_dir / "status.json"
        done_marker = task_dir / ".done"
        if not status_file.exists():
            import time
            payload = {
                "name": task_name,
                "backend": "interactive",
                "submit_time": time.time(),
                "status": "COMPLETED",
                "progress": "100% (Done)"
            }
            with open(status_file, "w") as f:
                json.dump(payload, f, indent=2)
        done_marker.touch()

    return df_manifest_final


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Consolidated Resource Downloader & Integrity Verifier for Pre-MZT Cis-Grammar Pipeline"
    )
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory containing study")
    parser.add_argument("--config", default=None, help="Path to config.json snapshot")
    parser.add_argument("--target-species", default=None, help="Target species name")
    parser.add_argument("--target-fa", default=None, help="Path to target genome FASTA")
    parser.add_argument("--target-gtf", default=None, help="Path to target GTF")
    parser.add_argument("--zf-fa", default=None, help="Path to zebrafish genome FASTA")
    parser.add_argument("--zf-gtf", default=None, help="Path to zebrafish GTF")
    parser.add_argument("--enable-medaka", action="store_true", help="Download medaka references")
    parser.add_argument("--disable-medaka", action="store_true", help="Explicitly disable medaka downloads")
    parser.add_argument("--medaka-fa", default=None, help="Path to medaka genome FASTA")
    parser.add_argument("--medaka-gtf", default=None, help="Path to medaka GTF")
    parser.add_argument("--jaspar-meme", default=None, help="Path to JASPAR CORE vertebrates MEME file")
    parser.add_argument("--tf-list", default=None, help="Path to pre-MZT TF list")
    parser.add_argument("--ortholog-map", default=None, help="Path to target-to-zf ortholog map")
    parser.add_argument("--skip-slamseq", action="store_true", help="Skip SLAM-seq FASTQ download")
    parser.add_argument("--slamseq-table", default=None, help="Path to pre-computed SLAM-seq table")
    parser.add_argument("--max-host-workers", type=int, default=3, help="Max distinct host domains in parallel")
    parser.add_argument("--max-per-host-workers", type=int, default=2, help="Max parallel download workers per host (e.g. EBI)")
    parser.add_argument("--max-retries", type=int, default=0, help="Max download retries (0 or None = infinite)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    cfg_file = Path(args.config) if args.config else work_dir / "config.json"
    cfg: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)

    target_species = args.target_species or cfg.get("TARGET_SPECIES_NAME")
    target_fa = args.target_fa or cfg.get("TARGET_GENOME_FA")
    target_gtf = args.target_gtf or cfg.get("TARGET_GTF")
    zf_fa = args.zf_fa or cfg.get("ZEBRAFISH_GENOME_FA")
    zf_gtf = args.zf_gtf or cfg.get("ZEBRAFISH_GTF")
    medaka_fa = args.medaka_fa or cfg.get("MEDAKA_GENOME_FA")
    medaka_gtf = args.medaka_gtf or cfg.get("MEDAKA_GTF")
    jaspar_meme = args.jaspar_meme or cfg.get("JASPAR_MEME_PATH")
    tf_list = args.tf_list or cfg.get("PREMZT_TF_LIST_PATH")
    ortholog_map = args.ortholog_map or cfg.get("ORTHOLOG_MAP_PATH")

    if args.disable_medaka:
        enable_medaka = False
    elif args.enable_medaka:
        enable_medaka = True
    else:
        enable_medaka = cfg.get("ENABLE_MEDAKA", False)

    slamseq_table = args.slamseq_table or cfg.get("SLAMSEQ_GENE_TABLE_PATH")
    zf_atac_cfg = cfg.get("ZEBRAFISH_ATAC_ACCESSIONS")
    med_atac_cfg = cfg.get("MEDAKA_ATAC_ACCESSIONS")
    slam_cfg = cfg.get("SLAM_ACCESSIONS")
    use_pretrained_bias = cfg.get("USE_PRETRAINED_BIAS", False)

    max_retries = None if args.max_retries <= 0 else args.max_retries

    download_all_resources(
        work_dir=work_dir,
        target_species=target_species,
        target_fa=target_fa,
        target_gtf=target_gtf,
        zf_fa=zf_fa,
        zf_gtf=zf_gtf,
        enable_medaka=enable_medaka,
        medaka_fa=medaka_fa,
        medaka_gtf=medaka_gtf,
        jaspar_meme_path=jaspar_meme,
        premzt_tf_list=tf_list,
        ortholog_map_path=ortholog_map,
        slamseq_gene_table=slamseq_table,
        skip_slamseq=args.skip_slamseq,
        max_host_workers=args.max_host_workers,
        max_per_host_workers=args.max_per_host_workers,
        max_retries=max_retries,
        atac_accessions_zf=zf_atac_cfg,
        atac_accessions_medaka=med_atac_cfg,
        slam_accessions=slam_cfg,
        use_pretrained_bias=use_pretrained_bias
    )


if __name__ == "__main__":
    main()
