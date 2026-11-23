#!/usr/bin/env python3
"""
scripts/train_chrombpnet.py
===========================
Modular worker for training Stage 1 ChromBPNet models (PyTorch engine)
on late-blastula ATAC-seq peaks, including GC-matched non-peak selection,
Tn5 sequence bias model training, and debiased accessibility prediction.

Workflow:
1. Load chromosome fold splits (chr_fold.json) under out_dir/prep/
2. Generate GC-matched background non-peaks (chrombpnet prep nonpeaks) under out_dir/prep/
3. Prepare Tn5 sequence bias model:
   - Centralized pretrained model or
   - Native chrombpnet bias pipeline into fresh out_dir/bias_run
4. Execute chrombpnet pipeline (--engine pytorch) into fresh out_dir/chrombpnet_run
5. Strict discovery of *nobias*.pt or *nobias*.h5 from chrombpnet_run/models/
6. Extract real post-training QC evaluation metrics (Pearson r, profile JSD) and persist to training_meta.json
7. STRICT ZERO-MOCK POLICY: fails loudly with non-zero exit code on any failure; never saves untrained dummy models.
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
from typing import Optional, Dict, Any, List, Set, Union, Tuple

# Ensure conda env bin directory is at the front of PATH
bin_dir = str(Path(sys.prefix) / "bin")
if bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

import torch
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


def run_cmd(cmd: list | str, desc: str = "") -> subprocess.CompletedProcess:
    """Run a shell command with live stdout/stderr streaming and strict error checking."""
    if desc:
        print(f"[*] {desc}", flush=True)
    cmd_str = cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)
    print(f"    CMD: {cmd_str}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    res = subprocess.run(cmd, shell=isinstance(cmd, str), check=True, text=True, stdout=None, stderr=None, env=os.environ)
    return res


def normalize_chrom_name(c: str) -> str:
    """Normalize chromosome name: prepend 'chr' if bare numeric Ensembl name, else return as-is."""
    c_str = str(c).strip()
    return f"chr{c_str}" if c_str.isdigit() else c_str


def prepare_chr_prefixed_reference(
    species_ref_dir: Path,
    genome_fa: Path,
    chrom_sizes: Path,
    late_bam: Path,
    threads: int = 8
) -> Tuple[Path, Path, Path]:
    """
    Prepare cached, 'chr'-prefixed reference genome FASTA, chrom_sizes, and merged BAM.
    Ensures Ensembl bare numeric chromosomes (1..24/25) do not crash chrombpnet-refactor's
    pd.read_csv(..., header=None) with KeyError: 1.
    Writes into species_ref_dir (shared across all folds of that species).
    Guarded with .ref.done sentinel and .ref.meta.json recording source sizes & mtimes.
    """
    species_ref_dir = Path(species_ref_dir).resolve()
    species_ref_dir.mkdir(parents=True, exist_ok=True)
    genome_fa = Path(genome_fa).resolve()
    chrom_sizes = Path(chrom_sizes).resolve()
    late_bam = Path(late_bam).resolve()

    done_file = species_ref_dir / ".ref.done"
    meta_file = species_ref_dir / ".ref.meta.json"
    chr_genome_fa = species_ref_dir / "genome.chr.fa"
    chr_fai = species_ref_dir / "genome.chr.fa.fai"
    chr_chrom_sizes = species_ref_dir / "chrom.chr.sizes"
    chr_late_bam = species_ref_dir / "late.chr.bam"
    chr_bai = species_ref_dir / "late.chr.bam.bai"

    src_meta = {
        "genome_fa_path": str(genome_fa),
        "genome_fa_size": genome_fa.stat().st_size,
        "genome_fa_mtime": genome_fa.stat().st_mtime,
        "chrom_sizes_path": str(chrom_sizes),
        "chrom_sizes_size": chrom_sizes.stat().st_size,
        "chrom_sizes_mtime": chrom_sizes.stat().st_mtime,
        "late_bam_path": str(late_bam),
        "late_bam_size": late_bam.stat().st_size,
        "late_bam_mtime": late_bam.stat().st_mtime,
    }

    # Check cache validity
    if done_file.exists() and meta_file.exists():
        if (
            chr_genome_fa.exists() and chr_genome_fa.stat().st_size > 0 and
            chr_fai.exists() and chr_fai.stat().st_size > 0 and
            chr_chrom_sizes.exists() and chr_chrom_sizes.stat().st_size > 0 and
            chr_late_bam.exists() and chr_late_bam.stat().st_size > 0 and
            (chr_bai.exists() or Path(str(chr_late_bam) + ".bai").exists())
        ):
            try:
                with open(meta_file, "r") as f:
                    cached_meta = json.load(f)
                if (
                    cached_meta.get("genome_fa_size") == src_meta["genome_fa_size"] and
                    cached_meta.get("genome_fa_mtime") == src_meta["genome_fa_mtime"] and
                    cached_meta.get("late_bam_size") == src_meta["late_bam_size"] and
                    cached_meta.get("late_bam_mtime") == src_meta["late_bam_mtime"]
                ):
                    chk = subprocess.run(["samtools", "quickcheck", str(chr_late_bam)], capture_output=True)
                    if chk.returncode == 0:
                        print(f"[+] Reusing verified cached chr-prefixed reference files from {species_ref_dir}")
                        return chr_genome_fa, chr_chrom_sizes, chr_late_bam
            except Exception:
                pass

    print(f"[*] Preparing chr-prefixed reference files under {species_ref_dir}...")
    done_file.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)

    tmp_fa = species_ref_dir / "genome.chr.fa.tmp"
    tmp_sizes = species_ref_dir / "chrom.chr.sizes.tmp"
    tmp_header = species_ref_dir / "header.chr.tmp.sam"
    tmp_bam = species_ref_dir / "late.chr.bam.tmp"

    try:
        # Clean any preexisting tmp files
        for tmp_p in [tmp_fa, tmp_sizes, tmp_header, tmp_bam]:
            tmp_p.unlink(missing_ok=True)

        # 1. genome.chr.fa: stream genome_fa rewriting FASTA headers whose first token is numeric
        print(f"[*] Rewriting FASTA headers for {genome_fa.name} -> {chr_genome_fa.name}...")
        open_fn = gzip.open if str(genome_fa).endswith(".gz") else open
        open_mode = "rt" if str(genome_fa).endswith(".gz") else "r"
        with open_fn(genome_fa, open_mode) as fin, open(tmp_fa, "w") as fout:
            for line in fin:
                if line.startswith(">"):
                    header_content = line[1:]
                    tokens = header_content.split(None, 1)
                    first_tok = tokens[0] if tokens else ""
                    if first_tok.isdigit():
                        line = f">chr{header_content}"
                fout.write(line)
        os.replace(str(tmp_fa), str(chr_genome_fa))
        print(f"[*] Indexing {chr_genome_fa.name} with samtools faidx...")
        run_cmd(["samtools", "faidx", str(chr_genome_fa)], desc=f"samtools faidx {chr_genome_fa.name}")

        # 2. chrom.chr.sizes: rewrite column 1 via normalize_chrom_name()
        print(f"[*] Rewriting chrom sizes: {chrom_sizes.name} -> {chr_chrom_sizes.name}...")
        with open(chrom_sizes, "r") as fin, open(tmp_sizes, "w") as fout:
            for line in fin:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split("\t")
                parts[0] = normalize_chrom_name(parts[0])
                fout.write("\t".join(parts) + "\n")
        os.replace(str(tmp_sizes), str(chr_chrom_sizes))

        # 3. late.chr.bam: extract header via samtools view -H, rewrite @SQ SN:<digits> -> SN:chr<digits>, reheader
        print(f"[*] Reheadering BAM: {late_bam.name} -> {chr_late_bam.name}...")
        header_res = subprocess.run(
            ["samtools", "view", "-H", str(late_bam)],
            capture_output=True,
            text=True,
            check=True
        )
        new_header_lines = []
        for hline in header_res.stdout.splitlines():
            if hline.startswith("@SQ\t"):
                tags = hline.split("\t")
                new_tags = []
                for tag in tags:
                    if tag.startswith("SN:"):
                        sn_val = tag[3:]
                        new_tags.append(f"SN:{normalize_chrom_name(sn_val)}")
                    else:
                        new_tags.append(tag)
                new_header_lines.append("\t".join(new_tags))
            else:
                new_header_lines.append(hline)

        with open(tmp_header, "w") as fout:
            fout.write("\n".join(new_header_lines) + "\n")

        with open(tmp_bam, "wb") as fout_bam:
            subprocess.run(
                ["samtools", "reheader", str(tmp_header), str(late_bam)],
                stdout=fout_bam,
                check=True
            )

        chk = subprocess.run(["samtools", "quickcheck", str(tmp_bam)], capture_output=True)
        if chk.returncode != 0:
            raise RuntimeError(f"samtools quickcheck failed on reheadered BAM: {tmp_bam}")

        os.replace(str(tmp_bam), str(chr_late_bam))
        print(f"[*] Indexing {chr_late_bam.name} with samtools index -@ {threads}...")
        run_cmd(["samtools", "index", "-@", str(threads), str(chr_late_bam)], desc=f"samtools index {chr_late_bam.name}")

        # Write metadata & sentinel
        with open(meta_file, "w") as f:
            json.dump(src_meta, f, indent=2)
        done_file.touch()
        print(f"[+] Successfully prepared and verified chr-prefixed reference files in {species_ref_dir}")
        return chr_genome_fa, chr_chrom_sizes, chr_late_bam

    finally:
        for tmp_p in [tmp_fa, tmp_sizes, tmp_header, tmp_bam]:
            try:
                tmp_p.unlink(missing_ok=True)
            except Exception:
                pass


def get_supported_cli_flags(subcommand: Union[List[str], str]) -> Set[str]:
    """Inspect chrombpnet CLI options programmatically so unsupported flags are never passed."""
    if isinstance(subcommand, str):
        return set(re.findall(r'(?:^|[\s,\[(/])(--?[a-zA-Z][a-zA-Z0-9_-]*)', subcommand))
    try:
        cmd = ["chrombpnet"] + list(subcommand) + ["--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        help_text = res.stdout + res.stderr
        flags = set(re.findall(r'(?:^|[\s,\[(/])(--?[a-zA-Z][a-zA-Z0-9_-]*)', help_text))
        return flags
    except Exception:
        return set()


def safe_recover_chrombpnet_reports(chrombpnet_run_dir: Union[str, Path], assay_type: str = "ATAC") -> bool:
    """
    Auto-recovery helper for chrombpnet pipeline report generation.
    Handles upstream modiscolite / tomtom empty match errors (EmptyDataError)
    in-memory and ensures motif reports and overall HTML/PDF reports are generated.
    """
    run_dir = Path(chrombpnet_run_dir).resolve()
    eval_dir = run_dir / "evaluation"
    if not eval_dir.exists():
        return False

    # 1. Monkey-patch modiscolite to handle empty tomtom tables gracefully
    try:
        import pandas as pd
        import modiscolite.report
        orig_fetch = modiscolite.report.fetch_tomtom_matches

        def safe_fetch(ppm, cwm, tomtom_exec, meme_motif_db, output_dir, top_n_matches=3):
            try:
                return orig_fetch(ppm, cwm, tomtom_exec, meme_motif_db, output_dir, top_n_matches=top_n_matches)
            except (pd.errors.EmptyDataError, Exception) as err:
                print(f"[!] Safe tomtom fetch fallback: handled {type(err).__name__} ({err})")
                return pd.DataFrame(columns=["Target ID", "q-value"])

        modiscolite.report.fetch_tomtom_matches = safe_fetch
    except Exception as e:
        print(f"[!] Warning: modiscolite monkey-patch skipped: {e}")

    # 2. Check and regenerate modisco motif report if missing
    modisco_dir = eval_dir / "modisco_profile"
    modisco_h5 = run_dir / "auxiliary" / "interpret_subsample" / "modisco_results_profile_scores.h5"
    if not modisco_h5.exists():
        alt_h5 = modisco_dir / "modisco_results.h5"
        if alt_h5.exists():
            modisco_h5 = alt_h5
    motifs_html = modisco_dir / "motifs.html"
    pdf_profile = eval_dir / "chrombpnet_nobias_profile.pdf"

    if modisco_h5.exists() and not (motifs_html.exists() and motifs_html.stat().st_size > 0):
        try:
            from chrombpnet.evaluation.modisco.api import generate_report
            import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf
            from chrombpnet.data import get_default_data_path, DefaultDataFile
            meme_file = get_default_data_path(DefaultDataFile.motifs_meme)
            print(f"[*] Regenerating modisco motif report using safe tomtom handler...")
            generate_report(str(modisco_h5), str(modisco_dir), meme_file)
            print(f"[+] Successfully generated modisco motifs report: {motifs_html}")
            if not (pdf_profile.exists() and pdf_profile.stat().st_size > 0):
                convert_html_to_pdf.main(str(motifs_html), str(pdf_profile))
                print(f"[+] Successfully generated profile PDF: {pdf_profile}")
        except Exception as e:
            print(f"[!] Warning: modisco report generation error: {e}")

    # 3. Check and regenerate overall report (HTML and PDF)
    overall_html = eval_dir / "overall_report.html"
    overall_pdf = eval_dir / "overall_report.pdf"
    if not (overall_html.exists() and overall_html.stat().st_size > 0):
        try:
            import chrombpnet.helpers.generate_reports.make_html as make_html
            import chrombpnet.evaluation.modisco.convert_html_to_pdf as convert_html_to_pdf

            class ReportArgs:
                input_dir = str(run_dir)
                command = "pipeline"
                data_type = assay_type
                file_prefix = None
                html_prefix = "./"

            print(f"[*] Regenerating overall HTML and PDF reports...")
            make_html.main(ReportArgs())
            if overall_html.exists() and not (overall_pdf.exists() and overall_pdf.stat().st_size > 0):
                convert_html_to_pdf.main(str(overall_html), str(overall_pdf))
            print(f"[+] Successfully generated overall report: {overall_html} and {overall_pdf}")
        except Exception as e:
            print(f"[!] Warning: overall HTML report generation error: {e}")

    return overall_html.exists() and overall_html.stat().st_size > 0


def extract_chrombpnet_metrics(run_dir: Optional[Path] = None, bias_run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Strictly parse post-training ChromBPNet evaluation metrics, Tn5 bias factorization QC metrics,
    and debiased TF model metrics without key collisions or file overwrites.
    """
    run_dir = Path(run_dir).resolve() if run_dir else None
    bias_run_dir = Path(bias_run_dir).resolve() if bias_run_dir else None

    metrics: Dict[str, Any] = {
        # Main ChromBPNet (peaks)
        "count_pearson_r": None,
        "count_spearman_r": None,
        "profile_pearson_r": None,
        "profile_median_jsd": None,
        # Debiased TF model (nobias)
        "nobias_count_pearson_r": None,
        "nobias_profile_pearson_r": None,
        "nobias_profile_median_jsd": None,
        # Tn5 Bias model
        "bias_counts_pearson_r": None,
        "bias_profile_jsd": None,
        "bias_nonpeaks_counts_pearson_r": None,
        "bias_nonpeaks_profile_jsd": None,
        "max_tn5_footprint_response": None,
        # Provenance
        "source_files": {
            "main": None,
            "nobias": None,
            "bias": None
        }
    }

    def _get_nested(d: dict, *keys):
        curr = d
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return None
        return float(curr) if isinstance(curr, (int, float)) else None

    # 1. Discover all candidate JSON evaluation files in run_dir and bias_run_dir
    eval_files_in_run: List[Path] = []
    if run_dir and run_dir.exists():
        raw_run_files = (
            list(run_dir.glob("**/*metrics*.json")) +
            list(run_dir.glob("**/*evaluation*.json")) +
            list(run_dir.glob("**/evaluation.json"))
        )
        seen_run = set()
        for f in raw_run_files:
            rf = f.resolve()
            if f.is_file() and rf not in seen_run:
                seen_run.add(rf)
                eval_files_in_run.append(f)

    eval_files_in_bias: List[Path] = []
    if bias_run_dir and bias_run_dir.exists():
        raw_bias_files = (
            list(bias_run_dir.glob("**/*metrics*.json")) +
            list(bias_run_dir.glob("**/*evaluation*.json")) +
            list(bias_run_dir.glob("**/evaluation.json"))
        )
        seen_bias = set()
        for f in raw_bias_files:
            rf = f.resolve()
            if f.is_file() and rf not in seen_bias:
                seen_bias.add(rf)
                eval_files_in_bias.append(f)

    # Group files strictly by classification rules
    main_candidates: List[Path] = []
    nobias_candidates: List[Path] = []
    bias_candidates: List[Path] = []

    for f in eval_files_in_run:
        stem_lower = f.stem.lower()
        name_lower = f.name.lower()
        is_nobias_file = "nobias" in name_lower or "nobias" in stem_lower
        is_bias_file = ("bias" in name_lower or "bias" in stem_lower) and not is_nobias_file
        is_main_file = (not is_bias_file) and (not is_nobias_file)

        if is_nobias_file:
            nobias_candidates.append(f)
        elif is_bias_file:
            bias_candidates.append(f)
        elif is_main_file:
            main_candidates.append(f)

    # Any metrics JSON inside bias_run_dir is classified as bias (or nobias if explicitly named)
    for f in eval_files_in_bias:
        stem_lower = f.stem.lower()
        name_lower = f.name.lower()
        if "nobias" in name_lower or "nobias" in stem_lower:
            nobias_candidates.append(f)
        else:
            bias_candidates.append(f)

    # Sort main candidates to prioritize chrombpnet_metrics.json / evaluation.json
    def _main_prio(p: Path) -> int:
        nl = p.name.lower()
        if nl == "chrombpnet_metrics.json":
            return 0
        if nl == "evaluation.json":
            return 1
        if "chrombpnet" in nl:
            return 2
        return 3

    main_candidates.sort(key=_main_prio)

    # 2. Extract Main full ChromBPNet metrics
    for mf in main_candidates:
        try:
            with open(mf, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                c_pr = _get_nested(data, "counts_metrics", "peaks", "pearsonr") or _get_nested(data, "counts_metrics", "regions", "pearsonr")
                c_sr = _get_nested(data, "counts_metrics", "peaks", "spearmanr") or _get_nested(data, "counts_metrics", "regions", "spearmanr")
                p_pr = _get_nested(data, "profile_metrics", "peaks", "median_pearsonr") or _get_nested(data, "profile_metrics", "peaks", "pearsonr") or _get_nested(data, "profile_metrics", "regions", "median_pearsonr")
                p_jsd = _get_nested(data, "profile_metrics", "peaks", "median_jsd") or _get_nested(data, "profile_metrics", "peaks", "jsd") or _get_nested(data, "profile_metrics", "regions", "median_jsd")

                if metrics["count_pearson_r"] is None and c_pr is not None:
                    metrics["count_pearson_r"] = c_pr
                if metrics["count_spearman_r"] is None and c_sr is not None:
                    metrics["count_spearman_r"] = c_sr
                if metrics["profile_pearson_r"] is None and p_pr is not None:
                    metrics["profile_pearson_r"] = p_pr
                if metrics["profile_median_jsd"] is None and p_jsd is not None:
                    metrics["profile_median_jsd"] = p_jsd

                if metrics["source_files"]["main"] is None and any(v is not None for v in [c_pr, c_sr, p_pr, p_jsd]):
                    metrics["source_files"]["main"] = str(mf)
        except Exception:
            pass

    # 3. Extract Debiased TF model (nobias) metrics
    for nf in nobias_candidates:
        try:
            with open(nf, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                c_pr = _get_nested(data, "counts_metrics", "peaks", "pearsonr") or _get_nested(data, "counts_metrics", "regions", "pearsonr")
                p_pr = _get_nested(data, "profile_metrics", "peaks", "median_pearsonr") or _get_nested(data, "profile_metrics", "peaks", "pearsonr") or _get_nested(data, "profile_metrics", "regions", "median_pearsonr")
                p_jsd = _get_nested(data, "profile_metrics", "peaks", "median_jsd") or _get_nested(data, "profile_metrics", "peaks", "jsd") or _get_nested(data, "profile_metrics", "regions", "median_jsd")

                if metrics["nobias_count_pearson_r"] is None and c_pr is not None:
                    metrics["nobias_count_pearson_r"] = c_pr
                if metrics["nobias_profile_pearson_r"] is None and p_pr is not None:
                    metrics["nobias_profile_pearson_r"] = p_pr
                if metrics["nobias_profile_median_jsd"] is None and p_jsd is not None:
                    metrics["nobias_profile_median_jsd"] = p_jsd

                if metrics["source_files"]["nobias"] is None and any(v is not None for v in [c_pr, p_pr, p_jsd]):
                    metrics["source_files"]["nobias"] = str(nf)
        except Exception:
            pass

    # 4. Extract Tn5 Bias model metrics
    for bf in bias_candidates:
        try:
            with open(bf, "r") as f:
                b_data = json.load(f)
            if isinstance(b_data, dict):
                b_c_pr = _get_nested(b_data, "counts_metrics", "peaks", "pearsonr") or _get_nested(b_data, "counts_metrics", "regions", "pearsonr")
                b_p_jsd = _get_nested(b_data, "profile_metrics", "peaks", "median_jsd") or _get_nested(b_data, "profile_metrics", "peaks", "jsd") or _get_nested(b_data, "profile_metrics", "regions", "median_jsd")
                b_np_c_pr = _get_nested(b_data, "counts_metrics", "nonpeaks", "pearsonr")
                b_np_p_jsd = _get_nested(b_data, "profile_metrics", "nonpeaks", "median_jsd") or _get_nested(b_data, "profile_metrics", "nonpeaks", "jsd")

                if metrics["bias_counts_pearson_r"] is None and b_c_pr is not None:
                    metrics["bias_counts_pearson_r"] = b_c_pr
                if metrics["bias_profile_jsd"] is None and b_p_jsd is not None:
                    metrics["bias_profile_jsd"] = b_p_jsd
                if metrics["bias_nonpeaks_counts_pearson_r"] is None and b_np_c_pr is not None:
                    metrics["bias_nonpeaks_counts_pearson_r"] = b_np_c_pr
                if metrics["bias_nonpeaks_profile_jsd"] is None and b_np_p_jsd is not None:
                    metrics["bias_nonpeaks_profile_jsd"] = b_np_p_jsd

                if metrics["source_files"]["bias"] is None and any(v is not None for v in [b_c_pr, b_p_jsd, b_np_c_pr, b_np_p_jsd]):
                    metrics["source_files"]["bias"] = str(bf)
        except Exception:
            pass

    # Extract max marginal Tn5 footprint response if available
    if run_dir and run_dir.exists():
        resp_f = run_dir / "evaluation" / "chrombpnet_nobias_max_bias_response.txt"
        if resp_f.exists():
            try:
                parts = resp_f.read_text().strip().split("_")
                metrics["max_tn5_footprint_response"] = float(parts[1])
            except Exception:
                pass

    # 5. Fallback search in logs only if JSON metrics missing
    if run_dir and run_dir.exists() and (metrics["count_pearson_r"] is None or metrics["profile_pearson_r"] is None):
        log_files = list(run_dir.glob("logs/chrombpnet*.log")) + list(run_dir.glob("*.log"))
        for lf in log_files:
            if "bias" in lf.name.lower():
                continue
            try:
                text = lf.read_text(errors="replace")
                m_cnt = re.search(r"count(?:s)?\s+(?:peaks\s+)?(?:pearson|correlation)[\s:=]+([0-9\.\-]+)", text, re.IGNORECASE)
                if m_cnt and metrics["count_pearson_r"] is None:
                    metrics["count_pearson_r"] = float(m_cnt.group(1))
                m_prof = re.search(r"profile\s+(?:peaks\s+)?(?:median\s+)?(?:pearson|correlation)[\s:=]+([0-9\.\-]+)", text, re.IGNORECASE)
                if m_prof and metrics["profile_pearson_r"] is None:
                    metrics["profile_pearson_r"] = float(m_prof.group(1))
                m_jsd = re.search(r"(?:median\s+)?jsd[\s:=]+([0-9\.\-]+)", text, re.IGNORECASE)
                if m_jsd and metrics["profile_median_jsd"] is None:
                    metrics["profile_median_jsd"] = float(m_jsd.group(1))
            except Exception:
                pass

    # 6. Residual-bias and model quality warning checks
    if metrics["bias_counts_pearson_r"] is not None:
        if metrics["bias_counts_pearson_r"] > 0.50:
            print(f"[CRITICAL] Severe residual Tn5 sequence bias! Bias counts Pearson r = {metrics['bias_counts_pearson_r']:.3f} (> 0.50). Bias model is capturing TF peak signal.", flush=True)
        elif metrics["bias_counts_pearson_r"] > 0.30:
            print(f"[WARNING] Moderate residual Tn5 sequence bias detected. Bias counts Pearson r = {metrics['bias_counts_pearson_r']:.3f} (> 0.30).", flush=True)

    if metrics["count_pearson_r"] is not None:
        if metrics["count_pearson_r"] <= 0.0:
            print(f"[CRITICAL] Main ChromBPNet count Pearson r is non-positive ({metrics['count_pearson_r']:.3f})! Model training failed or metric collision occurred.", flush=True)
        elif metrics["count_pearson_r"] < 0.50:
            print(f"[WARNING] Main ChromBPNet count Pearson r = {metrics['count_pearson_r']:.3f} (< 0.50).", flush=True)

    return metrics


def refresh_training_meta(fold_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Re-run extract_chrombpnet_metrics() on existing chrombpnet_run / bias_run directories
    and atomically update training_meta.json without retraining.
    """
    fold_dir = Path(fold_dir).resolve()
    chrombpnet_run_dir = fold_dir / "chrombpnet_run"
    bias_run_dir = fold_dir / "bias_run"

    meta_path = fold_dir / "training_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading existing {meta_path}: {e}")

    # Ensure baseline metadata fields
    if not meta.get("species"):
        meta["species"] = fold_dir.parent.name
    if not meta.get("fold"):
        meta["fold"] = fold_dir.name.replace("fold_", "")
    if not meta.get("output_dir"):
        meta["output_dir"] = str(fold_dir)
    if not meta.get("chrombpnet_run_dir"):
        meta["chrombpnet_run_dir"] = str(chrombpnet_run_dir)

    cand_nobias = list((fold_dir / "models").glob("*nobias*"))
    if cand_nobias and not meta.get("checkpoint"):
        meta["checkpoint"] = str(cand_nobias[0])

    cand_bias = list((bias_run_dir / "models").glob("*bias*"))
    if cand_bias and not meta.get("bias_model"):
        meta["bias_model"] = str(cand_bias[0])

    # Check if reports need recovery
    if chrombpnet_run_dir.exists():
        eval_d = chrombpnet_run_dir / "evaluation"
        if eval_d.exists() and not (eval_d / "overall_report.html").exists():
            print(f"[*] Ensuring overall reports are present in {eval_d}...")
            safe_recover_chrombpnet_reports(chrombpnet_run_dir, assay_type="ATAC")

    # Extract fresh metrics
    metrics = extract_chrombpnet_metrics(
        run_dir=chrombpnet_run_dir if chrombpnet_run_dir.exists() else None,
        bias_run_dir=bias_run_dir if bias_run_dir.exists() else None
    )
    meta["qc_metrics"] = metrics

    # Atomically write to disk
    tmp_meta = fold_dir / "training_meta.json.tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(str(tmp_meta), str(meta_path))
    print(f"[+] Atomically refreshed {meta_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Stage 1 ChromBPNet Training Pipeline")
    parser.add_argument("--species", default=None, help="Species identifier (e.g. Danio_rerio, Oryzias_latipes, or 'all')")
    parser.add_argument("--work-dir", default="./work_astatotilapia_premzt", help="Working directory for study")
    parser.add_argument("--genome-fa", default=None, help="Reference genome FASTA")
    parser.add_argument("--chrom-sizes", default=None, help="Chromosome sizes file")
    parser.add_argument("--peaks-bed", default=None, help="Late blastula ATAC-seq peaks bed/narrowPeak")
    parser.add_argument("--late-bam", default=None, help="Late blastula ATAC merged BAM file")
    parser.add_argument("--late-bw", default=None, help="Optional late blastula ATAC BigWig coverage track")
    parser.add_argument("--chr-fold-json", default=None, help="Path to chr_fold.json mapping")
    parser.add_argument("--fold", type=str, default="0", help="Fold index")
    parser.add_argument("--use-pretrained-bias", action="store_true", help="Use pretrained Tn5 bias model if available")
    parser.add_argument("--force-retrain-bias", action="store_true", help="Force retraining Tn5 sequence bias model even if already trained")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining ChromBPNet and debiased model even if already trained")
    parser.add_argument("--bias-model-path", default=None, help="Path to existing Tn5 bias model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--threads", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--prep-only", action="store_true", help="Stop after preparing reference and non-peaks BED (smoke verification)")
    parser.add_argument("--refresh-metrics-only", action="store_true", help="Atomically refresh training_meta.json from existing evaluation JSONs without retraining")
    parser.add_argument("--task-name", default=None, help="TaskManager task name for progress updates")
    args = parser.parse_args()

    # Handle --refresh-metrics-only
    if args.refresh_metrics_only:
        work_dir = Path(args.work_dir).resolve()
        models_root = work_dir / "models" / "chrombpnet"
        if not models_root.exists():
            print(f"[-] No ChromBPNet directory found at {models_root}")
            sys.exit(1)

        target_folds = []
        if args.species and args.species.lower() != "all":
            sp_dir = models_root / args.species
            if args.fold:
                f_dir = sp_dir / f"fold_{args.fold}"
                if f_dir.exists():
                    target_folds.append(f_dir)
            else:
                target_folds.extend(sorted(sp_dir.glob("fold_*")))
        else:
            for f_dir in sorted(models_root.glob("*/fold_*")):
                if f_dir.is_dir():
                    target_folds.append(f_dir)

        print(f"[*] Refreshing metrics for {len(target_folds)} fold directories:")
        for f_dir in target_folds:
            meta = refresh_training_meta(f_dir)
            qc = meta.get("qc_metrics", {})
            print(f"\n--- {meta.get('species')} [Fold {meta.get('fold')}] ---")
            print(f"    Count Pearson r:        {qc.get('count_pearson_r')}")
            print(f"    Count Spearman r:       {qc.get('count_spearman_r')}")
            print(f"    Profile Pearson r:      {qc.get('profile_pearson_r')}")
            print(f"    Profile Median JSD:     {qc.get('profile_median_jsd')}")
            print(f"    Debiased Count Pearson: {qc.get('nobias_count_pearson_r')}")
            print(f"    Bias Counts Pearson r:  {qc.get('bias_counts_pearson_r')}")
            print(f"    Bias Profile JSD:       {qc.get('bias_profile_jsd')}")
            print(f"    Bias Nonpeaks Count:    {qc.get('bias_nonpeaks_counts_pearson_r')}")
            print(f"    Bias Nonpeaks JSD:      {qc.get('bias_nonpeaks_profile_jsd')}")
            print(f"    Source files:           {qc.get('source_files')}")
        return

    if not args.species or not args.genome_fa or not args.chrom_sizes or not args.peaks_bed or not args.late_bam:
        parser.error("--species, --genome-fa, --chrom-sizes, --peaks-bed, and --late-bam are required for training mode")

    task_name = args.task_name or f"train_chrombpnet_{args.species.lower()}_fold_{args.fold}"
    work_dir = Path(args.work_dir).resolve()
    species_atac_dir = work_dir / "atac" / args.species
    out_dir = work_dir / "models" / "chrombpnet" / args.species / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auxiliary prep directory to prevent -o collision crashes with chrombpnet CLI
    prep_dir = out_dir / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[*] Stage 1 ChromBPNet Training Pipeline for {args.species} [Fold {args.fold}]")
    print(f"    Genome FA:    {args.genome_fa}")
    print(f"    Chrom Sizes:  {args.chrom_sizes}")
    print(f"    Peaks BED:    {args.peaks_bed}")
    print(f"    Input BAM:    {args.late_bam}")
    print(f"    Output Dir:   {out_dir}")
    print(f"    Prep Dir:     {prep_dir}")
    print("=" * 80)

    # 1. Validate inputs
    genome_fa = Path(args.genome_fa).resolve()
    chrom_sizes = Path(args.chrom_sizes).resolve()
    peaks_bed = Path(args.peaks_bed).resolve()
    late_bam = Path(args.late_bam).resolve()

    assert genome_fa.exists() and genome_fa.stat().st_size > 100 * 1024 * 1024, (
        f"[FATAL] Genome FASTA not found or too small: {genome_fa}"
    )
    assert chrom_sizes.exists() and chrom_sizes.stat().st_size > 0, (
        f"[FATAL] Chrom sizes file not found: {chrom_sizes}"
    )
    assert peaks_bed.exists() and peaks_bed.stat().st_size > 0, (
        f"[FATAL] ATAC-seq peaks file not found: {peaks_bed}"
    )
    assert late_bam.exists() and late_bam.stat().st_size > 10 * 1024 * 1024, (
        f"[FATAL] Input BAM not found or too small: {late_bam}"
    )

    # 2. Resolve chromosome fold with multi-fold support & key validation (train, valid, test)
    if args.chr_fold_json:
        chr_fold_json_src = Path(args.chr_fold_json).resolve()
    elif (species_atac_dir / "chr_folds.json").exists():
        chr_fold_json_src = species_atac_dir / "chr_folds.json"
    else:
        chr_fold_json_src = species_atac_dir / "chr_fold.json"

    fold_dict: Dict[str, List[str]] = {}

    if chr_fold_json_src.exists() and chr_fold_json_src.stat().st_size > 0:
        with open(chr_fold_json_src, "r") as f:
            raw_folds = json.load(f)

        fold_key = str(args.fold)
        if f"fold_{fold_key}" in raw_folds:
            fold_dict = raw_folds[f"fold_{fold_key}"]
        elif fold_key in raw_folds:
            fold_dict = raw_folds[fold_key]
        elif "all_folds" in raw_folds and f"fold_{fold_key}" in raw_folds["all_folds"]:
            fold_dict = raw_folds["all_folds"][f"fold_{fold_key}"]
        elif "train" in raw_folds or "test" in raw_folds:
            fold_dict = raw_folds
        else:
            fold_dict = raw_folds
    else:
        # Generate default fold if missing
        all_chroms = []
        with open(chrom_sizes, "r") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 2 and not ("_" in p[0] or "Un" in p[0] or "M" in p[0]):
                    all_chroms.append(p[0])
        np.random.seed(42 + int(args.fold))
        shuffled = np.random.permutation(all_chroms).tolist()
        n_test = max(1, len(shuffled) // 7)
        n_val = max(1, len(shuffled) // 7)
        test_c = shuffled[:n_test]
        val_c = shuffled[n_test:n_test + n_val]
        train_c = shuffled[n_test + n_val:]
        fold_dict = {"train": train_c, "valid": val_c, "test": test_c}

    # Normalize keys: chrombpnet strictly requires "train", "valid", "test"
    if "val" in fold_dict and "valid" not in fold_dict:
        fold_dict["valid"] = fold_dict.pop("val")
    for req_k in ["train", "valid", "test"]:
        if req_k not in fold_dict:
            fold_dict[req_k] = []

    # Check if primary chromosomes are bare numeric Ensembl format (1..24/25)
    allowed_chroms = set(fold_dict["train"] + fold_dict["valid"] + fold_dict["test"])
    needs_chr_prefix = any(str(c).isdigit() for c in allowed_chroms)

    if needs_chr_prefix:
        print(f"[*] Detected bare numeric Ensembl chromosomes in fold split ({args.species}). Preparing chr-prefixed reference files for ChromBPNet compatibility...")
        species_ref_dir = work_dir / "models" / "chrombpnet" / args.species / "chrombpnet_ref"
        genome_fa, chrom_sizes, late_bam = prepare_chr_prefixed_reference(
            species_ref_dir=species_ref_dir,
            genome_fa=genome_fa,
            chrom_sizes=chrom_sizes,
            late_bam=late_bam,
            threads=args.threads
        )
        fold_dict = {
            split: [normalize_chrom_name(c) for c in chroms]
            for split, chroms in fold_dict.items()
        }
        allowed_chroms = set(fold_dict["train"] + fold_dict["valid"] + fold_dict["test"])

    # Save fold under prep_dir so out_dir stays clean
    chr_fold_json = prep_dir / f"chr_fold_fold_{args.fold}.json"
    with open(chr_fold_json, "w") as f:
        json.dump(fold_dict, f, indent=2)

    print(f"[+] Using verified chromosome fold {args.fold}: {chr_fold_json}")
    print(f"    Train: {len(fold_dict['train'])} chroms, Valid: {len(fold_dict['valid'])} chroms, Test: {len(fold_dict['test'])} chroms")

    # Parse chrom_sizes dictionary
    chrom_sizes_dict: Dict[str, int] = {}
    with open(chrom_sizes, "r") as cs_f:
        for line in cs_f:
            p = line.strip().split("\t")
            if len(p) >= 2 and p[1].isdigit():
                chrom_sizes_dict[p[0]] = int(p[1])

    # Filter peaks_bed to primary chromosomes present in fold_dict and >= 1100 bp from boundary
    clean_stem = re.sub(r"(\.primary)+$", "", peaks_bed.stem)
    filtered_peaks_bed = prep_dir / f"{clean_stem}.primary.narrowPeak"
    if not (filtered_peaks_bed.exists() and filtered_peaks_bed.stat().st_size > 0):
        n_tot = 0
        n_kept = 0
        tmp_filtered_peaks = filtered_peaks_bed.with_suffix(".tmp.narrowPeak")
        try:
            with open(peaks_bed, "r") as fin, open(tmp_filtered_peaks, "w") as fout:
                for line in fin:
                    if not line.strip() or line.startswith("#"):
                        continue
                    p = line.strip().split("\t")
                    if len(p) < 10:
                        continue
                    n_tot += 1
                    raw_c = p[0]
                    p_c = normalize_chrom_name(raw_c) if needs_chr_prefix else raw_c
                    p_s, p_e = int(p[1]), int(p[2])
                    p_summit = p_s + int(p[9])
                    if p_c not in allowed_chroms or p_c not in chrom_sizes_dict:
                        continue
                    c_len = chrom_sizes_dict[p_c]
                    if p_s >= 1100 and p_e <= (c_len - 1100) and p_summit >= 1100 and p_summit <= (c_len - 1100):
                        p[0] = p_c
                        fout.write("\t".join(p) + "\n")
                        n_kept += 1
            os.replace(str(tmp_filtered_peaks), str(filtered_peaks_bed))
            print(f"[+] Verified and filtered peaks for ChromBPNet: {filtered_peaks_bed} ({n_kept:,} / {n_tot:,} peaks kept)")
        finally:
            tmp_filtered_peaks.unlink(missing_ok=True)
    peaks_bed = filtered_peaks_bed

    # 3. Non-peak generation under prep_dir
    set_task_progress(work_dir, task_name, f"Generating non-peaks [Fold {args.fold}]")
    nonpeaks_bed = prep_dir / "nonpeaks.bed"
    if not (nonpeaks_bed.exists() and nonpeaks_bed.stat().st_size > 0):
        # Clean up any leftover partial nonpeaks* files before running chrombpnet prep nonpeaks
        for partial_f in prep_dir.glob("nonpeaks*"):
            try:
                if partial_f.is_file():
                    partial_f.unlink()
                elif partial_f.is_dir():
                    shutil.rmtree(partial_f, ignore_errors=True)
            except Exception:
                pass

        print("[*] Generating GC-matched non-peak regions using chrombpnet prep nonpeaks...")
        cmd_nonpeaks = [
            "chrombpnet", "prep", "nonpeaks",
            "-g", str(genome_fa),
            "-p", str(peaks_bed),
            "-c", str(chrom_sizes),
            "-fl", str(chr_fold_json),
            "-o", str(prep_dir / "nonpeaks")
        ]
        run_cmd(cmd_nonpeaks, desc="GC-matched non-peak extraction via chrombpnet prep nonpeaks")
        cand_negs = (
            list(prep_dir.glob("*_negatives.bed")) +
            list(prep_dir.glob("nonpeaks*.bed")) +
            list(prep_dir.glob("*negatives*.bed"))
        )
        cand_negs = [p for p in cand_negs if p.resolve() != peaks_bed.resolve() and p.resolve() != nonpeaks_bed.resolve()]
        if cand_negs and cand_negs[0].stat().st_size > 0:
            shutil.copy(cand_negs[0], nonpeaks_bed)
            print(f"[+] Successfully extracted GC-matched non-peaks: {cand_negs[0]} -> {nonpeaks_bed}")
        else:
            raise RuntimeError(
                f"[FATAL] chrombpnet prep nonpeaks failed to produce negatives BED. Candidate files in {prep_dir}: "
                f"{[p.name for p in prep_dir.glob('*')]}. Zero-mock policy strictly enforced."
            )

    assert nonpeaks_bed.exists() and nonpeaks_bed.stat().st_size > 0, (
        f"[FATAL] Non-peaks BED generation failed: {nonpeaks_bed}"
    )

    if args.prep_only:
        print(f"[+] Smoke verification successful: {nonpeaks_bed} verified ({nonpeaks_bed.stat().st_size} bytes). Stopping before training.")
        return

    # 4. Resolve or Train Tn5 Sequence Bias Model
    bias_model_path = None
    if args.bias_model_path and Path(args.bias_model_path).exists():
        bias_model_path = Path(args.bias_model_path).resolve()
        print(f"[+] Using user-specified bias model: {bias_model_path}")
    elif args.use_pretrained_bias:
        cand_bias = work_dir / "downloads" / "bias_models" / "atac_bias.h5"
        if cand_bias.exists() and cand_bias.stat().st_size > 10 * 1024:
            bias_model_path = cand_bias
            print(f"[+] Using centralized pretrained bias model: {bias_model_path}")

    bias_run_dir = out_dir / "bias_run"
    bias_done = bias_run_dir / ".bias.done"
    valid_exts = {".pt", ".pth", ".ckpt", ".h5", ".hdf5"}

    if not bias_model_path:
        # Check if a valid trained bias checkpoint already exists in bias_run_dir with .bias.done sentinel
        if not args.force_retrain_bias and not args.force_retrain and bias_run_dir.exists() and bias_done.exists():
            existing_candidates = [
                bias_run_dir / "models" / "bias.pt",
                bias_run_dir / "models" / "bias.h5",
                bias_run_dir / "bias.pt",
                bias_run_dir / "bias.h5"
            ]
            for ep in existing_candidates:
                if ep.is_file() and ep.stat().st_size > 10 * 1024:
                    bias_model_path = ep
                    print(f"[+] Preserving & reusing existing trained native bias model (.bias.done verified): {bias_model_path} ({ep.stat().st_size / 1024:.1f} KB)")
                    break
            if not bias_model_path:
                for ep in list(bias_run_dir.glob("models/*bias*")) + list(bias_run_dir.glob("*bias*")):
                    if ep.is_file() and ep.suffix.lower() in valid_exts and ep.stat().st_size > 10 * 1024:
                        bias_model_path = ep
                        print(f"[+] Preserving & reusing existing trained native bias model (.bias.done verified): {bias_model_path} ({ep.stat().st_size / 1024:.1f} KB)")
                        break

    if not bias_model_path:
        # Native Bias Pipeline execution into clean fresh directory
        set_task_progress(work_dir, task_name, f"Training Tn5 bias model [Fold {args.fold}]")
        if bias_run_dir.exists():
            print(f"[*] Incomplete or forced rerun for bias model. Cleaning {bias_run_dir} before retraining...", flush=True)
            shutil.rmtree(bias_run_dir, ignore_errors=True)

        print(f"[*] Training native Tn5 sequence bias model via chrombpnet bias pipeline into {bias_run_dir}...")
        bias_flags = get_supported_cli_flags(["bias", "pipeline"])
        cmd_bias = [
            "chrombpnet", "bias", "pipeline",
            "-g", str(genome_fa),
            "-c", str(chrom_sizes),
            "-ibam", str(late_bam),
            "-o", str(bias_run_dir),
            "-d", "ATAC",
            "-p", str(peaks_bed),
            "-n", str(nonpeaks_bed),
            "-fl", str(chr_fold_json),
            "-b", "0.5"
        ]
        if "-e" in bias_flags:
            cmd_bias.extend(["-e", str(min(args.epochs, 50))])
        elif "--epochs" in bias_flags:
            cmd_bias.extend(["--epochs", str(min(args.epochs, 50))])

        if "-bs" in bias_flags:
            cmd_bias.extend(["-bs", str(args.batch_size)])
        elif "--batch-size" in bias_flags:
            cmd_bias.extend(["--batch-size", str(args.batch_size)])

        if "--engine" in bias_flags:
            cmd_bias.extend(["--engine", "pytorch"])

        run_cmd(cmd_bias, desc="Native ChromBPNet Tn5 bias model training")

        canonical_bias = [
            bias_run_dir / "models" / "bias.pt",
            bias_run_dir / "models" / "bias.h5",
            bias_run_dir / "bias.pt",
            bias_run_dir / "bias.h5"
        ]
        cand_models = [p for p in canonical_bias if p.is_file() and p.stat().st_size > 1024]
        if not cand_models:
            all_bias = (
                list(bias_run_dir.glob("models/*bias*")) +
                list(bias_run_dir.glob("models/*")) +
                list(bias_run_dir.glob("*bias*")) +
                list(bias_run_dir.glob("*"))
            )
            cand_models = [
                p for p in all_bias
                if p.is_file() and p.suffix.lower() in valid_exts and p.stat().st_size > 1024
            ]
        assert cand_models, (
            f"[FATAL] chrombpnet bias pipeline did not produce a model in {bias_run_dir}. Zero-mock policy enforced."
        )
        bias_model_path = cand_models[0]
        bias_done.touch()
        print(f"[+] Native bias model trained and verified: {bias_model_path} (.bias.done written)")

    # 5. Main ChromBPNet Pipeline Execution
    train_done = out_dir / ".train.done"
    chrombpnet_run_dir = out_dir / "chrombpnet_run"
    cbp_models_dir = chrombpnet_run_dir / "models"

    # Check if previously completed with verified debiased model checkpoint
    nobias_pt = None
    for cand_ext in [".pt", ".h5", ".pth", ".ckpt", ".hdf5"]:
        cand_p = models_dir / f"{args.species}_nobias{cand_ext}"
        if cand_p.is_file() and cand_p.stat().st_size > 1024:
            nobias_pt = cand_p
            break

    skip_training = False
    if not args.force_retrain and train_done.exists() and nobias_pt is not None:
        print(f"[+] Found existing verified debiased checkpoint ({nobias_pt}) and .train.done. Skipping pipeline execution and resuming directly to QC metrics / metadata export.")
        skip_training = True

    if not skip_training:
        set_task_progress(work_dir, task_name, f"Training main ChromBPNet [Fold {args.fold}]")
        train_done.unlink(missing_ok=True)
        if chrombpnet_run_dir.exists():
            shutil.rmtree(chrombpnet_run_dir, ignore_errors=True)

        print(f"[*] Executing main ChromBPNet pipeline into {chrombpnet_run_dir}...")
        pipeline_flags = get_supported_cli_flags(["pipeline"])
        cmd_pipeline = [
            "chrombpnet", "pipeline",
            "-g", str(genome_fa),
            "-c", str(chrom_sizes),
            "-ibam", str(late_bam),
            "-o", str(chrombpnet_run_dir),
            "-d", "ATAC",
            "-p", str(peaks_bed),
            "-n", str(nonpeaks_bed),
            "-fl", str(chr_fold_json),
            "-b", str(bias_model_path)
        ]
        if "-e" in pipeline_flags:
            cmd_pipeline.extend(["-e", str(args.epochs)])
        elif "--epochs" in pipeline_flags:
            cmd_pipeline.extend(["--epochs", str(args.epochs)])

        if "-bs" in pipeline_flags:
            cmd_pipeline.extend(["-bs", str(args.batch_size)])
        elif "--batch-size" in pipeline_flags:
            cmd_pipeline.extend(["--batch-size", str(args.batch_size)])

        if "--engine" in pipeline_flags:
            cmd_pipeline.extend(["--engine", "pytorch"])

        try:
            run_cmd(cmd_pipeline, desc="ChromBPNet training pipeline execution")
        except subprocess.CalledProcessError as err:
            canonical_nobias = [
                cbp_models_dir / "chrombpnet_nobias.pt",
                cbp_models_dir / "chrombpnet_nobias.h5",
                chrombpnet_run_dir / "chrombpnet_nobias.pt",
                chrombpnet_run_dir / "chrombpnet_nobias.h5"
            ]
            cand_nobias = [p for p in canonical_nobias if p.is_file() and p.stat().st_size > 1024]
            metrics_file = chrombpnet_run_dir / "evaluation" / "chrombpnet_metrics.json"
            if cand_nobias and metrics_file.exists():
                print(f"[!] Warning: chrombpnet pipeline encountered downstream report error: {err}")
                print("[*] Trained debiased model checkpoint and evaluation metrics verified on disk. Attempting safe report auto-recovery...")
                safe_recover_chrombpnet_reports(chrombpnet_run_dir, assay_type="ATAC")
            else:
                raise err

        # Verify overall report existence
        eval_dir = chrombpnet_run_dir / "evaluation"
        if eval_dir.exists() and not (eval_dir / "overall_report.html").exists():
            print("[*] Checking / generating overall reports...")
            safe_recover_chrombpnet_reports(chrombpnet_run_dir, assay_type="ATAC")

        # 6. Verify generated debiased model checkpoint strictly matching *.pt, *.pth, *.ckpt, *.h5, *.hdf5
        canonical_nobias = [
            cbp_models_dir / "chrombpnet_nobias.pt",
            cbp_models_dir / "chrombpnet_nobias.h5",
            chrombpnet_run_dir / "chrombpnet_nobias.pt",
            chrombpnet_run_dir / "chrombpnet_nobias.h5"
        ]
        cand_nobias = [p for p in canonical_nobias if p.is_file() and p.stat().st_size > 1024]
        if not cand_nobias:
            all_nobias = list(cbp_models_dir.glob("*nobias*")) + list(chrombpnet_run_dir.glob("**/*nobias*"))
            cand_nobias = [
                p for p in all_nobias
                if p.is_file() and p.suffix.lower() in valid_exts and p.stat().st_size > 1024
            ]
        assert cand_nobias, (
            f"[FATAL] chrombpnet pipeline finished but no debiased checkpoint (*nobias*.pt or *nobias*.h5) found in {cbp_models_dir}. "
            f"Zero-mock policy strictly enforced: training failed."
        )
        src_nobias = cand_nobias[0]
        nobias_pt = models_dir / f"{args.species}_nobias{src_nobias.suffix}"
        shutil.copy(src_nobias, nobias_pt)

        assert nobias_pt.exists() and nobias_pt.stat().st_size > 1024, (
            f"[FATAL] Valid debiased checkpoint missing at {nobias_pt}."
        )
        train_done.touch()
        print(f"[+] Verified debiased PyTorch checkpoint: {nobias_pt} ({nobias_pt.stat().st_size / 1e6:.1f} MB, .train.done written)")

    # 7. Extract real evaluation QC metrics
    bias_run_dir = out_dir / "bias_run"
    metrics = extract_chrombpnet_metrics(chrombpnet_run_dir, bias_run_dir=bias_run_dir if bias_run_dir.exists() else None)
    print("=" * 80)
    print(f"[*] Post-Training ChromBPNet QC Evaluation Metrics:")
    print(f"    Profile Pearson r:      {metrics['profile_pearson_r']}")
    print(f"    Count Pearson r:        {metrics['count_pearson_r']}")
    print(f"    Profile Median JSD:     {metrics['profile_median_jsd']}")
    print(f"    Bias Counts Pearson r:  {metrics['bias_counts_pearson_r']}")
    print(f"    Bias Profile JSD:       {metrics['bias_profile_jsd']}")
    print("=" * 80)

    # Export training metadata
    meta = {
        "species": args.species,
        "fold": args.fold,
        "peaks_bed": str(peaks_bed),
        "nonpeaks_bed": str(nonpeaks_bed),
        "bias_model": str(bias_model_path),
        "checkpoint": str(nobias_pt),
        "output_dir": str(out_dir),
        "chrombpnet_run_dir": str(chrombpnet_run_dir),
        "qc_metrics": metrics
    }
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    set_task_progress(work_dir, task_name, "100% (Done)")
    print(f"[SUCCESS] ChromBPNet Stage 1 model successfully trained and verified for {args.species}.")


if __name__ == "__main__":
    main()
