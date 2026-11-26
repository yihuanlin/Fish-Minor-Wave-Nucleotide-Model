#!/usr/bin/env python3
"""
scripts/premzt_inference.py
===========================
Shared High-Performance Inference API for Pre-MZT Cis-Regulatory Modeling.
Provides unified, cached, and batched execution for ChromBPNet and NTv3 models:
- Entry points: infer_premzt(...) and infer_premzt_batch(...)
- Cached InferenceContext (ChromBPNet/NTv3 models, pyfaidx Fasta handles, parsed GTF tables)
- Strand-aware Locus Range DSL Parser (parse_locus_range)
- Direct sequence mode (short center-pad, exact single-shot, long tiled)
- Zero-mock policy, zero retraining, publication-ready generic design.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import sys
import json
import gzip
import time
import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import pyfaidx
    from pyfaidx import Fasta
except ImportError:
    pyfaidx = None
    Fasta = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("premzt_inference")


# ==============================================================================
# Sequence Processing & Helper Utilities
# ==============================================================================

def sanitize_sequence(seq: str) -> str:
    """Sanitize DNA sequence: uppercase and map non-ACGT to N."""
    seq = seq.upper()
    return re.sub(r"[^ACGT]", "N", seq)


def one_hot_encode_batch(seqs: List[str], seq_len: int = 2114) -> np.ndarray:
    """One-hot encode a batch of DNA sequences to shape (N, 4, L)."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    n = len(seqs)
    arr = np.zeros((n, 4, seq_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        s_clean = s.upper()
        cur_len = min(len(s_clean), seq_len)
        for j in range(cur_len):
            b = s_clean[j]
            idx = mapping.get(b)
            if idx is not None:
                arr[i, idx, j] = 1.0
    return arr


def reverse_complement_onehot(x_4xl: np.ndarray) -> np.ndarray:
    """Compute reverse complement of (N, 4, L) one-hot array."""
    # Reverse complementary channel swap: A(0)<->T(3), C(1)<->G(2) and reverse sequence length
    return x_4xl[:, [3, 2, 1, 0], ::-1].copy()


def dinuc_shuffle_sequence(seq: str, seed: Optional[int] = None) -> str:
    """Perform dinucleotide shuffle preserving dinucleotide frequencies."""
    import random
    rng = random.Random(seed) if seed is not None else random
    seq = seq.upper()
    if len(seq) <= 2:
        return seq

    words = [seq[i:i+2] for i in range(len(seq) - 1)]
    adj = {}
    for w in words:
        u, v = w[0], w[1]
        adj.setdefault(u, []).append(v)

    last_char = seq[-1]
    curr = seq[0]
    result = [curr]
    for _ in range(len(seq) - 1):
        if curr in adj and adj[curr]:
            nxt = rng.choice(adj[curr])
            adj[curr].remove(nxt)
            result.append(nxt)
            curr = nxt
        else:
            remaining = [c for clist in adj.values() for c in clist]
            if remaining:
                nxt = rng.choice(remaining)
                for k, clist in adj.items():
                    if nxt in clist:
                        clist.remove(nxt)
                        break
                result.append(nxt)
                curr = nxt
            else:
                result.append(last_char)
                break
    return "".join(result[:len(seq)])


# ==============================================================================
# GTF Data Structures & Canonical Feature Representations
# ==============================================================================

@dataclass
class GeneRecord:
    gene_id: str
    gene_name: str
    chrom: str
    start: int          # 0-based coordinate
    end: int            # 0-based coordinate (exclusive)
    strand: str         # '+' or '-'
    tss: int            # 0-based TSS
    tes: int            # 0-based TES
    canonical_transcript_id: Optional[str] = None
    exons: List[Tuple[int, int]] = field(default_factory=list)      # sorted 0-based [(s, e), ...]
    intron1: Optional[Tuple[int, int]] = None                      # 0-based (s, e)
    utr5: List[Tuple[int, int]] = field(default_factory=list)       # 0-based [(s, e), ...]
    utr3: List[Tuple[int, int]] = field(default_factory=list)       # 0-based [(s, e), ...]
    cds: List[Tuple[int, int]] = field(default_factory=list)        # 0-based [(s, e), ...]


# ==============================================================================
# InferenceContext: Singleton Caching & Resource Management
# ==============================================================================

class InferenceContext:
    """
    Central cached execution context:
    - Maintains loaded PyTorch model instances (ChromBPNet, NTv3, Tokenizer)
    - Caches pyfaidx.Fasta genome readers
    - Caches parsed GTF feature structures, canonical transcripts, and neighbor TSS arrays
    """

    def __init__(self, work_dir: Union[str, Path] = "./work_astatotilapia_premzt", device: Optional[str] = None):
        self.work_dir = Path(work_dir).resolve()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.bf16_supported = (self.device == "cuda" and torch.cuda.is_bf16_supported())
        self.autocast_dtype = torch.bfloat16 if self.bf16_supported else torch.float32

        # Caches
        self.models: Dict[str, Any] = {}
        self.fastas: Dict[str, Any] = {}
        self.gtf_data: Dict[str, Dict[str, Any]] = {}
        self.study_cfg: Dict[str, Any] = self._load_study_config()

        # Cache directory for pre-parsed indices
        self.cache_dir = self.work_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_study_config(self) -> Dict[str, Any]:
        cfg_path = self.work_dir / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config.json: {e}")
        return {}

    def resolve_species_key(self, species: Optional[str]) -> str:
        """Resolve species input aliases to standardized species name."""
        if not species:
            return self.study_cfg.get("TARGET_SPECIES_NAME", "Astatotilapia_calliptera")

        s = species.strip().lower()
        if s in ("ac", "astatotilapia", "astatotilapia_calliptera", "target"):
            return "Astatotilapia_calliptera"
        elif s in ("zf", "danio", "danio_rerio", "zebrafish"):
            return "Danio_rerio"
        elif s in ("ol", "medaka", "oryzias", "oryzias_latipes"):
            return "Oryzias_latipes"
        return species

    def get_species_paths(self, species: Optional[str]) -> Tuple[Path, Path]:
        """Return (genome_fa_path, gtf_path) for the given species."""
        sp = self.resolve_species_key(species)
        downloads = self.work_dir / "downloads"

        if sp == "Astatotilapia_calliptera":
            fa = Path(self.study_cfg.get("TARGET_GENOME_FA", downloads / "Astatotilapia_calliptera.fAstCal1.3.dna_sm.toplevel.fa"))
            gtf = Path(self.study_cfg.get("TARGET_GTF", downloads / "Astatotilapia_calliptera.fAstCal1.3.112.gtf.gz"))
        elif sp == "Danio_rerio":
            fa = Path(self.study_cfg.get("ZEBRAFISH_GENOME_FA", downloads / "Danio_rerio.GRCz11.dna_sm.primary_assembly.fa"))
            gtf = Path(self.study_cfg.get("ZEBRAFISH_GTF", downloads / "Danio_rerio.GRCz11.112.gtf.gz"))
        elif sp == "Oryzias_latipes":
            fa = Path(self.study_cfg.get("MEDAKA_GENOME_FA", downloads / "Oryzias_latipes.ASM223467v1.dna_sm.toplevel.fa"))
            gtf = Path(self.study_cfg.get("MEDAKA_GTF", downloads / "Oryzias_latipes.ASM223467v1.112.gtf.gz"))
        else:
            fa = downloads / f"{sp}.fa"
            gtf = downloads / f"{sp}.gtf.gz"

        if not fa.exists() and fa.with_suffix(".fa.gz").exists():
            fa = fa.with_suffix(".fa.gz")

        return fa, gtf

    def get_fasta(self, species: Optional[str]) -> Any:
        """Return cached pyfaidx.Fasta handle for the specified species."""
        sp = self.resolve_species_key(species)
        if sp in self.fastas:
            return self.fastas[sp]

        fa_path, _ = self.get_species_paths(sp)
        if not fa_path.exists():
            raise FileNotFoundError(f"FASTA file not found for species '{sp}': {fa_path}")

        if Fasta is None:
            raise ImportError("pyfaidx is required for genome sequence extraction.")

        fa_handle = Fasta(str(fa_path), as_raw=True)
        self.fastas[sp] = fa_handle
        return fa_handle

    def get_gtf_data(self, species: Optional[str]) -> Dict[str, Any]:
        """
        Parse and return cached GTF annotations:
        - 'genes': Dict[str, GeneRecord] (indexed by both gene_id and gene_name, lowercase)
        - 'neighbor_idx': Dict[str, np.ndarray] (sorted TSS coordinates per chromosome)
        - 'chrom_lengths': Dict[str, int]
        """
        sp = self.resolve_species_key(species)
        if sp in self.gtf_data:
            return self.gtf_data[sp]

        cache_file = self.cache_dir / f"gtf_parsed_{sp}.pkl"
        fa_path, gtf_path = self.get_species_paths(sp)

        if not gtf_path.exists():
            raise FileNotFoundError(f"GTF file not found for species '{sp}': {gtf_path}")

        # Check if pickled cache exists and is newer than gtf_path
        if cache_file.exists():
            try:
                if cache_file.stat().st_mtime >= gtf_path.stat().st_mtime:
                    with open(cache_file, "rb") as f_pkl:
                        data = pickle.load(f_pkl)
                        self.gtf_data[sp] = data
                        logger.info(f"Loaded cached GTF indices for {sp} ({len(data['genes'])//2} genes).")
                        return data
            except Exception as e_c:
                logger.debug(f"Cache load skipped: {e_c}")

        logger.info(f"Parsing GTF for {sp} from {gtf_path.name}...")
        t0 = time.time()

        genes_raw: Dict[str, Dict[str, Any]] = {}
        tx_exons: Dict[str, List[Tuple[int, int]]] = {}
        tx_cds: Dict[str, List[Tuple[int, int]]] = {}
        tx_utr5: Dict[str, List[Tuple[int, int]]] = {}
        tx_utr3: Dict[str, List[Tuple[int, int]]] = {}
        tx_to_gene: Dict[str, str] = {}
        gene_transcripts: Dict[str, List[str]] = {}

        opener = gzip.open(gtf_path, "rt") if str(gtf_path).endswith(".gz") else open(gtf_path, "rt")
        with opener as f_g:
            for line in f_g:
                if line.startswith("#"):
                    continue
                parts = line.rstrip().split("\t")
                if len(parts) < 9:
                    continue

                feat = parts[2]
                if feat not in ("gene", "transcript", "exon", "CDS", "five_prime_utr", "three_prime_utr"):
                    continue

                chrom = parts[0]
                start = int(parts[3]) - 1  # 0-based
                end = int(parts[4])        # 0-based exclusive
                strand = parts[6]
                attr_str = parts[8]

                # Quick attribute extraction
                gid, gname, tx_id = None, None, None
                for token in attr_str.split(";"):
                    token = token.strip()
                    if token.startswith("gene_id"):
                        gid = token.split('"', 2)[1] if '"' in token else token.split()[1]
                    elif token.startswith("gene_name"):
                        gname = token.split('"', 2)[1] if '"' in token else token.split()[1]
                    elif token.startswith("transcript_id"):
                        tx_id = token.split('"', 2)[1] if '"' in token else token.split()[1]

                if not gid:
                    continue

                if feat == "gene":
                    genes_raw[gid] = {
                        "gene_id": gid,
                        "gene_name": gname or gid,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "tss": start if strand == "+" else end,
                        "tes": end if strand == "+" else start,
                    }
                elif feat == "exon" and tx_id:
                    tx_exons.setdefault(tx_id, []).append((start, end))
                    tx_to_gene[tx_id] = gid
                    gene_transcripts.setdefault(gid, []).append(tx_id)
                elif feat == "CDS" and tx_id:
                    tx_cds.setdefault(tx_id, []).append((start, end))
                elif feat == "five_prime_utr" and tx_id:
                    tx_utr5.setdefault(tx_id, []).append((start, end))
                elif feat == "three_prime_utr" and tx_id:
                    tx_utr3.setdefault(tx_id, []).append((start, end))

        # Build finalized GeneRecord lookup
        gene_lookup: Dict[str, GeneRecord] = {}
        chrom_tss_dict: Dict[str, List[int]] = {}

        for gid, ginfo in genes_raw.items():
            strand = ginfo["strand"]
            tx_ids = list(dict.fromkeys(gene_transcripts.get(gid, [])))

            # Select canonical transcript: longest combined exon length
            best_tx = None
            best_len = -1
            for tid in tx_ids:
                exs = tx_exons.get(tid, [])
                tlen = sum(e[1] - e[0] for e in exs)
                if tlen > best_len:
                    best_len = tlen
                    best_tx = tid

            exons_sorted = sorted(tx_exons.get(best_tx, []), key=lambda x: x[0]) if best_tx else []
            cds_sorted = sorted(tx_cds.get(best_tx, []), key=lambda x: x[0]) if best_tx else []
            utr5_sorted = sorted(tx_utr5.get(best_tx, []), key=lambda x: x[0]) if best_tx else []
            utr3_sorted = sorted(tx_utr3.get(best_tx, []), key=lambda x: x[0]) if best_tx else []

            # First intron calculation
            intron1 = None
            if len(exons_sorted) >= 2:
                if strand == "+":
                    intron1 = (exons_sorted[0][1], exons_sorted[1][0])
                else:
                    intron1 = (exons_sorted[-2][1], exons_sorted[-1][0])

            # UTR fallback proxies if annotations lack explicit UTR entries
            if not utr5_sorted and cds_sorted:
                cds_min = min(c[0] for c in cds_sorted)
                cds_max = max(c[1] for c in cds_sorted)
                if strand == "+" and cds_min > ginfo["tss"]:
                    utr5_sorted = [(ginfo["tss"], cds_min)]
                elif strand == "-" and cds_max < ginfo["tss"]:
                    utr5_sorted = [(cds_max, ginfo["tss"])]

            if not utr3_sorted and cds_sorted:
                cds_min = min(c[0] for c in cds_sorted)
                cds_max = max(c[1] for c in cds_sorted)
                if strand == "+" and cds_max < ginfo["tes"]:
                    utr3_sorted = [(cds_max, ginfo["tes"])]
                elif strand == "-" and cds_min > ginfo["tes"]:
                    utr3_sorted = [(ginfo["tes"], cds_min)]

            rec = GeneRecord(
                gene_id=gid,
                gene_name=ginfo["gene_name"],
                chrom=ginfo["chrom"],
                start=ginfo["start"],
                end=ginfo["end"],
                strand=strand,
                tss=ginfo["tss"],
                tes=ginfo["tes"],
                canonical_transcript_id=best_tx,
                exons=exons_sorted,
                intron1=intron1,
                utr5=utr5_sorted,
                utr3=utr3_sorted,
                cds=cds_sorted,
            )

            gene_lookup[gid.lower()] = rec
            gene_lookup[ginfo["gene_name"].lower()] = rec
            chrom_tss_dict.setdefault(ginfo["chrom"], []).append(ginfo["tss"])

        # Build neighbor TSS index
        neighbor_idx = {ch: np.array(sorted(set(tss_list)), dtype=np.int64) for ch, tss_list in chrom_tss_dict.items()}

        # Fetch chromosome lengths
        chrom_lengths = {}
        if fa_path.exists() and Fasta is not None:
            try:
                fa = self.get_fasta(sp)
                for ch in neighbor_idx:
                    if ch in fa:
                        chrom_lengths[ch] = len(fa[ch])
            except Exception:
                pass

        data = {
            "genes": gene_lookup,
            "neighbor_idx": neighbor_idx,
            "chrom_lengths": chrom_lengths
        }

        # Save to disk cache
        try:
            with open(cache_file, "wb") as f_out:
                pickle.dump(data, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e_w:
            logger.debug(f"Could not write GTF cache: {e_w}")

        self.gtf_data[sp] = data
        logger.info(f"GTF parsed for {sp} in {time.time()-t0:.2f}s ({len(genes_raw):,} genes indexed).")
        return data

    def get_chrombpnet(self, fold: int = 0) -> nn.Module:
        """Load and cache grafted ChromBPNet model checkpoint."""
        key = f"chrombpnet_fold_{fold}"
        if key in self.models:
            return self.models[key]

        from scripts.infer_and_attribute import load_grafted_model
        cand_paths = [
            self.work_dir / "models" / "chrombpnet" / "Danio_rerio" / f"fold_{fold}" / "grafted_chrombpnet.pt",
            self.work_dir / "models" / "chrombpnet" / "Danio_rerio" / f"fold_{fold}" / "grafted_model.pt",
            self.work_dir / "models" / "chrombpnet" / f"fold_{fold}" / "grafted_chrombpnet.pt",
        ]
        ckpt = next((p for p in cand_paths if p.exists()), None)
        if not ckpt:
            raise FileNotFoundError(f"ChromBPNet grafted checkpoint not found under {self.work_dir}/models/chrombpnet/")

        logger.info(f"Loading ChromBPNet model from {ckpt}...")
        model = load_grafted_model(ckpt, device=self.device, work_dir=self.work_dir)
        model.to(self.device)
        model.eval()
        self.models[key] = model
        return model

    def get_ntv3(self) -> Tuple[nn.Module, Any, Dict[str, Any]]:
        """Load and cache NTv3 model, tokenizer, and config dictionary."""
        if "ntv3" in self.models:
            return self.models["ntv3"]

        from scripts.train_ntv3 import build_ntv3_model
        ntv3_dir = self.work_dir / "models" / "ntv3"
        cfg_file = ntv3_dir / "ntv3_train_config.json"
        if not cfg_file.exists():
            cfg_file = ntv3_dir / "ntv3_train_config_fold_0.json"

        if not cfg_file.exists():
            raise FileNotFoundError(f"NTv3 configuration not found under {ntv3_dir}")

        with open(cfg_file, "r") as f:
            n_cfg = json.load(f)

        ckpt_file = ntv3_dir / "ntv3_best_checkpoint.pt"
        if not ckpt_file.exists():
            ckpt_file = ntv3_dir / "ntv3_best_checkpoint_fold_0.pt"

        if not ckpt_file.exists():
            raise FileNotFoundError(f"NTv3 checkpoint not found under {ntv3_dir}")

        species_id = int(n_cfg.get("species_id", 8))
        context_len = int(n_cfg.get("context_len", 65536))
        model_id = n_cfg.get("model_id", "InstaDeepAI/NTv3_tracks")
        train_mode = n_cfg.get("train_mode", "lora")
        tracks = n_cfg.get("tracks", [
            "atac_256cell", "atac_dome",
            "slam_0.75hpf_fwd", "slam_0.75hpf_rev",
            "slam_2.5hpf_fwd", "slam_2.5hpf_rev",
            "slam_3.0_3.5hpf_fwd", "slam_3.0_3.5hpf_rev",
            "slam_4.5hpf_fwd", "slam_4.5hpf_rev"
        ])

        logger.info(f"Loading NTv3 model ({model_id}, species_id={species_id}, context_len={context_len})...")
        model, tok = build_ntv3_model(
            model_id=model_id,
            num_tracks=len(tracks),
            train_mode=train_mode,
            context_len=context_len,
            grad_checkpointing=False,
            default_species_id=species_id,
            hf_token=os.environ.get("HF_TOKEN")
        )

        payload = torch.load(ckpt_file, map_location=self.device, weights_only=True)
        model.load_state_dict(payload["model_state"])
        model.to(self.device)
        model.eval()

        meta = {
            "config": n_cfg,
            "species_id": species_id,
            "context_len": context_len,
            "crop_ratio": getattr(model, "crop_ratio", 0.375),
            "crop_len": int(round(context_len * getattr(model, "crop_ratio", 0.375))),
            "crop_start": (context_len - int(round(context_len * getattr(model, "crop_ratio", 0.375)))) // 2,
            "tracks": tracks,
            "idx_atac_256": tracks.index("atac_256cell") if "atac_256cell" in tracks else 0,
            "idx_atac_dome": tracks.index("atac_dome") if "atac_dome" in tracks else 1,
            "idx_slam_25_fwd": tracks.index("slam_2.5hpf_fwd") if "slam_2.5hpf_fwd" in tracks else 4,
            "idx_slam_25_rev": tracks.index("slam_2.5hpf_rev") if "slam_2.5hpf_rev" in tracks else 5,
        }
        meta["crop_end"] = meta["crop_start"] + meta["crop_len"]

        self.models["ntv3"] = (model, tok, meta)
        return model, tok, meta


# Global default context instance
_GLOBAL_CONTEXT: Optional[InferenceContext] = None

def get_inference_context(work_dir: Union[str, Path] = "./work_astatotilapia_premzt", device: Optional[str] = None) -> InferenceContext:
    """Retrieve or initialize the global shared InferenceContext."""
    global _GLOBAL_CONTEXT
    if _GLOBAL_CONTEXT is None:
        _GLOBAL_CONTEXT = InferenceContext(work_dir=work_dir, device=device)
    return _GLOBAL_CONTEXT


# ==============================================================================
# Flexible locus_range String DSL Parser
# ==============================================================================

def parse_locus_range(
    locus_range: str,
    gene_record: GeneRecord,
    chrom_len: Optional[int] = None,
    intron1_cap_bp: Optional[int] = 10000,
    neighbor_idx: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Strand-aware DSL parser mapping flexible locus strings to resolved genomic intervals.
    
    Supports:
    - Named presets: 'promoter_1kb', 'proximal_5kb', 'bounded_25kb', 'promoter_intron1_capped', 'gene', 'exons', 'tes_1kb'
    - DSL patterns (comma-separated):
      * 'TSS+-200' or 'TSS±200': symmetric ±200 bp around TSS
      * 'TSS+200-100': upstream 200 bp through downstream 100 bp relative to TSS
      * 'TSS+200': 200 bp upstream of TSS through TSS [TSS - 200, TSS] (strand-aware)
      * '+2000-1000': 2000 bp upstream of TSS through 1000 bp downstream of TES
      * 'TES+-500' or 'TES±500': symmetric ±500 bp around TES
      * 'TES+500-200': 500 bp upstream of TES through 200 bp downstream of TES
      * 'intron1', 'intron1:10000', 'intron1:uncapped': canonical first intron
      * '5utr', '3utr', 'exons', 'gene'
      * Raw genomic coordinates 'chr:start-end'
    
    Returns:
    - resolved_intervals: List[Tuple[int, int]] clamped to [0, chrom_len] and merged.
    - metadata: Dict[str, Any] detailing intermediate parsed tokens and fallback proxies.
    """
    g = gene_record
    strand = g.strand
    tss = g.tss
    tes = g.tes
    c_len = chrom_len if chrom_len is not None else 10**9

    raw_tokens = [tok.strip() for tok in locus_range.split(",") if tok.strip()]
    raw_intervals: List[Tuple[int, int]] = []
    metadata: Dict[str, Any] = {"gene": g.gene_name, "strand": strand, "tokens": raw_tokens, "fallbacks": []}

    for token in raw_tokens:
        tok_clean = token.replace("±", "+-")

        # 1. Preset: promoter_1kb (TSS - 1000 to TSS + 500 strand-aware)
        if tok_clean == "promoter_1kb":
            if strand == "+":
                st, en = tss - 1000, tss + 500
            else:
                st, en = tss - 500, tss + 1000
            raw_intervals.append((st, en))

        # 2. Preset: proximal_5kb (TSS ± 5000 bp)
        elif tok_clean == "proximal_5kb":
            raw_intervals.append((tss - 5000, tss + 5000))

        # 3. Preset: bounded_25kb (TSS ± 25 kb bounded by midpoint to nearest neighbor TSS)
        elif tok_clean == "bounded_25kb":
            lo, hi = tss - 25000, tss + 25000
            if neighbor_idx and g.chrom in neighbor_idx:
                arr = neighbor_idx[g.chrom]
                if len(arr) > 1:
                    pos = int(np.searchsorted(arr, tss))
                    l_idx = pos - 1
                    while l_idx >= 0 and arr[l_idx] >= tss:
                        l_idx -= 1
                    if l_idx >= 0:
                        lo = max(lo, int((tss + arr[l_idx]) // 2))
                    r_idx = pos
                    while r_idx < len(arr) and arr[r_idx] <= tss:
                        r_idx += 1
                    if r_idx < len(arr):
                        hi = min(hi, int((tss + arr[r_idx]) // 2))
            raw_intervals.append((lo, hi))

        # 4. Preset: promoter_intron1_capped
        # Strand-aware TSS - 2kb to min(first_intron_end, TSS + 10kb); fallback TSS ± 2kb
        elif tok_clean == "promoter_intron1_capped":
            cap = intron1_cap_bp if intron1_cap_bp is not None else 10000
            if g.intron1 is not None:
                i_st, i_en = g.intron1
                if strand == "+":
                    p_st = tss - 2000
                    p_en = min(i_en, tss + cap)
                else:
                    p_st = max(i_st, tss - cap)
                    p_en = tss + 2000
            else:
                metadata["fallbacks"].append("promoter_intron1_capped: single-exon fallback to TSS ± 2kb")
                p_st, p_en = tss - 2000, tss + 2000
            raw_intervals.append((p_st, p_en))

        # 5. Preset: gene body
        elif tok_clean == "gene":
            raw_intervals.append((min(g.start, g.end), max(g.start, g.end)))

        # 6. Preset: exons
        elif tok_clean == "exons":
            if g.exons:
                raw_intervals.extend(g.exons)
            else:
                raw_intervals.append((min(g.start, g.end), max(g.start, g.end)))

        # 7. Preset: tes_1kb
        elif tok_clean == "tes_1kb":
            if strand == "+":
                raw_intervals.append((max(0, tes - 1000), tes))
            else:
                raw_intervals.append((tes, tes + 1000))

        # 8. DSL: Symmetric around TSS ('TSS+-X' or 'TSS±X')
        elif re.match(r"^TSS\+\-(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TSS\+\-(\d+)$", tok_clean, re.IGNORECASE)
            flank = int(m.group(1))
            raw_intervals.append((tss - flank, tss + flank))

        # 9. DSL: Symmetric around TES ('TES+-X' or 'TES±X')
        elif re.match(r"^TES\+\-(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TES\+\-(\d+)$", tok_clean, re.IGNORECASE)
            flank = int(m.group(1))
            raw_intervals.append((tes - flank, tes + flank))

        # 10. DSL: Asymmetric TSS anchor ('TSS+A-B')
        # +A = upstream of TSS, -B = downstream of TSS
        elif re.match(r"^TSS\+(\d+)\-(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TSS\+(\d+)\-(\d+)$", tok_clean, re.IGNORECASE)
            up = int(m.group(1))
            dn = int(m.group(2))
            if strand == "+":
                st, en = tss - up, tss + dn
            else:
                st, en = tss - dn, tss + up
            raw_intervals.append((st, en))

        # 11. DSL: Upstream-only TSS anchor ('TSS+A') -> [TSS - A, TSS] strand-aware
        elif re.match(r"^TSS\+(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TSS\+(\d+)$", tok_clean, re.IGNORECASE)
            up = int(m.group(1))
            if strand == "+":
                st, en = tss - up, tss
            else:
                st, en = tss, tss + up
            raw_intervals.append((st, en))

        # 12. DSL: Downstream-only TSS anchor ('TSS-B') -> [TSS, TSS + B] strand-aware
        elif re.match(r"^TSS\-(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TSS\-(\d+)$", tok_clean, re.IGNORECASE)
            dn = int(m.group(1))
            if strand == "+":
                st, en = tss, tss + dn
            else:
                st, en = tss - dn, tss
            raw_intervals.append((st, en))

        # 13. DSL: Asymmetric TES anchor ('TES+A-B')
        elif re.match(r"^TES\+(\d+)\-(\d+)$", tok_clean, re.IGNORECASE):
            m = re.match(r"^TES\+(\d+)\-(\d+)$", tok_clean, re.IGNORECASE)
            up = int(m.group(1))
            dn = int(m.group(2))
            if strand == "+":
                st, en = tes - up, tes + dn
            else:
                st, en = tes - dn, tes + up
            raw_intervals.append((st, en))

        # 14. DSL: Gene-body anchored (+A-B) -> [TSS - A upstream, TES + B downstream]
        elif re.match(r"^\+(\d+)\-(\d+)$", tok_clean):
            m = re.match(r"^\+(\d+)\-(\d+)$", tok_clean)
            up = int(m.group(1))
            dn = int(m.group(2))
            if strand == "+":
                st, en = tss - up, tes + dn
            else:
                st, en = tes - dn, tss + up
            raw_intervals.append((st, en))

        # 15. DSL: intron1 (with optional cap override 'intron1:CAP' or 'intron1:uncapped')
        elif tok_clean.startswith("intron1"):
            cap = intron1_cap_bp
            if ":" in tok_clean:
                spec = tok_clean.split(":", 1)[1].strip().lower()
                cap = None if spec == "uncapped" else int(spec)

            if g.intron1 is not None:
                i_st, i_en = g.intron1
                if cap is not None:
                    if strand == "+":
                        i_en = min(i_en, tss + cap)
                    else:
                        i_st = max(i_st, tss - cap)
                raw_intervals.append((i_st, i_en))
            else:
                metadata["fallbacks"].append("intron1: single-exon gene lacks intron1; falling back to proximal exon proxy")
                if g.exons:
                    raw_intervals.append(g.exons[0])
                else:
                    raw_intervals.append((tss - 1000, tss + 1000))

        # 16. DSL: 5utr
        elif tok_clean.lower() in ("5utr", "five_prime_utr"):
            if g.utr5:
                raw_intervals.extend(g.utr5)
            else:
                metadata["fallbacks"].append("5utr: explicit 5' UTR annotation missing; falling back to core promoter proxy")
                if strand == "+":
                    raw_intervals.append((max(0, tss - 250), tss + 100))
                else:
                    raw_intervals.append((max(0, tss - 100), tss + 250))

        # 17. DSL: 3utr
        elif tok_clean.lower() in ("3utr", "three_prime_utr"):
            if g.utr3:
                raw_intervals.extend(g.utr3)
            else:
                metadata["fallbacks"].append("3utr: explicit 3' UTR annotation missing; falling back to terminal 1 kb proxy")
                if strand == "+":
                    raw_intervals.append((max(0, tes - 1000), tes))
                else:
                    raw_intervals.append((tes, tes + 1000))

        # 18. DSL: Raw genomic coordinates 'chr:start-end'
        elif ":" in tok_clean and "-" in tok_clean:
            try:
                coords = tok_clean.split(":", 1)[1]
                s_str, e_str = coords.split("-", 1)
                raw_intervals.append((int(s_str), int(e_str)))
            except Exception as e_c:
                logger.warning(f"Failed to parse raw coordinate interval '{tok_clean}': {e_c}")

        else:
            logger.warning(f"Unrecognized locus range DSL token '{token}', ignoring.")

    # Clamp to chromosome bounds and sort
    clamped = [(max(0, min(c_len, st)), max(0, min(c_len, en))) for st, en in raw_intervals if st < en]
    if not clamped:
        clamped = [(max(0, tss - 1000), min(c_len, tss + 1000))]

    clamped.sort(key=lambda x: x[0])

    # Merge overlapping or abutting intervals
    merged: List[Tuple[int, int]] = []
    for cur_st, cur_en in clamped:
        if not merged:
            merged.append((cur_st, cur_en))
        else:
            prev_st, prev_en = merged[-1]
            if cur_st <= prev_en:
                merged[-1] = (prev_st, max(prev_en, cur_en))
            else:
                merged.append((cur_st, cur_en))

    metadata["resolved_intervals"] = merged
    metadata["total_span_bp"] = sum(e - s for s, e in merged)
    return merged, metadata


# ==============================================================================
# Model Forward Engine & Window Aggregation
# ==============================================================================

def _score_chrombpnet_sequences(
    cbp_model: nn.Module,
    sequences: List[str],
    batch_size: int = 256,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """Score a list of 2,114 bp sequences with ChromBPNet in batched bf16 autocast."""
    n = len(sequences)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return {"y_abs": empty, "y_rel": empty, "late": empty}

    out_abs = np.empty(n, dtype=np.float32)
    out_rel = np.empty(n, dtype=np.float32)
    out_late = np.empty(n, dtype=np.float32)
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    def _forward_x(x_4xl):
        xt = torch.from_numpy(x_4xl).to(device, non_blocking=True)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    o = cbp_model(xt)
            else:
                o = cbp_model(xt)

        if isinstance(o, dict):
            a = o["y_abs"].detach().float().cpu().numpy().reshape(-1)
            r = o["y_rel"].detach().float().cpu().numpy().reshape(-1)
            l = o["late"].detach().float().cpu().numpy().reshape(-1)
        elif isinstance(o, (list, tuple)):
            a = o[0].detach().float().cpu().numpy().reshape(-1)
            r = o[1].detach().float().cpu().numpy().reshape(-1)
            l = o[2].detach().float().cpu().numpy().reshape(-1)
        else:
            raise TypeError(f"Unexpected model output type: {type(o)}")
        return a, r, l

    for i in range(0, n, batch_size):
        sub_seqs = sequences[i:min(n, i + batch_size)]
        x_fwd = one_hot_encode_batch(sub_seqs, seq_len=2114)
        x_rev = reverse_complement_onehot(x_fwd)

        a_f, r_f, l_f = _forward_x(x_fwd)
        a_r, r_r, l_r = _forward_x(x_rev)

        out_abs[i:min(n, i + batch_size)] = 0.5 * (a_f + a_r)
        out_rel[i:min(n, i + batch_size)] = 0.5 * (r_f + r_r)
        out_late[i:min(n, i + batch_size)] = 0.5 * (l_f + l_r)

    return {"y_abs": out_abs, "y_rel": out_rel, "late": out_late}


def _score_ntv3_sequences(
    ntv3_model: nn.Module,
    ntv3_tok: Any,
    meta: Dict[str, Any],
    seqs: List[str],
    batch_size: int = 8,
    device: str = "cuda"
) -> np.ndarray:
    """Score context_len (65,536 bp) sequences through NTv3 tracks model in batches."""
    if not seqs:
        return np.empty((0, 0, 0), dtype=np.float32)

    context_len = meta["context_len"]
    species_id = meta["species_id"]
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    outputs: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    for st in range(0, len(seqs), bs):
        sub_seqs = [sanitize_sequence(s) for s in seqs[st:st + bs]]
        enc = ntv3_tok(
            sub_seqs,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=context_len,
            truncation=True,
            padding="max_length"
        )
        inp_ids = enc["input_ids"].to(device, non_blocking=True)
        cond_tok = [torch.full((len(sub_seqs),), species_id, dtype=torch.long, device=device)]
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                preds = ntv3_model(inp_ids, conditions_tokens=cond_tok)
                outputs.append(preds.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _score_ntv3_sequence(
    ntv3_model: nn.Module,
    ntv3_tok: Any,
    meta: Dict[str, Any],
    seq: str,
    device: str = "cuda"
) -> np.ndarray:
    """Score a single context_len (65,536 bp) sequence through NTv3 tracks model."""
    return _score_ntv3_sequences(ntv3_model, ntv3_tok, meta, [seq], batch_size=1, device=device)[0]


def _plan_minimal_crop_centers(
    intervals: List[Tuple[int, int]],
    crop_len: int
) -> List[int]:
    """Greedily select minimal crop centers covering all disjoint intervals within central crop."""
    if not intervals:
        return []
    iv_sorted = sorted(intervals, key=lambda x: x[0])
    centers: List[int] = []
    cov_end = -1
    margin = 250
    eff_span = max(1000, crop_len - 2 * margin)
    for st, en in iv_sorted:
        if en <= cov_end:
            continue
        cur_st = max(st, cov_end)
        while cur_st < en:
            c_mid = cur_st + eff_span // 2
            centers.append(c_mid)
            cov_end = c_mid + eff_span // 2
            cur_st = cov_end
    return centers


# ==============================================================================
# Public Inference Entry Point: infer_premzt
# ==============================================================================

def infer_premzt(
    model: Union[str, List[str]] = "all",
    species: Optional[str] = None,
    gene: Optional[Union[str, List[str]]] = None,
    locus_range: Optional[str] = "auto",
    sequence: Optional[Union[str, List[str]]] = None,
    work_dir: Union[str, Path] = "./work_astatotilapia_premzt",
    intron1_cap_bp: int = 10000,
    stride_bp: int = 250,
    batch_size: int = 256,
    device: Optional[str] = None,
    return_profile: bool = False,
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Unified entry point for Pre-MZT genomic inference across ChromBPNet and NTv3 models.
    
    Supports:
    1. Direct Sequence Mode (sequence provided; bypasses species/gene/locus_range):
       - ChromBPNet: pads <2114 bp to 2114; scores exact 2114; tiles >2114 bp.
       - NTv3 ATAC & SLAM: pads <= crop_len inside central crop; tiles cleanly across crop if > crop_len.
    2. Gene Scoring Mode (species, gene, and locus_range provided):
       - 'auto': model-specific optimal length-matched core promoter and nascent tracks.
       - named presets ('promoter_1kb', 'promoter_intron1_capped', 'proximal_5kb', 'bounded_25kb', etc.)
       - flexible DSL strings ('TSS+200-100,intron1', '+2000-1000', 'TSS+200,intron1,5utr,3utr', etc.)
    """
    ctx = get_inference_context(work_dir=work_dir, device=device)

    # Normalize models parameter
    if isinstance(model, str):
        if model.lower() == "all":
            requested_models = ["chrombpnet", "ntv3atac", "ntv3slam"]
        else:
            requested_models = [model.lower()]
    else:
        requested_models = [m.lower() for m in model]

    # --------------------------------------------------------------------------
    # MODE 1: Direct Sequence Mode
    # --------------------------------------------------------------------------
    if sequence is not None:
        is_single = isinstance(sequence, str)
        seq_list = [sequence] if is_single else list(sequence)
        results = []

        for s_idx, raw_seq in enumerate(seq_list):
            seq_len = len(raw_seq)
            res_rec: Dict[str, Any] = {"sequence_idx": s_idx, "length": seq_len}

            # ChromBPNet
            if "chrombpnet" in requested_models:
                cbp = ctx.get_chrombpnet()
                if seq_len < 2114:
                    pad_l = (2114 - seq_len) // 2
                    pad_r = 2114 - seq_len - pad_l
                    p_seq = "N" * pad_l + raw_seq + "N" * pad_r
                    sc = _score_chrombpnet_sequences(cbp, [p_seq], batch_size=1, device=ctx.device)
                    res_rec["chrombpnet_y_abs_max"] = float(sc["y_abs"][0])
                    res_rec["chrombpnet_y_abs_mean"] = float(sc["y_abs"][0])
                    res_rec["chrombpnet_y_rel_max"] = float(sc["y_rel"][0])
                    res_rec["chrombpnet_late_max"] = float(sc["late"][0])
                    if return_profile:
                        res_rec["chrombpnet_profile"] = sc["y_abs"]
                elif seq_len == 2114:
                    sc = _score_chrombpnet_sequences(cbp, [raw_seq], batch_size=1, device=ctx.device)
                    res_rec["chrombpnet_y_abs_max"] = float(sc["y_abs"][0])
                    res_rec["chrombpnet_y_abs_mean"] = float(sc["y_abs"][0])
                    res_rec["chrombpnet_y_rel_max"] = float(sc["y_rel"][0])
                    res_rec["chrombpnet_late_max"] = float(sc["late"][0])
                    if return_profile:
                        res_rec["chrombpnet_profile"] = sc["y_abs"]
                else:
                    # Tile across sequence
                    tile_windows = [raw_seq[st:st+2114] for st in range(0, seq_len - 2114 + 1, stride_bp)]
                    if (seq_len - 2114) % stride_bp != 0:
                        tile_windows.append(raw_seq[seq_len - 2114:])
                    sc = _score_chrombpnet_sequences(cbp, tile_windows, batch_size=batch_size, device=ctx.device)
                    res_rec["chrombpnet_y_abs_max"] = float(np.max(sc["y_abs"]))
                    res_rec["chrombpnet_y_abs_mean"] = float(np.mean(sc["y_abs"]))
                    res_rec["chrombpnet_y_rel_max"] = float(np.max(sc["y_rel"]))
                    res_rec["chrombpnet_late_max"] = float(np.max(sc["late"]))
                    if return_profile:
                        res_rec["chrombpnet_profile"] = sc["y_abs"]

            # NTv3 ATAC and/or SLAM
            if "ntv3atac" in requested_models or "ntv3slam" in requested_models or "ntv3" in requested_models:
                ntv3_mod, tok, meta = ctx.get_ntv3()
                c_len = meta["context_len"]
                crop_len = meta["crop_len"]
                crop_st = meta["crop_start"]

                if seq_len <= crop_len:
                    # Center input sequence inside central crop
                    pad_in_crop_l = (crop_len - seq_len) // 2
                    pad_l = crop_st + pad_in_crop_l
                    pad_r = c_len - pad_l - seq_len
                    full_padded_seq = "N" * pad_l + raw_seq + "N" * pad_r

                    p_mat = _score_ntv3_sequence(ntv3_mod, tok, meta, full_padded_seq, device=ctx.device)
                    # Squeeze strictly to input bins within central crop
                    valid_bin_st = pad_in_crop_l
                    valid_bin_en = valid_bin_st + seq_len
                    valid_bins = p_mat[valid_bin_st:valid_bin_en]

                    if "ntv3atac" in requested_models or "ntv3" in requested_models:
                        res_rec["ntv3_pred_atac_256cell_max"] = float(np.max(valid_bins[:, meta["idx_atac_256"]]))
                        res_rec["ntv3_pred_atac_256cell_mean"] = float(np.mean(valid_bins[:, meta["idx_atac_256"]]))
                        res_rec["ntv3_pred_atac_dome_max"] = float(np.max(valid_bins[:, meta["idx_atac_dome"]]))

                    if "ntv3slam" in requested_models or "ntv3" in requested_models:
                        slam_fwd = valid_bins[:, meta["idx_slam_25_fwd"]]
                        slam_rev = valid_bins[:, meta["idx_slam_25_rev"]]
                        # Report terminal 1 kb signal in unified per-1kb units (mean * 1000 == sum over 1 kb)
                        res_rec["ntv3_pred_slam_2.5hpf_3p"] = float(max(np.mean(slam_fwd[-1000:]), np.mean(slam_rev[:1000])) * 1000.0)
                        res_rec["ntv3_pred_slam_2.5hpf_all_exons"] = float(max(np.sum(slam_fwd), np.sum(slam_rev)))
                        res_rec["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = float(max(np.mean(slam_fwd), np.mean(slam_rev)) * 1000.0)
                        res_rec["slam_3p_tes_centered"] = True

                    if return_profile:
                        res_rec["ntv3_profile"] = valid_bins
                else:
                    # Tile across long sequence stepping across central crop
                    step = crop_len // 2
                    tile_starts = list(range(0, seq_len - crop_len + 1, step))
                    if (seq_len - crop_len) % step != 0:
                        tile_starts.append(seq_len - crop_len)

                    atac_maxs = []
                    atac_means = []
                    slam_means = []
                    tile_seqs = [
                        "N" * crop_st + raw_seq[t_st:t_st + crop_len] + "N" * (c_len - crop_st - crop_len)
                        for t_st in tile_starts
                    ]
                    p_tiles = _score_ntv3_sequences(ntv3_mod, tok, meta, tile_seqs, batch_size=min(8, max(1, batch_size)), device=ctx.device)
                    for p_mat in p_tiles:
                        atac_maxs.append(np.max(p_mat[:, meta["idx_atac_256"]]))
                        atac_means.append(np.mean(p_mat[:, meta["idx_atac_256"]]))
                        slam_means.append(max(np.mean(p_mat[:, meta["idx_slam_25_fwd"]]), np.mean(p_mat[:, meta["idx_slam_25_rev"]])))

                    if "ntv3atac" in requested_models or "ntv3" in requested_models:
                        res_rec["ntv3_pred_atac_256cell_max"] = float(np.max(atac_maxs))
                        res_rec["ntv3_pred_atac_256cell_mean"] = float(np.mean(atac_means))

                    if "ntv3slam" in requested_models or "ntv3" in requested_models:
                        res_rec["ntv3_pred_slam_2.5hpf_3p"] = float(np.max(slam_means) * 1000.0)
                        res_rec["ntv3_pred_slam_2.5hpf_all_exons"] = float(np.sum(slam_means) * step)
                        res_rec["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = float(np.mean(slam_means) * 1000.0)
                        res_rec["slam_3p_tes_centered"] = True

            results.append(res_rec)

        if is_single:
            return results[0]
        return pd.DataFrame(results)

    # --------------------------------------------------------------------------
    # MODE 2: Gene Scoring Mode
    # --------------------------------------------------------------------------
    if gene is None:
        raise ValueError("Either 'sequence' or 'gene' must be provided to infer_premzt.")

    # Multi-gene batch call
    if isinstance(gene, list) and len(gene) > 1:
        return infer_premzt_batch(
            genes=gene,
            model=model,
            species=species,
            locus_range=locus_range,
            work_dir=work_dir,
            intron1_cap_bp=intron1_cap_bp,
            stride_bp=stride_bp,
            batch_size=batch_size,
            device=device,
            context=ctx
        )

    # Single gene call
    target_gene = gene[0] if isinstance(gene, list) else gene
    sp_key = ctx.resolve_species_key(species)
    gtf_data = ctx.get_gtf_data(sp_key)
    gene_lookup = gtf_data["genes"]
    g_rec = gene_lookup.get(target_gene.lower())

    if g_rec is None:
        raise KeyError(f"Gene '{target_gene}' not found in GTF annotations for {sp_key}.")

    fa = ctx.get_fasta(sp_key)
    ch_len = len(fa[g_rec.chrom]) if g_rec.chrom in fa else 10**9

    res: Dict[str, Any] = {
        "gene": g_rec.gene_name,
        "gene_id": g_rec.gene_id,
        "species": sp_key,
        "chrom": g_rec.chrom,
        "strand": g_rec.strand,
        "tss": g_rec.tss,
        "tes": g_rec.tes,
        "locus_range": locus_range,
    }

    # ==========================================================================
    # Auto Mode: Model-Specific Optimal Presets
    # ==========================================================================
    if locus_range == "auto":
        # 1. ChromBPNet Suite
        if "chrombpnet" in requested_models:
            cbp = ctx.get_chrombpnet()
            # Standard length-matched suite
            windows = ["promoter_1kb", "promoter_intron1_capped", "proximal_5kb", "bounded_25kb"]
            for w in windows:
                intervals, _ = parse_locus_range(w, g_rec, chrom_len=ch_len, intron1_cap_bp=intron1_cap_bp, neighbor_idx=gtf_data["neighbor_idx"])
                # Extract window centers
                centers: List[int] = []
                for st, en in intervals:
                    centers.extend(list(range(st + 1057, en - 1057 + 1, stride_bp)))
                    if not centers:
                        centers.append((st + en) // 2)
                centers = sorted(set(centers))

                # Fetch and score sequences
                seqs = [str(fa[g_rec.chrom][max(0, c - 1057): min(ch_len, c + 1057)]).upper() for c in centers]
                seqs = [s if len(s) == 2114 else s.ljust(2114, "N") for s in seqs]
                sc = _score_chrombpnet_sequences(cbp, seqs, batch_size=batch_size, device=ctx.device)

                res[f"pred_{w}_y_abs"] = float(np.max(sc["y_abs"])) if len(sc["y_abs"]) else np.nan
                res[f"pred_{w}_y_rel"] = float(np.max(sc["y_rel"])) if len(sc["y_rel"]) else np.nan
                res[f"pred_{w}_late"] = float(np.max(sc["late"])) if len(sc["late"]) else np.nan

            res["pred_promoter_1kb_y_rel_gated"] = res["pred_promoter_1kb_y_rel"] if res.get("pred_promoter_1kb_late", 0) >= 4.70 else np.nan

        # 2. NTv3 Suite
        if "ntv3atac" in requested_models or "ntv3slam" in requested_models or "ntv3" in requested_models:
            ntv3_mod, tok, meta = ctx.get_ntv3()
            c_len = meta["context_len"]
            crop_len = meta["crop_len"]
            crop_st = meta["crop_start"]
            crop_en = meta["crop_end"]

            # TSS-centered 65 kb window
            w_tss_st = max(0, g_rec.tss - (crop_st + crop_len // 2))
            w_tss_en = min(ch_len, w_tss_st + c_len)
            seq_tss = str(fa[g_rec.chrom][w_tss_st:w_tss_en])
            if len(seq_tss) < c_len:
                seq_tss = seq_tss.ljust(c_len, "N")

            p_mat_tss = _score_ntv3_sequence(ntv3_mod, tok, meta, seq_tss, device=ctx.device)
            crop_gen_st = w_tss_st + crop_st
            crop_gen_en = w_tss_st + crop_en

            # Accessibility: Slices inside TSS-centered crop
            if "ntv3atac" in requested_models or "ntv3" in requested_models:
                idx_atac = meta["idx_atac_256"]
                res["ntv3_pred_atac_256cell_max"] = float(np.max(p_mat_tss[:, idx_atac]))
                res["ntv3_pred_atac_256cell_mean"] = float(np.mean(p_mat_tss[:, idx_atac]))

                # promoter_1kb slice
                p1_int, _ = parse_locus_range("promoter_1kb", g_rec, chrom_len=ch_len)
                s1, e1 = p1_int[0]
                b_s1 = max(0, min(crop_len - 1, s1 - crop_gen_st))
                b_e1 = max(b_s1 + 1, min(crop_len, e1 - crop_gen_st))
                res["ntv3_pred_atac_promoter_1kb_max"] = float(np.max(p_mat_tss[b_s1:b_e1, idx_atac]))

                # promoter_intron1_capped slice
                pi_int, _ = parse_locus_range("promoter_intron1_capped", g_rec, chrom_len=ch_len, intron1_cap_bp=intron1_cap_bp)
                si, ei = pi_int[0]
                b_si = max(0, min(crop_len - 1, si - crop_gen_st))
                b_ei = max(b_si + 1, min(crop_len, ei - crop_gen_st))
                res["ntv3_pred_atac_promoter_intron1_capped_max"] = float(np.max(p_mat_tss[b_si:b_ei, idx_atac]))

                # proximal_5kb slice
                b_s5 = max(0, min(crop_len - 1, (g_rec.tss - 5000) - crop_gen_st))
                b_e5 = max(b_s5 + 1, min(crop_len, (g_rec.tss + 5000) - crop_gen_st))
                res["ntv3_pred_atac_proximal_5kb_max"] = float(np.max(p_mat_tss[b_s5:b_e5, idx_atac]))

            # Nascent Expression: TES-centered vs TSS-centered pass
            if "ntv3slam" in requested_models or "ntv3" in requested_models:
                slam_track_idx = meta["idx_slam_25_fwd"] if g_rec.strand == "+" else meta["idx_slam_25_rev"]
                tes = g_rec.tes

                # Check if TES sits safely >= 1,500 bp inside central crop
                if (crop_gen_st + 1500) <= tes <= (crop_gen_en - 1500):
                    p_mat_slam = p_mat_tss
                    crop_slam_st = crop_gen_st
                    crop_slam_en = crop_gen_en
                    res["slam_3p_tes_centered"] = False
                else:
                    # Dedicated TES-centered pass
                    w_tes_st = max(0, tes - (crop_st + crop_len // 2))
                    w_tes_en = min(ch_len, w_tes_st + c_len)
                    seq_tes = str(fa[g_rec.chrom][w_tes_st:w_tes_en])
                    if len(seq_tes) < c_len:
                        seq_tes = seq_tes.ljust(c_len, "N")
                    p_mat_slam = _score_ntv3_sequence(ntv3_mod, tok, meta, seq_tes, device=ctx.device)
                    crop_slam_st = w_tes_st + crop_st
                    crop_slam_en = w_tes_st + crop_en
                    res["slam_3p_tes_centered"] = True

                # Terminal 1,000 bp at TES
                if g_rec.strand == "+":
                    t_st, t_en = max(g_rec.start, tes - 1000), tes
                else:
                    t_st, t_en = tes, min(g_rec.end, tes + 1000)

                b_t_st = max(0, min(crop_len - 1, t_st - crop_slam_st))
                b_t_en = max(b_t_st + 1, min(crop_len, t_en - crop_slam_st))
                res["ntv3_pred_slam_2.5hpf_3p"] = float(np.mean(p_mat_slam[b_t_st:b_t_en, slam_track_idx]) * 1000.0)

                # Exon nascent expression & length normalization
                exons = g_rec.exons or [(g_rec.start, g_rec.end)]
                total_exon_kb = max(0.1, sum(e - s for s, e in exons) / 1000.0)
                gene_len = abs(g_rec.end - g_rec.start)

                if gene_len <= (crop_len - 2000) and crop_gen_st <= g_rec.start and g_rec.end <= crop_gen_en:
                    # All exons lie within TSS crop
                    exon_sum = 0.0
                    for ex_s, ex_e in exons:
                        b_es = max(0, min(crop_len - 1, ex_s - crop_gen_st))
                        b_ee = max(b_es + 1, min(crop_len, ex_e - crop_gen_st))
                        exon_sum += float(np.sum(p_mat_tss[b_es:b_ee, slam_track_idx]))
                    res["ntv3_pred_slam_2.5hpf_all_exons"] = exon_sum
                    res["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = exon_sum / total_exon_kb
                else:
                    # Minimal greedy crop set + cached TSS/TES crops for fast exon scoring
                    cached_crops: List[Tuple[int, int, np.ndarray]] = [(crop_gen_st, crop_gen_en, p_mat_tss)]
                    if (crop_slam_st, crop_slam_en) != (crop_gen_st, crop_gen_en):
                        cached_crops.append((crop_slam_st, crop_slam_en, p_mat_slam))

                    uncovered = []
                    for ex_s, ex_e in exons:
                        if not any(c_s <= ex_s and ex_e <= c_e for c_s, c_e, _ in cached_crops):
                            uncovered.append((ex_s, ex_e))

                    extra_centers = _plan_minimal_crop_centers(uncovered, crop_len)
                    if extra_centers:
                        extra_seqs = []
                        extra_bounds = []
                        for c_mid in extra_centers:
                            w_st = max(0, c_mid - (crop_st + crop_len // 2))
                            w_en = min(ch_len, w_st + c_len)
                            seq_ex = str(fa[g_rec.chrom][w_st:w_en])
                            if len(seq_ex) < c_len:
                                seq_ex = seq_ex.ljust(c_len, "N")
                            extra_seqs.append(seq_ex)
                            extra_bounds.append((w_st + crop_st, w_st + crop_en))
                        extra_mats = _score_ntv3_sequences(ntv3_mod, tok, meta, extra_seqs, batch_size=8, device=ctx.device)
                        for (c_gs, c_ge), p_ex in zip(extra_bounds, extra_mats):
                            cached_crops.append((c_gs, c_ge, p_ex))

                    exon_sum = 0.0
                    for ex_s, ex_e in exons:
                        best_crop = next(((c_s, c_e, pm) for c_s, c_e, pm in cached_crops if c_s <= ex_s and ex_e <= c_e), None)
                        if best_crop is None:
                            best_crop = max(cached_crops, key=lambda x: max(0, min(x[1], ex_e) - max(x[0], ex_s)))
                        c_gst, _, p_ex = best_crop
                        b_es = max(0, min(crop_len - 1, ex_s - c_gst))
                        b_ee = max(b_es + 1, min(crop_len, ex_e - c_gst))
                        exon_sum += float(np.sum(p_ex[b_es:b_ee, slam_track_idx]))

                    res["ntv3_pred_slam_2.5hpf_all_exons"] = exon_sum
                    res["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = exon_sum / total_exon_kb

    # ==========================================================================
    # Custom DSL / Preset Mode
    # ==========================================================================
    else:
        resolved_intervals, meta_dsl = parse_locus_range(
            locus_range=locus_range,
            gene_record=g_rec,
            chrom_len=ch_len,
            intron1_cap_bp=intron1_cap_bp,
            neighbor_idx=gtf_data["neighbor_idx"]
        )
        res["resolved_intervals"] = resolved_intervals
        res["resolved_span_bp"] = meta_dsl["total_span_bp"]

        # ChromBPNet scoring over resolved intervals
        if "chrombpnet" in requested_models:
            cbp = ctx.get_chrombpnet()
            centers = []
            for st, en in resolved_intervals:
                c_list = list(range(st + 1057, en - 1057 + 1, stride_bp))
                if not c_list:
                    c_list.append((st + en) // 2)
                centers.extend(c_list)
            centers = sorted(set(centers))

            seqs = [str(fa[g_rec.chrom][max(0, c - 1057): min(ch_len, c + 1057)]).upper() for c in centers]
            seqs = [s if len(s) == 2114 else s.ljust(2114, "N") for s in seqs]
            sc = _score_chrombpnet_sequences(cbp, seqs, batch_size=batch_size, device=ctx.device)

            res["chrombpnet_y_abs_max"] = float(np.max(sc["y_abs"])) if len(sc["y_abs"]) else np.nan
            res["chrombpnet_y_abs_mean"] = float(np.mean(sc["y_abs"])) if len(sc["y_abs"]) else np.nan
            res["chrombpnet_y_rel_max"] = float(np.max(sc["y_rel"])) if len(sc["y_rel"]) else np.nan
            res["chrombpnet_late_max"] = float(np.max(sc["late"])) if len(sc["late"]) else np.nan

        # NTv3 scoring over resolved intervals
        if "ntv3atac" in requested_models or "ntv3slam" in requested_models or "ntv3" in requested_models:
            ntv3_mod, tok, meta = ctx.get_ntv3()
            c_len = meta["context_len"]
            crop_len = meta["crop_len"]
            crop_st = meta["crop_start"]

            # Cover all intervals with minimum number of 65 kb windows
            atac_vals = []
            slam_vals = []
            for st, en in resolved_intervals:
                mid_int = (st + en) // 2
                w_st = max(0, mid_int - (crop_st + crop_len // 2))
                w_en = min(ch_len, w_st + c_len)
                seq_w = str(fa[g_rec.chrom][w_st:w_en])
                if len(seq_w) < c_len:
                    seq_w = seq_w.ljust(c_len, "N")
                p_mat = _score_ntv3_sequence(ntv3_mod, tok, meta, seq_w, device=ctx.device)

                c_gst = w_st + crop_st
                b_st = max(0, min(crop_len - 1, st - c_gst))
                b_en = max(b_st + 1, min(crop_len, en - c_gst))

                if "ntv3atac" in requested_models or "ntv3" in requested_models:
                    atac_vals.append(float(np.max(p_mat[b_st:b_en, meta["idx_atac_256"]])))

                if "ntv3slam" in requested_models or "ntv3" in requested_models:
                    slam_idx = meta["idx_slam_25_fwd"] if g_rec.strand == "+" else meta["idx_slam_25_rev"]
                    slam_vals.append(float(np.mean(p_mat[b_st:b_en, slam_idx])))

            if atac_vals:
                res["ntv3_pred_atac_max"] = float(np.max(atac_vals))
                res["ntv3_pred_atac_mean"] = float(np.mean(atac_vals))
            if slam_vals:
                res["ntv3_pred_slam_mean"] = float(np.mean(slam_vals))
                res["ntv3_pred_slam_per_kb"] = float(np.mean(slam_vals) * 1000.0)

    return res


# ==============================================================================
# Batched High-Performance Inference: infer_premzt_batch
# ==============================================================================

def infer_premzt_batch(
    genes: Union[List[Dict[str, Any]], pd.DataFrame, List[str]],
    model: Union[str, List[str]] = "all",
    species: Optional[str] = None,
    locus_range: Optional[str] = "auto",
    work_dir: Union[str, Path] = "./work_astatotilapia_premzt",
    intron1_cap_bp: int = 10000,
    stride_bp: int = 250,
    batch_size: int = 256,
    device: Optional[str] = None,
    context: Optional[InferenceContext] = None,
) -> pd.DataFrame:
    """
    Batched, high-throughput inference across multiple genes with coordinate deduplication.
    Guarantees full GPU saturation across ChromBPNet (unique 2114 bp windows) and NTv3 (unique 65 kb windows).
    """
    ctx = context or get_inference_context(work_dir=work_dir, device=device)
    sp_key = ctx.resolve_species_key(species)
    gtf_data = ctx.get_gtf_data(sp_key)
    gene_lookup = gtf_data["genes"]
    fa = ctx.get_fasta(sp_key)

    # Normalize gene list input
    gene_names: List[str] = []
    if isinstance(genes, pd.DataFrame):
        col = next((c for c in ("gene_name", "gene_id", "symbol") if c in genes.columns), genes.columns[0])
        gene_names = genes[col].astype(str).tolist()
    elif isinstance(genes, list):
        for item in genes:
            if isinstance(item, dict):
                gene_names.append(str(item.get("gene_name", item.get("gene_id", ""))))
            else:
                gene_names.append(str(item))
    else:
        raise TypeError(f"Unsupported type for genes: {type(genes)}")

    if isinstance(model, str):
        req_models = ["chrombpnet", "ntv3atac", "ntv3slam"] if model.lower() == "all" else [model.lower()]
    else:
        req_models = [m.lower() for m in model]

    # Fast deduplicated batched execution for auto mode
    if locus_range == "auto":
        g_recs: List[Optional[GeneRecord]] = [gene_lookup.get(gn.lower()) for gn in gene_names]
        records: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        for idx, (gn, gre) in enumerate(zip(gene_names, g_recs)):
            if gre is None:
                records.append({"gene": gn, "error": f"Gene '{gn}' not found in GTF for {sp_key}."})
            else:
                valid_indices.append(idx)
                records.append({
                    "gene": gre.gene_name,
                    "gene_id": gre.gene_id,
                    "species": sp_key,
                    "chrom": gre.chrom,
                    "strand": gre.strand,
                    "tss": gre.tss,
                    "tes": gre.tes,
                    "locus_range": "auto",
                })

        if "chrombpnet" in req_models and valid_indices:
            cbp = ctx.get_chrombpnet()
            windows = ["promoter_1kb", "promoter_intron1_capped", "proximal_5kb", "bounded_25kb"]
            win_centers_map: Dict[Tuple[int, str], List[Tuple[str, int]]] = {}
            uniq_coords: List[Tuple[str, int]] = []
            seen_c = set()

            for idx in valid_indices:
                gre = g_recs[idx]
                ch_len = len(fa[gre.chrom]) if gre.chrom in fa else 10**9
                for w in windows:
                    ivs, _ = parse_locus_range(w, gre, chrom_len=ch_len, intron1_cap_bp=intron1_cap_bp, neighbor_idx=gtf_data["neighbor_idx"])
                    cs: List[int] = []
                    for st, en in ivs:
                        cs.extend(list(range(st + 1057, en - 1057 + 1, stride_bp)))
                        if not cs:
                            cs.append((st + en) // 2)
                    c_tuples = [(gre.chrom, c) for c in sorted(set(cs))]
                    win_centers_map[(idx, w)] = c_tuples
                    for ct in c_tuples:
                        if ct not in seen_c:
                            seen_c.add(ct)
                            uniq_coords.append(ct)

            seqs_cb = []
            for ch, c in uniq_coords:
                ch_len = len(fa[ch]) if ch in fa else 10**9
                s_bp = str(fa[ch][max(0, c - 1057): min(ch_len, c + 1057)]).upper()
                seqs_cb.append(s_bp if len(s_bp) == 2114 else s_bp.ljust(2114, "N"))
            sc_all = _score_chrombpnet_sequences(cbp, seqs_cb, batch_size=batch_size, device=ctx.device)
            coord_to_idx = {ct: i for i, ct in enumerate(uniq_coords)}

            for idx in valid_indices:
                rec = records[idx]
                for w in windows:
                    w_idxs = [coord_to_idx[ct] for ct in win_centers_map[(idx, w)]]
                    rec[f"pred_{w}_y_abs"] = float(np.max(sc_all["y_abs"][w_idxs])) if w_idxs else np.nan
                    rec[f"pred_{w}_y_rel"] = float(np.max(sc_all["y_rel"][w_idxs])) if w_idxs else np.nan
                    rec[f"pred_{w}_late"] = float(np.max(sc_all["late"][w_idxs])) if w_idxs else np.nan
                rec["pred_promoter_1kb_y_rel_gated"] = rec["pred_promoter_1kb_y_rel"] if rec.get("pred_promoter_1kb_late", 0) >= 4.70 else np.nan

        if ("ntv3atac" in req_models or "ntv3slam" in req_models or "ntv3" in req_models) and valid_indices:
            ntv3_mod, tok, meta = ctx.get_ntv3()
            c_len = meta["context_len"]
            crop_len = meta["crop_len"]
            crop_st = meta["crop_start"]
            crop_en = meta["crop_end"]
            ntv3_bs = min(16, max(1, batch_size // 32)) if batch_size >= 64 else min(16, max(1, batch_size))

            # Collect unique required 65 kb windows across TSS, TES, and uncovered long-gene exons
            uniq_wins: List[Tuple[str, int]] = []
            win_set = set()
            gene_tss_key: Dict[int, Tuple[str, int]] = {}
            gene_tes_key: Dict[int, Tuple[str, int]] = {}
            gene_extra_keys: Dict[int, List[Tuple[str, int]]] = {}

            def _reg_win(ch: str, center_pos: int) -> Tuple[str, int]:
                ch_len = len(fa[ch]) if ch in fa else 10**9
                w_st = max(0, center_pos - (crop_st + crop_len // 2))
                w_en = w_st + c_len
                if w_en > ch_len:
                    w_st = max(0, ch_len - c_len)
                key = (ch, w_st)
                if key not in win_set:
                    win_set.add(key)
                    uniq_wins.append(key)
                return key

            for idx in valid_indices:
                gre = g_recs[idx]
                tk = _reg_win(gre.chrom, gre.tss)
                gene_tss_key[idx] = tk
                c_gst = tk[1] + crop_st
                c_gen = tk[1] + crop_en
                if ("ntv3slam" in req_models or "ntv3" in req_models):
                    if (c_gst + 1500) <= gre.tes <= (c_gen - 1500):
                        gene_tes_key[idx] = tk
                    else:
                        gene_tes_key[idx] = _reg_win(gre.chrom, gre.tes)
                    exons = gre.exons or [(gre.start, gre.end)]
                    tss_b = (tk[1] + crop_st, tk[1] + crop_en)
                    tes_b = (gene_tes_key[idx][1] + crop_st, gene_tes_key[idx][1] + crop_en)
                    uncovered = [
                        (es, ee) for es, ee in exons
                        if not ((tss_b[0] <= es and ee <= tss_b[1]) or (tes_b[0] <= es and ee <= tes_b[1]))
                    ]
                    extra_cs = _plan_minimal_crop_centers(uncovered, crop_len)
                    gene_extra_keys[idx] = [_reg_win(gre.chrom, cm) for cm in extra_cs]

            win_seqs: List[str] = []
            for ch, w_st in uniq_wins:
                ch_len = len(fa[ch]) if ch in fa else 10**9
                s_w = str(fa[ch][w_st:min(ch_len, w_st + c_len)])
                if len(s_w) < c_len:
                    s_w = s_w.ljust(c_len, "N")
                win_seqs.append(s_w)

            preds_all = _score_ntv3_sequences(ntv3_mod, tok, meta, win_seqs, batch_size=ntv3_bs, device=ctx.device)
            win_pred_map = {wk: preds_all[i] for i, wk in enumerate(uniq_wins)}

            for idx in valid_indices:
                gre = g_recs[idx]
                rec = records[idx]
                ch_len = len(fa[gre.chrom]) if gre.chrom in fa else 10**9
                tk = gene_tss_key[idx]
                p_mat_tss = win_pred_map[tk]
                crop_gen_st = tk[1] + crop_st
                crop_gen_en = tk[1] + crop_en

                if "ntv3atac" in req_models or "ntv3" in req_models:
                    idx_atac = meta["idx_atac_256"]
                    rec["ntv3_pred_atac_256cell_max"] = float(np.max(p_mat_tss[:, idx_atac]))
                    rec["ntv3_pred_atac_256cell_mean"] = float(np.mean(p_mat_tss[:, idx_atac]))
                    p1_int, _ = parse_locus_range("promoter_1kb", gre, chrom_len=ch_len)
                    s1, e1 = p1_int[0]
                    b_s1 = max(0, min(crop_len - 1, s1 - crop_gen_st))
                    b_e1 = max(b_s1 + 1, min(crop_len, e1 - crop_gen_st))
                    rec["ntv3_pred_atac_promoter_1kb_max"] = float(np.max(p_mat_tss[b_s1:b_e1, idx_atac]))

                    pi_int, _ = parse_locus_range("promoter_intron1_capped", gre, chrom_len=ch_len, intron1_cap_bp=intron1_cap_bp)
                    si, ei = pi_int[0]
                    b_si = max(0, min(crop_len - 1, si - crop_gen_st))
                    b_ei = max(b_si + 1, min(crop_len, ei - crop_gen_st))
                    rec["ntv3_pred_atac_promoter_intron1_capped_max"] = float(np.max(p_mat_tss[b_si:b_ei, idx_atac]))

                    b_s5 = max(0, min(crop_len - 1, (gre.tss - 5000) - crop_gen_st))
                    b_e5 = max(b_s5 + 1, min(crop_len, (gre.tss + 5000) - crop_gen_st))
                    rec["ntv3_pred_atac_proximal_5kb_max"] = float(np.max(p_mat_tss[b_s5:b_e5, idx_atac]))

                if "ntv3slam" in req_models or "ntv3" in req_models:
                    slam_track_idx = meta["idx_slam_25_fwd"] if gre.strand == "+" else meta["idx_slam_25_rev"]
                    tk_slam = gene_tes_key[idx]
                    p_mat_slam = win_pred_map[tk_slam]
                    crop_slam_st = tk_slam[1] + crop_st
                    rec["slam_3p_tes_centered"] = (tk_slam != tk)

                    if gre.strand == "+":
                        t_st, t_en = max(gre.start, gre.tes - 1000), gre.tes
                    else:
                        t_st, t_en = gre.tes, min(gre.end, gre.tes + 1000)
                    b_ts = max(0, min(crop_len - 1, t_st - crop_slam_st))
                    b_te = max(b_ts + 1, min(crop_len, t_en - crop_slam_st))
                    rec["ntv3_pred_slam_2.5hpf_3p"] = float(np.mean(p_mat_slam[b_ts:b_te, slam_track_idx]) * 1000.0)

                    exons = gre.exons or [(gre.start, gre.end)]
                    total_exon_kb = max(0.1, sum(e - s for s, e in exons) / 1000.0)
                    avail_crops = [(tk[1] + crop_st, tk[1] + crop_en, p_mat_tss)]
                    if tk_slam != tk:
                        avail_crops.append((tk_slam[1] + crop_st, tk_slam[1] + crop_en, p_mat_slam))
                    for ek in gene_extra_keys.get(idx, []):
                        avail_crops.append((ek[1] + crop_st, ek[1] + crop_en, win_pred_map[ek]))

                    exon_sum = 0.0
                    for ex_s, ex_e in exons:
                        best_c = next(((cs, ce, pm) for cs, ce, pm in avail_crops if cs <= ex_s and ex_e <= ce), None)
                        if best_c is None:
                            best_c = max(avail_crops, key=lambda x: max(0, min(x[1], ex_e) - max(x[0], ex_s)))
                        c_gst, _, p_ex = best_c
                        b_es = max(0, min(crop_len - 1, ex_s - c_gst))
                        b_ee = max(b_es + 1, min(crop_len, ex_e - c_gst))
                        exon_sum += float(np.sum(p_ex[b_es:b_ee, slam_track_idx]))
                    rec["ntv3_pred_slam_2.5hpf_all_exons"] = exon_sum
                    rec["ntv3_pred_slam_2.5hpf_exon_mean_per_kb"] = exon_sum / total_exon_kb

        return pd.DataFrame(records)

    # Custom DSL fallback
    records = []
    for gname in gene_names:
        try:
            r = infer_premzt(
                model=model,
                species=sp_key,
                gene=gname,
                locus_range=locus_range,
                work_dir=work_dir,
                intron1_cap_bp=intron1_cap_bp,
                stride_bp=stride_bp,
                batch_size=batch_size,
                device=device,
            )
            if isinstance(r, dict):
                records.append(r)
            elif isinstance(r, pd.DataFrame):
                records.extend(r.to_dict("records"))
        except Exception as e_g:
            logger.warning(f"Error scoring gene '{gname}' in batch: {e_g}")
            records.append({"gene": gname, "error": str(e_g)})

    return pd.DataFrame(records)


if __name__ == "__main__":
    logger.info("premzt_inference module loaded directly.")
