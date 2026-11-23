# Dissecting Pre-MZT Cis-Regulatory Grammar via Cross-Species Sequence-to-Function Transfer Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Conda](https://img.shields.io/badge/conda-nucleotide-green.svg)](https://docs.conda.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A publication-ready, species- and locus-agnostic bioinformatics framework to dissect the cis-regulatory grammar of pre-MZT (minor-wave zygotic gene activation) gene expression. This platform leverages cross-species transfer learning across teleost embryogenesis (Zebrafish, Medaka, Cichlid) combining high-resolution base-pair convolutional networks (**ChromBPNet**) and multi-task genomic foundation models (**Nucleotide Transformer v3**).

---

## 🔬 Scientific & Architectural Overview

During animal embryogenesis, developmental control shifts from maternal transcripts to nascent zygotic transcription in two distinct waves: the **minor wave (pre-MZT)** and the canonical **major wave (MZT)**. Unraveling the causal motifs driving early pioneer accessibility before widespread chromatin opening has been hindered by low cell numbers and rapid embryonic cell cycles.

This repository implements a modular sequence-to-function transfer learning platform:
1. **Target-Species Minor-Wave Calling & Transcriptional Diagnostics (Block 1):**
   - Replicate-aware normalization via `PyDESeq2` ($\ge 2$ replicates per stage) or geometric median-of-ratios size factors (unreplicated single-BAM fallback with explicit diagnostic notice).
   - Standardized gene classifications: `strictly_zygotic_minor_wave` ($CPM_{\text{early}} < 0.5$, $\le 5$ counts, $CPM_{\text{premzt}} \ge 2.0$, $\ge 20$ counts, $\log_2\text{FC} \ge 2.0$), `all_upregulated` (unfiltered diagnostic set), and `control_non_activated` (expression- and GC-matched).
   - Strand-corrected 5'-to-3' spliced exonic coverage diagnostic (reversing bin order for minus-strand genes so 0% is TSS and 100% is TES, with automated polarity sanity assertions).
   - Post-transcriptional vs. transcriptional diagnostics: 3' UTR CPE enrichment test (Fisher's exact test on canonical CPEs `TTTTAT`, `TTTTAAT`, eCPEs, and `AATAAA`/`ATTAAA` hexamers with $+500\text{ bp}$ downstream PAS checks), and exonic SNP allele-frequency shift checks (`pysam`) over target genes.
2. **Zebrafish SLAM-seq Nascent RNA Ortholog Overlay (Block 1b & Block 7):**
   - Unified processing of Zebrafish 4sU SLAM-seq nascent RNA-seq ($0.75\text{ hpf}$, $2.5\text{ hpf}$, $3.0–3.5\text{ hpf}$, $4.5\text{ hpf}$) exporting `zf_slamseq_gene_classes.tsv` and BigWigs.
   - Cross-species ortholog mapping (1-to-1 Ensembl / symbol matching with teleost paralog normalization) overlaying nascent transcriptional validation on the target-species MA plot to distinguish true minor-wave activation from cytoplasmic polyadenylation.
3. **Deep-Learned Blastula Chromatin Accessibility (ChromBPNet, Block 2):** Trains debiased Stage 1 base models (Kundaje Lab PyTorch engine) on late-blastula ATAC-seq peaks (*Danio rerio* dome, *Oryzias latipes* st11) with Tn5 bias correction and GC-matched negative selection.
4. **Continuous Pre-MZT Probe & Grafted End-to-End Models (Block 3):** Extrapolates early chromatin dynamics across late blastula summits using chromosome-split `StandardScaler + RidgeCV` heads trained on continuous pre-MZT targets:
   $$\text{y}_{\text{abs}} = \log_2(\text{CPM}_{\text{early}} + 1)$$
   $$\text{y}_{\text{rel}} = \log_2\left(\frac{\text{CPM}_{\text{early}} + \epsilon}{\text{CPM}_{\text{late}} + \epsilon}\right)$$
   Reports both all-peak and repeat-filtered metrics (excluding `miR-430` tandem repeats). Probes are mathematically folded into `nn.Linear(dim, 1)` and grafted onto frozen encoders for forward + reverse-complement averaged inference.
5. **Zero-Shot Cross-Species Validation & Medaka Benchmark (Block 4):**
   - Primary evaluation on clean positive sets (`strictly_zygotic_minor_wave` and `slam_confirmed_minor_wave`) vs. `control_non_activated`.
   - Secondary diagnostic series (`all_upregulated`) clearly segregated to prevent cytoplasmic polyadenylation confounding.
   - Zero-shot Zebrafish $\rightarrow$ Medaka pre-MZT ATAC benchmark (`st9` vs `st11`-only summits).
6. **Unbiased Sliding-Window Scanning (Block 5):** Tiles the target locus (e.g. `notch2_slc35a3a` ± 75 kb) across target and reference species to discover putative gained enhancers.
7. **Motif Attribution, Pre-MZT TF Filtering & Evolutionary Dissection (Block 6):** Base-resolution gradient saliency / DeepLIFT and single-nucleotide *in silico* mutagenesis (ISM). Candidate seqlets are queried against both:
   - Filtered `premzt_tfs.meme`: High-confidence subset derived from `TF_gene_names.txt` with teleost paralog normalization (`POU5F3` $\rightarrow$ `POU5F1`, `MXTX2` $\rightarrow$ `MIXL1`, `VOX` $\rightarrow$ `VENTX`).
   - Full JASPAR CORE vertebrates non-redundant `.meme`: Unbiased screening annotated with maternal TPM expression.
   - Reciprocal *in silico* evolutionary swap and TE / homologous DNA alignment.
8. **Multi-Task Foundation Model (`NTv3` Tracks), Publication Baselines & Distal Knockout (Block 7):**
   - Direct reuse of Block 1 SLAM-seq BigWigs alongside ATAC tracks.
   - Publication baselines comparison (`NTV3_RUN_BASELINES`): Raw pretrained `NTv3_tracks`, frozen trunk + linear head, and LoRA/full fine-tuning.
   - QuantSeq-aware 3'-end scoring for distal *in silico* knockout: evaluates terminal 1,000 bp / 3' UTR nascent RNA accumulation (primary QuantSeq metric) alongside all-exons sum, properly handling `NTv3`'s central 37.5% output crop.
   - Optional Option 2 target-species RNA-seq ablation toggle (`ENABLE_OPTION2_TARGET_RNA_ABLATION`).

---

## 📁 Repository Layout

```
.
├── requirements.txt           # Python dependencies for Python 3.12 + CUDA PyTorch
├── setup_env.sh               # Idempotent environment installer & self-check verification
├── premzt_cis_grammar.ipynb   # Main generic analysis and discovery notebook
├── README.md                  # System architecture, profiles & documentation
├── TF_gene_names.txt          # Study pre-MZT expressed transcription factors list
├── RNA-seq/                   # Target species maternal and pre-MZT BAM alignments
│   ├── AC_2C_rep1.markdup.sorted.bam
│   └── AC_1K_rep1.markdup.sorted.bam
├── scripts/                   # Modular, CLI-callable worker scripts
│   ├── task_manager.py        # Persistent background job executor (Local + SLURM) with live progress tracking
│   ├── download_resources.py  # Consolidated resource downloader, integrity verifier & cleanup
│   ├── prep_atac.py           # ATAC preprocessing, native Poisson peaks & continuous targets
│   ├── train_chrombpnet.py    # GC-matched nonpeaks & Stage 1 ChromBPNet training
│   ├── prep_slamseq.py        # Nascent RNA-seq SLAM-seq STAR alignment & SNP masking
│   ├── train_ntv3.py          # Multi-task NTv3 fine-tuning (LoRA / DDP full)
│   └── infer_and_attribute.py # Saliency, ISM, TOMTOM motif matching & in silico swap
└── work_<STUDY_NAME>/         # Study work directory (strictly inside repo)
    ├── config.json            # Centralized snapshot of Block 0 parameters
    ├── tasks/                 # Background task logs, status.json, and .done markers
    ├── downloads/             # Auto-downloaded genomes, GTFs, and JASPAR databases
    ├── atac/                  # Processed BAMs, native Poisson peaks, summit target TSVs
    ├── slamseq_zf/            # Processed nascent SLAM-seq tracks & STAR indices
    ├── models/                # Trained ChromBPNet and NTv3 model checkpoints
    └── results/               # Publication figures (PNG/SVG), IGV tracks, and summary TSVs
```

---

## ⚡ Environment Installation (`nucleotide`)

The repository includes an idempotent installer script `setup_env.sh` that sets up Python 3.12, installs bioinformatics CLIs via Conda (`conda-forge` and `bioconda`), installs CUDA-enabled PyTorch 2.6 and `requirements.txt`, registers the Jupyter kernel, and executes an automated verification self-check.

```bash
# Run installer from within repository root
chmod +x setup_env.sh
./setup_env.sh
```

To activate the environment manually:
```bash
conda activate nucleotide
```

### Verified Toolchain
- **Python:** 3.12
- **PyTorch:** 2.11.0 + CUDA 13.0 (`torch.cuda.is_available() == True`, RTX 3080 Laptop GPU 16 GB VRAM)
- **Genomic sequence models:** `chrombpnet-refactor` (PyTorch engine), `transformers>=4.45.0,<5.0.0`, `peft`, `accelerate`
- **Bioinformatics CLIs:** `fastp`, `bowtie2`, `samtools`, `bedtools`, `subread` (`featureCounts`), `minimap2`, `meme` (`tomtom`), `star`, `ucsc-bedgraphtobigwig`, `aria2`
- **Jupyter Kernel:** `Python (nucleotide)`

---

## 🖥️ Compute Profiles (`COMPUTE_PROFILE` in Block 0)

Every study and locus configuration is defined strictly in **Block 0** of `premzt_cis_grammar.ipynb`. Users can switch between hardware profiles with a single variable:

| Parameter | `"laptop_16gb"` (Default) | `"hpc_1mb"` |
| :--- | :--- | :--- |
| **Hardware Target** | Laptop GPU (16 GB VRAM, 64 GB RAM) | Multi-GPU HPC Node (A100 / H100) |
| **Model ID** | `InstaDeepAI/NTv3_tracks` (~650M) | `InstaDeepAI/NTv3_tracks_1mb` |
| **Context Length** | 65,536 bp (65 kb) | 1,048,576 bp (1 Mb) |
| **Central Crop Ratio** | 37.5% (24,576 bp output) | 37.5% (393,216 bp output) |
| **Fine-Tuning Mode** | LoRA (`peft`, $r=8, \alpha=16$) | Full parameter / LoRA |
| **Precision** | `bfloat16` / `float16` autocast | `bfloat16` |
| **Gradient Checkpointing** | `True` | `True` |
| **Batch Size & Accum** | Batch Size 1, Accum 16 (Effective = 16) | Batch Size 4, Accum 4 (Effective = 16) |
| **Execution Backend** | `"local_background"` (Detached session) | `"slurm"` (sbatch auto-generation) |

---

## 🔄 Persistent Background Task Manager (`scripts/task_manager.py`)

All computations running longer than a standard interactive turn are managed by `TaskManager`:
- **Fully Detached:** Local jobs are executed with `subprocess.Popen(..., start_new_session=True, stdin=subprocess.DEVNULL, close_fds=True)`, allowing tasks to continue uninterrupted even if the notebook kernel is restarted or the terminal disconnects.
- **SLURM Cluster Ready:** When `EXECUTION_BACKEND = "slurm"`, the manager writes `.sbatch` submission scripts, schedules jobs via `sbatch`, and queries `squeue`/`sacct`.
- **Strict Idempotency:** Before launching, the task manager checks `WORK_DIR/tasks/<task_name>/status.json` and the `.done` marker. Re-executing a cell while a job is running outputs `[RUNNING]` with elapsed runtime and recent log tails, preventing duplicate processes. If `.done` exists and `FORCE_RERUN=False`, the step skips cleanly.
- **Process Verification:** Uses `/proc/<pid>/stat` start-time checks to prevent erroneous status reporting due to OS PID recycling.
- **Real-Time Progress Tracking:** Worker scripts and the downloader stream status updates into `progress.txt`, dynamically surfaced in the `Progress` column of `tm.status()` (`45% (1.2G/2.7G, 10.5MiB/s, ETA 2m25s)` or `100% (Done)`).

### Interactive Task Controls in Python / Notebook:
```python
from scripts.task_manager import TaskManager

tm = TaskManager(work_dir="./work_astatotilapia_premzt", backend="local_background")

# Display status of all registered jobs
display(tm.status())

# Tail recent logs
tm.tail("prep_atac_danio_rerio", n=30)

# Cancel a running task
tm.cancel("train_chrombpnet_danio_rerio")
```

---

## 📊 Summary of Notebook Execution (Blocks 0–7)

1. **Block 0:** Configure study name, target species, target genes (`notch2`, `slc35a3a`), BAM paths, compute profile, and auto-download reference resources.
2. **Block 1:** Run `featureCounts` and `PyDESeq2` on `AC_2C` vs. `AC_1K` RNA-seq BAMs. Classify minor-wave genes and plot volcano and coverage diagnostics.
3. **Block 2:** Preprocess ATAC-seq for Zebrafish (*GRCz11*) and Medaka (*ASM223467v1*), calling native local-lambda Poisson late peaks and generating continuous targets `y_abs` and `y_rel`. Launch Stage 1 ChromBPNet training.
4. **Block 3:** Extract penultimate encoder embeddings, train `RidgeCV` linear probes, verify held-out test chromosome metrics ($r$ and AUROC), and graft linear heads onto frozen encoders.
5. **Block 4:** Perform zero-shot transfer prediction on target-species TSS regions (±25 kb), calculating ROC-AUC and Mann-Whitney $U$ statistics.
6. **Block 5:** Run an unbiased sliding-window scan across the target locus (`notch2_slc35a3a` ± 75 kb) in target species, Zebrafish, and Medaka.
7. **Block 6:** Compute DeepLIFT / gradient saliency and single-nucleotide ISM on candidate elements. Match seqlets against filtered `premzt_tfs.meme` and full JASPAR `.meme` using `tomtom`. Perform *in silico* evolutionary swap.
8. **Block 7:** Multi-task fine-tuning of `NTv3` on Zebrafish ATAC + SLAM-seq nascent RNA-seq. Evaluate distal *in silico* knockout of causal motifs and export publication figures (PNG/SVG) and IGV tracks to `WORK_DIR/results/`.

---

## 📜 License
This repository is licensed under the MIT License.
