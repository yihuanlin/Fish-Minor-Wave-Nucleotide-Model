# Dissecting Pre-MZT Cis-Regulatory Grammar via Cross-Species Sequence-to-Function Transfer Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.11%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Conda](https://img.shields.io/badge/conda-nucleotide-green.svg)](https://docs.conda.io/)
[![Code License: MIT](https://img.shields.io/badge/Code%20%26%20ChromBPNet-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NTv3 Weights: InstaDeep Non-Commercial](https://img.shields.io/badge/NTv3%20Weights%20%26%20Outputs-InstaDeep%20Non--Commercial-orange.svg)](#licensing--redistribution-guide)

A species- and locus-agnostic deep learning framework to decode the **cis-regulatory grammar of pre-MZT (minor-wave) zygotic genome activation (ZGA)** across teleost fishes. It combines base-resolution chromatin accessibility networks (**ChromBPNet**) with a 65 kb multi-task genomic transformer (**Nucleotide Transformer v3, `NTv3`**) trained exclusively on public reference embryos (*Danio rerio*), then transfers **zero-shot** purely from DNA sequence to score enhancers, promoters, and nascent transcription potential in target teleost genomes (default case study: the **`notch2_slc35a3a`** locus in the Lake Malawi cichlid *Astatotilapia calliptera*).

---

## 🧬 Scientific Background & Methodology (For Biologists)

### 1. Resolving Pre-MZT Zygotic Activation Without Early ATAC-seq Confounders
In early teleost embryos prior to the mid-blastula transition (MZT), massive maternal mRNA stores and rapid cleavage cycles create two fundamental hurdles for identifying minor-wave ZGA regulators:
1. **Cytoplasmic polyadenylation masks true transcription in standard RNA-seq:** Many maternally deposited mRNAs acquire longer poly(A) tails after fertilization, causing their apparent abundance to rise in oligo(dT) RNA-seq libraries despite zero nuclear RNA Polymerase II synthesis (indeed, **`25.3%` [`1,506 / 5,953`]** of *A. calliptera* orthologs of non-transcribed maternal zebrafish genes exhibit $\log_2\text{FC} \ge 1.5$ between 2-cell and 1,000-cell oligo(dT) libraries).
2. **Low chromatin input at early cleavage stages:** Isolating pure chromatin from hundreds of pre-MZT blastomeres yields lower library signal-to-noise than later blastula stages and is often unavailable in non-model species.

This pipeline overcomes both obstacles by combining **4sU nascent metabolic labeling (SLAM-seq)** ortholog filtering (`Block 1b`) with **cross-species sequence-to-function deep learning (`Blocks 2–8`)** trained solely on *Danio rerio* DNA, ATAC-seq, and SLAM-seq tracks and evaluated **zero-shot** purely from target-species genomic sequence.

---

### 2. Transcriptomic Cohort Definitions & Biological Rationale (`Block 1` & `Block 1b`)
To separate chromatin-opening syntax from post-transcriptional mRNA stability, target-species genes (`33,341` nuclear genes in *A. calliptera*, excluding mitochondrial and zero-span contigs) are partitioned via `scripts/extract_rds_tpm.R` and `scripts/prep_slamseq.py` into five biologically distinct classes:

* **`strictly_zygotic_minor_wave` (`n=481`, including `75` `strictly_zygotic_highconf`):** Genes virtually absent at the 2-cell maternal stage (`mean_early < 0.5 TPM`; `< 0.2 TPM` for `highconf`) that activate strongly before MZT (`mean_premzt >= 2.0 TPM`, `log2FC >= 2.0`; `>= 3.0 TPM`, `log2FC >= 2.5`, `padj < 0.05` for `highconf`). Requiring near-zero maternal baseline isolates **true *de novo* minor-wave transcription** free from cytoplasmic polyadenylation of abundant maternal transcripts.
* **`all_upregulated` (`n=2,732`):** Pre-MZT induced genes (`log2FC >= 1.5`, `mean_premzt >= 2.0 TPM`, `padj < 0.05`) without an upper cap on 2-cell maternal RNA. This captures biologically critical developmental genes that are simultaneously maternally supplied and early-induced—including default targets **`NOTCH2` (`4.39 -> 32.23`, `+2.85 log2FC`)** and **`slc35a3a` (`48.78 -> 217.55`, `+2.15 log2FC`)**—which require direct chromatin/sequence attribution (`Blocks 4–8`) to verify zygotic promoter activity.
* **`is_slam_confirmed_minor_wave` (`SLAM High-Conf`, `n=11–13` across benchmarks):** Pre-MZT upregulated cichlid genes whose 1:1 zebrafish orthologs exhibit direct 4sU metabolic `T->C` nascent RNA labeling before MZT (`preMZT_strictly_zygotic` or `preMZT_maternal_zygotic`).
* **`control_silent` (`Control Silent`, `n=14,245`):** Inactive genes (`mean_early < 0.2 TPM`, `mean_premzt < 0.2 TPM`), serving as the primary negative baseline to test whether sequence models distinguish early-activated loci from inert chromatin.
* **`control_non_activated` (`Control Maternal`, `n=3,820`):** Abundant, stable maternal mRNAs (`mean_early >= 2.0 TPM`, `|log2FC| <= 0.5`). Although also **transcriptionally silent zygotically** between 2-cell and 1,000-cell, `Control Maternal` is deliberately separated from `Control Silent` because maternally deposited mRNAs overwhelmingly represent **constitutive housekeeping, translational, and cell-cycle genes**. Their promoters carry constitutively open CpG islands and Pol II bookmarking right from fertilization, giving them chromatin openness nearly equal to (or exceeding) minor-wave genes (`Figure 8A–C`). Comparing minor wave against `Control Silent` tests **promoter opening capability**, whereas comparing against `Control Maternal` tests **early-timing specificity over housekeeping chromatin**.

---

### 3. Leak-Free Whole-Chromosome Holdout Across Both ATAC-seq and SLAM-seq (`chr_fold.json`)
In genomic deep learning, splitting randomly by genomic bin leaks overlapping sliding windows between training and test sets. Instead, `assign_leak_free_splits(cv_folds=5, val_frac=0.15)` in [scripts/prep_atac.py](file:///home/home/NucleotideModel/scripts/prep_atac.py) partitions *Danio rerio*'s **25 assembled primary chromosomes ($\ge 1$ Mb; `1,345.1 Mb` total)** cleanly into **5 disjoint cross-validation test folds**:
* **Train (`64.0%`):** 16 chromosomes (`863.9 Mb`)
* **Validation (`12.0%`):** 3 chromosomes (`chr1, chr10, chr12` — `154.2 Mb`, used for checkpoint selection / early stopping)
* **Test (`24.0%`):** **6 strictly held-out chromosomes (`chr7, chr8, chr11, chr17, chr18, chr19` — `327.0 Mb`, containing `12,252` high-confidence peaks)**

Holding out ~20–25% of chromosomes (and ~10–15% for validation) is standard practice in sequence-to-function genomics (ChromBPNet, Enformer, NTv3) because evaluation requires thousands of unmemorized peaks spanning diverse GC and gene-density regimes rather than a single small chromosome (`fold_0` explicitly forces `chr8`—carrying zebrafish `notch2` and `slc35a3a`—into `test`). Crucially, **both `ChromBPNet` (ATAC-seq) and `NTv3` (jointly trained on ATAC-seq + nascent SLAM-seq BigWigs) share this exact `chr_fold.json` chromosome split**, while target-species *A. calliptera* RNA-seq and DNA remain **100% unseen** during training. Repeat-masking additionally excludes the hyper-tandem `miR-430` array during evaluation so copy-number repeats cannot inflate correlation metrics.

---

### 4. Model Architectures, Bias Factorization & Derived Metrics

#### Stage 1 & Stage 2 Grafted `ChromBPNet` (`2,114 bp` CNN)
1. **How the Debiased Model (`nobias`) Is Trained:** ChromBPNet (`scripts/train_chrombpnet.py`) does *not* subtract Tn5 bias post-hoc from a pretrained network. First, a standalone **Tn5 bias subnetwork (`bias.pt`)** is trained solely on GC-matched closed background regions (`nonpeaks`), where real TF occupancy is minimal, forcing it to learn pure Tn5 transposase nicking sequence preferences (`bias_counts_r = -0.1438` on peaks). Next, `bias.pt` is **frozen** and summed in log-count and profile space with a trainable **TF regulatory network (`tf_model`)** across high-depth **dome-stage (`105.2M` reads, `TSSe = 6.85`)** ATAC-seq peaks (`prediction = tf_model(DNA) + frozen_bias(DNA)`). Because the frozen bias subnetwork accounts for Tn5 enzymatic texture during backpropagation, `tf_model` (`*_nobias.pt`) isolates genuine TF binding syntax:
   * **Profile Median JSD (*Jensen–Shannon Divergence*, `0` = identical base distribution, `1` = disjoint):** Evaluates shape similarity across 1,000 bp peak profiles (`0.6212` for Full ChromBPNet). For `nobias`, JSD against raw ATAC cuts rises intentionally to `0.8226` because `nobias` removes sharp Tn5 cutting spikes present in experimental reads, while preserving quantitative peak count accuracy (**Debiased Count $r = 0.7001$** vs. Full `0.7044`).
   * **Max Tn5 Footprint Response (`0.001` ZF / `0.002` Medaka; `< 0.003` pass threshold):** Probes synthetic background sequences carrying pure canonical Tn5 motifs, proving directly that `nobias` produces near-zero artifactual response to Tn5 enzyme specificity.
2. **Why Stage 2 Uses `RidgeCV` (and Not Minor-Wave RNA-seq Fine-Tuning):** Stage 2 `RidgeCV` (`alphas = np.logspace(-2, 4, 20)` with chromosome-grouped 5-fold CV) is trained **on the `49,377` high-confidence zebrafish ATAC peaks (`FE >= 4.0, -log10(q) >= 5.0`)**—*not* on minor-wave RNA-seq genes. We extract penultimate embeddings from the frozen dome `nobias` encoder and fit convex L2 linear heads predicting early `256-cell` chromatin openness (`48.0M` reads, `TSSe = 3.16`) rather than fine-tuning all CNN weights directly, because end-to-end retraining on lower-SNR early ATAC risks overfitting noise and distorting clean Stage 1 TF filters. Scaling and Ridge weights are folded analytically into `grafted_chrombpnet.pt` alongside the dome `count_dense` head:
   * **$y_{\text{abs}} = \log_2(\text{early CPM} + 1.0)$:** Predicted absolute pre-MZT (`256-cell`) chromatin accessibility (`scripts/prep_atac.py`).
   * **$\text{late}$ (`pred_late`):** Predicted late-blastula (`dome`) log accessibility from Stage 1 ChromBPNet.
   * **$y_{\text{rel}}$ (plot alias `y_resid`):** $\log_2\!\big((\text{early CPM}+1.0)/(\text{late CPM}+1.0)\big)$ (with residual regression parameters `0.4448 / -1.2265` stored in `grafted_chrombpnet.meta.json`), measuring whether chromatin is *relatively enriched early* versus late blastula.
   * **Openness Gate (`late >= 5.70`) & `y_resid_gated`:** Calibrated at the 15th percentile of held-out dome ATAC peaks (`17.3%` background silent pass rate in `openness_gate.json`). In closed heterochromatin (`late < 5.70`) or `'N'`-padded contigs (`late = -5.47, y_abs = -3.33`), near-zero ratios generate noisy positive `y_rel` values (`+0.31`); gating sets `y_rel` to `NaN` whenever `late < 5.70`.

#### Multi-Task `NTv3` (`65,536 bp` Transformer, `LoRA` vs. `Frozen Head` vs. `Raw Pretrained`)
To capture distal `>2 kb` enhancer–promoter looping and directly model nascent RNA (`scripts/train_ntv3.py`), we adapt **`InstaDeepAI/NTv3_tracks`** (~650M-parameter U-Net DNA transformer over `65,536 bp` inputs predicting the central `24,576 bp` / `37.5%` crop) jointly across **`10` tracks**: zebrafish `256-cell` & `dome` ATAC plus `+`/`-` strand nascent SLAM-seq at `0.75, 2.5, 3.0–3.5, 4.5 hpf` (up-weighting pre-MZT `256-cell`, `0.75 hpf`, and `2.5 hpf` tracks `3×` in `TrackBalancedLoss`):
* **`NTv3_raw_pretrained`:** Evaluates out-of-the-box pretrained zebrafish U-Net tracks without task training (`Spearman r = 0.0704`, near random because pretrained tracks represent later/differentiated tissues).
* **`NTv3_frozen_head`:** Freezes the entire 650M-parameter transformer trunk (`requires_grad=False`) and trains **only a `1024 -> 10` linear head** (`2` epochs). Because NTv3's pretrained multi-species representations already encode generic promoter/CpG grammar, `NTv3_frozen_head` jumps to `r = 0.6741` (`target ATAC AUROC = 0.672`), yet fails completely on nascent SLAM (`AUROC = 0.495`, pure chance).
* **`NTv3_multitask_lora` (`LoRA`):** Injects lightweight trainable low-rank adapter pairs (`r=8, alpha=16`) into attention (`q_layer`, `k_layer`, `v_layer`) and `linear_embedding` projections plus the `F.softplus` 10-track head (fitting in ~7.5 GB VRAM via activation checkpointing for `3` epochs, `best_val_loss = 0.0942`). Fine-tuning internal layers raises held-out `256-cell` ATAC Spearman $r$ to **`0.8457`**, target ATAC AUROC to **`0.7546`**, and lifts SLAM AUROC above random (`0.5400`/`0.5649`). Exact run parameters (`context_len`, `species_id=8`, tracks, loss) are saved in **`ntv3_train_config.json`** (hyperparameter snapshot) so downstream inference scripts rebuild the exact architecture reproducibly.

#### Attribution, Statistical Benchmarking & Gene Parity Metrics
* **Motif Saliency ($|\nabla_{\text{seq}}|$):** Backpropagation derivatives summing gradient magnitude across `A/C/G/T` channels (`50 bp` boxcar-smoothed for locus tracks), highlighting regulatory DNA bases and TF motifs (*seqlets*).
* **In Silico Perturbations ($\Delta$) & Evolutionary Swaps:** Quantifies predicted accessibility/SLAM shifts upon deleting a discovered motif (ablation), shuffling 500 bp preserving dinucleotide counts (`dinuc_shuffle`), ablating CpGs (`CG -> TG`), or replacing *A. calliptera* motifs with *D. rerio* ortholog sequence.
* **TOMTOM & Teleost Paralog Normalization:** Matches attribution seqlets to `JASPAR 2024` (`q`-value = FDR-adjusted match probability) with teleost-to-vertebrate paralog aliases (`POU5F3 -> POU5F1`, `SOX19B -> SOX2`, `MXTX2 -> MIXL1`, `VOX -> VENTX`).
* **Gene Parity & MAD Robust $z$-Score:** Evaluates whether target genes occupy the predicted regulatory activity band (**Within IQR `Q25–Q75`** or **`Above Q75`**) of *bona fide* minor-wave ZGA genes (`strictly_zygotic_minor_wave`), calibrated against negative controls via `robust_z = (score - median) / (1.4826 * MAD)` (**Median Absolute Deviation** `median(|x - median(x)|)` scaled to Gaussian SD so extreme housekeeping/rDNA outliers do not inflate variance), leave-one-out empirical $p$-value, and Benjamini–Hochberg FDR $q$-value.

---

## 📊 Scientific Audit & Default Case Study (`notch2_slc35a3a` in *A. calliptera*)

### Stage 1 ChromBPNet Training & Medaka Exclusion (`Block 2`)
| Species | Peaks (`FE>=4, q>=5`) | Late TSSe (Early) | Full Count $r$ | Debiased (`nobias`) Count $r$ | Full JSD (`nobias` JSD) | Max Tn5 Response | QC Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| ***Danio rerio*** | 146,778 (`49,377`) | **6.85** (`3.16`) | **0.7044** | **0.7001** | **0.6212** (`0.8226`) | **0.001** | **PASS** (Sole CNN encoder) |
| ***Oryzias latipes*** | 12,141 (`245`) | **2.23** (`1.35`) | **0.3931** | **0.3913** | **0.6101** (`0.7715`) | **0.002** | **FAIL (`MEDAKA_PASSED_QC = False`)** |

* **Quick Interpretation:** `0.00%` mitochondrial contamination across all BAMs ([qc_audit_report.md](file:///home/home/NucleotideModel/work_astatotilapia_premzt/qc_audit_report.md)) proves Medaka's `r=0.3931` reflects intrinsically low experimental library enrichment (`st9` TSSe `1.35`, comparable to zebrafish naked gDNA `1.72`) plus missing gDNA background suppression. Excluding Medaka (`MEDAKA_PASSED_QC = False`) prevents noise dilution while retaining Medaka `st9` as an external cross-species test benchmark (`AUROC = 0.7113`).

### Cross-Architecture Benchmark Summary (`Block 7` & `Block 8`)
Evaluated on identical held-out zebrafish test chromosomes (`chr7, 8, 11, 17, 18, 19`, excluding `miR-430`) and zero-shot *A. calliptera* cohorts ([ntv3_baselines_benchmark.tsv](file:///home/home/NucleotideModel/work_astatotilapia_premzt/models/ntv3/ntv3_baselines_benchmark.tsv) & [notch2_slc35a3a_cross_architecture_summary.tsv](file:///home/home/NucleotideModel/work_astatotilapia_premzt/results/notch2_slc35a3a/notch2_slc35a3a_cross_architecture_summary.tsv)):

| Model | Architecture | ZF Test 256-Cell Spearman $r$ | Early vs. Late Timing AUROC | Medaka `st9` Zero-Shot AUROC | Target ATAC AUROC (vs. Silent / Maternal) | Target SLAM AUROC (vs. Silent / Maternal) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`ChromBPNet_grafted`** | 2.1 kb Debiased CNN + Ridge | 0.6573 | **0.9807** | 0.7113 | 0.6925 / 0.4342 | N/A (ATAC-only) |
| **`NTv3_raw_pretrained`** | Pretrained 65 kb Transformer (no head) | 0.0704 | 0.5356 | N/A | N/A | N/A |
| **`NTv3_frozen_head`** | Frozen NTv3 Trunk + Linear Head | 0.6741 | 0.8758 | 0.6611 | 0.6720 / 0.4244 | 0.4950 / 0.4897 |
| **`NTv3_multitask_lora`** | 65 kb Multi-Task LoRA Transformer | **0.8457** | 0.9553 | **0.7264** | **0.7546 / 0.4501** | **0.5400 / 0.5649** |

---

## 🖼️ Publication Figure & Table Legends (With Biological Interpretations)

All primary outputs live under `work_astatotilapia_premzt/results/notch2_slc35a3a/` (plus training/QC files under `models/`, `atac/`, and `slamseq/`).

### Figure Legends & Key Biological Takeaways

#### `fig1_volcano_ma_plot.png` — Pre-MZT Transcriptome Induction & Two-Cichlid Concordance (`Block 1`, RDS Mode)
* **Panels A & B (MA Plots — *Astatotilapia calliptera* [AC] & *Tropheops* sp. 'mauve' [TM]):** Plot mean expression ($\log_{10}(\text{mean TPM} + 0.1)$, x-axis) against pre-MZT induction ($\log_2\text{FC}$ between 1,000-cell pre-MZT and 2-cell baseline, y-axis) across `33,378` annotated genes. Color encodes cohort assignment: `Strictly Zygotic Minor Wave` (red; AC `n=481`, TM `n=698`), `All Upregulated` (orange; AC `n=2,732`, TM `n=2,611`), `Control Non-Activated / Maternal` (blue, `|log2FC| <= 0.5`; AC `n=3,835`, TM `n=3,756`), and remaining genes (`gray`, including `Control Silent` at bottom left). Target genes `notch2` (`AC +2.85`, `TM +3.01`) and `slc35a3a` (`AC +2.15`, `TM +1.28`) sit within the robustly upregulated cloud alongside canonical ZGA markers `vox` (`AC +4.96`, `TM +6.54`) and `klf17` (`AC +2.51`, `TM +3.08`).
* **Panel C (Cross-Cichlid $\log_2\text{FC}$ Concordance):** Plots AC $\log_2\text{FC}$ (x) vs. TM $\log_2\text{FC}$ (y) along the $y=x$ concordance diagonal, isolating `247` conserved two-cichlid minor-wave genes (red) vs. lineage-specific AC-only (orange) or TM-only (light blue) genes.
* **Quick Interpretation:** Although `vox` has zero detectable 2-cell RNA (`strictly_zygotic_minor_wave`), `klf17`, `notch2`, and `slc35a3a` begin with detectable maternal mRNA (`4.4–48.8 CPM`) and still surge `4–8×` before MZT consistently across both cichlid species—making them strong candidate pre-MZT targets requiring cis-regulatory testing against maternal poly(A) artifacts.

#### `fig1b_slamseq_ortholog_overlay.png` — Zebrafish 4sU Nascent SLAM-seq Ortholog Filter (`Block 1b`)
* **Panel A (AC MA Plot Recolored by Orthologous Zebrafish Nascent Timing):** Maps `17,131 / 33,378` (`51.3%`) AC genes onto 1:1 zebrafish SLAM-seq kinetic classes: `preMZT_strictly_zygotic` (`n=1`, `RABEPK`), `preMZT_maternal_zygotic` (`n=46`), `major_wave_only` (`n=469`), `maternal_non_transcribed` (`n=5,953`), `silent_or_low` (`n=10,662`), and `unmapped_ortholog` (`n=16,247`). Labeled genes show that zebrafish `notch2` is `maternal_non_transcribed` and `slc35a3a` is `silent_or_low` (whereas `vox` and `klf17` activate at the major wave in zebrafish), consistent with **cichlid lineage-gained early activation** at `notch2` and `slc35a3a`.
* **Panel B (AC $\log_2\text{FC}$ Violins Grouped by Orthologous SLAM Class):** Displays target $\log_2\text{FC}$ distributions per zebrafish timing class with one-sided Mann–Whitney *U* tests against `Maternal Non-Transcribed (Control)` and a global one-sided Fisher's exact enrichment test (`Pre-MZT Active n=47` enriched in cichlid upregulated genes: `OR = 1.93, p = 7.79e-02`). Notice the broad right-skewed upper tail (`log2FC` up to `+6..+8`) in `Maternal Non-Transcribed (median = +0.60, 25.3% >= 1.5)`, which washes out group-median differences (`MWU p = 0.59–0.94`).
* **Quick Interpretation:** Roughly one-quarter (`25.3%`) of conserved non-transcribed maternal orthologs still rise $\ge 2.8\times$ in cichlid oligo(dT) RNA-seq due to cytoplasmic polyadenylation, proving bulk RNA-seq fold-change alone cannot reliably separate nuclear transcription from poly(A) elongation—motivating the sequence-based chromatin/SLAM models in `Blocks 2–8`.

#### `fig1c_snp_allele_shift.png` — Zygotic Allelic Onset via Exonic SNP Shifts (`Block 1`, BAM Mode)
* **What It Plots:** Histogram and kernel density curve (`Null Background Density`) of heterozygous exonic alternate-allele frequency shifts ($|\Delta \text{alt\_freq}|$ between 1,000-cell and 2-cell BAMs, `Q>=20`, `depth>=30`, `MAF>=0.15`) across `N=379` `control_non_activated` SNPs, alongside the empirical `95th percentile` cutoff (`0.089`, dashed orange) and median shifts for `notch2` (`0.038`, purple) and `slc35a3a` (`0.172`, red).
* **Quick Interpretation:** Purely maternal transcripts stay balanced near their 2-cell allelic ratio (`<= 0.05` sampling noise), whereas `slc35a3a` exhibits a large shift (`0.172 > 0.089` cutoff) consistent with newly synthesized paternal/zygotic transcripts shifting allelic proportions before MZT (`notch2` shows modest shift `0.038`, consistent with either homozygous/low SNP power or balanced biallelic onset).

#### `fig2_zeroshot_benchmark.png` / `.svg` — Zero-Shot ChromBPNet Promoters Validation & Gate Mechanics (`Block 4`)
* **Difference vs. Figure 7:** `Figure 2` evaluates **ChromBPNet alone across all `33,341` genome-wide AC nuclear genes**, four cis-windows, and three ChromBPNet scores (`y_abs`, `y_resid_gated`, `late`), whereas `Figure 7` evaluates **ChromBPNet vs. NTv3 ATAC vs. NTv3 SLAM head-to-head on the exact intersected subset** (`n=90 vs. 587`) so Precision-Recall curves share one baseline. Thus the two figures are complementary, not redundant.
* **Panel A (Zero-Shot ROC, `promoter_1kb | y_abs vs. control_silent`):** Plots True Positive Rate (Sensitivity) vs. False Positive Rate (`1 - Specificity`) using core promoter pre-MZT openness `y_abs` to rank unseen AC cohorts against `control_silent` (`n=14,245`). Stronger bowing toward top-left indicates better separation (`SLAM-Confirmed Orthologs n=11 AUC=0.86`; `All Upregulated n=3,213 AUC=0.75`; `Strictly Zygotic All n=481 AUC=0.69`; `Strictly Zygotic High-Conf n=83 AUC=0.69`).
* **Panel B (Zero-Shot Precision-Recall vs. Dotted Cohort Prevalence):** Plots Precision (`TP / (TP + FP)`) vs. Recall against dotted random prevalence lines (`0.0008` for `n=11` up to `0.184` for `n=3,213`). *Why does Precision drop sharply to `0.0` near `Recall = 0`?* Ranking all `14,328` genes by descending `y_abs` puts a handful of silent CpG-rich developmentally poised promoters (`ENSACLG00000029139` `y_abs=3.47`, `FOXB1` `2.84`, `5_8S_rRNA` `2.73`) slightly above the highest minor-wave gene; until the first true positive enters, Precision plunges before rebounding strongly (`SLAM-Confirmed PR = 0.011` achieves **`14.4×` its random prevalence**; `Strictly Zygotic All PR = 0.075` is `2.3×` baseline).
* **Panel C (`Primary: vs. Silent Controls, n=14,245`):** Compares AUROC across four windows (`promoter_1kb`, `promoter_intron1_capped`, `proximal_5kb`, `bounded_25kb`) for three ChromBPNet metrics: absolute early openness **`y_abs`** (`0.68–0.71`), late-dome baseline openness **`late`** (`0.73–0.75`), and **`y_resid_gated (late >= 5.70)`** (`0.49–0.54`), plus the held-out **Zebrafish Test-Chr 1:5 GC-Matched ATAC→SLAM reference** (`0.645` at `1kb` dropping to `0.461` at `25kb`). *Why do `y_abs`/`late` dominate here while `y_resid_gated` sits near `0.50`?* Only `14.0%` of silent genes pass `late >= 5.70` vs. `54.7%` of minor-wave genes, so overall openness already separates silent genes (`median y_abs 1.53 vs. 1.17`), whereas within the small open subset, silent poised promoters match minor-wave early/late ratios (`0.279 vs. 0.275`). Meanwhile, the ZF Test-Chr curve drops at `25kb` because taking the `max` across ±25 kb windows picks up neighboring constitutively open promoters on gene-dense chromosomes in `n=16` positives.
* **Panel D (`Diagnostic: vs. Maternal-Stable Controls, n=3,820`):** Reverses Panel C! Because maternal housekeeping promoters are already open (`59.5%` pass `late >= 5.70`; `median y_abs = 1.516` vs. minor-wave `1.531`), raw `y_abs` (`0.47–0.57`) and `late` (`0.44–0.53`) fail to distinguish minor-wave promoters from housekeeping promoters—whereas **`y_resid_gated` climbs to `AUROC = 0.61` (`promoter_1kb`)**, demonstrating that early-biased chromatin timing above dome baseline specifically separates minor-wave regulation from maternal housekeeping chromatin.

#### `fig3_locus_scan_profile.png` / `.svg` — 5-Row Sliding-Window Locus Scan vs. Held-Out Zebrafish Ortholog (`Block 5`)
* **Two-Column Layout:** Left column scans *A. calliptera* target locus `chr23:10,907,626–11,058,130` (`594` sliding `2,114 bp` windows, `250 bp` step); Right column scans the **strictly held-out** orthologous *D. rerio* locus `chr8:19,498,985–19,688,362` for direct evolutionary comparison.
* **Row 0 (Full GTF Genes):** Non-overlapping gene spans preserving neighborhood synteny (`zgc:92140`..`TMEM59L` vs. `sec22bb`..`foxe3`), with `NOTCH2` and `slc35a3a` highlighted in red.
* **Row 1 (`pred_late` + `5.70` Gate):** Stage 1 predicted dome-stage baseline openness against the horizontal `5.70` Openness Gate (`66/594` = `11.1%` AC windows pass).
* **Row 2 (`pred_y_abs` & Candidates `C1–C14`):** Predicted pre-MZT `256-cell` accessibility marking `14` non-redundant open peaks ranked strictly by peak `y_abs` (`C1=2.65`, `C2=2.56`, `C3=2.29`, down to `C14=1.46`; target promoters `C9 NOTCH2 = 1.71`, `C11 slc35a3a = 1.62`).
* **Row 3 (`y_resid_gated` Timing Bias — Why Discrete Bars Over Gray Background):** Comparing `Predicted y_resid_gated` (Left, AC `chr23`) vs. `ZF y_resid_gated` (Right, ZF `chr8`), the faint **gray line** shows *un-gated* `y_rel`, which scatters noisily between `-0.3` and `+0.7` across the ~`89%` closed heterochromatin where tiny early/late counts fluctuate randomly. Applying the **`late >= 5.70` Openness Gate** removes closed noise, leaving crisp vertical **blue (`y_resid_gated >= 0`, early-biased)** and **orange (`< 0`, late-biased)** bars strictly on genuine open regulatory summits (`414 bp` median peak width across a 150 kb span, led by `C1 = +0.96`, `C2 = +0.71`, `C3 = +0.52`).
* **Row 4 (`Candidate Spans` vs. `Experimental ZF ATAC + MACS2/Poisson Dome Reference Peaks`):** Left flags `C1, C2, C3` in pink as **cichlid lineage-gained peaks** surpassing the entire zebrafish ortholog max (`1.987`, dashed line). Right overlays **observed experimental zebrafish `256-cell` (green) & `dome` (purple) ATAC signal** with orange **`MACS2/Poisson Dome Reference Peaks`** (`37` peaks called across this `chr8` interval). *Why plot dome reference peaks rather than 256-cell peak calls?* Because `256-cell` ATAC libraries have half the sequencing depth (`48M` vs. `105M`) and lower promoter signal-to-noise (`TSSe = 3.16` vs. `6.85`), discrete reference peak intervals are called on high-SNR dome reads via local-lambda Poisson modeling (capturing both constitutive pre-MZT enhancers and dome peaks), and *both* `256-cell` and `dome` cut densities are then quantified inside those exact reference intervals—visually confirming here that held-out ChromBPNet predictions (Rows 1–3 Right) recover real accessible chromatin on unseen `chr8`.

#### `fig4a_locus_cis_regulatory_landscape.png` / `.svg` — Locus Landscape & Evolutionary Swap Perturbation (`Block 6`)
* **Panel A (`Pre-MZT Cis-Regulatory Landscape`):** Summarizes ChromBPNet `y_abs` (blue) and `y_rel_gated` (purple dashed) on a signed coordinate axis centered on `notch2` TSS (`0 kb`), shading lineage-gained elements (red), conserved syntenic promoters (green), and enhancers (blue) above dedicated gene tracks, annotating top TF-matched drivers (`R3 NFYA★`, `R10 ZNF281`, `R12 MNT`, `R5 Foxf1`, `R2 ZBTB14`).
* **Panel B (`Top 15 Causal Motifs by In Silico Ablation`):** Ranks all `56` extracted seqlets by predicted accessibility drop when deleted (`-0.064` to `-0.107`), distinguishing maternal TFs (`NFYA★ CCAATC` at R3, red) from general JASPAR vertebrate motifs (`GGCCGCGC` R13, `MNT` R12, `Foxf1` R5, `ZBTB14` R2, blue).
* **Panel C (`Ablation vs. Evolutionary Swap Perturbation`):** Plots target motif ablation effect (x-axis) against replacing *A. calliptera* motif DNA with *D. rerio* homologous sequence (y-axis) across all `24` syntenically aligned seqlets (`32` unaligned lineage-specific motifs lie outside alignment), coding feature class (`Promoter` green / `Intronic` orange), dot size hierarchy (`Large = primary causal motif`, `Small = secondary sub-motif`), gold maternal TF ring, and point labels for every motif.
* **Quick Interpretation:** Most points cluster in the **bottom-left quadrant** (`R1`, `R2`, `R4`, `R9`, `R13`: both deleting the cichlid motif *and* swapping in the zebrafish allele reduce accessibility by `-0.025` to `-0.075`), demonstrating that nucleotide substitutions at these conserved promoters (`chr23` vs. `chr8`) directly increased TF affinity and pre-MZT openness in *A. calliptera*.

#### `fig4b_single_nt_attribution_tracks.png` / `.svg` — Base-Resolution Saliency Sequence Logos (`Block 6`)
* **What It Plots:** Individual nucleotide gradient contribution weights (`±80 bp` centered on element summits) for `Rank 1 (fam78bb, +61.8 kb)`, `Rank 2 (TMEM59L, +79.6 kb)`, `Rank 3 (zgc:92140, -39.7 kb)`, and `Rank 9 (NOTCH2, +0.3 kb)`. Positive letters (`A/C/G/T` above 0) increase predicted `y_abs`, shaded boxes mark JASPAR seqlets overlapping the central `160 bp`, top-left badges flag top element motifs located just outside `±80 bp` (e.g., `NFYA ★ CCAATC` at `+222 bp` in Rank 3; `ZBTB14 GTGCGCG` at `-93 bp` in Rank 2; `Unmatched GTCGCTGCCG` at `-350 bp` in Rank 9), and bottom alignments compare cichlid vs. homologous zebrafish DNA.
* **Quick Interpretation:** Reveals GC/CpG-rich regulatory clusters across cichlid summits (`R1 CGCGCT Δ=-0.072`, `R2 ZBTB14 GTGCGCG` where *A. calliptera* gained `cpg_oe = 0.99` vs. zebrafish `0.00` via alignment divergence, and unaligned cichlid-specific `R3`).

#### `fig4c_genomewide_minor_wave_motifs.png` / `.svg` — Genome-Wide Promoter Motif & CpG Enrichment (`Block 6`)
* **Panel A (`Top TF Motifs in Minor-Wave Loci vs. Silent Controls`):** Paired horizontal bars comparing JASPAR seqlet occurrence (`%`) across `n=265` minor-wave promoters (red) vs. `n=265` GC-matched silent controls (blue), with odds ratios and Fisher exact $p$-values (`Zbtb2 2.7% vs. 0%, OR=inf, p=7.5e-03`; `ZKSCAN3 5.8% vs. 2.7%, OR=2.2, p=6.3e-02`; `CREM 2.7% vs. 0.8%, OR=3.6, p=8.8e-02`).
* **Panel B (`CpG-Island Priming`):** Violin distributions of promoter `Observed / Expected CpG Ratio` showing genome-wide minor-wave enrichment (`Mean CpG O/E = 0.68` vs. Silent `0.44`), with `R2 (TMEM59L, 0.99)`, `R1 (fam78bb, 0.78)`, and `R9 (NOTCH2, 0.74)` marked inside the elevated CpG-primed zone.
* **Quick Interpretation:** Beyond sequence-specific TFs (`Zbtb2`, `ZKSCAN3`, `CREM`), broad **CpG-island priming** (`0.68` vs. `0.44`) distinguishes active pre-MZT minor-wave promoters genome-wide and characterizes the `notch2_slc35a3a` elements.

#### `fig5_cross_architecture_locus_overlay.png` / `.svg` — Five-Track `NTv3` vs. `ChromBPNet` Locus Overlay (`Block 8b`)
* **Panel A (Pre-MZT ATAC):** Overlays ChromBPNet `y_abs` (`blue`, 2.1 kb CNN) and `NTv3` 256-cell ATAC (`red`, 65 kb transformer) across signed distance from `notch2` TSS (`0 kb`), confirming concordant accessibility peaks at `R3`, `R10`, `NOTCH2` (`0 kb`), `slc35a3a` (`+42.8 kb`), `R12`, `R1` (`+61.8 kb`), `R5`, and `R2` (`+79.6 kb`).
* **Panel B (Strand-Resolved `NTv3` `2.5 hpf` Nascent SLAM — Why `+` and `-` Co-Vary With an Offset):** Plots predicted `log1p` `2.5 hpf` nascent transcription for `+` (dark green, `mean=0.0842`) and `-` (purple dashed, `mean=0.1050`) strands. *Why do both curves track closely (`r = 0.9887`) separated by a `+0.0208` shift?* Training `2.5 hpf` nascent SLAM BigWigs are extraordinarily sparse (`<0.035%` non-zero bases genome-wide, `0.0000` across orthologous `chr8`), so both linear output heads draw largely on a shared multi-task promoter-openness representation plus `+0.021` head bias/weight offset—superimposed on coding sequence periodicity inside `NOTCH2` exons (`0–40 kb`).
* **Panel C (Dual-Axis Base Saliency — ATAC vs. `Total Locus` SLAM vs. `notch2 Exon` SLAM):** Plots ATAC $|\nabla|$ (pink, **left y-axis**) against nascent SLAM $|\nabla|$ (**right `twinx` y-axis**, green). *Why do Total SLAM saliency peaks largely overlap ATAC peaks across the entire plot?* First, **`Total Locus 2.5 hpf SLAM Saliency`** (dashed green, max `234.91` at `-20.6 kb`) differentiates the **sum of predicted SLAM signal across the whole 150.5 kb `notch2_slc35a3a` neighborhood** (all 10 genes: `zgc:92140`, `EDEM3`, `NPL`, `sec22bb`, `NOTCH2`, `slc35a3a`, `fam78bb`, `TMEM59L`), NOT `notch2` alone—and because Pol II initiation depends on open chromatin motifs, total-locus SLAM saliency correlates strongly with ATAC saliency (`r = 0.8348`). Conversely, **`notch2 Exon-Specific SLAM Saliency`** (dotted black, max `42.15`, **right y-axis**) differentiates *only* bins overlapping `notch2`'s own exons (`r = 0.0569` with whole-locus ATAC), isolating gradients concentrated directly inside `NOTCH2`'s promoter and 3' exonic domain (`0–40 kb`, peak at `+38.15 kb`).
* **Panels D & E:** Aligned *A. calliptera* gene models (`zgc:92140`..`TMEM59L`) and the orthologous held-out zebrafish `chr8:19.50–19.69 Mb` NTv3 scan (`ATAC + SLAM`).

#### `fig6_knockout_and_motif_saliency.png` / `.svg` — `NTv3` Element Perturbations & Single-Base Promoter Saliency (`Block 8b`)
* **Panel A (`NTv3 In Silico Perturbation of Candidate Elements`):** Quantifies `% change` in local 256-cell ATAC across `Rank 1, 2, 3, 9 (NOTCH2), 11 (slc35a3a)` under 500 bp dinucleotide shuffle (blue), 500 bp `CG -> TG` CpG ablation (red), and single causal motif swap (green).
  * *Quick Interpretation:* **`CG -> TG` CpG ablation causes massive collapse across all five elements** (`-29.0%` Rank 3, `-27.4%` Rank 2, `-13.1%` Rank 11, `-12.1%` Rank 1, `-8.5%` Rank 9), exceeding single-motif swaps (`-1.6%` to `-5.4%`). Notably, dinucleotide shuffle reduces distal peaks (`-1.2%` to `-21.7%`) yet slightly increases `NOTCH2` (`+5.4%`) and `slc35a3a` (`+9.1%`) because dinucleotide shuffling preserves total CpG/GC density while disrupting local basal TSS spacing/repressive elements, confirming that **regional CpG-rich grammar**—rather than any single motif alone—anchors `NTv3`'s pre-MZT chromatin prediction.
* **Panel B (`Single-Nucleotide Attribution at notch2 Promoter`):** Base-resolution `NTv3` ATAC saliency (`raw |\nabla|` gray, 50 bp smoothed red) centered directly on `notch2` TSS (`±150 bp`), pinpointing sharp nucleotide hotspots right at TSS (`0 bp`) and `+55..+75 bp` downstream initiator/CpG sequence.

#### `fig7_cross_architecture_benchmark.png` / `.svg` — Matched-Cohort Cross-Architecture Benchmark (`Block 8b`)
* **Fair Matched Cohort Design (`n=90` Minor Wave vs. `n=587` Silent, Shared Baseline `0.133`):** Unlike `Figure 2` (which evaluates ChromBPNet alone across all `33k` genome-wide genes), `Figure 7` restricts evaluation strictly to the **identical intersected gene set scored by both ChromBPNet and NTv3**, making Precision-Recall AUPRC values directly comparable against one common `0.133` chance baseline.
* **Panels A & B (ROC & Precision-Recall):** **`NTv3 ATAC`** (`AUC = 0.760`, `AUPRC = 0.310`) outperforms **`ChromBPNet Promoter + Intron 1`** (`AUC = 0.732`, `AUPRC = 0.307`) and `ChromBPNet 1 kb` (`0.685`), showing that 65 kb long-range transformer context captures distal regulatory cues better than 2.1 kb promoters. **`NTv3 SLAM 3'-End`** achieves `AUC = 0.671` and strong early precision (`AUPRC = 0.309`, reaching `>0.55` precision up to ~15% recall), outperforming full exon-mean SLAM (`0.660` / `0.176`). *(In cached `ntv3_target_genes_scored.tsv`, `NTv3` `1 kb` and `Intron 1` display identical values prior to running `--subtask score --force-rescore` manually with the updated window slicer in [scripts/ntv3_infer.py](file:///home/home/NucleotideModel/scripts/ntv3_infer.py).)*
* **Panel C (`Training Progression`):** Compares `ChromBPNet_grafted` (`0.69` Target ATAC AUROC, `0.66` ZF $r$, `"ATAC only"`), `NTv3_raw` (`0.07` ZF $r$), `NTv3_frozen` (`0.67` Target ATAC AUROC & `0.67` ZF $r$, yet `0.49` random SLAM), and **`NTv3_multitask` (`0.75` Target ATAC AUROC, `0.85` ZF $r$, `0.54` SLAM)**. *Quick Interpretation:* Pretrained frozen representations give reasonable promoter classification (`0.67`) from generic DNA features alone, but LoRA adaptation across `65 kb` is essential to achieve high quantitative accessibility accuracy in embryos (`0.67 -> 0.85`) and lift nascent SLAM prediction above chance (`0.49 -> 0.54`).

#### `fig8_target_gene_parity_benchmark.png` / `.svg` — Target Gene Statistical Parity Benchmark (`Block 8c`)
* **Panels A & B (`ChromBPNet TSS ±1 kb` & `NTv3 Promoter + Intron 1 ATAC`):** Violins/boxes compare `Control Silent` (gray, full cohort `n=14,245` / NTv3 `n=715`), `Control Maternal` (purple, `n=3,820` / `300`), `Minor Wave Zygotic` (teal, `n=481` / `108`), and `SLAM High-Conf` (blue, `n=13`). Notice `Control Maternal` mirrors `Minor Wave Zygotic` (`ChromBPNet median = 1.52 vs. 1.45`; `NTv3 = 0.293 vs. 0.287`) because constitutively active housekeeping/maternal promoters carry open CpG chromatin from fertilization. Both **`notch2` (`1.71`, `73%ile MW`; `0.289`, `57%ile`)** and **`slc35a3a` (`1.67`, `70%ile MW`; `0.299`, `68%ile`)** exceed `Control Silent` (`medians 1.17` / `0.242`) and align cleanly with `vox`, `klf17`, and `SLAM High-Conf`.
* **Panel C (`NTv3 Nascent SLAM 3'-End`, `log(1 + Signal/kb at TES)`):** After filtering legacy un-TES-centered zeros (`n=581` Silent, `n=78` Maternal, `n=90` Minor Wave, `n=13` High-Conf), `notch2` (`94.26/kb`, `log1p=4.56`, `37%ile MW`) and `slc35a3a` (`95.05/kb`, `log1p=4.56`, `39%ile MW`) match `vox` and `klf17` inside the minor-wave IQR. *Why do silent controls still average `~91–93/kb` (`log1p ~4.53`)?* `NTv3TrackHead` applies a strictly positive smooth `F.softplus` output trained with `3×` weight on ultra-sparse `2.5 hpf` T→C bases (`<0.04%` genome positive), yielding a baseline floor of `~0.085–0.095 / bp` (`~85–95 / kb`) across most genes; poised/CGI silent 3' ends cannot yet be driven to zero from sequence alone (consistent with moderate target SLAM AUROC `0.54–0.67`).
* **Panel D (`Cross-Architecture Parity vs. Silent Calibration` — How to Read):** Each curve plots **Control-Silent Percentile (`x`-axis, `0–100%`)** against **Minor-Wave Percentile (`y`-axis, `0–100%`)** across matched thresholds $v$, with **`Minor-Wave Parity IQR (Q25–Q75)`** shaded green and **`Top Quartile (>Q75)`** shaded blue. Both axes increase from `0` to `100`, where the **dashed diagonal $y = x$ indicates no separation (`AUROC = 0.50`)**. **Curves bowing farther down and right beneath $y = x$ demonstrate stronger enrichment over silent genes** (meaning high silent percentiles correspond to lower minor-wave percentiles—e.g., an **`85th-percentile` Control-Silent** promoter reaches only the **`~50th–60th` percentile** of minor-wave genes). Because all six target gene points (`notch2`: `91.6%`→`73%` ChromBPNet, `83.2%`→`57%` NTv3 ATAC, `50.9%`→`37%` SLAM; `slc35a3a`: `90.4%`→`70%`, `87.4%`→`68%`, `51.3%`→`39%`) fall cleanly right of the diagonal and inside the green `Q25–Q75` zone, both genes show reproducible multi-modal parity with *bona fide* minor-wave ZGA regulators.

---

### Generated Tables & Genomic Tracks (With Key Results)
| Output File (`results/notch2_slc35a3a/` unless noted) | Block | Schema & Key Result / Interpretation |
| :--- | :---: | :--- |
| `qc_audit_report.md`, `qc_audit_data.json` *(in `work_.../`)* | Audit | Library/peak-caller/holdout audit proving `0.00%` MT reads, Medaka low TSSe (`1.35`/`2.23`) vs. zebrafish (`6.85`), and strict `chr8` exclusion. |
| `atac_sample_qc.tsv`, `atac_species_qc.tsv`, `chrombpnet_qc.tsv` | 2 | Alignment depths, FRiP, TSSe, `49,377` ZF high-conf peaks, Stage 1 count $r$ (`0.7044` full / `0.7001` `nobias`), JSD, and `0.001` Tn5 footprint response. |
| `nascent_rnaseq_qc_summary.tsv`, `slamseq/zf_slamseq_gene_classes.tsv` | 1b | Zebrafish 4sU kinetic classifications across nuclear genes (`1` `preMZT_strictly_zygotic`, `52` `preMZT_maternal_zygotic`, `521` `major_wave_only`). |
| `zero_shoot_validation_metrics.json`, `zero_shoot_window_control_comparison.tsv` | 4 | Full `4 windows × 3 scores × 2 controls` zero-shot AUROC/AUPRC matrix confirming `y_abs`/`late` dominance vs. silent (`0.69–0.86`) and `y_resid_gated` flip (`0.61`) vs. maternal. |
| `candidate_elements.tsv`/`.bed`, `notch2_slc35a3a_candidate_enhancers.tsv`/`.bed` | 5 | Coordinates & summit scores for `14` open AC elements (`C1=+2.65`, `C2=+2.56`, `C3=+2.29` gaining `+0.30..+0.67` above ZF ortholog max `1.99`). |
| `notch2_slc35a3a_pred_y_abs.bedgraph`, `_pred_late.bedgraph`, `_pred_y_rel_gated.bedgraph` | 5 | IGV-loadable 250 bp step ChromBPNet tracks spanning `chr23:10,908,558–11,057,058`. |
| `locus_cis_regulatory_motifs.tsv`, `evolutionary_swap_summary.tsv`, `candidate_1_causal_motif.meme` | 6 | All `56` attributed seqlets (`24` aligned / `32` lineage-specific), CpG O/E, JASPAR & maternal TOMTOM matches (`NFYA★` R3, `ZBTB14` R2), and swap deltas. |
| `minor_wave_genomewide_top_motifs.tsv` | 6 | Fisher enrichment across minor-wave vs. GC-matched silent promoters led by `Zbtb2` (`p=0.0075`), `ZKSCAN3` (`p=0.063`), and `CREM` (`p=0.088`). |
| `models/ntv3/ntv3_train_config.json`, `ntv3_baselines_benchmark.tsv` | 7 | Snapshot (`best_val_loss=0.0942`, `species_id=8`, `epochs=3`) + raw (`0.0704`), frozen-head (`0.6741`), LoRA (`0.8457`), and ChromBPNet (`0.6573`) benchmarks. |
| `notch2_slc35a3a_cross_architecture_summary.tsv` | 8 | Unified table merging Block 7 held-out metrics with Block 8 matched-cohort ATAC (`0.755`) & SLAM (`0.540`/`0.565`) AUROCs. |
| `ntv3_locus_scan.tsv`, `ntv3_zf_locus_scan.tsv`, `*_ntv3_*.bedgraph`, `ntv3_zf_pred_*.bedgraph` | 8a | Base-resolution (1 bp) predicted ATAC and strand-resolved SLAM tracks across AC `chr23` and ZF `chr8`. |
| `ntv3_distal_coupling_screen.tsv`, `ntv3_knockout_saliency_summary.tsv`, `ntv3_saliency_tracks.npz` | 8a | Perturbation matrix confirming `-8.5%` to `-29.0%` CpG-ablation sensitivity across R1/2/3/9/11 plus locus & TSS saliency arrays. |
| `notch2_slc35a3a_target_gene_ranking_stats.tsv` | 8c | Leave-one-out ranking (`26` tests): `NOTCH2` (`72.8%ile` ChromBPNet, `+0.422` over ZF) and `slc35a3a` (`69.6%ile` ChromBPNet, `+0.715` over ZF) fall *Within IQR Q25–Q75*. |
| *(Ignored private RNA-seq tables)* | 1,4,8 | `notch2_slc35a3a_gene_sets.tsv`, `target_genes_scored_metrics.tsv`, `ntv3_target_genes_scored.tsv` (ignored from git because they contain unpublished RNA-seq TPM/CPM values). |

---

## ⚙️ Installation, Usage & Customization Guide

### 1. Environment Setup & Public Tracking Policy
```bash
chmod +x setup_env.sh
./setup_env.sh
conda activate nucleotide
```
Add your Hugging Face access token (required to download gated `InstaDeepAI/NTv3_tracks`) in `.env` at the repository root:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```
* **`.gitignore` & `.gitattributes` Tracking Scope:**
  * **Published in git / LFS:** All scripts & notebook, study `config.json` & QC audits, ChromBPNet checkpoints (`25 MB` `grafted_chrombpnet.pt` + Stage 1 weights), fine-tuned `ntv3_best_checkpoint.pt` (**1.4 GB via `git-lfs`**), summary benchmark TSVs, locus BED/bedGraph/motif/swap tables, and publication figures (`fig1..fig8`).
  * **Excluded from git:** Raw FASTQs/BAMs, STAR indices, reference FASTA caches (~130 GB), plus unpublished cichlid `RNA-seq/`, `TF_gene_names.txt`, `work_.../rnaseq/`, `*_gene_sets.tsv`, and gene-level tables embedding raw replicate RNA-seq columns (`target_genes_scored_metrics.tsv`, `ntv3_target_genes_scored.tsv`).

### 2. Inferring Custom Target Genes (Reusing Pretrained Zebrafish Checkpoints)
Because both `ChromBPNet` and `NTv3` were trained strictly on zebrafish embryos and generalize zero-shot, **do NOT retrain Blocks 2, 3, or 7** to test new genes or species:

1. **Modify Block 0 in [`premzt_cis_grammar.ipynb`](premzt_cis_grammar.ipynb) (or `work_.../config.json`):**
   * `STUDY_NAME`: Set to `"astatotilapia_premzt"` (reuses trained checkpoints) or a new study name.
   * `TARGET_SPECIES_NAME`, `TARGET_GENOME_FA`, `TARGET_GTF`: Your target genome and annotations.
   * `TARGET_LOCUS_NAME`, `TARGET_GENES`: Locus identifier and list of gene symbols.
   * `LOCUS_FLANK_BP`: Upstream/downstream scan flanking span (default `75000`).
   * `CANONICAL_ZGA_MARKERS`: Reference early ZGA marker symbols (e.g., `["vox", "klf17", "mxtx2"]`).
   * `RNASEQ_TPM_RDS` or `RNASEQ_EARLY_BAMS`/`RNASEQ_PREMZT_BAMS`: Your expression source.
2. **Execute ONLY these blocks:** `Block 0` → `Block 1` & `Block 1b` → **`Block 4`** *(if scoring a new species genome)* → **`Block 5`** → **`Block 6`** → **`Block 8a` (`--subtask all` or `locus --force-rescore`)** → **`Block 8b`** → **`Block 8c`**.
3. **Or call the Python Inference API directly (`scripts/premzt_inference.py`, Block 8d):**
   ```python
   from scripts.premzt_inference import infer_premzt

   df_out = infer_premzt(
       species="Astatotilapia_calliptera",
       gene=["notch2", "slc35a3a"],
       model="all",            # "chrombpnet", "ntv3atac", "ntv3slam", or "all"
       locus_range="auto",     # or flexible DSL e.g. "TSS+2000-500,intron1"
       work_dir="./work_astatotilapia_premzt"
   )
   ```
   > **Optional Manual Refresh of Cached `NTv3` Cohort Windows (`Figure 7`):** To re-score the `1,427` cached NTv3 genes with separated `promoter_1kb` vs. `promoter_intron1_capped` columns, run:
   > ```bash
   > python -u scripts/ntv3_infer.py --work-dir ./work_astatotilapia_premzt --config ./work_astatotilapia_premzt/config.json --subtask score --force-rescore
   > ```

### 3. Training Custom Models on New Reference Embryos (Full Pipeline)
To retrain from scratch on new blastula ATAC-seq / SLAM-seq datasets:
1. Update `ZEBRAFISH_ATAC_ACCESSIONS`, `MEDAKA_ATAC_ACCESSIONS`, `SLAM_ACCESSIONS`, `COMPUTE_PROFILE` (`"laptop_16gb"` vs. `"hpc_1mb"`), `CV_FOLDS`, `CHROMBPNET_EPOCHS`, `ENABLE_MEDAKA`, `STAGE2_MIN_FE`, and `NTV3_RUN_BASELINES` in **Block 0**.
2. Execute **`Block 0`** → **`Block 2`** → **`Block 3`** → **`Block 4`** → **`Block 7`** *(requires Block 1b `prep_slamseq.py` BigWigs)* → **`Blocks 5, 6, 8a–8c`**.

---

## 📚 Data & Foundation Model Citations

1. **ChromBPNet:** Pampari, A., Shcherbina, A., Kvon, E. Z., *et al.* (Kundaje Lab). *ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and regulatory variants* (2024/2025, `kundajelab/chrombpnet`).
2. **Nucleotide Transformer v3 (`NTv3`):** Boshar, S., Evans, B., Tang, Z., Picard, A., *et al.* (InstaDeep). *A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction*, *bioRxiv* (2025), doi:[10.64898/2025.12.22.695963](https://doi.org/10.64898/2025.12.22.695963) (`InstaDeepAI/NTv3_tracks`).
3. **Zebrafish Blastula ATAC-seq:** Pálfy, M., Schulze, G., Valen, E. & Vastenhouw, N. L. Chromatin accessibility established by Pou5f3, Sox19b and Nanog primes genes for activity during zebrafish genome activation. *PLoS Genetics* 16(1), e1008546 (2020). GEO Accession: **`GSE130944`** (`GSM3756599`, `GSM3756600`, `GSM3756603`–`GSM3756605`; gDNA control `SRR9032640`).
4. **Medaka Blastula ATAC-seq:** Nakamura, R. *et al.* *3D genome structure in medaka embryos* (2021). DDBJ BioProject: **`PRJDB7492`** (`DRX143405`–`DRX143407`, `DRX143412`, `DRX143413`).
5. **Zebrafish Embryonic SLAM-seq:** Bhat, P. *et al.* SLAM-seq resolves the kinetics of maternal and zygotic gene expression during early zebrafish embryogenesis. *Cell Reports* (2023). BioProject: **`PRJNA768450`** (`GSM5610039`, `GSM5610040`, `GSM5610043`–`GSM5610048`, `GSM5610051`, `GSM5610052`, `GSM5610075`–`GSM5610077`).
6. **JASPAR Motif Database:** Rauluseviciute, I., Riudavets-Puig, R., Blanc-Mathieu, R., *et al.* JASPAR 2024: 20th anniversary of the open-access database of transcription factor binding profiles. *Nucleic Acids Research* 52(D1), D174–D182 (2024) (`JASPAR2024_CORE_vertebrates_non-redundant`).

---

## 📜 Licensing & Redistribution Guide

Because this repository distributes both original code/ChromBPNet weights and fine-tuned `NTv3` weights and outputs, it applies **Dual Licensing**:

1. **MIT License — Repository Code, Notebook & ChromBPNet Artifacts:**
   All original Python/R scripts (`scripts/`), [`premzt_cis_grammar.ipynb`](premzt_cis_grammar.ipynb), ChromBPNet models (`kundajelab/chrombpnet` is upstream MIT-licensed; our trained zebrafish Stage 1 & grafted ChromBPNet checkpoints), and ChromBPNet-only outputs (`fig1*`–`fig4*`, `candidate_elements.tsv`, `locus_cis_regulatory_motifs.tsv`) are licensed under the **MIT License**.
2. **InstaDeep Open Model Licence (Non-Commercial Purposes Only) — Fine-Tuned `NTv3` Weights & Outputs:**
   Upstream `InstaDeepAI/NTv3_tracks` is distributed under the **InstaDeep Open Model Licence**. Because `work_astatotilapia_premzt/models/ntv3/ntv3_best_checkpoint.pt` (`1.4 GB` via Git LFS) retains the full adapted `NTv3` backbone weights alongside our LoRA adapters and 10-track head:
   * Derivative/fine-tuned `NTv3` checkpoints **must remain under the InstaDeep Open Model Licence (Non-Commercial Use Only)** and cannot be re-licensed under MIT.
   * **Outputs** generated by `NTv3` (`ntv3_locus_scan.tsv`, `ntv3_distal_coupling_screen.tsv`, `fig5`–`fig8` NTv3 panels, and NTv3 rows in `notch2_slc35a3a_target_gene_ranking_stats.tsv`) may be shared and published freely for **non-commercial research**, but may not be resold, used commercially, or used to train commercial third-party models without a separate license from InstaDeep.
   * *(Optional Lightweight Alternative)* To keep a public repository 100% MIT-licensed without redistributing the `1.4 GB` `NTv3` backbone weights, exclude `ntv3_best_checkpoint.pt` in `.gitignore` and release only the small LoRA adapter + `nn.Linear(1024, 10)` head weights alongside [scripts/train_ntv3.py](scripts/train_ntv3.py) so users fetch base `NTv3` weights directly from Hugging Face.
