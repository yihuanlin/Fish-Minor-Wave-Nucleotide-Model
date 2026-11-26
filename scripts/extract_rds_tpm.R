#!/usr/bin/env Rscript
# scripts/extract_rds_tpm.R
# ==============================================================================
# Helper to parse two-species pre-MZT RNA-seq TPM RDS file (e.g. TPM_2C_1K_both_species.rds)
# - Performs replicate-aware differential expression & regularized log2FC
# - Classifies gene sets for both cichlid species
# - Annotates GTF coordinates (chrom, strand, start, end, tss, tes)
# - Computes conserved two-cichlid minor-wave gene set (is_both_cichlids_minor_wave)
# - Exports standardized tables for primary target species and second species
# ==============================================================================

suppressPackageStartupMessages({
    library(methods)
})

parse_args <- function() {
    args <- commandArgs(trailingOnly = TRUE)
    params <- list(
        rds = "./TPM_2C_1K_both_species.rds",
        gtf = NULL,
        outdir = "./work_astatotilapia_premzt/rnaseq",
        resdir = NULL,
        locus_name = "notch2_slc35a3a",
        primary_prefix = "AC",
        second_prefix = "TM"
    )
    
    i <- 1
    while (i <= length(args)) {
        arg <- args[i]
        if (arg == "--rds" && i < length(args)) {
            params$rds <- args[i + 1]; i <- i + 2
        } else if (arg == "--gtf" && i < length(args)) {
            params$gtf <- args[i + 1]; i <- i + 2
        } else if (arg == "--outdir" && i < length(args)) {
            params$outdir <- args[i + 1]; i <- i + 2
        } else if (arg == "--resdir" && i < length(args)) {
            params$resdir <- args[i + 1]; i <- i + 2
        } else if (arg == "--locus-name" && i < length(args)) {
            params$locus_name <- args[i + 1]; i <- i + 2
        } else if (arg == "--primary-prefix" && i < length(args)) {
            params$primary_prefix <- args[i + 1]; i <- i + 2
        } else if (arg == "--second-prefix" && i < length(args)) {
            params$second_prefix <- args[i + 1]; i <- i + 2
        } else {
            i <- i + 1
        }
    }
    return(params)
}

process_species_tpm <- function(df, prefix) {
    early_cols <- grep(paste0("^", prefix, "_(2C|early).*rep"), colnames(df), value = TRUE)
    premzt_cols <- grep(paste0("^", prefix, "_(1K|premzt).*rep"), colnames(df), value = TRUE)
    
    if (length(early_cols) == 0 || length(premzt_cols) == 0) {
        stop(sprintf("Could not find early or premzt replicate columns for prefix '%s' in RDS!", prefix))
    }
    
    mat_early <- as.matrix(df[, early_cols, drop = FALSE])
    mat_premzt <- as.matrix(df[, premzt_cols, drop = FALSE])
    
    mean_early <- rowMeans(mat_early, na.rm = TRUE)
    mean_premzt <- rowMeans(mat_premzt, na.rm = TRUE)
    
    # Regularized log2 fold change
    log2fc <- log2((mean_premzt + 0.1) / (mean_early + 0.1))
    
    # Replicate-aware statistics: Welch two-sample t-test on log2(TPM + 1)
    n_genes <- nrow(df)
    pvals <- rep(1.0, n_genes)
    
    n_e <- length(early_cols)
    n_p <- length(premzt_cols)
    
    if (n_e >= 2 && n_p >= 2) {
        log_e <- log2(mat_early + 1.0)
        log_p <- log2(mat_premzt + 1.0)
        
        m_e <- rowMeans(log_e)
        m_p <- rowMeans(log_p)
        v_e <- rowSums((log_e - m_e)^2) / (n_e - 1)
        v_p <- rowSums((log_p - m_p)^2) / (n_p - 1)
        
        se <- sqrt(v_e / n_e + v_p / n_p)
        t_stat <- (m_p - m_e) / (se + 1e-9)
        
        # Welch-Satterthwaite degrees of freedom
        df_num <- (v_e / n_e + v_p / n_p)^2
        df_den <- (v_e / n_e)^2 / (n_e - 1) + (v_p / n_p)^2 / (n_p - 1)
        deg_f <- ifelse(df_den > 0, df_num / df_den, 1)
        deg_f <- pmax(deg_f, 1)
        
        # Two-sided p-value
        pv <- 2 * pt(-abs(t_stat), df = deg_f)
        # For completely zero genes, set p-value to 1.0
        zero_mask <- (mean_early == 0) & (mean_premzt == 0)
        pv[zero_mask] <- 1.0
        pvals <- pv
    }
    
    padj <- p.adjust(pvals, method = "BH")
    padj[is.na(padj)] <- 1.0
    
    # Standard classification thresholds
    is_minor <- (mean_early < 0.5) & (mean_premzt >= 2.0) & (log2fc >= 2.0)
    is_highconf <- (mean_early < 0.2) & (mean_premzt >= 3.0) & (log2fc >= 2.5) & (padj < 0.05)
    is_upreg <- (log2fc >= 1.5) & (mean_premzt >= 2.0) & (padj < 0.05)
    is_silent <- (mean_early < 0.2) & (mean_premzt < 0.2)
    is_control_nonact <- (mean_early >= 2.0) & (abs(log2fc) <= 0.5)
    
    category <- rep("unclassified", n_genes)
    category[is_silent] <- "control_silent"
    category[is_control_nonact] <- "control_non_activated"
    category[is_upreg] <- "all_upregulated"
    category[is_minor] <- "strictly_zygotic_minor_wave"
    category[is_highconf] <- "strictly_zygotic_highconf"
    
    res <- data.frame(
        tpm_early = mean_early,
        tpm_premzt = mean_premzt,
        log2fc = log2fc,
        pval = pvals,
        padj = padj,
        is_strictly_zygotic_minor_wave = is_minor,
        is_strictly_zygotic_highconf = is_highconf,
        is_all_upregulated = is_upreg,
        is_control_silent = is_silent,
        is_control_non_activated = is_control_nonact,
        category = category,
        stringsAsFactors = FALSE
    )
    return(res)
}

parse_gtf_coords <- function(gtf_path) {
    if (is.null(gtf_path) || !file.exists(gtf_path)) {
        return(NULL)
    }
    cat(sprintf("[*] Fast-parsing coordinates from GTF: %s\n", gtf_path))
    cmd <- if (grepl("\\.gz$", gtf_path)) {
        sprintf("gzip -dc '%s' | awk -F'\\t' '$3 == \"gene\" {print $1 \"\\t\" $4 \"\\t\" $5 \"\\t\" $7 \"\\t\" $9}'", gtf_path)
    } else {
        sprintf("awk -F'\\t' '$3 == \"gene\" {print $1 \"\\t\" $4 \"\\t\" $5 \"\\t\" $7 \"\\t\" $9}' '%s'", gtf_path)
    }
    con <- pipe(cmd, "r")
    lines <- readLines(con)
    close(con)
    
    if (length(lines) == 0) return(NULL)
    
    parts <- strsplit(lines, "\t")
    g_chrom <- sapply(parts, `[`, 1)
    g_start <- as.integer(sapply(parts, `[`, 2))
    g_end <- as.integer(sapply(parts, `[`, 3))
    g_strand <- sapply(parts, `[`, 4)
    info <- sapply(parts, `[`, 5)
    
    g_id <- sub('.*gene_id "([^"]+)".*', '\\1', info)
    has_name <- grepl('gene_name "([^"]+)"', info)
    g_name <- ifelse(has_name, sub('.*gene_name "([^"]+)".*', '\\1', info), g_id)
    
    coords_df <- data.frame(
        gene_id = g_id,
        gtf_gene_name = g_name,
        chrom = g_chrom,
        strand = g_strand,
        start = g_start,
        end = g_end,
        stringsAsFactors = FALSE
    )
    coords_df <- coords_df[!duplicated(coords_df$gene_id), ]
    coords_df$tss <- ifelse(coords_df$strand == "+", coords_df$start, coords_df$end)
    coords_df$tes <- ifelse(coords_df$strand == "+", coords_df$end, coords_df$start)
    return(coords_df)
}

main <- function() {
    params <- parse_args()
    cat("======================================================================\n")
    cat("[*] Extracting Multi-Species TPM from RDS\n")
    cat(sprintf("    RDS File:        %s\n", params$rds))
    cat(sprintf("    Primary Prefix:  %s\n", params$primary_prefix))
    cat(sprintf("    Second Prefix:   %s\n", params$second_prefix))
    cat(sprintf("    Output Dir:      %s\n", params$outdir))
    cat("======================================================================\n")
    
    dir.create(params$outdir, recursive = TRUE, showWarnings = FALSE)
    
    df <- readRDS(params$rds)
    cat(sprintf("[+] Loaded RDS with %d rows and %d columns.\n", nrow(df), ncol(df)))
    
    # Process primary species
    res_primary <- process_species_tpm(df, params$primary_prefix)
    # Process second species
    res_second <- process_species_tpm(df, params$second_prefix)
    
    # Two-cichlid minor-wave conservation
    is_both_minor <- res_primary$is_strictly_zygotic_minor_wave & res_second$is_strictly_zygotic_minor_wave
    is_both_highconf <- res_primary$is_strictly_zygotic_highconf & res_second$is_strictly_zygotic_highconf
    is_primary_only <- res_primary$is_strictly_zygotic_minor_wave & !res_second$is_strictly_zygotic_minor_wave
    is_second_only <- res_second$is_strictly_zygotic_minor_wave & !res_primary$is_strictly_zygotic_minor_wave
    
    cat(sprintf("[+] %s minor wave genes: %d (highconf: %d)\n", 
                params$primary_prefix, sum(res_primary$is_strictly_zygotic_minor_wave), sum(res_primary$is_strictly_zygotic_highconf)))
    cat(sprintf("[+] %s minor wave genes: %d (highconf: %d)\n", 
                params$second_prefix, sum(res_second$is_strictly_zygotic_minor_wave), sum(res_second$is_strictly_zygotic_highconf)))
    cat(sprintf("[+] Conserved two-cichlid minor wave genes: %d (highconf: %d)\n", 
                sum(is_both_minor), sum(is_both_highconf)))
    cat(sprintf("[+] %s-only minor wave genes: %d | %s-only: %d\n", 
                params$primary_prefix, sum(is_primary_only), params$second_prefix, sum(is_second_only)))
    
    # Map GTF coordinates
    coords <- parse_gtf_coords(params$gtf)
    
    # Build primary table
    primary_df <- data.frame(
        Geneid = df$gene_id,
        gene_id = df$gene_id,
        gene_name = df$gene_name,
        stringsAsFactors = FALSE
    )
    
    if (!is.null(coords)) {
        m <- match(primary_df$gene_id, coords$gene_id)
        # Fallback to match by gene_name if gene_id did not match
        na_m <- which(is.na(m))
        if (length(na_m) > 0) {
            m_by_name <- match(tolower(primary_df$gene_name[na_m]), tolower(coords$gtf_gene_name))
            found <- !is.na(m_by_name)
            m[na_m[found]] <- m_by_name[found]
        }
        
        primary_df$chrom <- ifelse(is.na(m), ifelse(grepl("^(ATP|COX|CYTB|ND[1-6])", primary_df$gene_name), "MT", "unknown"), coords$chrom[m])
        primary_df$strand <- ifelse(is.na(m), "+", coords$strand[m])
        primary_df$start <- ifelse(is.na(m), 0L, coords$start[m])
        primary_df$end <- ifelse(is.na(m), 0L, coords$end[m])
        primary_df$tss <- ifelse(is.na(m), 0L, coords$tss[m])
        primary_df$tes <- ifelse(is.na(m), 0L, coords$tes[m])
        primary_df$Chr <- primary_df$chrom
        primary_df$Start <- primary_df$start
        primary_df$End <- primary_df$end
        primary_df$Strand <- primary_df$strand
        primary_df$Length <- ifelse(is.na(m), 1000L, pmax(coords$end[m] - coords$start[m] + 1L, 100L))
    } else {
        primary_df$chrom <- "unknown"
        primary_df$strand <- "+"
        primary_df$start <- 0
        primary_df$end <- 0
        primary_df$tss <- 0
        primary_df$tes <- 0
        primary_df$Chr <- "unknown"
        primary_df$Start <- 0
        primary_df$End <- 0
        primary_df$Strand <- "+"
        primary_df$Length <- 1000
    }
    
    # Primary species expression columns (alias cpm_* to tpm_* for full backward compatibility)
    primary_df$cpm_early <- res_primary$tpm_early
    primary_df$cpm_premzt <- res_primary$tpm_premzt
    primary_df$tpm_early <- res_primary$tpm_early
    primary_df$tpm_premzt <- res_primary$tpm_premzt
    primary_df$log2fc <- res_primary$log2fc
    primary_df$pval <- res_primary$pval
    primary_df$padj <- res_primary$padj
    primary_df$is_strictly_zygotic_minor_wave <- res_primary$is_strictly_zygotic_minor_wave
    primary_df$is_strictly_zygotic_highconf <- res_primary$is_strictly_zygotic_highconf
    primary_df$is_all_upregulated <- res_primary$is_all_upregulated
    primary_df$is_control_silent <- res_primary$is_control_silent
    primary_df$is_control_non_activated <- res_primary$is_control_non_activated
    primary_df$category <- res_primary$category
    primary_df$is_slam_confirmed_minor_wave <- FALSE  # Updated in Block 1b
    
    # Conserved & species-specific flags
    primary_df$is_both_cichlids_minor_wave <- is_both_minor
    primary_df$is_both_cichlids_highconf <- is_both_highconf
    primary_df[[paste0("is_", tolower(params$primary_prefix), "_only_minor_wave")]] <- is_primary_only
    primary_df[[paste0("is_", tolower(params$second_prefix), "_only_minor_wave")]] <- is_second_only
    
    # Attach secondary species summary columns
    sec_prefix_lower <- tolower(params$second_prefix)
    primary_df[[paste0(sec_prefix_lower, "_tpm_early")]] <- res_second$tpm_early
    primary_df[[paste0(sec_prefix_lower, "_tpm_premzt")]] <- res_second$tpm_premzt
    primary_df[[paste0(sec_prefix_lower, "_log2fc")]] <- res_second$log2fc
    primary_df[[paste0(sec_prefix_lower, "_pval")]] <- res_second$pval
    primary_df[[paste0(sec_prefix_lower, "_padj")]] <- res_second$padj
    primary_df[[paste0(sec_prefix_lower, "_is_strictly_zygotic_minor_wave")]] <- res_second$is_strictly_zygotic_minor_wave
    primary_df[[paste0(sec_prefix_lower, "_is_strictly_zygotic_highconf")]] <- res_second$is_strictly_zygotic_highconf
    primary_df[[paste0(sec_prefix_lower, "_is_all_upregulated")]] <- res_second$is_all_upregulated
    primary_df[[paste0(sec_prefix_lower, "_is_control_silent")]] <- res_second$is_control_silent
    primary_df[[paste0(sec_prefix_lower, "_is_control_non_activated")]] <- res_second$is_control_non_activated
    primary_df[[paste0(sec_prefix_lower, "_category")]] <- res_second$category
    
    # Attach raw replicate columns from original df
    rep_cols <- grep("rep[0-9]+", colnames(df), value = TRUE)
    for (rc in rep_cols) {
        primary_df[[rc]] <- df[[rc]]
    }
    
    # Build secondary species standalone table
    second_df <- data.frame(
        Geneid = df$gene_id,
        gene_id = df$gene_id,
        gene_name = df$gene_name,
        tpm_early = res_second$tpm_early,
        tpm_premzt = res_second$tpm_premzt,
        log2fc = res_second$log2fc,
        pval = res_second$pval,
        padj = res_second$padj,
        is_strictly_zygotic_minor_wave = res_second$is_strictly_zygotic_minor_wave,
        is_strictly_zygotic_highconf = res_second$is_strictly_zygotic_highconf,
        is_all_upregulated = res_second$is_all_upregulated,
        is_control_silent = res_second$is_control_silent,
        is_control_non_activated = res_second$is_control_non_activated,
        category = res_second$category,
        is_both_cichlids_minor_wave = is_both_minor,
        stringsAsFactors = FALSE
    )
    
    # Write files
    target_tsv <- file.path(params$outdir, "target_species_gene_sets.tsv")
    write.table(primary_df, target_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
    cat(sprintf("[+] Written primary target table to: %s (%d rows)\n", target_tsv, nrow(primary_df)))
    
    if (!is.null(params$resdir)) {
        dir.create(params$resdir, recursive = TRUE, showWarnings = FALSE)
        res_locus_tsv <- file.path(params$resdir, sprintf("%s_gene_sets.tsv", params$locus_name))
        write.table(primary_df, res_locus_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
        cat(sprintf("[+] Written locus gene sets table to: %s\n", res_locus_tsv))
    }
    
    second_tsv <- file.path(params$outdir, sprintf("%s_gene_sets.tsv", tolower(params$second_prefix)))
    write.table(second_df, second_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
    cat(sprintf("[+] Written second species table to: %s\n", second_tsv))
}

if (!interactive()) {
    main()
}
