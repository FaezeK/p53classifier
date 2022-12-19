#import pandas as pd
from email import message

rule all:
    input:
        'results/RF_better_clssfctn.jpg'
    shell: 'rm -rf tmp_data'


rule preprocess_data:
    input:
        pog_tpm = 'data/POG_TPM_matrix.txt',
        pog_snv = 'data/POG_small_mutations.txt',
        pog_meta = 'data/pog_cohort_details.txt',
        tcga_tpm = 'data/tcga_rsem_gene_tpm.txt',
        tcga_snv = 'data/mc3.v0.2.8.PUBLIC.txt',
        tss = 'data/tissueSourceSite.tsv',
        cmmn_genes = 'data/common_genes.txt'
    output:
        'results/merged_pca.jpg',
        'results/POG_pca.jpg',
        'results/TCGA_pca.jpg',
        tcga_expr_prcssd = 'tmp_data/TCGA_expr_prcssd.txt',
        tcga_snv_prcssd = 'tmp_data/TCGA_snv_prcssd.txt',
        tcga_types = 'tmp_data/TCGA_types.txt',
        pog_expr_prcssd = 'tmp_data/POG_expr_prcssd.txt',
        pog_snv_prcssd = 'tmp_data/POG_snv_prcssd.txt',
        pog_meta_prcssd = 'tmp_data/POG_meta_prcssd.txt'
    message: 'Preprocessing datasets!'
    script: 'preprocess_input.py'


rule find_best_hyperparam:
    input:
        pog_expr_prcssd = 'tmp_data/POG_expr_prcssd.txt',
        tcga_expr_prcssd = 'tmp_data/TCGA_expr_prcssd.txt',
        pog_snv_prcssd = 'tmp_data/POG_snv_prcssd.txt',
        tcga_snv_prcssd = 'tmp_data/TCGA_snv_prcssd.txt',
        pog_meta_prcssd = 'tmp_data/POG_meta_prcssd.txt',
        tss = 'data/tissueSourceSite.tsv'
    output:
        pog_hyper_param_results = 'results/pog_hyper_param_results.txt',
        pog_set_best_hyperparam = 'results/pog_set_best_hyperparam.txt',
        tcga_hyper_param_results = 'results/tcga_hyper_param_results.txt',
        tcga_set_best_hyperparam = 'results/tcga_set_best_hyperparam.txt',
        both_hyper_param_results = 'results/both_hyper_param_results.txt',
        both_sets_best_hyperparam = 'results/both_sets_best_hyperparam.txt'
    message: 'Finding the most suitable hyperparameters!'
    script: 'best_hyperparam.py'


rule find_threshold:
    input:
        pog_expr_prcssd = 'tmp_data/POG_expr_prcssd.txt',
        tcga_expr_prcssd = 'tmp_data/TCGA_expr_prcssd.txt',
        pog_snv_prcssd = 'tmp_data/POG_snv_prcssd.txt',
        tcga_snv_prcssd = 'tmp_data/TCGA_snv_prcssd.txt',
        pog_meta_prcssd = 'tmp_data/POG_meta_prcssd.txt',
        tcga_types = 'tmp_data/TCGA_types.txt',
        both_sets_best_hyperparam = 'results/both_sets_best_hyperparam.txt'
    output:
        TrueVsRandomScoresByRank = 'results/TrueVsRandomScoresByRank.png',
        TrueVsRandomScoresByRankZoomedIn = 'results/TrueVsRandomScoresByRankZoomedIn.png',
        TrueVsRandomScoresByRankZoomedInMore = 'results/TrueVsRandomScoresByRankZoomedInMore.png',
        num_important_genes = 'tmp_data/num_important_genes.txt'
    message: 'Finding a threshold to find the most important genes in classification!'
    script: 'threshold.py'


rule train_rf_and_analyze_results:
    input:
        pog_expr_prcssd = 'tmp_data/POG_expr_prcssd.txt',
        tcga_expr_prcssd = 'tmp_data/TCGA_expr_prcssd.txt',
        pog_snv_prcssd = 'tmp_data/POG_snv_prcssd.txt',
        tcga_snv_prcssd = 'tmp_data/TCGA_snv_prcssd.txt',
        pog_meta_prcssd = 'tmp_data/POG_meta_prcssd.txt',
        tcga_types = 'tmp_data/TCGA_types.txt',
        num_important_genes = 'tmp_data/num_important_genes.txt',
        pog_set_best_hyperparam = 'results/pog_set_best_hyperparam.txt',
        tcga_set_best_hyperparam = 'results/tcga_set_best_hyperparam.txt',
        both_sets_best_hyperparam = 'results/both_sets_best_hyperparam.txt'
    output:
        rf_pred_on_merged = 'results/rf_pred_on_merged.txt',
        auroc_auprc = 'results/auroc_auprc.jpg',
        TCGA_cancer_types_metrics = 'results/TCGA_cancer_types_metrics.txt',
        accr_vs_oob_plot = 'results/accr_vs_oob_plot.jpg',
        pred_prob_all4 = 'results/pred_prob_all4.jpg',
        outliers = 'results/outliers.txt',
        top_genes_w_mut_rate_reg_stat = 'results/top_genes_w_mut_rate_reg_stat.txt',
        top_10_expr = 'results/top_10_expr.jpg',
        not_impact_mut_pred = 'results/not_impact_mut_pred.txt'
    message: 'Training the random forest model with TCGA, POG, and merged sets and analyzing the results.'
    script: 'p53.py'


rule treatment_results:
    input:
        rf_pred_on_merged = 'results/rf_pred_on_merged.txt',
        pog_drugs = 'data/pog_drugs.tsv'
    output:
        RF_better_clssfctn = 'results/RF_better_clssfctn.jpg',
        TruLab_better_clssfctn = 'results/TruLab_better_clssfctn.jpg'
    message: 'Make boxplots of number of days on treatment for patients receiving different therapies.'
    script: 'treatment.py'
