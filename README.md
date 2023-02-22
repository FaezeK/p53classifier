# Classifying Tumour Samples based on TP53 Mutational Status

## Overview

This repo contains the scripts used to classify the TCGA (The Cancer Genome Atlas) primary and the POG (Personalized Oncogenomics) metastatic tumour samples based on the presence or absence of mutations in the TP53 gene using a random forest model. The input to the classifier is an expression data set containing TPM (Transcript per Million) values. The samples were labeled as 'p53 wt' (p53 wild type) or 'p53 mut' (p53 mutant) based on the MAF file containing SNVs (Single Nucleotide Variation) and small INDELs (Insertions and deletions) downloaded from [here](https://gdc.cancer.gov/about-data/publications/mc3-2017) (more info in the "Input Data" section). The performance of the random forest was evaluated using performance metrics including precision, recall, f1-score, AUROC and AUPRC.

After training and evaluating the model, the top features (genes) were extracted and their expression modifications in presence of p53 mutations were assessed. The fully trained model was also used to classify the tumours containing the mutations that are not expected to change the p53 protein (non-impactful mutations).

## Input Data

TCGA expression data (tcga_RSEM_gene_tpm) was downloaded on June 21, 2021 from: https://xenabrowser.net/ \
TCGA mutation data (mc3.v0.2.8.PUBLIC.maf.gz) was downloaded on September 30, 2020 from: https://gdc.cancer.gov/about-data/publications/mc3-2017 \
POG datasets are not publicly available.

## Workflow

The workflow is visualized below: \
<img src="dag.pdf" width="200" height="400">
