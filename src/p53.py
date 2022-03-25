import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

# find_tcga_p53_mut function finds the samples with and w/out p53 mutation in tcga
def find_tcga_p53_mut(tcga_expr, tcga_snv):

    tcga_snv['donor_id'] = tcga_snv.sample_id.str.slice(start=0, stop=15)
    tcga_expr['donor_id'] = tcga_expr.sample_id.str.slice(start=0, stop=15)

    # filter tcga expr and mut data to include samples that exist in both
    tcga_snv = tcga_snv[tcga_snv.donor_id.isin(tcga_expr.donor_id)]
    tcga_expr = tcga_expr[tcga_expr.donor_id.isin(tcga_snv.donor_id)]

    # find samples with mutated and wt p53 genes in TCGA
    tcga_snv_fltr = tcga_snv[tcga_snv.gene_id == "TP53"]

    tcga_impactful_p53_mut = tcga_snv_fltr[tcga_snv_fltr.effect.isin(['missense_variant', 'frameshift_variant', 'splice_acceptor_variant', 'stop_gained', 'inframe_deletion', 'splice_donor_variant', 'inframe_insertion', 'protein_altering_variant', 'start_lost'])]
    tcga_not_impactful_p53_mut = tcga_snv_fltr[tcga_snv_fltr.effect.isin(['synonymous_variant', 'splice_region_variant', '3_prime_UTR_variant', 'intron_variant', '5_prime_UTR_variant'])]

    tcga_impactful_p53_mut_samples = tcga_impactful_p53_mut.donor_id.unique()
    tcga_not_impactful_p53_mut_samples = tcga_not_impactful_p53_mut.donor_id.unique()
    tcga_not_impactful_p53_mut_samples = tcga_not_impactful_p53_mut_samples[np.isin(tcga_not_impactful_p53_mut_samples, tcga_impactful_p53_mut_samples)==False]

    all_tcga_snv_samples = tcga_snv.donor_id.unique()
    tcga_wt_p53_samples = all_tcga_snv_samples[np.isin(all_tcga_snv_samples, tcga_impactful_p53_mut_samples)==False]
    tcga_wt_p53_samples = tcga_wt_p53_samples[np.isin(tcga_wt_p53_samples, tcga_not_impactful_p53_mut_samples)==False]

    # divide tcga tpm dataset by above groups
    tcga_tpm_impactful_p53_mut = tcga_expr[tcga_expr.donor_id.isin(tcga_impactful_p53_mut_samples)]
    tcga_tpm_not_impactful_p53_mut = tcga_expr[tcga_expr.donor_id.isin(tcga_not_impactful_p53_mut_samples)]
    tcga_tpm_wt_p53 = tcga_expr[tcga_expr.donor_id.isin(tcga_wt_p53_samples)]

    tcga_tpm_impactful_p53_mut = tcga_tpm_impactful_p53_mut.drop(columns='donor_id')
    tcga_tpm_not_impactful_p53_mut = tcga_tpm_not_impactful_p53_mut.drop(columns='donor_id')
    tcga_tpm_wt_p53 = tcga_tpm_wt_p53.drop(columns='donor_id')

    return tcga_tpm_impactful_p53_mut, tcga_tpm_not_impactful_p53_mut, tcga_tpm_wt_p53


# find_pog_p53_mut function finds the samples with and w/out p53 mutation in pog
def find_pog_p53_mut(pog_expr, pog_snv):

    pog_snv_fltr = pog_snv[pog_snv.gene_id=="TP53"]
    pog_snv_fltr_smpl = pog_snv_fltr.sample_id.unique()

    pog_snv_impacful = pog_snv_fltr[pog_snv_fltr.effect.isin(['splice_donor_variant+intron_variant', 'missense_variant', 'splice_acceptor_variant+intron_variant', 'disruptive_inframe_deletion', 'stop_gained', 'frameshift_variant', 'missense_variant+splice_region_variant', 'splice_acceptor_variant+disruptive_inframe_deletion+splice_region_variant+splice_region_variant+intron_variant', 'disruptive_inframe_deletion+splice_region_variant', 'stop_lost', 'inframe_deletion'])]
    pog_snv_impacful_smpl = pog_snv_impacful.sample_id.unique()

    pog_snv_not_impacful = pog_snv_fltr[pog_snv_fltr.effect.isin(['splice_region_variant+intron_variant', 'intron_variant', 'downstream_gene_variant', 'upstream_gene_variant', '5_prime_UTR_variant', 'synonymous_variant', '3_prime_UTR_variant'])]
    pog_snv_not_impacful_smpl = pog_snv_not_impacful.sample_id.unique()
    pog_snv_not_impacful_smpl = pog_snv_not_impacful_smpl[np.isin(pog_snv_not_impacful_smpl, pog_snv_impacful_smpl)==False]

    pog_all_smpl = pog_snv.sample_id.unique()
    pog_snv_wt_smpl = pog_all_smpl[np.isin(pog_all_smpl, pog_snv_fltr_smpl)==False]

    # divide pog tpm data by above groups
    pog_tpm_p53_wt = pog_expr[pog_expr.sample_id.isin(pog_snv_wt_smpl)]
    pog_tpm_p53_impactful_mut = pog_expr[pog_expr.sample_id.isin(pog_snv_impacful_smpl)]
    pog_tpm_p53_not_impact_mut = pog_expr[pog_expr.sample_id.isin(pog_snv_not_impacful_smpl)]

    return pog_tpm_p53_impactful_mut, pog_tpm_p53_not_impact_mut, pog_tpm_p53_wt


# make_X_y function creates the feature matrix and label array using two dfs w/ and w/out mut
def make_X_y(exp_mut_df, exp_wt_df, mut_label, wt_label):

    X=pd.concat([exp_mut_df, exp_wt_df])
    y=pd.concat([pd.Series([mut_label]*exp_mut_df.shape[0]),pd.Series([wt_label]*exp_wt_df.shape[0])])

    return X, y


# make_X_y_merged function creates the feature matrix and label array for merged data
def make_X_y_merged(exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2, mut_label, wt_label):

    X = pd.concat([exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2])
    y = pd.concat([pd.Series([mut_label]*exp_mut_df1.shape[0]), pd.Series([mut_label]*exp_mut_df2.shape[0]),
                   pd.Series([wt_label]*exp_wt_df1.shape[0]), pd.Series([wt_label]*exp_wt_df2.shape[0])])

    return X, y


# split_80_20 function splits X and y and keeps the proportion of class label
def split_80_20(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # set samples ids as index
    X_train = X_train.set_index('sample_id')
    X_test = X_test.set_index('sample_id')

    return X_train, X_test, y_train, y_test


# split_90_10 function splits X and y and keeps the proportion of class label
def split_90_10(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)

    # set samples ids as index
    X_train = X_train.set_index('sample_id')
    X_test = X_test.set_index('sample_id')

    return X_train, X_test, y_train, y_test
