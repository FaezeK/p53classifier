############################################################################
# This script processes expression data and produces a matrix that can be
# used as the random forest input
############################################################################

def process_POG_expr(pog_expr, pog_genes):
    pog_expr = pog_expr.rename(columns={"genes": "ensembl_ids"})

    ensembl_ids = pog_genes.genes.str.split(pat='_', expand=True)[1] # extract ensembl ids
    pog_genes['ensembl_ids'] = ensembl_ids

    pog_tpm_fltrd = pd.merge(pog_expr, pog_genes, on='ensembl_ids') # filter for genes also used for tcga samples
    pog_tpm_fltrd_genes = pog_tpm_fltrd.ensembl_ids

    pog_tpm_fltrd = pog_tpm_fltrd.drop(columns=['ensembl_ids'])
    pog_tpm_fltrd = pog_tpm_fltrd.set_index('genes')

    pog_tpm_trnspsd = pog_tpm_fltrd.transpose()  # transpose the df

    s_id = pd.DataFrame({'sample_id': pog_tpm_trnspsd.index}, index=pog_tpm_trnspsd.index) # add samples as a column
    pog_tpm_trnspsd = pd.concat([s_id, pog_tpm_trnspsd], axis=1)

    return pog_tpm_trnspsd, pog_tpm_fltrd_genes


def process_TCGA_expr(tcga_expr, pog_genes, pog_fltrd_genes):
    tcga_ensembl_ids = tcga_expr['sample'].str.split(pat='.', expand=True)[0]
    tcga_expr['ensembl_ids'] = tcga_ensembl_ids

    tcga_tpm_ucsc_fltrd = tcga_expr[tcga_expr.ensembl_ids.isin(pog_fltrd_genes)] # filter TCGA dataset for genes also in POG dataset
    tcga_tpm_ucsc_fltrd = pd.merge(tcga_tpm_ucsc_fltrd, pog_genes, on='ensembl_ids')
    tcga_tpm_ucsc_fltrd = tcga_tpm_ucsc_fltrd.drop(columns=['sample', 'ensembl_ids']) # get the gene names for ensembl ids
    tcga_tpm_ucsc_fltrd = tcga_tpm_ucsc_fltrd.set_index('genes')

    tcga_tpm_ucsc_t = tcga_tpm_ucsc_fltrd.transpose()  # transpose the dataset

    tcga_tpm_ucsc_t['sample_id'] = tcga_tpm_ucsc_t.index
    tcga_tpm_ucsc_t = tcga_tpm_ucsc_t.reset_index(drop=True)
    tcga_tpm_ucsc_t = tcga_tpm_ucsc_t.set_index('sample_id')

    tcga_tpm_ucsc_samples_df = pd.DataFrame({'sample_id':tcga_tpm_ucsc_t.index, 'code':tcga_tpm_ucsc_t.index.str.slice(start=13, stop=15)}) # find primary tumours in TCGA
    tcga_tpm_ucsc_primary_samples_df = tcga_tpm_ucsc_samples_df[tcga_tpm_ucsc_samples_df.code.isin(['01', '03', '05'])]

    tcga_tpm_ucsc_t = tcga_tpm_ucsc_t[tcga_tpm_ucsc_t.index.isin(tcga_tpm_ucsc_primary_samples_df.sample_id)]

    tcga_actual_tpm_ucsc = (2**tcga_tpm_ucsc_t) - 0.001 # convert to actual TPM values

    s_id_tcga = pd.DataFrame({'sample_id': tcga_actual_tpm_ucsc.index}, index=tcga_actual_tpm_ucsc.index) # add samples as a column
    tcga_actual_tpm_ucsc = pd.concat([s_id_tcga, tcga_actual_tpm_ucsc], axis=1)

    return tcga_actual_tpm_ucsc
