############################################################################
# This script contains the code for preprocessing the input
############################################################################

import pandas as pd
import process_expr
import process_mut
import make_pca

# read expression datasets
pog_tpm = pd.read_csv('POG_TPM_matrix.txt', delimiter = '\t', header=0)
tcga_tpm = pd.read_csv('tcga_rsem_gene_tpm.txt', delimiter='\t', header=0)

# read mutation datasets
pog_snv = pd.read_csv('POG_small_mutations.txt', delimiter='\t', header=0)
tcga_snv = pd.read_csv('mc3.v0.2.8.PUBLIC.txt', delimiter='\t', header=0)

# read metadata
common_genes = pd.read_csv('common_genes.txt', delimiter = '\t', header=0) # list of common genes between POG and TCGA datasets
pog_meta = pd.read_csv('pog_cohort_details.txt', delimiter = '\t', header=0) # POG metadata
tss = pd.read_csv('tissueSourceSite.tsv', delimiter='\t', header=0) # TCGA metadata
print('The input files have been read')
print('')

# process expr data
pog_tpm_trnspsd, pog_tpm_fltrd_genes = process_expr.process_POG_expr(pog_tpm, common_genes)
tcga_actual_tpm_ucsc = process_expr.process_TCGA_expr(tcga_tpm, common_genes, pog_tpm_fltrd_genes)

# process mutation data
pog_snv = process_mut.process_POG_mut(pog_snv)
tcga_snv = process_mut.process_TCGA_mut(tcga_snv)

# process metadata
pog_meta_fltrd = pog_meta[['ID','PRIMARY SITE']]
pog_meta_fltrd = pog_meta_fltrd.rename(columns={'ID':'sample_id'})

# find TCGA cancer types
tcga_sample_ids = tcga_actual_tpm_ucsc.sample_id
tcga_type=[]

for i in tcga_sample_ids:
    st_ind = i.find('-')
    end_ind = i.find('-',(i.find('-')+1))
    code = i[st_ind+1 : end_ind]
    t_type = tss[tss['TSS Code']==code]['Study Name'].to_string(index=False)
    tcga_type.append(t_type)

tcga_type_df = pd.DataFrame({'sample_id':tcga_sample_ids, 'type':tcga_type})

print('Data preprocessing is done . . .')
print('')
print('Making PCA plots . . .')
print('')

# generate PCA plots
make_pca.generate_PCA(pog_tpm_trnspsd, pog_meta_fltrd, 'POG')

tcga_type_df = make_pca.extract_tcga_types(tcga_expr, tcga_meta)

make_pca.generate_PCA(tcga_actual_tpm_ucsc, tcga_type_df, 'TCGA')

make_pca.generate_PCA_merged(tcga_actual_tpm_ucsc, pog_tpm_trnspsd)

print('PCA plots are made . . .')
print('')

tcga_actual_tpm_ucsc.to_csv('TCGA_expr_prcssd.txt', sep='\t', index=False)
tcga_snv.to_csv('TCGA_snv_prcssd.txt', sep='\t', index=False)
tcga_type_df.to_csv('TCGA_types.txt', sep='\t', index=False)
pog_tpm_trnspsd.to_csv('POG_expr_prcssd.txt', sep='\t', index=False)
pog_snv.to_csv('POG_snv_prcssd.txt', sep='\t', index=False)
pog_meta_fltrd.to_csv('POG_meta_prcssd.txt', sep='\t', index=False)
