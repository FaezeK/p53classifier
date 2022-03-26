############################################################################
# This script generates the PCA plots using expression data for the TCGA
# and POG data sets as well as the merged set of all samples.
############################################################################

def extract_tcga_types(tcga_expr, tcga_meta):
    tcga_sample_ids = tcga_expr.sample_id
    tcga_type=[]

    for i in tcga_sample_ids:
        st_ind = i.find('-')
        end_ind = i.find('-',(i.find('-')+1))
        code = i[st_ind+1 : end_ind]
        t_type = tss[tss['TSS Code']==code]['Study Name'].to_string(index=False)
        tcga_type.append(t_type)

        tcga_type_df = pd.DataFrame({'sample_id':tcga_sample_ids, 'type':tcga_type})

    return tcga_type_df

def generate_PCA(expr_df, metadata, dat):
    expr_no_ids = expr_df.set_index('sample_id')
    expr_tpm_log2 = np.log2(expr_no_ids + 0.001)

    pca = PCA(n_components=2)
    expr_principalComponents = pca.fit_transform(expr_tpm_log2)
    expr_principalDf = pd.DataFrame(data = expr_principalComponents, columns = ['pc1', 'pc2'], index=expr_no_ids.index)
    expr_principalDf['sample_id'] = expr_principalDf.index
    expr_principalDf = expr_principalDf.reset_index(drop=True)

    expr_finalDf = pd.merge(expr_principalDf, metadata, on='sample_id')

    fig_dims = (12, 9)
    fig, ax = plt.subplots(figsize=fig_dims)
    if dat=='POG':
        pog_pca_plt = sns.scatterplot(x="pc1", y="pc2", hue="PRIMARY SITE", data=expr_finalDf)
    else:
        tcga_pca_plt = sns.scatterplot(x="pc1", y="pc2", hue="type", data=expr_finalDf)
    plt.legend(bbox_to_anchor=(1, 1.02))
    ax.figure.savefig(dat+'_pca.jpg',format='jpeg',dpi=300,bbox_inches='tight')

def generate_PCA_merged(tcga_expr, pog_expr):
    tcga_no_ids = tcga_expr.set_index('sample_id')
    tcga_tpm_log2 = np.log2(tcga_no_ids + 0.001)

    pog_no_ids = pog_expr.set_index('sample_id')
    pog_tpm_log2 = np.log2(pog_no_ids + 0.001)

    # merge two datasets
    both_tpm_log2 = pd.concat([tcga_tpm_log2, pog_tpm_log2], axis=0)
    both_principalComponents = pca.fit_transform(both_tpm_log2)

    both_principalDf = pd.DataFrame(data = both_principalComponents, columns = ['pc1', 'pc2'], index=both_tpm_log2.index)
    both_principalDf['sample_id'] = both_principalDf.index
    both_principalDf = both_principalDf.reset_index(drop=True)

    both_labels_pca=pd.DataFrame({'sample_id':both_principalDf.sample_id, 'cohort':'tcga'})
    both_labels_pca.loc[both_labels_pca['sample_id'].isin(pog_expr.sample_id), 'cohort'] = 'pog'

    both_finalDf = pd.merge(both_principalDf, both_labels_pca, on='sample_id')

    fig_dims = (12, 9)
    fig, ax = plt.subplots(figsize=fig_dims)
    both_pca_plt = sns.scatterplot(x="pc1", y="pc2", hue="cohort", data=both_finalDf)#, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    ax.figure.savefig('merged_pca.jpg',format='jpeg',dpi=300,bbox_inches='tight')
