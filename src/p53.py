############################################################################
# This script contains the code for training the random forest to classify
# tumour samples based on presence or absence of mutations in TP53 gene
# and evaluating the model performance.
############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
import p53_helper as p53h
import five_fold_cv as fcv
import oob
from sklearn.ensemble import RandomForestClassifier

# read expression datasets
pog_tpm_pr = pd.read_csv('POG_expr_prcssd.txt', delimiter = '\t', header=0)
tcga_tpm_pr = pd.read_csv('TCGA_expr_prcssd.txt', delimiter='\t', header=0)

# read mutation datasets
pog_snv_pr = pd.read_csv('POG_snv_prcssd.txt', delimiter='\t', header=0)
tcga_snv_pr = pd.read_csv('TCGA_snv_prcssd.txt', delimiter='\t', header=0)

# read metadata
pog_meta_pr = pd.read_csv('POG_meta_prcssd.txt', delimiter = '\t', header=0) # POG metadata
tcga_type_df = pd.read_csv('TCGA_types.txt', delimiter='\t', header=0) # TCGA metadata
print('The input files have been read')
print('')

# make X and y matrices
tcga_tpm_impactful_p53_mut, tcga_tpm_not_impactful_p53_mut, tcga_tpm_wt_p53 = p53h.find_tcga_p53_mut(tcga_tpm_pr, tcga_snv_pr)
pog_tpm_p53_impactful_mut, pog_tpm_p53_not_impact_mut, pog_tpm_p53_wt = p53h.find_pog_p53_mut(pog_tpm_pr, pog_snv_pr)

tcga_feature_matrix, tcga_p53_labels = p53h.make_X_y(tcga_tpm_impactful_p53_mut, tcga_tpm_wt_p53, 'p53_mut', 'p53_wt')
pog_feature_matrix, pog_p53_labels = p53h.make_X_y(pog_tpm_p53_impactful_mut, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')
merged_feature_matrix, merged_p53_labels = p53h.make_X_y_merged(tcga_tpm_impactful_p53_mut, pog_tpm_p53_impactful_mut, tcga_tpm_wt_p53, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')

###############################################################################################
###############################################################################################
############################## Random Forest Overall Performance ##############################
###############################################################################################
###############################################################################################

# obtain random forest predictions on TCGA, POG, and merged datasets in a 5-fold CV analysis
tcga_all_pred_df, tcga_all_prob, tcga_true_label_prob = fcv.predict_5_fold_cv(tcga_feature_matrix, tcga_p53_labels, 50, 25, 2)
pog_all_pred_df, pog_all_prob, pog_true_label_prob = fcv.predict_5_fold_cv(pog_feature_matrix, pog_p53_labels, 100, 50, 1)
merged_all_pred_df, merged_all_prob, merged_true_label_prob = fcv.predict_5_fold_cv(merged_feature_matrix, merged_p53_labels, 50, 2, 2)

# random forest performance (make the AUROC and AUPRC graphs)
fig, axes =plt.subplots(3, 2, figsize=(12, 18), dpi=300)
sns.set_theme()

fcv.generate_auroc_curve(tcga_all_pred_df.p53_status, tcga_true_label_prob, axes[0,0])
fcv.generate_auprc_curve(tcga_all_pred_df.p53_status, tcga_true_label_prob, axes[0,1])

fnt_size = 18
axes[0,0].set_title("AUROC", fontsize=fnt_size)
axes[0,1].set_title("AUPRC", fontsize=fnt_size)

axes[0,0].text(-0.45, 0.5,'TCGA', fontsize=fnt_size)

tcga_auprc = sklearn.metrics.average_precision_score(tcga_all_pred_df.p53_status, tcga_true_label_prob, pos_label="p53_wt")
tcga_auroc = sklearn.metrics.roc_auc_score(tcga_all_pred_df.p53_status, tcga_true_label_prob)

axes[0,0].text(0.20, 0.60, 'AUROC='+str(round(tcga_auroc, 2)))
axes[0,1].text(0.50, 0.60, 'AUPRC='+str(round(tcga_auprc, 2)))

fcv.generate_auroc_curve(pog_all_pred_df.p53_status, pog_true_label_prob, axes[1,0])
fcv.generate_auprc_curve(pog_all_pred_df.p53_status, pog_true_label_prob, axes[1,1])

axes[1,0].text(-0.45, 0.5,'POG', fontsize=fnt_size)

pog_auprc = sklearn.metrics.average_precision_score(pog_all_pred_df.p53_status, pog_true_label_prob, pos_label="p53_wt")
pog_auroc = sklearn.metrics.roc_auc_score(pog_all_pred_df.p53_status, pog_true_label_prob)

axes[1,0].text(0.20, 0.60, 'AUROC='+str(round(pog_auroc, 2)))
axes[1,1].text(0.50, 0.60, 'AUPRC='+str(round(pog_auprc, 2)))

fcv.generate_auroc_curve(merged_all_pred_df.p53_status, merged_true_label_prob, axes[2,0])
fcv.generate_auprc_curve(merged_all_pred_df.p53_status, merged_true_label_prob, axes[2,1])

axes[2,0].text(-0.5, 0.5,'Merged', fontsize=fnt_size)

both_auprc = sklearn.metrics.average_precision_score(merged_all_pred_df.p53_status, merged_true_label_prob, pos_label="p53_wt")
both_auroc = sklearn.metrics.roc_auc_score(merged_all_pred_df.p53_status, merged_true_label_prob)

axes[2,0].text(0.20, 0.60, 'AUROC='+str(round(both_auroc, 2)))
axes[2,1].text(0.50, 0.60, 'AUPRC='+str(round(both_auprc, 2)))

fig.savefig('auroc_auprc.jpg',format='jpeg',dpi=300,bbox_inches='tight')

###############################################################################################
###############################################################################################
#################### Random Forest Performance across 33 TCGA Cancer Types ####################
###############################################################################################
###############################################################################################

tcga_cancer_types_performance = fcv.performance_tcga_cancer_types(tcga_all_pred_df, tcga_type_df)
tcga_cancer_types_performance.to_csv('results/TCGA_cancer_types_metrics.txt', sep='\t', index=False)

# compare performance when all samples are used to train the RF vs when each cancer type samples are used
tcga_cancer_types_oob = oob.evaluate_RF_on_each_cancer_type(tcga_all_pred_df, tcga_type_df, tcga_tpm_pr, tcga_snv_pr)
tcga_cancer_types_accuracy_oob = pd.merge(tcga_cancer_types_performance, tcga_cancer_types_oob, on='type')

cancer_type_abbv_df = oob.make_abbv(tcga_type_df)
tcga_cancer_types_accuracy_oob_abbv = pd.merge(tcga_cancer_types_accuracy_oob, cancer_type_abbv_df, on='type')

# make accuracy vs oob score graph
tcga_cancer_types_accuracy_oob_abbv = tcga_cancer_types_accuracy_oob_abbv.sort_values(by='accuracy')
tcga_cancer_types_accuracy_oob_abbv = tcga_cancer_types_accuracy_oob_abbv[['type','accuracy','oob_score','abbv']]
tcga_cancer_types_accuracy_oob_abbv = tcga_cancer_types_accuracy_oob_abbv.set_index('abbv')
sns.set(rc={'figure.figsize':(11.7,5.77)})
fig = plt.figure()
oob_vs_accr_plot = sns.lineplot(data=tcga_cancer_types_accuracy_oob_abbv, markers=True, dashes=False)
plt.xticks(rotation=90)
plt.xlabel('Cancer Type')
plt.ylabel('Mean Accuracy and OOB Scores')
fig.savefig('accr_vs_oob_plot.jpg',format='jpeg',dpi=400,bbox_inches='tight')

###############################################################################################
###############################################################################################
############################ Prediction Probabilities and Outliers ############################
###############################################################################################
###############################################################################################

sns.set_style('white')
fig, axes =plt.subplots(2, 2, figsize=(12, 10), dpi=300)

# A
merged_all_pred_df['pred_prob'] = merged_all_prob
merged_all_pred_df['pred_crrctness'] = 'correct'
merged_all_pred_df.loc[merged_all_pred_df['p53_status'] != merged_all_pred_df['predict'], 'pred_crrctness'] = 'mispredicted'

pred_crrctness_boxplot = sns.boxplot(data = merged_all_pred_df, x = "pred_crrctness", y = "pred_prob", palette="Paired", ax=axes[0,0])
axes[0,0].set(xlabel='', ylabel='Prediction Probability')

# B
merged_all_pred_df['cohort'] = 'TCGA'
merged_all_pred_df.loc[merged_all_pred_df['expr_sa_ids'].str.contains('POG'), 'cohort'] = 'POG'

pred_crrctness_cohort_boxplot = sns.boxplot(data = merged_all_pred_df, x = "cohort", y = "pred_prob", hue = "pred_crrctness",
                                            palette="Paired", ax=axes[0,1])
axes[0,1].set(xlabel='', ylabel='Prediction Probability')
axes[0,1].legend(bbox_to_anchor=(0.35, 1))

# C
pred_crrctness_p53stat_boxplot = sns.boxplot(data = merged_all_pred_df, x = "p53_status", y = "pred_prob", hue = "pred_crrctness",
                                            palette="Greens", order=['p53_wt', 'p53_mut'], ax=axes[1,0])
axes[1,0].set(xlabel='p53 status', ylabel='Prediction Probability')
axes[1,0].legend(bbox_to_anchor=(0.35, 1))

# D
pred_crrctness_p53class_boxplot = sns.boxplot(data = merged_all_pred_df, x = "predict", y = "pred_prob", hue = "pred_crrctness",
                                            palette="Greens", order=['p53_wt', 'p53_mut'], ax=axes[1,1])
axes[1,1].set(xlabel='p53 Classification by RF', ylabel='Prediction Probability')
axes[1,1].legend(bbox_to_anchor=(0.35, 1))

# add titles
axes[0,0].set_title('A', fontsize=22)
axes[0,0].title.set_position([-0.1, 1.1])

axes[0,1].set_title('B', fontsize=22)
axes[0,1].title.set_position([-0.1, 1.1])

axes[1,0].set_title('C', fontsize=22)
axes[1,0].title.set_position([-0.1, 1.1])

axes[1,1].set_title('D', fontsize=22)
axes[1,1].title.set_position([-0.1, 1.1])

fig.tight_layout()
fig.savefig('pred_prob_all4.jpg',format='jpeg',dpi=300,bbox_inches='tight')

# extract the mispredicted samples with high Probabilities
merged_mis_gt95 = merged_all_pred_df[(merged_all_pred_df.pred_crrctness=='mispredicted') & (merged_all_pred_df.pred_prob > 0.95)]
merged_mis_gt95 = merged_mis_gt95.sort_values(by='pred_prob', ascending=False)

merged_mis_gt95 = merged_mis_gt95.rename(columns={"expr_sa_ids": "sample_id"})
merged_mis_gt95_w_type = pd.merge(merged_mis_gt95, tcga_type_df, on='sample_id')
merged_mis_gt95_w_type = merged_mis_gt95_w_type.sort_values(by='type')
merged_mis_gt95_w_type.to_csv('results/outliers.txt', sep='\t', index=False)

###############################################################################################
###############################################################################################
############################ Important Features in Classification #############################
###############################################################################################
###############################################################################################

clf = RandomForestClassifier(n_estimators=3000, max_depth=50, max_features=0.05, max_samples=0.99, min_samples_split=2, min_samples_leaf=2, n_jobs=40)
clf.fit(merged_feature_matrix, merged_p53_labels)
rand_f_scores = clf.feature_importances_
indices = np.argsort(rand_f_scores)
rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(merged_feature_matrix.columns[indices]), 'importance_score':rand_f_scores_sorted})
rand_forest_importance_scores_df = rand_forest_importance_scores_df.sort_values(by='importance_score', ascending=False)

top_67_genes = rand_forest_importance_scores_df.iloc[0:67,:]

# combine the TCGA and POG sets of samples with impactful mutations
both_tpm_impact_p53 = pog_tpm_p53_impactful_mut.append(tcga_tpm_impactful_p53_mut, ignore_index=True)
both_tpm_impact_p53 = both_tpm_impact_p53.set_index('sample_id')

# combine the TCGA and POG sets of samples with WT p53 copies
both_tpm_wt_p53 = pog_tpm_p53_wt.append(tcga_tpm_wt_p53, ignore_index=True)
both_tpm_wt_p53 = both_tpm_wt_p53.set_index('sample_id')

# extract the mutation rates and modification in expression in presence of p53 mutations
# for the top 67 genes
#pd.set_option('display.max_rows', 500)
top_genes_mut_rate_n_reg_stat = pd.DataFrame({'gene':['a'],'imp_score':[-1.0],'mut_rate_tcga':[-1],'mut_rate_pog':[-1],'reg_stat':['up_or_down']})

for i in top_67_genes.gene:
    i_p53_mut = both_tpm_impact_p53[i]
    i_p53_wt = both_tpm_wt_p53[i]

    i_p53_mut_mean = i_p53_mut.mean()
    i_p53_mut_median = i_p53_mut.median()
    i_p53_wt_mean = i_p53_wt.mean()
    i_p53_wt_median = i_p53_wt.median()

    if i_p53_mut_median > i_p53_wt_median:
        reg_stat = 'Up-reg'
    elif i_p53_mut_median < i_p53_wt_median:
        reg_stat = 'Down-reg'
    elif (i_p53_mut_median == i_p53_wt_median) & (i_p53_mut_mean > i_p53_wt_mean):
        reg_stat = 'Up-reg'
    elif (i_p53_mut_median == i_p53_wt_median) & (i_p53_mut_mean < i_p53_wt_mean):
        reg_stat = 'Down-reg'
    else:
        reg_stat = 'No diff'

    g = i.split('_')[0]
    g_mut_rate_tcga = round((len(tcga_snv_pr[tcga_snv_pr.gene_id==g].donor_id.unique()) / len(tcga_snv_pr.donor_id.unique())) * 100, ndigits=2)
    g_mut_rate_pog = round((len(pog_snv_pr[pog_snv_pr.gene_id==g].sample_id.unique()) / len(pog_snv_pr.sample_id.unique())) * 100, ndigits=2)

    scr = top_67_genes.importance_score[top_67_genes.gene==i].iloc[0]

    i_row = pd.Series({'gene':g,'imp_score':scr,'mut_rate_tcga':g_mut_rate_tcga,'mut_rate_pog':g_mut_rate_pog,'reg_stat':reg_stat})
    top_genes_mut_rate_n_reg_stat = top_genes_mut_rate_n_reg_stat.append(i_row, ignore_index=True)

top_genes_mut_rate_n_reg_stat = top_genes_mut_rate_n_reg_stat[top_genes_mut_rate_n_reg_stat.gene != 'a']
top_genes_mut_rate_n_reg_stat.to_csv('top67_genes_w_mut_rate_reg_stat.txt', sep='\t', index=False)

# boxplots of expr for the top 10 genes
sns.set_style("whitegrid")
fig, axes =plt.subplots(5, 2, figsize=(12, 26), dpi=400)
sns.set_style("whitegrid")

p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'EDA2R_ENSG00000131080', axes[0,0])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'RPS27L_ENSG00000185088', axes[0,1])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'MDM2_ENSG00000135679', axes[1,0])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'MYBL2_ENSG00000101057', axes[1,1])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'UBE2C_ENSG00000175063', axes[2, 0])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'CDC20_ENSG00000117399', axes[2,1])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'RP11-115D19.1_ENSG00000251095', axes[3,0])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'FAM83D_ENSG00000101447', axes[3,1])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'RP11-611O2.5_ENSG00000257181', axes[4,0])
p53h.gene_expr_boxplot_mutVsWt_multi(both_tpm_impact_p53, both_tpm_wt_p53, 'DDB2_ENSG00000134574', axes[4,1])

fig.savefig('top_10_expr.jpg',format='jpeg',dpi=400,bbox_inches='tight')

###############################################################################################
###############################################################################################
################################## Non-impactful Mutations ####################################
###############################################################################################
###############################################################################################

both_p53_expr_test = pd.concat([tcga_tpm_not_impactful_p53_mut, pog_tpm_p53_not_impact_mut])

not_impactful_p53_predictions = clf.predict(both_p53_expr_test)
not_impactful_p53_predictions_df = pd.DataFrame({'sample_id':both_p53_expr_test.index, 'pred':not_impactful_p53_predictions})

# add mutation effect to the above df
tcga_mut_effect = tcga_snv_pr[tcga_snv_pr.gene_id=="TP53"]
pog_mut_effect = pog_snv_pr[pog_snv_pr.gene_id=="TP53"]
all_mut_effect = pd.concat([tcga_mut_effect, pog_mut_effect])

not_impactful_p53_predictions_df = pd.merge(not_impactful_p53_predictions_df, all_mut_effect, on='sample_id')
not_impactful_p53_predictions_df = not_impactful_p53_predictions_df.drop_duplicates()
not_impactful_p53_predictions_df.to_csv('not_impact_mut_pred.txt', sep='\t', index=False)
