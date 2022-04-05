############################################################################
# This script contains the code to find a threshold for the number of
# important genes (features) in classification.
############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import p53_helper as p53h
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

X, y = p53h.make_X_y_merged(tcga_tpm_impactful_p53_mut, pog_tpm_p53_impactful_mut, tcga_tpm_wt_p53, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')
X = X.set_index('sample_id')

print("Input files are read and processed.")
print("Algorithm training starts . . . ")

clf = RandomForestClassifier(n_estimators=3000, max_depth=50, max_features=0.05, max_samples=0.99, min_samples_split=2, min_samples_leaf=2, n_jobs=40)
clf.fit(X,y)
rand_f_scores = clf.feature_importances_
indices = np.argsort(rand_f_scores)
rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(X.columns[indices]), 'importance_score':rand_f_scores_sorted})
rand_forest_importance_scores_df = rand_forest_importance_scores_df.sort_values(by='importance_score', ascending=False)
rand_forest_importance_scores_df = rand_forest_importance_scores_df.head(n=1000)
rand_forest_importance_scores_df['iter'] = 'true_lab'
rand_forest_importance_scores_df['rank'] = range(1,(len(rand_forest_importance_scores_df.gene)+1))

rand_forest_importance_scores_df_all = rand_forest_importance_scores_df

print("RF is trained and features ranks are extracted for random assignments.")
print("Start to shuffle labels and train the algorithm 100 times . . . ")

X_1000 = X.iloc[:,X.columns.isin(rand_forest_importance_scores_df.gene)]

y_imp_feat = y.sample(frac=1)
for iter in range(100):
    clf.fit(X_1000, y_imp_feat)

    rand_f_scores = clf.feature_importances_
    indices = np.argsort(rand_f_scores)
    rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
    rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(X.columns[indices]), 'importance_score':rand_f_scores_sorted, 'iter':iter})
    rand_forest_importance_scores_df = rand_forest_importance_scores_df[rand_forest_importance_scores_df.importance_score != 0]
    rand_forest_importance_scores_df = rand_forest_importance_scores_df.sort_values(by='importance_score', ascending=False)
    rand_forest_importance_scores_df['rank'] = range(1,(len(rand_forest_importance_scores_df.gene)+1))
    rand_forest_importance_scores_df_all = rand_forest_importance_scores_df_all.append(rand_forest_importance_scores_df, ignore_index=True)
    y_imp_feat = y.sample(frac=1)

print("100 random training is done.")
print("Starts making graphs . . . ")

rand_frst_imp_feat_scr_df2 = rand_forest_importance_scores_df_all.pivot(index='rank', columns='iter', values='importance_score')
rand_frst_imp_feat_scr_df2 = rand_frst_imp_feat_scr_df2.dropna()

rand_frst_imp_feat_scr_df2.columns = ['iter' + str(s) for s in list(rand_frst_imp_feat_scr_df2.columns)]
rand_frst_imp_feat_scr_df2 = rand_frst_imp_feat_scr_df2.rename(columns={"itertrue_lab": "true_lab"})
rand_frst_imp_feat_scr_df2 = rand_frst_imp_feat_scr_df2.sort_values(by='true_lab', ascending=False)
n, d = rand_frst_imp_feat_scr_df2.shape

rand_frst_imp_feat_scr_df2_copy = rand_frst_imp_feat_scr_df2
rand_frst_imp_feat_scr_df2_copy['mean'] = rand_frst_imp_feat_scr_df2_copy.iloc[:,0:(d-1)].mean(axis=1)
rand_frst_imp_feat_scr_df2_copy['sd'] = rand_frst_imp_feat_scr_df2_copy.iloc[:,0:(d-1)].std(axis=1)

plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_feat_scr_df2_copy.index, rand_frst_imp_feat_scr_df2_copy.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_feat_scr_df2_copy.index, rand_frst_imp_feat_scr_df2_copy['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_feat_scr_df2_copy.index, rand_frst_imp_feat_scr_df2_copy['mean']+rand_frst_imp_feat_scr_df2_copy.sd, rand_frst_imp_feat_scr_df2_copy['mean']-rand_frst_imp_feat_scr_df2_copy.sd, color='mediumslateblue')
plt.xlabel('Gene Rank', fontsize=16)
plt.ylabel('Importance (Gini) Score', fontsize=16)
plt.savefig('TrueVsRandomScoresByRank.png', bbox_inches='tight', dpi=300)
plt.close()

rand_frst_imp_feat_scr_df2_copy2 = rand_frst_imp_feat_scr_df2_copy.iloc[30:100,:]
plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean']+rand_frst_imp_feat_scr_df2_copy2.sd, rand_frst_imp_feat_scr_df2_copy2['mean']-rand_frst_imp_feat_scr_df2_copy2.sd, color='mediumslateblue')
plt.savefig('TrueVsRandomScoresByRankZoomedIn.png', bbox_inches='tight', dpi=300)
plt.close()

rand_frst_imp_feat_scr_df2_copy2 = rand_frst_imp_feat_scr_df2_copy.iloc[60:75,:]
plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean']+rand_frst_imp_feat_scr_df2_copy2.sd, rand_frst_imp_feat_scr_df2_copy2['mean']-rand_frst_imp_feat_scr_df2_copy2.sd, color='mediumslateblue')
plt.savefig('TrueVsRandomScoresByRankZoomedInMore.png', bbox_inches='tight', dpi=300)
plt.close()

print("Graphs are made")
