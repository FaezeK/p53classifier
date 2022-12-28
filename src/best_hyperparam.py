############################################################################
# This script contains the code for hyperparameter tuning for the POG,
# TCGA, and the merged datasets
############################################################################

import pandas as pd
import timeit
import p53_helper as p53h
import grid_search as gs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

start_time = timeit.default_timer()

# read expression datasets
pog_tpm_pr = pd.read_csv(snakemake.input.pog_expr_prcssd, delimiter = '\t', header=0)
tcga_tpm_pr = pd.read_csv(snakemake.input.tcga_expr_prcssd, delimiter='\t', header=0)

# read mutation datasets
pog_snv_pr = pd.read_csv(snakemake.input.pog_snv_prcssd, delimiter='\t', header=0)
tcga_snv_pr = pd.read_csv(snakemake.input.tcga_snv_prcssd, delimiter='\t', header=0)

# read metadata
pog_meta_pr = pd.read_csv(snakemake.input.pog_meta_prcssd, delimiter = '\t', header=0) # POG metadata
tss = pd.read_csv(snakemake.input.tss, delimiter='\t', header=0) # TCGA metadata
print('The input files have been read')
print('')

# make X and y matrices
tcga_tpm_impactful_p53_mut, tcga_tpm_not_impactful_p53_mut, tcga_tpm_wt_p53 = p53h.find_tcga_p53_mut(tcga_tpm_pr, tcga_snv_pr)
pog_tpm_p53_impactful_mut, pog_tpm_p53_not_impact_mut, pog_tpm_p53_wt = p53h.find_pog_p53_mut(pog_tpm_pr, pog_snv_pr)

tcga_feature_matrix, tcga_p53_labels = p53h.make_X_y(tcga_tpm_impactful_p53_mut, tcga_tpm_wt_p53, 'p53_mut', 'p53_wt')
pog_feature_matrix, pog_p53_labels = p53h.make_X_y(pog_tpm_p53_impactful_mut, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')
both_feature_matrix, both_p53_labels = p53h.make_X_y_merged(tcga_tpm_impactful_p53_mut, pog_tpm_p53_impactful_mut, tcga_tpm_wt_p53, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')

# divide feature matrices to 90% test and 10% validation sets
tcga_X_train, tcga_X_test, tcga_y_train, tcga_y_test = p53h.split_90_10(tcga_feature_matrix, tcga_p53_labels)
pog_X_train, pog_X_test, pog_y_train, pog_y_test = p53h.split_90_10(pog_feature_matrix, pog_p53_labels)
both_X_train, both_X_test, both_y_train, both_y_test = p53h.split_90_10(both_feature_matrix, both_p53_labels)

print('feature matrices and label arrays are ready!')
print('')

# find the hyperparameters resulting in the best performance in a CV analysis
tcga_grid_result = gs.findHyperparam(tcga_X_train, tcga_y_train)
pog_grid_result = gs.findHyperparam(pog_X_train, pog_y_train)
both_grid_result = gs.findHyperparam(both_X_train, both_y_train)

##################################################################################
### Write the Grid Search results for POG to a file and test on validation set ###
##################################################################################

f = open(snakemake.output.pog_hyper_param_results, 'w')
print('best score:', file=f)
print(pog_grid_result.best_score_, file=f)
print('', file=f)
print('best params:', file=f)
print(pog_grid_result.best_params_, file=f)
print('', file=f)

pog_max_depth = pog_grid_result.best_params_.get('max_depth')
pog_max_features = pog_grid_result.best_params_.get('max_features')
pog_max_samples = pog_grid_result.best_params_.get('max_samples')
pog_min_samples_leaf = pog_grid_result.best_params_.get('min_samples_leaf')
pog_min_samples_split = pog_grid_result.best_params_.get('min_samples_split')
pog_n_estimators = pog_grid_result.best_params_.get('n_estimators')

pog_best_params = pd.DataFrame({'max_depth':[pog_max_depth], 'max_features':[pog_max_features],
                                 'max_samples':[pog_max_samples], 'min_samples_leaf':[pog_min_samples_leaf],
                                 'min_samples_split':[pog_min_samples_split], 'n_estimators':[pog_n_estimators]})

pog_best_params.to_csv(snakemake.output.pog_set_best_hyperparam, sep='\t', index=False)

clf = RandomForestClassifier(n_estimators=pog_n_estimators, max_depth=pog_max_depth,
                             max_features=pog_max_features, max_samples=pog_max_samples, n_jobs=40)
clf.fit(pog_X_train, pog_y_train)
print('Test validation set:', file=f)
print(clf.score(pog_X_train, pog_y_train), file=f)

pog_sample_ids = pog_X_test.index.values
pog_p53_predictions = clf.predict(pog_X_test)
pog_p53_pred_df = pd.DataFrame({'expr_sa_ids':pog_sample_ids, 'p53_status':pog_y_test, 'predict':pog_p53_predictions})

print(classification_report(pog_p53_pred_df.p53_status, pog_p53_pred_df.predict), file=f)

print('Validation set results with clf default params:', file=f)
clf2 = RandomForestClassifier(n_jobs=40)
clf2.fit(pog_X_train, pog_y_train)
print(clf2.score(pog_X_train, pog_y_train), file=f)

pog_p53_predictions2 = clf2.predict(pog_X_test)
pog_p53_pred_df2 = pd.DataFrame({'expr_sa_ids':pog_sample_ids, 'p53_status':pog_y_test, 'predict':pog_p53_predictions2})

print(classification_report(pog_p53_pred_df2.p53_status, pog_p53_pred_df2.predict), file=f)

###################################################################################
### Write the Grid Search results for TCGA to a file and test on validation set ###
###################################################################################

f2 = open(snakemake.output.tcga_hyper_param_results, 'w')
print('best score:', file=f2)
print(tcga_grid_result.best_score_, file=f2)
print('', file=f2)
print('best params:', file=f2)
print(tcga_grid_result.best_params_, file=f2)
print('', file=f2)

tcga_max_depth = tcga_grid_result.best_params_.get('max_depth')
tcga_max_features = tcga_grid_result.best_params_.get('max_features')
tcga_max_samples = tcga_grid_result.best_params_.get('max_samples')
tcga_min_samples_leaf = tcga_grid_result.best_params_.get('min_samples_leaf')
tcga_min_samples_split = tcga_grid_result.best_params_.get('min_samples_split')
tcga_n_estimators = tcga_grid_result.best_params_.get('n_estimators')

tcga_best_params = pd.DataFrame({'max_depth':[tcga_max_depth], 'max_features':[tcga_max_features],
                                 'max_samples':[tcga_max_samples], 'min_samples_leaf':[tcga_min_samples_leaf],
                                 'min_samples_split':[tcga_min_samples_split], 'n_estimators':[tcga_n_estimators]})

tcga_best_params.to_csv(snakemake.output.tcga_set_best_hyperparam, sep='\t', index=False)

clf3 = RandomForestClassifier(n_estimators=tcga_n_estimators, max_depth=tcga_max_depth,
                              max_features=tcga_max_features, max_samples=tcga_max_samples, n_jobs=40)
clf3.fit(tcga_X_train, tcga_y_train)
print('Test validation set:', file=f2)
print(clf3.score(tcga_X_train, tcga_y_train), file=f2)

tcga_sample_ids = tcga_X_test.index.values
tcga_p53_predictions = clf3.predict(tcga_X_test)
tcga_p53_pred_df = pd.DataFrame({'expr_sa_ids':tcga_sample_ids, 'p53_status':tcga_y_test, 'predict':tcga_p53_predictions})

print(classification_report(tcga_p53_pred_df.p53_status, tcga_p53_pred_df.predict), file=f2)

print('Validation set results with clf default params:', file=f2)
clf4 = RandomForestClassifier(n_jobs=40)
clf4.fit(tcga_X_train, tcga_y_train)
print(clf4.score(tcga_X_train, tcga_y_train), file=f2)

tcga_p53_predictions2 = clf4.predict(tcga_X_test)
tcga_p53_pred_df2 = pd.DataFrame({'expr_sa_ids':tcga_sample_ids, 'p53_status':tcga_y_test, 'predict':tcga_p53_predictions2})

print(classification_report(tcga_p53_pred_df2.p53_status, tcga_p53_pred_df2.predict), file=f2)

###################################################################################
### Write the Grid Search results for both to a file and test on validation set ###
###################################################################################

f3 = open(snakemake.output.both_hyper_param_results, 'w')
print('best score:', file=f3)
print(both_grid_result.best_score_, file=f3)
print('', file=f3)
print('best params:', file=f3)
print(both_grid_result.best_params_, file=f3)
print('', file=f3)

both_max_depth = both_grid_result.best_params_.get('max_depth')
both_max_features = both_grid_result.best_params_.get('max_features')
both_max_samples = both_grid_result.best_params_.get('max_samples')
both_min_samples_leaf = both_grid_result.best_params_.get('min_samples_leaf')
both_min_samples_split = both_grid_result.best_params_.get('min_samples_split')
both_n_estimators = both_grid_result.best_params_.get('n_estimators')

both_best_params = pd.DataFrame({'max_depth':[both_max_depth], 'max_features':[both_max_features],
                                 'max_samples':[both_max_samples], 'min_samples_leaf':[both_min_samples_leaf],
                                 'min_samples_split':[both_min_samples_split], 'n_estimators':[both_n_estimators]})

both_best_params.to_csv(snakemake.output.both_sets_best_hyperparam, sep='\t', index=False)

clf5 = RandomForestClassifier(n_estimators=both_n_estimators, max_depth=both_max_depth,
                              max_features=both_max_features, max_samples=both_max_samples, n_jobs=40)
clf5.fit(both_X_train, both_y_train)
print('Test validation set:', file=f3)
print(clf5.score(both_X_train, both_y_train), file=f3)

both_sample_ids = both_X_test.index.values
both_p53_predictions = clf5.predict(both_X_test)
both_p53_pred_df = pd.DataFrame({'expr_sa_ids':both_sample_ids, 'p53_status':both_y_test, 'predict':both_p53_predictions})

print(classification_report(both_p53_pred_df.p53_status, both_p53_pred_df.predict), file=f3)

print('Validation set results with clf default params:', file=f3)
clf6 = RandomForestClassifier(n_jobs=40)
clf6.fit(both_X_train, both_y_train)
print(clf6.score(both_X_train, both_y_train), file=f3)

both_p53_predictions2 = clf6.predict(both_X_test)
both_p53_pred_df2 = pd.DataFrame({'expr_sa_ids':both_sample_ids, 'p53_status':both_y_test, 'predict':both_p53_predictions2})

print(classification_report(both_p53_pred_df2.p53_status, both_p53_pred_df2.predict), file=f3)

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
