import pandas as pd
import numpy as np
import timeit
import p53
import process_expr
import process_mut
import p53_hyperparam as p53hp
import make_pca
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

start_time = timeit.default_timer()

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
pog_snv = process_POG_mut(pog_snv)
tcga_snv = process_TCGA_mut(tcga_snv)

print('Data preprocessing is done . . .')
print('')
print('Making PCA plots . . .')
print('')

# process metadata
pog_meta_fltrd = pog_meta[['ID','PRIMARY SITE']]
pog_meta_fltrd = pog_meta_fltrd.rename(columns={'ID':'sample_id'})

# generate PCA plots
make_pca.generate_PCA(pog_tpm_trnspsd, pog_meta_fltrd, 'POG')

tcga_type_df = make_pca.extract_tcga_types(tcga_expr, tcga_meta)

make_pca.generate_PCA(tcga_actual_tpm_ucsc, tcga_type_df, 'TCGA')

make_pca.generate_PCA_merged(tcga_actual_tpm_ucsc, pog_tpm_trnspsd)

print('PCA plots are made . . .')
print('')

# make X and y matrices
tcga_tpm_impactful_p53_mut, tcga_tpm_not_impactful_p53_mut, tcga_tpm_wt_p53 = p53.find_tcga_p53_mut(tcga_actual_tpm_ucsc, tcga_snv)
pog_tpm_p53_impactful_mut, pog_tpm_p53_not_impact_mut, pog_tpm_p53_wt = p53.find_pog_p53_mut(pog_tpm_trnspsd, pog_snv)

tcga_feature_matrix, tcga_p53_labels = p53.make_X_y(tcga_tpm_impactful_p53_mut, tcga_tpm_wt_p53, 'p53_mut', 'p53_wt')
pog_feature_matrix, pog_p53_labels = p53.make_X_y(pog_tpm_p53_impactful_mut, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')
both_feature_matrix, both_p53_labels = p53.make_X_y_merged(tcga_tpm_impactful_p53_mut, pog_tpm_p53_impactful_mut, tcga_tpm_wt_p53, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')

# divide feature matrices to 90% test and 10% validation sets
tcga_X_train, tcga_X_test, tcga_y_train, tcga_y_test = p53.split_90_10(tcga_feature_matrix, tcga_p53_labels)
pog_X_train, pog_X_test, pog_y_train, pog_y_test = p53.split_90_10(pog_feature_matrix, pog_p53_labels)
both_X_train, both_X_test, both_y_train, both_y_test = p53.split_90_10(both_feature_matrix, both_p53_labels)

print('feature matrices and label arrays are ready!')
print('')

# find the hyperparameters resulting in the best performance in a CV analysis
tcga_grid_result = p53hp.findHyperparam(tcga_X_train, tcga_y_train)
pog_grid_result = p53hp.findHyperparam(pog_X_train, pog_y_train)
both_grid_result = p53hp.findHyperparam(both_X_train, both_y_train)

##################################################################################
### Write the Grid Search results for POG to a file and test on validation set ###
##################################################################################

f = open('res/pog_hyper_param_results.txt', 'w')
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

f2 = open('res/tcga_hyper_param_results.txt', 'w')
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

f3 = open('res/both_hyper_param_results.txt', 'w')
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
