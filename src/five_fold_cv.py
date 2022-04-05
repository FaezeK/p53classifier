############################################################################
# This script contains the code for training the RF and predicting the
# samples labels in a 5 fold CV.
############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True)

# function to train and evaluate RF in a 5-fold CV
def predict_5_fold_cv(X, y, max_depth, min_smpl_split, min_smpl_leaf):
    all_pred_df = pd.DataFrame({'expr_sa_ids':['a'], 'p53_status':['mut_wt'], 'predict':['mut_wt']})
    all_prob = []
    true_label_prob = np.empty([0,])

    for train_index, test_index in skf.split(X, y):

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        clf = RandomForestClassifier(n_estimators=3000, max_depth=max_depth, max_features=0.05,
            max_samples=0.99, min_samples_split=min_smpl_split, min_samples_leaf=min_smpl_leaf, n_jobs=40)
        clf.fit(X_train, y_train)

        sample_ids = X_test.index.values

        p53_predictions = clf.predict(X_test)
        p53_pred_df = pd.DataFrame({'expr_sa_ids':sample_ids, 'p53_status':y_test, 'predict':p53_predictions})
        all_pred_df = all_pred_df.append(p53_pred_df, ignore_index=True)

        p53_prob = clf.predict_proba(X_test)
        for i in p53_prob:
            p53_prob_max=max(i)
            all_prob.append(p53_prob_max)

        if clf.classes_[0]=="p53_wt":
            p53_true_prob = p53_prob[:,0]
        else:
            p53_true_prob = p53_prob[:,1]
        true_label_prob = np.concatenate((true_label_prob, p53_true_prob), axis=0)

    all_pred_df = all_pred_df[all_pred_df.expr_sa_ids != 'a']

    return all_pred_df, all_prob, true_label_prob


# function to make AUROC graphs
def generate_auroc_curve(tru_p53_stat, tru_lab_prob, pos):
    data_fpr, data_tpr, data_thresholds = sklearn.metrics.roc_curve(tru_p53_stat, tru_lab_prob, pos_label="p53_wt")
    data_fpr_tpr = pd.DataFrame({'fpr':data_fpr, 'tpr':data_tpr})
    sns.lineplot(data=data_fpr_tpr, x='fpr', y='tpr', ax=pos)
    pos.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    pos.plot([0, 1], [0, 1], color='black', ls='--')


# function to make AUPRC graphs
def generate_auprc_curve(tru_p53_stat, tru_lab_prob, pos):
    data_prcsn, data_rcll, data_thrshlds = sklearn.metrics.precision_recall_curve(tru_p53_stat, tru_lab_prob, pos_label="p53_wt")
    data_prcsn_rcll = pd.DataFrame({'prcsn':data_prcsn, 'rcll':data_rcll})
    sns.lineplot(data=data_prcsn_rcll, x='rcll', y='prcsn', ax=pos)
    pos.set(xlabel='Recall', ylabel='Precision')
    pos.plot([0, 1], [1, 0], color='black', ls='--')


# function to evaluate RF across 33 TCGA cancer types
def performance_tcga_cancer_types(pred_df, type_df):
    pred_df = pred_df.rename(columns={"expr_sa_ids": "sample_id"})
    all_pred_w_type = pd.merge(pred_df, type_df, on='sample_id')

    tcga_cancer_types = all_pred_w_type.type.unique()
    tcga_cancer_types_measures = pd.DataFrame({'type':['a'], 'num_mut':[-1], 'num_wt':[-1], 'precision':[-0.1], 'recall':[-1.0], 'f1_score':[-0.1], 'accuracy':[-0.1]})

    for i in tcga_cancer_types:
        df = all_pred_w_type[all_pred_w_type.type == i]
        measures = sklearn.metrics.precision_recall_fscore_support(df.p53_status, df.predict, pos_label='p53_wt', average='macro')
        numerator = df[df.p53_status == df.predict].shape[0]
        accr = numerator / df.shape[0]

        cancer_type = i
        num_mut_smpl = df[df.p53_status == 'p53_mut'].shape[0]
        num_wt_smpl = df[df.p53_status == 'p53_wt'].shape[0]
        prcsn = measures[0]
        rcll = measures[1]
        f1scr = measures[2]

        tcga_type_and_measure = pd.DataFrame({'type':[i], 'num_mut':num_mut_smpl, 'num_wt':num_wt_smpl, 'precision':prcsn, 'recall':rcll, 'f1_score':f1scr, 'accuracy':accr})
        tcga_cancer_types_measures = tcga_cancer_types_measures.append(tcga_type_and_measure, ignore_index=True)

    tcga_cancer_types_measures = tcga_cancer_types_measures[tcga_cancer_types_measures.type != 'a']

    return tcga_cancer_types_measures
