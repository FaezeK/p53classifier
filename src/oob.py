############################################################################
# This script contains the code for training the RF using the samples in
# each cancer type only (using OOB approach to train the RF)
############################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# function to calculate oob score for each cancer type
def evaluate_RF_on_each_cancer_type(type_df, tcga_tpm_df, tcga_snv):
    tcga_cancer_types = type_df.type.unique()

    tcga_cancer_types_oob = pd.DataFrame({'type':['a'], 'oob_score':[-0.1]})

    for i in tcga_cancer_types:
        samples_in_i = all_pred_w_type[all_pred_w_type.type == i].sample_id
        tcga_tpm_i = tcga_tpm_df[tcga_tpm_df.sample_id.isin(samples_in_i)]

        tcga_snv_i = tcga_snv[tcga_snv.donor_id.isin(samples_in_i)]
        tcga_snv_fltr_i = tcga_snv_i[tcga_snv_i.gene_id == "TP53"]

        tcga_impactful_p53_mut_samples_i = tcga_snv_fltr_i.donor_id.unique()
        all_tcga_snv_samples_i = tcga_snv_i.donor_id.unique()
        tcga_wt_p53_samples_i = all_tcga_snv_samples_i[np.isin(all_tcga_snv_samples_i, tcga_impactful_p53_mut_samples_i)==False]

        tcga_tpm_impactful_p53_mut_i = tcga_tpm_i[tcga_tpm_i.sample_id.isin(tcga_impactful_p53_mut_samples_i)]
        tcga_tpm_wt_p53_i = tcga_tpm_i[tcga_tpm_i.sample_id.isin(tcga_wt_p53_samples_i)]

        X_i=pd.concat([tcga_tpm_impactful_p53_mut_i, tcga_tpm_wt_p53_i])
        X_i = X_i.set_index('sample_id')
        X_i = X_i.drop(columns=['donor_id'])
        y_i=pd.concat([pd.Series(['p53_mut']*tcga_tpm_impactful_p53_mut_i.shape[0]),pd.Series(['p53_wt']*tcga_tpm_wt_p53_i.shape[0])])

        clf = RandomForestClassifier(n_estimators=3000, max_depth=50, max_features=0.05, max_samples=0.99, min_samples_split=2, min_samples_leaf=2, oob_score=True, n_jobs=40)
        clf.fit(X_i, y_i)
        oob_score_i = clf.oob_score_

        tcga_type_and_oob = pd.DataFrame({'type':[i], 'oob_score':oob_score_i})
        tcga_cancer_types_oob = tcga_cancer_types_oob.append(tcga_type_and_oob, ignore_index=True)

    tcga_cancer_types_oob = tcga_cancer_types_oob[tcga_cancer_types_oob.type != 'a']

    return tcga_cancer_types_oob


# function to make an abbreviation table for tcga cancer types
def make_abbv(type_df):
    tcga_cancer_types = type_df.type.unique()

    cancer_type_abbv_df = pd.DataFrame({'type':tcga_cancer_types, 'abbv':['abbv']*len(tcga_cancer_types)})
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Esophageal carcinoma '] = 'ESCA'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Lung squamous cell carcinoma'] = 'LUSC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Head and Neck squamous cell carcinoma'] = 'HNSC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Liver hepatocellular carcinoma'] = 'LIHC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Ovarian serous cystadenocarcinoma'] = 'OV'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Lung adenocarcinoma'] = 'LUAD'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Sarcoma'] = 'SARC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Brain Lower Grade Glioma'] = 'LGG'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Colon adenocarcinoma'] = 'COAD'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Stomach adenocarcinoma'] = 'STAD'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Glioblastoma multiforme'] = 'GBM'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Bladder Urothelial Carcinoma'] = 'BLCA'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Rectum adenocarcinoma'] = 'READ'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Pancreatic adenocarcinoma'] = 'PAAD'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Prostate adenocarcinoma'] = 'PRAD'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Testicular Germ Cell Tumors'] = 'TGCT'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Breast invasive carcinoma'] = 'BRCA'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Uterine Corpus Endometrial Carcinoma'] = 'UCEC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Cholangiocarcinoma'] = 'CHOL'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Kidney renal papillary cell carcinoma'] = 'KIRP'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Lymphoid Neoplasm Diffuse Large B-cell Lymphoma'] = 'DLBC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Uterine Carcinosarcoma'] = 'UCS'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Cervical squamous cell carcinoma and endocervi...'] = 'CESC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Kidney renal clear cell carcinoma'] = 'KIRC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Acute Myeloid Leukemia'] = 'LAML'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Adrenocortical carcinoma'] = 'ACC'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Skin Cutaneous Melanoma'] = 'SKCM'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Mesothelioma'] = 'MESO'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Kidney Chromophobe'] = 'KICH'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Thyroid carcinoma'] = 'THCA'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Pheochromocytoma and Paraganglioma'] = 'PCPG'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Uveal Melanoma'] = 'UVM'
    cancer_type_abbv_df.abbv[cancer_type_abbv_df.type==' Thymoma'] = 'THYM'

    return cancer_type_abbv_df
