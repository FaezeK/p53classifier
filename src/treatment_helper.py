############################################################################
# This script contains functions required in the treatment.py script.
############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from datetime import date
from statannot import add_stat_annotation
from PIL import Image

def process_pog_drugs(pog_drugs):
    pog_drugs = pog_drugs.iloc[:,[1,2,6,7,8,10,12,13,14]]
    num_days_on_treatment = []

    for i in range(len(pog_drugs.id)):
        diff_date = date.fromisoformat(pog_drugs['drug_treatment.course_end_on'][i]) - date.fromisoformat(pog_drugs['drug_treatment.course_begin_on'][i])
        num_days_on_treatment.append(diff_date.days)

    pog_drugs['num_days_on_treatment'] = num_days_on_treatment

    # remove entries with the same start and end dates
    pog_drugs = pog_drugs[pog_drugs.num_days_on_treatment != 0]

    return pog_drugs


def get_uniq_drugs(pog_drugs_w_pred):
    drug_list = []
    for j in pog_drugs_w_pred['drug_treatment.drug_list']:
        for k in j.split(','):
            k = (k.split("'")[1]).split("'")[0]
            drug_list.append(k)

    drug_list = pd.Series((pd.Series(drug_list)).unique())

    # remove placebo from pog data
    drug_list = drug_list[-drug_list.str.contains('PLACEBO')]

    return drug_list


# functions to make the boxplots
def make_trtmnt_RF_pred_bxplt(drug_data, drug_group, file_name):
    sns.set_theme()
    sns.set_style("white")
    drug_fig = sns.boxplot(data = drug_data, x = "predict", y = "num_days_on_treatment", order=['p53_wt', 'p53_mut'], palette="Paired")
    drug_fig.set_yscale("log")
    drug_fig.set(xlabel='Prediction by Random Forest', ylabel='Log10 of Number of days on '+drug_group)
    add_stat_annotation(drug_fig, data=drug_data, x="predict", y="num_days_on_treatment", order=['p53_wt', 'p53_mut'], box_pairs=[("p53_wt", "p53_mut")], test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
    p53_pred_val_cnts = drug_data.predict.value_counts()
    num_log = np.log(drug_data.num_days_on_treatment)
    if (np.max(num_log) - np.min(num_log)) < 2:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.15
    elif (np.max(num_log) - np.min(num_log)) < 4.5:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.4
    else:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.65
    drug_fig.text(-0.25,text_y_data,'n='+str(p53_pred_val_cnts['p53_wt']),size='small',color='black',weight='semibold')
    drug_fig.text(1.03,text_y_data,'n='+str(p53_pred_val_cnts['p53_mut']),size='small',color='black',weight='semibold')
    plt.savefig('results/treatment_boxplots/'+ file_name +'_rf.jpg',format='jpeg',dpi=300,bbox_inches='tight')


def make_trtmnt_p53_status_bxplt(drug_data, drug_group, file_name):
    drug_fig2 = sns.boxplot(data=drug_data, x = "p53_status", y = "num_days_on_treatment", order=['p53_wt', 'p53_mut'], palette="Reds")
    drug_fig2.set_yscale("log")
    drug_fig2.set(xlabel='Real p53 status', ylabel='Log10 of Number of days on '+drug_group)
    add_stat_annotation(drug_fig2, data=drug_data, x="p53_status", y="num_days_on_treatment", order=['p53_wt', 'p53_mut'], box_pairs=[("p53_wt", "p53_mut")], test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
    p53_stat_val_cnts = drug_data.p53_status.value_counts()
    num_log = np.log(drug_data.num_days_on_treatment)
    if (np.max(num_log) - np.min(num_log)) < 2:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.15
    elif (np.max(num_log) - np.min(num_log)) < 4.5:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.4
    else:
        text_y_data = np.max(drug_data.num_days_on_treatment) * 1.65
    drug_fig2.text(-0.25,text_y_data,'n='+str(p53_stat_val_cnts['p53_wt']),size='small',color='black',weight='semibold')
    drug_fig2.text(1.03,text_y_data,'n='+str(p53_stat_val_cnts['p53_mut']),size='small',color='black',weight='semibold')
    plt.savefig('results/treatment_boxplots/'+ file_name +'_tru_lab.jpg',format='jpeg',dpi=300,bbox_inches='tight')


# function to combine the boxplots
def make_comb_graph(rf_imgs, tl_imgs, i):
    rf_min_shape = sorted([(np.sum(i.size), i.size) for i in rf_imgs])[0][1]
    rf_imgs_comb = np.hstack(list(np.asarray(i.resize(rf_min_shape)) for i in rf_imgs))
    rf_imgs_comb = Image.fromarray(rf_imgs_comb)

    tru_lab_min_shape = sorted([(np.sum(i.size), i.size) for i in tl_imgs])[0][1]
    tru_lab_imgs_comb = np.hstack(list(np.asarray(i.resize(tru_lab_min_shape)) for i in tl_imgs))
    tru_lab_imgs_comb = Image.fromarray(tru_lab_imgs_comb)

    all_six = np.vstack([rf_imgs_comb, tru_lab_imgs_comb])
    all_six = Image.fromarray(all_six)
    all_six.save('results/RF_and_TruLab_same'+i+'.jpg', quality=400)
