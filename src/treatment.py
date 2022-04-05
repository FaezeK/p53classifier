############################################################################
# This script contains the code to analyze treatments efficacy in patients
# with WT p53 copies vs patients with mutated TP53.
############################################################################

import pandas as pd
import numpy as np
import p53_helper as p53h
import treatment_helper as th
from PIL import Image

# read expression datasets
pog_tpm_pr = pd.read_csv('POG_expr_prcssd.txt', delimiter = '\t', header=0)
tcga_tpm_pr = pd.read_csv('TCGA_expr_prcssd.txt', delimiter='\t', header=0)

# read mutation datasets
pog_snv_pr = pd.read_csv('POG_snv_prcssd.txt', delimiter='\t', header=0)
tcga_snv_pr = pd.read_csv('TCGA_snv_prcssd.txt', delimiter='\t', header=0)

# read treatment data
pog_drugs = pd.read_csv('pog_drugs.tsv', delimiter = '\t', header=0)

print('The input files have been read')
print('')

# add num of days on treatment to pog_drugs
pog_drugs_pr = th.process_pog_drugs(pog_drugs)

# make X and y matrices
tcga_tpm_impactful_p53_mut, tcga_tpm_not_impactful_p53_mut, tcga_tpm_wt_p53 = p53h.find_tcga_p53_mut(tcga_tpm_pr, tcga_snv_pr)
pog_tpm_p53_impactful_mut, pog_tpm_p53_not_impact_mut, pog_tpm_p53_wt = p53h.find_pog_p53_mut(pog_tpm_pr, pog_snv_pr)

X, y = p53h.make_X_y_merged(tcga_tpm_impactful_p53_mut, pog_tpm_p53_impactful_mut, tcga_tpm_wt_p53, pog_tpm_p53_wt, 'p53_mut', 'p53_wt')
X = X.set_index('sample_id')

# predict labels in a 5-fold CV
both_p53_all_pred_df = th.predict_labs_5cv(X, y)

# combine prediction and treatment data
pog_drugs = pog_drugs.rename(columns={"participant_project_identifier": "expr_sa_ids"})
pog_drugs_w_pred = pd.merge(both_p53_all_pred_df, pog_drugs, on='expr_sa_ids')

# find the list of all unique drugs
drug_list = th.get_uniq_drugs(pog_drugs_w_pred)

# make boxplots per drug group
# ALK inhibitors
alk_nhbtrs_drugs = drug_list[drug_list.isin(['ALECTINIB','CERITINIB','CRIZOTINIB','LORLATINIB','BRIGATINIB'])]
pog_drugs_w_pred_alk_nhbtrs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(alk_nhbtrs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_alk_nhbtrs, 'ALK inhibitors', 'alk_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_alk_nhbtrs, 'ALK inhibitors', 'alk_nhbtr')

# alkylating agents
nitro_must_drugs = drug_list[drug_list.isin(['CYCLOPHOSPHAMIDE','MELPHALAN','BENDAMUSTINE','IFOSFAMIDE','LOMUSTINE','CARMUSTINE','BUSULFAN',
                                            'PROCARBAZINE','TEMOZOLOMIDE','DACARBAZINE','TRABECTEDIN'])]
pog_drugs_w_pred_nitro_must = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(nitro_must_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_nitro_must, 'AAs', 'alkyl_agnts')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_nitro_must, 'AAs', 'alkyl_agnts')

# anti-CTLA-4
anti_ctla4_drugs = drug_list[drug_list.isin(['IPILIMUMAB','TREMELIMUMAB'])]
pog_drugs_w_pred_anti_ctla4 = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(anti_ctla4_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_anti_ctla4, 'Anti-CTLA-4', 'anti_CTLA4')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_anti_ctla4, 'Anti-CTLA-4', 'anti_CTLA4')

# anti-EGFR
anti_egfr_drugs = drug_list[drug_list.isin(['CETUXIMAB','AFATINIB','ERLOTINIB','GEFITINIB','OSIMERTINIB','PANITUMUMAB','ROCILETINIB'])]
pog_drugs_w_pred_anti_egfr = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(anti_egfr_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_anti_egfr, 'Anti-EGFR', 'anti_EGFR')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_anti_egfr, 'Anti-EGFR', 'anti_EGFR')

# anti-HER2
anti_her2_drugs = drug_list[drug_list.isin(['TRASTUZUMAB','PERTUZUMAB','LAPATINIB'])]
pog_drugs_w_pred_anti_her2 = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(anti_her2_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_anti_her2, 'Anti-HER2', 'anti_HER2')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_anti_her2, 'Anti-HER2', 'anti_HER2')

# anti-PD-1/PD-L1
anti_pdl1_drugs = drug_list[drug_list.isin(['ATEZOLIZUMAB','AVELUMAB','DURVALUMAB','NIVOLUMAB','PEMBROLIZUMAB'])]
pog_drugs_w_pred_anti_pdl1 = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(anti_pdl1_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_anti_pdl1, 'Anti-PD1/PDL1', 'anti_PD1_PDL1')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_anti_pdl1, 'Anti-PD1/PDL1', 'anti_PD1_PDL1')

# anti-VEGF
anti_vegf_drugs = drug_list[drug_list.isin(['BEVACIZUMAB', 'CEDIRANIB','RAMUCIRUMAB'])]
pog_drugs_w_pred_anti_vegf = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(anti_vegf_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_anti_vegf, 'Anti-VEGF', 'anti_VEGF')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_anti_vegf, 'Anti-VEGF', 'anti_VEGF')

# antiandrogens
ntndrgns_drugs = drug_list[drug_list.isin(['BICALUTAMIDE','FLUTAMIDE','ABIRATERONE','ENZALUTAMIDE'])]
pog_drugs_w_pred_ntndrgns = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(ntndrgns_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_ntndrgns, 'Antiandrogens', 'antiandrogens')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_ntndrgns, 'Antiandrogens', 'antiandrogens')

# antiestrogens
ntstrgns_drugs = drug_list[drug_list.isin(['FULVESTRANT','TAMOXIFEN','EXEMESTANE','ANASTROZOLE','LETROZOLE'])]
pog_drugs_w_pred_ntstrgns = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(ntstrgns_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_ntstrgns, 'Antiestrogens', 'antiestrogens')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_ntstrgns, 'Antiestrogens', 'antiestrogens')

# antihyperglycemic agents
metform_drugs = drug_list[drug_list.isin(['METFORMIN'])]
pog_drugs_w_pred_metform = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(metform_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_metform, 'AntiHGs', 'antihyperglycemic')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_metform, 'AntiHGs', 'antihyperglycemic')

# BRAF inhibitors
brf_nhbtrs_drugs = drug_list[drug_list.isin(['DABRAFENIB','VEMURAFENIB'])]
pog_drugs_w_pred_brf_nhbtrs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(brf_nhbtrs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_brf_nhbtrs, 'BRAF Inhibitors', 'BRAF_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_brf_nhbtrs, 'BRAF Inhibitors', 'BRAF_nhbtr')

# bisphosphonate
bsphsphnt_drugs = drug_list[drug_list.isin(['ZOLEDRONIC ACID','PAMIDRONATE','CLODRONATE'])]
pog_drugs_w_pred_bsphsphnt = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(bsphsphnt_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_bsphsphnt, 'Bisphosphonate', 'bisphosphonate')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_bsphsphnt, 'Bisphosphonate', 'bisphosphonate')

# CDK4/6 inhibitors
cdk46_nhbtrs_drugs = drug_list[drug_list.isin(['PALBOCICLIB','RIBOCICLIB'])]
pog_drugs_w_pred_cdk46_nhbtrs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(cdk46_nhbtrs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_cdk46_nhbtrs, 'CDK4/6 Is', 'CDK4_6_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_cdk46_nhbtrs, 'CDK4/6 Is', 'CDK4_6_nhbtr')

# corticosteroids
crtcstrds_drugs = drug_list[drug_list.isin(['PREDNISONE','CORTISONE','FLUDROCORTISONE','HYDROCORTISONE','DEXAMETHASONE'])]
pog_drugs_w_pred_crtcstrds = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(crtcstrds_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_crtcstrds, 'Corticosteroids', 'corticosteroids')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_crtcstrds, 'Corticosteroids', 'corticosteroids')

# cytotoxic antibiotics
cttxc_ntbtcs_drugs = drug_list[drug_list.isin(['DOXORUBICIN','EPIRUBICIN','MITOMYCIN','BLEOMYCIN','DACTINOMYCIN'])]
pog_drugs_w_pred_cttxc_ntbtcs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(cttxc_ntbtcs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_cttxc_ntbtcs, 'CAs', 'cytotox_antibio')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_cttxc_ntbtcs, 'CAs', 'cytotox_antibio')

# epothilones
epothilones_drugs = drug_list[drug_list.isin(['ERIBULIN'])]
pog_drugs_w_pred_epothilones = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(epothilones_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_epothilones, 'Epothilones', 'epothilones')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_epothilones, 'Epothilones', 'epothilones')

# folate antagonists
flt_antag_drugs = drug_list[drug_list.isin(['METHOTREXATE','PEMETREXED','RALTITREXED','LEUCOVORIN'])]
pog_drugs_w_pred_flt_antag = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(flt_antag_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_flt_antag, 'FAs', 'flt_anta')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_flt_antag, 'FAs', 'flt_anta')

# GnRH analouges
grha_drugs = drug_list[drug_list.isin(['GOSERELIN', 'BUSERELIN', 'LEUPROLIDE', 'DEGARELIX'])]
pog_drugs_w_pred_grha = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(grha_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_grha, 'GnRH Analouges', 'GnRH_analouge')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_grha, 'GnRH Analouges', 'GnRH_analouge')

# IDO1 inhibitors
ido1i_drugs = drug_list[drug_list.isin(['BMS-986205'])] # Linrodostat
pog_drugs_w_pred_ido1i = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(ido1i_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_ido1i, 'IDO1 Inhibitor', 'IDO1_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_ido1i, 'IDO1 Inhibitor', 'IDO1_nhbtr')

# mTOR inhibitors
mtori_drugs = drug_list[drug_list.isin(['EVEROLIMUS','TEMSIROLIMUS'])]
pog_drugs_w_pred_mtori = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(mtori_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_mtori, 'mTOR Inhibitors', 'mTOR_inhibitors')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_mtori, 'mTOR Inhibitors', 'mTOR_inhibitors')

# multikinase inhibitors
mltkns_nhbtrs_drugs = drug_list[drug_list.isin(['AXITINIB','CABOZANTINIB','IMATINIB','LENVATINIB','NERATINIB','PAZOPANIB','REGORAFENIB',
                                                'SORAFENIB','SUNITINIB','VANDETANIB','BAY 73-4506','DOVITINIB'])]
pog_drugs_w_pred_mltkns_nhbtrs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(mltkns_nhbtrs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_mltkns_nhbtrs, 'MIs', 'mltkns_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_mltkns_nhbtrs, 'MIs', 'mltkns_nhbtr')

# other monoclonal antibodies
antibody_drugs = drug_list[drug_list.isin(['ALEMTUZUMAB','RITUXIMAB','OLARATUMAB','MONALIZUMAB','MOXR0916','AGS67E','CEMIPLIMAB','HERCEPTIN'])]
pog_drugs_w_pred_antibody = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(antibody_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_antibody, 'Other MAs', 'other_mnclnl_ntbd')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_antibody, 'Other MAs', 'other_mnclnl_ntbd')

# PARP inhibitors
parpi_drugs = drug_list[drug_list.isin(['OLAPARIB','NIRAPARIB','VELIPARIB','RUCAPARIB','TALAZOPARIB'])]
pog_drugs_w_pred_parpi = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(parpi_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_parpi, 'PARP Inhibitors', 'PARP_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_parpi, 'PARP Inhibitors', 'PARP_nhbtr')

# platinum therapies
plat_drugs = drug_list[drug_list.str.contains('PLAT')]
pog_drugs_w_pred_plat = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(plat_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_plat, 'Platinum', 'platinum')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_plat, 'Platinum', 'platinum')

# progestins
prgstns_drugs = drug_list[drug_list.isin(['MEGESTROL','MEDROXYPROGESTERONE'])]
pog_drugs_w_pred_prgstns = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(prgstns_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_prgstns, 'Progestins', 'progestins')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_prgstns, 'Progestins', 'progestins')

# pyrimidine analouges
capec_drugs = drug_list[drug_list.isin(['CAPECITABINE','FLUOROURACIL','CYTARABINE','GEMCITABINE','LONSURF'])]
pog_drugs_w_pred_capec = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(capec_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_capec, 'PAs', 'pyr_ana')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_capec, 'PAs', 'pyr_ana')

# taxanes
taxan_drugs = drug_list[drug_list.isin(['PACLITAXEL','DOCETAXEL','GENETAXYL'])]
pog_drugs_w_pred_taxan = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(taxan_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_taxan, 'Taxanes', 'taxanes')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_taxan, 'Taxanes', 'taxanes')

# topoisomerase inhibitors
tpsmrs1nhbtrs_drugs = drug_list[drug_list.isin(['IRINOTECAN','TOPOTECAN','ETOPOSIDE','ETIRINOTECAN PEGOL'])]
pog_drugs_w_pred_tpsmrs1nhbtrs = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(tpsmrs1nhbtrs_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_tpsmrs1nhbtrs, 'TIs', 'tpsmrs_nhbtr')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_tpsmrs1nhbtrs, 'TIs', 'tpsmrs_nhbtr')

# vinca alkaloids
vnc_lklds_drugs = drug_list[drug_list.isin(['VINORELBINE','VINCRISTINE'])]
pog_drugs_w_pred_vnc_lklds = pog_drugs_w_pred[pog_drugs_w_pred['drug_treatment.drug_list'].str.contains('|'.join(vnc_lklds_drugs))]
th.make_trtmnt_RF_pred_bxplt(pog_drugs_w_pred_vnc_lklds, 'VAs', 'vinca_alkal')
th.make_trtmnt_p53_status_bxplt(pog_drugs_w_pred_vnc_lklds, 'VAs', 'vinca_alkal')

# put the graphs where RF performs better than true label together
rf_images = [Image.open(x) for x in ['treatment_boxplots/platinum_rf.jpg',
                                    'treatment_boxplots/taxanes_rf.jpg']]

rf_min_shape = sorted([(np.sum(i.size), i.size) for i in rf_images])[0][1]
rf_imgs_comb = np.hstack(list(np.asarray(i.resize(rf_min_shape)) for i in rf_images))
rf_imgs_comb = Image.fromarray(rf_imgs_comb)

tru_lab_images = [Image.open(x) for x in ['treatment_boxplots/platinum_tru_lab.jpg',
                                        'treatment_boxplots/taxanes_tru_lab.jpg']]

tru_lab_min_shape = sorted([(np.sum(i.size), i.size) for i in tru_lab_images])[0][1]
tru_lab_imgs_comb = np.hstack(list(np.asarray(i.resize(tru_lab_min_shape)) for i in tru_lab_images))
tru_lab_imgs_comb = Image.fromarray(tru_lab_imgs_comb)

all_four = np.vstack([rf_imgs_comb, tru_lab_imgs_comb])
all_four = Image.fromarray(all_four)
all_four.save('RF_better_clssfctn.jpg', quality=400)

# put the graphs where true label performs better than RF together
epo_images = [Image.open(x) for x in ['treatment_boxplots/epothilones_rf.jpg',
                                    'treatment_boxplots/epothilones_tru_lab.jpg']]

epo_min_shape = sorted([(np.sum(i.size), i.size) for i in epo_images])[0][1]
epo_imgs_comb = np.hstack(list(np.asarray(i.resize(epo_min_shape)) for i in epo_images))
epo_imgs_comb = Image.fromarray(epo_imgs_comb)

epo_imgs_comb.save('TruLab_better_clssfctn.jpg', quality=400)

# put all other graphs together
dir='treatment_boxplots/'

rf_images1 = [Image.open(x) for x in [dir+'alk_nhbtr_rf'+'.jpg',
                                    dir+'alkyl_agnts_rf'+'.jpg',
                                    dir+'anti_CTLA4_rf'+'.jpg']]

tru_lab_images1 = [Image.open(x) for x in [dir+'alk_nhbtr_tru_lab'+'.jpg',
                                           dir+'alkyl_agnts_tru_lab'+'.jpg',
                                           dir+'anti_CTLA4_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images1,tru_lab_images1,'1')

# img2
rf_images2 = [Image.open(x) for x in [dir+'anti_EGFR_rf'+'.jpg',
                                    dir+'anti_HER2_rf'+'.jpg',
                                    dir+'anti_PD1_PDL1_rf'+'.jpg']]

tru_lab_images2 = [Image.open(x) for x in [dir+'anti_EGFR_tru_lab'+'.jpg',
                                           dir+'anti_HER2_tru_lab'+'.jpg',
                                           dir+'anti_PD1_PDL1_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images2,tru_lab_images2,'2')

# img3
rf_images3 = [Image.open(x) for x in [dir+'anti_VEGF_rf'+'.jpg',
                                    dir+'antiandrogens_rf'+'.jpg',
                                    dir+'antiestrogens_rf'+'.jpg']]

tru_lab_images3 = [Image.open(x) for x in [dir+'anti_VEGF_tru_lab'+'.jpg',
                                           dir+'antiandrogens_tru_lab'+'.jpg',
                                           dir+'antiestrogens_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images3,tru_lab_images3,'3')

# img4
rf_images4 = [Image.open(x) for x in [dir+'antihyperglycemic_rf'+'.jpg',
                                    dir+'bisphosphonate_rf'+'.jpg',
                                    dir+'BRAF_nhbtr_rf'+'.jpg']]

tru_lab_images4 = [Image.open(x) for x in [dir+'antihyperglycemic_tru_lab'+'.jpg',
                                           dir+'bisphosphonate_tru_lab'+'.jpg',
                                           dir+'BRAF_nhbtr_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images4,tru_lab_images4,'4')

# img5
rf_images5 = [Image.open(x) for x in [dir+'CDK4_6_nhbtr_rf'+'.jpg',
                                    dir+'corticosteroids_rf'+'.jpg',
                                    dir+'cytotox_antibio_rf'+'.jpg']]

tru_lab_images5 = [Image.open(x) for x in [dir+'CDK4_6_nhbtr_tru_lab'+'.jpg',
                                           dir+'corticosteroids_tru_lab'+'.jpg',
                                           dir+'cytotox_antibio_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images5,tru_lab_images5,'5')

# img6
rf_images6 = [Image.open(x) for x in [dir+'flt_anta_rf'+'.jpg',
                                    dir+'GnRH_analouge_rf'+'.jpg',
                                    dir+'IDO1_nhbtr_rf'+'.jpg']]

tru_lab_images6 = [Image.open(x) for x in [dir+'flt_anta_tru_lab'+'.jpg',
                                           dir+'GnRH_analouge_tru_lab'+'.jpg',
                                           dir+'IDO1_nhbtr_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images6, tru_lab_images6, '6')

# img7
rf_images7 = [Image.open(x) for x in [dir+'mltkns_nhbtr_rf'+'.jpg',
                                    dir+'mTOR_inhibitors_rf'+'.jpg',
                                    dir+'other_mnclnl_ntbd_rf'+'.jpg']]

tru_lab_images7 = [Image.open(x) for x in [dir+'mltkns_nhbtr_tru_lab'+'.jpg',
                                           dir+'mTOR_inhibitors_tru_lab'+'.jpg',
                                           dir+'other_mnclnl_ntbd_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images7, tru_lab_images7, '7')

# img8
rf_images8 = [Image.open(x) for x in [dir+'PARP_nhbtr_rf'+'.jpg',
                                    dir+'progestins_rf'+'.jpg',
                                    dir+'pyr_ana_rf'+'.jpg']]

tru_lab_images8 = [Image.open(x) for x in [dir+'PARP_nhbtr_tru_lab'+'.jpg',
                                           dir+'progestins_tru_lab'+'.jpg',
                                           dir+'pyr_ana_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images8, tru_lab_images8, '8')

# img9
rf_images9 = [Image.open(x) for x in [dir+'tpsmrs_nhbtr_rf'+'.jpg',
                                    dir+'vinca_alkal_rf'+'.jpg']]

tru_lab_images9 = [Image.open(x) for x in [dir+'tpsmrs_nhbtr_tru_lab'+'.jpg',
                                            dir+'vinca_alkal_tru_lab'+'.jpg']]

th.make_comb_graph(rf_images9, tru_lab_images9, '9')
