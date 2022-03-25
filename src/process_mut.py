############################################################################
# This script processes mutation data and produces a filtered data frame of
# mutations to be used to generate labels.
############################################################################

def process_POG_mut(pog_snv):
    pog_snv = pog_snv[['pog_id','gene_id','aa_change','effect']]

    HGVSp_pog = pog_snv.aa_change.str.split(pat='/', expand=True)[0] # convert aa_change to match HGVSp
    HGVSp_pog = pd.Series(HGVSp_pog, name='HGVSp')
    pog_snv = pd.concat([pog_snv, HGVSp_pog], axis=1)
    pog_snv = pog_snv.drop(columns=["aa_change"])

    pog_snv = pog_snv.rename(columns={"pog_id": "sample_id"})

    return pog_snv

def process_TCGA_mut(tcga_snv):
    tcga_snv = tcga_snv[['Hugo_Symbol','Tumor_Sample_Barcode','HGVSp','Consequence']] # extract the required columns
    tcga_snv = tcga_snv.rename(columns={"Tumor_Sample_Barcode": "sample_id", "Hugo_Symbol":"gene_id", "Consequence":"effect"})

    return tcga_snv