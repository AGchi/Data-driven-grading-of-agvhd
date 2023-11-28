# read libraries
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from functions.grading_functions import calc_p_0, mean_confidence_interval,\
    myplot_biplot, calc_gvhd_score, calc_gvhd_score_simplified
import seaborn as sns
import os.path
import yaml
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# folder with config files
CONFIG_PATH = "../config/"

# load configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as config_file:
        config = yaml.safe_load(config_file)
    return config
grading_config = load_config('grading.yaml')

# parameter settings
scaling = grading_config['others']['scaling']
skin_stage = grading_config['columns']['skin_stage']
liver_stage = grading_config['columns']['liver_stage']
gi_stage = grading_config['columns']['gi_stage']
has_aGVHD = grading_config['columns']['has_gvhd']
center = grading_config['columns']['center']
clinics = grading_config['others']['clinics']
skin_label = grading_config['plotsettings']['skin_label']
liver_label = grading_config['plotsettings']['liver_label']
gi_label = grading_config['plotsettings']['gi_label']
PC1_label = grading_config['plotsettings']['PC1_label']
PC2_label = grading_config['plotsettings']['PC2_label']
PC3_label = grading_config['plotsettings']['PC3_label']
screeplot_xlabel = grading_config['plotsettings']['screeplot_xlabel']
screeplot_ylabel = grading_config['plotsettings']['screeplot_ylabel']
screeplot_title = grading_config['plotsettings']['screeplot_title']
outpath = grading_config['out']['pca_outpath']
outpath_data = grading_config['out']['data_outpath']
ML_stages_PC1 = grading_config['columns_out']['ML_stages_PC1']
ML_stages_simplified_PC1 = grading_config['columns_out'][
    'ML_stages_simplified_PC1']

clinical_data = os.path.join(grading_config['dataset']['clinical_data_path'],
                             grading_config['dataset']['clinical_data_name'])

clinical_data_df = pd.read_excel(clinical_data)
print("Number of rows in dataset: ", len(clinical_data_df))
print(clinical_data_df.info())

# Create Output Structure for PCA
outname_tag = '_'.join([str(item) for item in clinics])+"_train"
OUTDIR = os.path.join(outpath, outname_tag+'/')
if scaling:
    OUTDIR = os.path.join(OUTDIR, "scaled"+'/')

CHECK_FOLDER = os.path.isdir(OUTDIR)
if not CHECK_FOLDER:
    os.makedirs(OUTDIR)
    print("new folder: ", OUTDIR)
else:
    print(OUTDIR, "already exists")

# Create Output Structure for PCA

OUTDIR_DATA = os.path.join(outpath_data, outname_tag+'/')
if scaling:
    OUTDIR_DATA = os.path.join(OUTDIR_DATA, "scaled"+'/')
CHECK_FOLDER = os.path.isdir(OUTDIR_DATA)
if not CHECK_FOLDER:
    os.makedirs(OUTDIR_DATA)
    print("new folder: ", OUTDIR_DATA)
else:
    print(OUTDIR_DATA, "already exists")


# Filter data by gvhd and clinics
# clinical data with agvhd
clinical_data_df_gvhd = clinical_data_df.loc[
    clinical_data_df[has_aGVHD] != 0]
# clinical data, which is not defined as training data
clinical_data_df_gvhd_testdata = clinical_data_df_gvhd.loc[
    ~clinical_data_df_gvhd[center].isin(clinics)]
# clinical data, which is defined as training data
clinical_data_df_gvhd_traindata = clinical_data_df_gvhd.loc[
    clinical_data_df_gvhd[center].isin(clinics)]
# select skin, liver and gi stages
clinics_slg = clinical_data_df_gvhd_traindata.loc[
              :, [skin_stage, liver_stage, gi_stage]].values
clinics_slg_df = clinical_data_df_gvhd_traindata.loc[
              :, [skin_stage, liver_stage, gi_stage]]

# Save Preprocessed Data as Excel Files
print("\nProcess: ---> Save preprocessed data as excel files <---\n")
print(os.path.join(OUTDIR, outname_tag+'.xlsx'))
pd.DataFrame(clinical_data_df_gvhd_traindata).to_excel(
    os.path.join(OUTDIR, outname_tag+'.xlsx'), index=False)
pd.DataFrame(clinics_slg,
             columns=[skin_stage, liver_stage, gi_stage]).to_excel(
             os.path.join(OUTDIR, outname_tag+'_UPNs.xlsx'), index=False)

# Scale data if needed: can be set in pca.yaml
if scaling:
    sc = StandardScaler()
    clinics_slg = sc.fit_transform(clinics_slg)

# Calculate p0 (part of PC grades calculation)
print("\nProcess: ---> Calculate p0 of other_clinics_slg and data2<---\n")
p0_other_clinics_slg = calc_p_0(
    pd.DataFrame(clinics_slg, columns=[skin_stage, liver_stage, gi_stage]),
    skin_stage, liver_stage, gi_stage)
# write result into .txt file
with open(os.path.join(OUTDIR, outname_tag+'_p0_slg.txt'), "w") as f:
    f.write("p0 of clinics_slg")
    f.write(str(p0_other_clinics_slg))
f.close()


# Do PCA
print("\nProcess: ---> Do PCA <---\n")
pca = PCA()
x = pca.fit_transform(clinics_slg)
xdf = pd.DataFrame(data=clinics_slg)
xdf.to_excel(os.path.join(OUTDIR, outname_tag+'_pca.xlsx'), index=False)
clinical_data_df_gvhd_traindata[PC1_label] = [ele[0] for ele in x]
clinical_data_df_gvhd_traindata[PC2_label] = [ele[1] for ele in x]
clinical_data_df_gvhd_traindata[PC3_label] = [ele[2] for ele in x]
clinical_data_df_gvhd_traindata.to_excel(
    os.path.join(OUTDIR, outname_tag+'_pca_withPC1.xlsx'), index=False)


# Creating a score based on aGVHD stages and PCA
print("Length clinics_slg: "+str(len(clinics_slg)))
print("PC variance, n_components=3, "+str(len(clinics_slg))+", all: "+str(
    pca.explained_variance_ratio_))
print("PC1 loadings, n_components=3, "+str(len(clinics_slg))+", all: "+str(
    pca.components_.T[:, 0]))

with open(os.path.join(OUTDIR, outname_tag+'_PCA_results.txt'), "w") as f:
    f.write("Length clinics_slg: "+str(len(clinics_slg))+"\n")
    f.write("c1: PC1 loadings, n_components=3, "+str(
        len(clinics_slg))+", all: "+str(pca.components_.T[:, 0])+"\n")
    f.write(
        "PC2 loadings, n_components=3, "+str(len(clinics_slg))+", all: "+str(
            pca.components_.T[:, 1])+"\n")
    f.write(
        "PC3 loadings, n_components=3, "+str(len(clinics_slg))+", all: "+str(
            pca.components_.T[:, 2])+"\n")
    f.write(
        "PC variance, n_components=3, "+str(len(clinics_slg))+", all: "+str(
            pca.explained_variance_ratio_)+"\n")
    f.write(
        "PC Eigenvalues, n_components=3, "+str(len(clinics_slg))+", all: "+str(
            pca.explained_variance_))
f.close()

p0_other_clinics_slg = calc_p_0(pd.DataFrame(clinics_slg,
                                columns=[skin_stage, liver_stage, gi_stage]),
                                skin_stage, liver_stage, gi_stage)


# Do PCA Bootstrapping
print("\nProcess: ---> Do PCA Bootstrapping <---\n")
bootstrap_array_loadings = []
bootstrap_array_PC1_eigenvalues = []
bootstrap_array_PC2_eigenvalues = []
bootstrap_array_PC3_eigenvalues = []

# calculate sample number: 2/3 of overall training cohort
no_samples = int(round(((len(clinics_slg)/3)*2), 0))


# set seed to reproduce bootstrapping
np.random.seed(42)
for i in range(500):
    boot = resample(clinics_slg, replace=True, n_samples=no_samples)
    # out of bag observations
    oob = [r for r in clinics_slg if r not in boot]
    pca = PCA()
    x_pca = pca.fit_transform(boot)
    bootstrap_array_loadings.append(pca.explained_variance_ratio_)
    bootstrap_array_PC1_eigenvalues.append(pca.explained_variance_[0])
    bootstrap_array_PC2_eigenvalues.append(pca.explained_variance_[1])
    bootstrap_array_PC3_eigenvalues.append(pca.explained_variance_[2])

print("loadings")
print(bootstrap_array_loadings)
print("bootstrap PC1 eigenvalues")
print(bootstrap_array_PC1_eigenvalues)
print("bootstrap PC2 eigenvalues")
print(bootstrap_array_PC2_eigenvalues)


#sns.histplot(bootstrap_array_PC1_eigenvalues, color="royalblue", kde=True)

bootstrap_data=[bootstrap_array_PC1_eigenvalues,
                bootstrap_array_PC2_eigenvalues,
                bootstrap_array_PC3_eigenvalues]

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=10)
fig = plt.figure(figsize=(3.54, 3.54), dpi=600)
ax = fig.add_axes([0.18, 0.1, 0.8, 0.82])
ax.set_ylabel('Eigenvalues', fontsize=15)
ax.set_title('Bootstrapping', fontsize=15)
ax.set_xticklabels(['PC1', 'PC2', 'PC3'])
ax.boxplot(bootstrap_data)
plt.grid()
plt.savefig(os.path.join(OUTDIR, outname_tag+"_bootstrapping.png"))
plt.close()

print(
    mean_confidence_interval(bootstrap_array_PC1_eigenvalues, confidence=0.95))
print(
    mean_confidence_interval(bootstrap_array_PC2_eigenvalues, confidence=0.95))
print(
    mean_confidence_interval(bootstrap_array_PC3_eigenvalues, confidence=0.95))

with open(os.path.join(OUTDIR, outname_tag+'_PCA_bootstrapping_results.txt'),
          "w") as f:
    f.write("bootstrapping results\n")
    f.write("             mean-----------------min------------------max\n")
    f.write("PC1: "+str(
        mean_confidence_interval(bootstrap_array_PC1_eigenvalues,
            confidence=0.95))+"\n")
    f.write("PC2: "+str(
        mean_confidence_interval(bootstrap_array_PC2_eigenvalues,
            confidence=0.95))+"\n")
    f.write("PC3: "+str(
        mean_confidence_interval(bootstrap_array_PC3_eigenvalues,
            confidence=0.95)))
f.close()
print("\nProcess: ---> Create PCA Plots: Biplot and Scree Plot <---\n")

# Explained variance
pca = PCA()
pca = PCA().fit(clinics_slg)
print("Length clinics_slg: "+str(len(clinics_slg)))
print("PC variance, n_components=3, n="+str(len(clinics_slg))+", all: "+str(
    pca.explained_variance_ratio_))
print("eigenvalues, n_components=3, n="+str(
    len(clinics_slg))+", no outlier: "+str(pca.explained_variance_))
print("PC1 loadings, n_components=3, n="+str(len(clinics_slg))+", all: "+str(
    pca.components_))

# biplot
pca_bi = PCA()
x_biplot = pca_bi.fit_transform(clinics_slg)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams["figure.figsize"] = (3.54, 3.54)
plt.figure(constrained_layout=True)
myplot_biplot(x_biplot[:, 0:2],
              np.transpose(pca_bi.components_[0:2, :]),
              -1.1, 1.1, -0.5, 1.1,
              labels=[skin_label, liver_label, gi_label])
plt.grid()
plt.savefig(os.path.join(OUTDIR, outname_tag+'_biplot.png'), dpi=600)
plt.close()

# Scree Plot, bars
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=10)
plt.rcParams["figure.figsize"] = (3.54, 3.54)
plt.figure(constrained_layout=True)
sk_pca = PCA(n_components=3, random_state=234)
sk_pca.fit(clinics_slg)
dset2 = pd.DataFrame()
dset2['pca'] = range(1, 4)
dset2['vari'] = pd.DataFrame(pca_bi.explained_variance_ratio_)

graph = sns.barplot(x='pca', y='vari', data=dset2, color="#a6cee3", ci=95,
                    palette="Blues_d")
for p in graph.patches:
    graph.annotate('{:.2f}'.format(p.get_height()),
                   (p.get_x()+0.28, p.get_height()),
                   ha='left', va='bottom',
                   color='black', fontsize=10)

plt.xticks([0, 1, 2], [PC1_label, PC2_label, PC3_label])
plt.yticks(np.arange(0, max(sk_pca.explained_variance_ratio_)+0.1, 0.1))
plt.ylabel(screeplot_ylabel, fontsize=15)
plt.xlabel(screeplot_xlabel, fontsize=10)
plt.title(screeplot_title, fontsize=16)
plt.savefig(os.path.join(OUTDIR, outname_tag+'_screePlotBars.png'), dpi=600)
plt.close()


# Create new table with ML Grading Column
p_0 = [round(num, 2) for num in p0_other_clinics_slg]
p_0 = p0_other_clinics_slg
c_1 = [round(num, 2) for num in pca.components_.T[:, 0]]
c_2 = [round(num, 2) for num in pca.components_.T[:, 1]]
c_3 = [round(num, 2) for num in pca.components_.T[:, 2]]

# calculate severity scores for the whole cohort (test and train) and create
# table with organ-stages, modified glucksberg and calculated severity score
gvhd_data_with_new_gvhd_grade_train = calc_gvhd_score(
    clinical_data_df_gvhd_traindata, p_0, c_1, skin_stage, liver_stage, gi_stage,
    ML_stages_PC1)
gvhd_data_with_new_gvhd_grade_train = calc_gvhd_score_simplified(
    gvhd_data_with_new_gvhd_grade_train, p_0, c_1, skin_stage, liver_stage,
    gi_stage, ML_stages_simplified_PC1)
gvhd_data_with_new_gvhd_grade_train.to_excel(
    os.path.join(OUTDIR_DATA, outname_tag+'_ML_stages_train.xlsx'),
    index=False)

# calculate severity scores for the whole cohort (test and train) and create
# table with organ-stages, modified glucksberg and calculated severity score
gvhd_data_with_new_gvhd_grade_overall = calc_gvhd_score(clinical_data_df_gvhd,
                                                        p_0, c_1, skin_stage,
                                                        liver_stage, gi_stage,
                                                        ML_stages_PC1)
gvhd_data_with_new_gvhd_grade_overall = calc_gvhd_score_simplified(
    gvhd_data_with_new_gvhd_grade_overall, p_0, c_1, skin_stage, liver_stage,
    gi_stage, ML_stages_simplified_PC1)
gvhd_data_with_new_gvhd_grade_overall.to_excel(
    os.path.join(OUTDIR_DATA, outname_tag+'_ML_stages_overall.xlsx'),
    index=False)


# calculate severity scores for the whole cohort (test and train) and create
# table with organ-stages, modified glucksberg and calculated severity score
gvhd_data_with_new_gvhd_grade_test = calc_gvhd_score(
    clinical_data_df_gvhd_testdata, p_0, c_1, skin_stage, liver_stage,
    gi_stage, ML_stages_PC1)
gvhd_data_with_new_gvhd_grade_test = calc_gvhd_score_simplified(
    gvhd_data_with_new_gvhd_grade_test, p_0, c_1, skin_stage, liver_stage,
    gi_stage, ML_stages_simplified_PC1)
gvhd_data_with_new_gvhd_grade_test.to_excel(
    os.path.join(OUTDIR_DATA, outname_tag+'_ML_stages_test.xlsx'), index=False)


