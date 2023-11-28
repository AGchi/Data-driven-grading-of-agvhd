import pandas as pd
import os
import os.path
import yaml

# folder with config files
CONFIG_PATH = "../config/"

# open configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as config_file:
        config = yaml.safe_load(config_file)
    return config

stages_to_grades_config = load_config('stages_to_grades.yaml')

scaling = stages_to_grades_config['others']['scaling']
print(stages_to_grades_config['ML_Grading']['PC1_grades']['columnname'])
ML_stages_data = os.path.join(
    stages_to_grades_config['dataset']['MLstages_data_path'],
    stages_to_grades_config['dataset']['MLstages_data_name'])

if scaling:
    ML_stages_data = os.path.join(
    stages_to_grades_config['dataset']['MLstages_data_path'],
                                                    "scaled",
    stages_to_grades_config['dataset']['MLstages_data_name'])

ML_stages_data_df = pd.read_excel(ML_stages_data)


# if data is scaled, add scaled folder
OUTDIR = stages_to_grades_config['out']['ML_grades_outpath']
if scaling:
    OUTDIR = os.path.join(OUTDIR, "scaled" + '/')

CHECK_FOLDER = os.path.isdir(OUTDIR)
if not CHECK_FOLDER:
    os.makedirs(OUTDIR)
    print("new folder: ", OUTDIR)
else:
    print(OUTDIR, "already exists")

# ML PC1 to grades

ML_PC1_grades = stages_to_grades_config['ML_Grading']['PC1_grades'][
    'columnname']
ML_stages_data_df[ML_PC1_grades] = ML_stages_data_df[
    (stages_to_grades_config['columns']['ML_stages_PC1'])]
for grade_no in range(1, len(
        stages_to_grades_config['ML_Grading']['PC1_grades'])):
    grade = 'grade'+str(grade_no)
    print(grade)
    ML_stages_data_df.loc[ML_stages_data_df.ML_PC1_grades.isin(
        stages_to_grades_config['ML_Grading']['PC1_grades'][
            grade]), ML_PC1_grades] = grade_no


ML_stages_data_df.to_excel(os.path.join(OUTDIR,
    stages_to_grades_config['out']['ML_grades_data_name']), index=False)
