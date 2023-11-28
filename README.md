## PCA-based grading of aGVHD

Usage information for the source code used to leverage a PCA-based approach to classify aGVHD severity grades using organ stages-data as presented in the reference article 
Bayraktar, E., Graf, T., Ayuk, F.A. et al. Data-driven grading of acute graft-versus-host disease. Nature Communications 14, 7799 (2023). https://doi.org/10.1038/s41467-023-43372-2

**This code is for research purposes only.**

This readme contains:

1. Short description of the main scripts
2. Information about programming languages and requiered packages and how to install them.
3. Instructions for running the scripts on simulated demo data.

**The simulation data is only used as an example to run the code and does not necessarily represent and behave like the data shown in the publication.**

##########################################  
##########################################  
##########################################  

#### INTRODUCTION

Application for aGVHD grading using principal component analysis (PCA).
The PCA output is adapted for the calculation of aGVHD severity scores which are then converted into aGVHD grades.

**1. SCRIPT DESCRIPTIONS**

- grading_analysis.py (grading_functions.py): PCA-based grading
- stages_to_grades.py: Example of manual conversion of pca-based scores to aGVHD grades
   
##########################################  
##########################################  
##########################################  


**2. PREREQUISITES**

Requirements:

python (3.10.2)
libraries:
- pandas (v1.4.1)
- numpy (v1.22.2)
- seaborn (v0.11.2)
- matplotlib (v3.5.1)
- PyYAML (v6.0)
- scikit-learn (v1.0.2)
- scipy (v1.8.0)
- os (python standard library)
- datetime (python standard library)

**References to the libraries are listed in the publication**.  

Installation guide:
Python and related libraries must be installed. 
Then, the script can be started in the recommended order.

DATA

clinical data as excel-files (.xlsx) which contain the following information in the respective columns:  
aGVHD grading (grading_analysis.py):  
- skin-stages (0-4)
- liver-stages (0-4)
- gastro-intestinal (GI)-stages (0-4)
- transplant center (string)
- 1/0 encoded column: 1: has gvhd, 0: no gvhd
	    
##########################################  
##########################################  
##########################################  


**3. DEMO RUN WITH SIMULATED TEST DATA**  
**RUNTIME < 5 minutes**


PCA analysis  
RUNTIME: 30 seconds  
IN: clincial data as excel files (.xlsx) with specific columns, see above.  
OUT: excel file (.xlsx)  

grading_analysis.py to do pca analysis.  

1. Go to config/grading.yaml file:
	If you use our simulated dataset, nothing needs to be changed in the configuration.  
	
	Set parameter if you use your own data set.  
	"dataset", "others", "columns" and "out" have to be defined to run the code.  
	"columns_out" and "plotsettings" only if required.  

2. After configuration, the script src/grading_analysis.py can be started.  

##########################################################################

Convert pca output to PC1 derived grades:  
RUNTIME: few seconds  
IN: excel file (.xlsx)  
OUT: excel file (.xlsx)  

stages_to_grades.py to convert pca stages to pca_grades. 

Plese not that due to the artificial data structure of the provided simulation data, the PC1 derived classification is yielding to a different spread of stages and its respective combination into aGVHD grades than in the reference article. Using the simulation data, stages are combined as follows:  
Stages 1-4 -> Simulation grade I,  5-8 -> Simulation grade II, 9-11 -> Simulation grade III, 12-14 -> Simulation grade IV  The present conifguration is already adapted to provide the appropriate output with the simulated dataset.  

The original PC1 derived classification spreads from 1 to 12 stages, which are combined as follows in PC1 grading using 4 severity grades:  
Stages 1-3 -> PC1 grade I,  4-6 -> PC1 grade II, 7-9 -> PC1 grade III, >9 -> PC1 grade IV  

1. Go to config/stages_to_grades.yaml file:  
The way the stages will be combined can be set under "ML_Grading".  
Under "dataset" you can define the input file, which is one of the output files of previous grading_analysis.py run.  
Under "out"  you can define the outputpath and the name of the output excel file.

2. After configuration, the script src/stages_to_grades.py can be started.

The output files contain the following PC1-derived aGVHD grading categories:  
"PC1_stages" (corresponding to "PC1 12 grades" in the reference publication)  
"PC1_stages_simplified" (corresponding to "PC1 6 grades" in the reference publication)  
"ML_PC1_grades" (corresponding to "PC1 4 grades" in the reference publication)  


Reference publication:
Bayraktar, E., Graf, T., Ayuk, F.A. et al. Data-driven grading of acute graft-versus-host disease. Nature Communications 14, 7799 (2023). https://doi.org/10.1038/s41467-023-43372-2




