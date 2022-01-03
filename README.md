# brain_tumor_diagnosis
This project includes all the MRI data and codes of the paper. The main contents are as follows:
## data folder
* See clinical_imform.xlsx for clinical data and original classification category labels;
* The results of feature extraction for all sequences (T1W, T2W, CE-T1W, ADC, ASL, SWI) using the PyRadiomics package are shown in extracted_features.xlsx;
* The original MRI images of all patients are available at https://zenodo.org/record/5815017#.YdL2uIgzZPY, and it is still being added.
## code folder
* The code for feature extraction using the PyRadiomics package is in feature_contract.py, and the configuration file is params.yaml;
* For feature selection, see feature_select.py;
* See classify.py for the training and testing process of SVM model and random forest model;
* See fig_heatmap.py and fig_feature_bars.py for the codes of part of the pictures in the paper.
