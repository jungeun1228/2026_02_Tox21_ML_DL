# 2026_02_Tox21_ML_DL
Predict bioassay activity in Tox21 dataset using machine learning

Three traditional machine learning models (logistic regression / random forest / lightGBM) and a neural network model were compared for their performance to predict bioassay activity based on Tox21 dataset. 

*models: 
-ML: multi-task binary classification - logistic regression / random forest / lightGBM
-DL: multi-task learning
*input: Morgan fingerprint and RDKit descriptors
*output: bioassay activity (active/inactive; 12 bioassays)
*Scaffold split ; Nested Cross-Validation (inner CV: hyperparameter tuning , model selection / outer split: evaluation)
*performance measures: PR-AUC, ROC-AUC, Precision, recall, Recall@precition>0.8

References
<img width="3385" height="531" alt="image" src="https://github.com/user-attachments/assets/b236deb9-57a3-48ea-9a8a-f31f46542fce" />
