# 2026_02_Tox21_ML_DL

Three traditional machine learning models (logistic regression / random forest / lightGBM) and a neural network model were compared for their performance to predict bioassay activity based on Tox21 dataset. 

*models:   
-ML: multi-task binary classification (logistic regression / random forest / lightGBM)  
-DL: multi-task learning  
*input: Morgan fingerprint and RDKit descriptors  
*output: bioassay activity (active/inactive; 12 bioassays)  
*Scaffold split ; Nested Cross-Validation (inner CV: hyperparameter tuning , model selection / outer split: evaluation)  
*performance measures: PR-AUC, ROC-AUC, Precision, recall, Recall@precition>0.8  

### References  
Tox21 challenge: https://tripod.nih.gov/tox21/challenge/data.jsp#
OpenAI. 2026. GPT-5, ChatGPT model. OpenAI, San Francisco, CA. Available at: ChatGPT (accessed on March 2026).
