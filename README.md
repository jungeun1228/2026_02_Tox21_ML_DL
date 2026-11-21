# 2026_02_Tox21_ML_DL

Three traditional machine learning models (logistic regression / random forest / lightGBM) and a neural network model were compared for their performance to predict bioassay activity based on Tox21 dataset. 

### Rendered Notebooks
> If GitHub notebook preview fails, please use the NBViewer links below.
- [Tox21_Traditional ML](https://nbviewer.org/github/jungeun1228/2026_02_Tox21_ML_DL/blob/main/2026_Tox21_Traditional_ML.ipynb)
- [Tox21_DL](https://nbviewer.org/github/jungeun1228/2026_02_Tox21_ML_DL/blob/main/2026_Tox21_DL.ipynb)


### Details of model structure
**input:** Morgan fingerprint and RDKit descriptors  
**output:** bioassay activity (active/inactive; 12 bioassays)  
**performance measures:** PR-AUC, ROC-AUC

1. Machine learning
- per-endpoint binary classification across 12 Tox21 assays (logistic regression / random forest / LightGBM)
- structure: Random/Scaffold baseline comparisons within the outer training set; scaffold-based outer hold-out split; 5-fold StratifiedGroupKFold inner CV for RF tuning; final evaluation on the held-out test set.

The modularized Traditional ML notebook is located at:
notebooks/Tox21_Traditional_ML.ipynb

The pipeline performs data preparation, feature generation, outer scaffold splitting, baseline comparisons, RF hyperparameter tuning, and final hold-out evaluation.

#### Project Structure

```text
notebooks/                # Experiment notebooks
src/tox21_ml/
    data.py               # SDF loading and data cleaning
    features.py           # Morgan FP and RDKit descriptors
    splitting.py          # Scaffold-based hold-out splitting
    experiments.py        # Random/Scaffold baseline experiments
    tuning.py             # RF hyperparameter tuning with inner CV
    evaluation.py         # Final hold-out evaluation
data/raw/                 # Raw Tox21 SDF data (not tracked by Git)
results/                  # Tuning checkpoints and evaluation results
pyproject.toml
```

#### Setup and Execution

The Traditional ML pipeline was developed and tested with Python 3.12 in a Conda environment (`tox21_ml`).

To create the environment:

```bash
conda create -n tox21_ml python=3.12
conda activate tox21_ml
python -m pip install -r requirements.txt
python -m pip install -e .
```

The package versions in `requirements.txt` reflect the validated local environment.

On Windows, a Conda DLL path workaround may be required when using the VS Code Jupyter kernel. The notebook includes this workaround.

Google Colab setup will be configured separately, depending on the available runtime and package compatibility.

Place `tox21_10k_data_all.sdf` in `data/raw/` before running the notebook.

RF tuning checkpoints are saved to:

`results/rf_roc_checkpoints_shared_holdout_v1/`

Final hold-out evaluation results are saved to:

`results/rf_holdout_results.csv`

Raw data and training checkpoints are excluded from Git and should be backed up separately.





2. Deep learning
- multi-task learning  




### References  
Tox21 challenge: https://tripod.nih.gov/tox21/challenge/data.jsp#  
OpenAI. 2026. GPT-5, ChatGPT model. OpenAI, San Francisco, CA. Available at: ChatGPT (accessed on March 2026).
