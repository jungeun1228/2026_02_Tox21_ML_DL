"""Baseline multi-endpoint model experiments for the Tox21 project.

These functions retain the original notebook's random/scaffold split,
model fitting, scoring, and return signatures. Pass the seed explicitly from
the notebook (e.g. ``random_state=SEED``). The caller should pass data from
the outer **training** partition, not the reserved hold-out test set.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def run_multitarget_random_split(
    valid_df, X_all, target_cols, base_model, *, random_state
):
    """Run a stratified random 80/20 split separately for each endpoint.

    Parameters
    ----------
    valid_df : pandas.DataFrame
        Rows must be in exactly the same order as those in X_all.
    X_all : numpy.ndarray or pandas.DataFrame or compatible feature matrix
        Feature matrix aligned positionally with valid_df.
    target_cols : list-like
        Endpoint columns in valid_df.
    base_model : sklearn-compatible classifier
        Must implement ``fit`` and ``predict_proba``.
    random_state : int
        Explicit random seed supplied by the calling notebook.

    Returns
    -------
    results_df : pandas.DataFrame
    models : dict
        Fitted model per endpoint.
    """
    results = []
    models = {}

    for target in target_cols:
        mask = valid_df[target].notna()
        X_task = X_all[mask.values]
        y_task = valid_df.loc[mask, target].astype(int).values

        if len(np.unique(y_task)) < 2:
            print(f"SKIP: {target} (only one class)")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_task,
            y_task,
            test_size=0.2,
            random_state=random_state,
            stratify=y_task,
        )

        model = clone(base_model)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "target": target,
                "n_samples": len(y_task),
                "positive_ratio": y_task.mean(),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
            }
        )
        models[target] = model

    return pd.DataFrame(results), models


def make_scaffold(smiles):
    """Compute a Bemis-Murcko scaffold from one SMILES (original behavior)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def run_multitarget_scaffold_split(
    valid_df, X_all, target_cols, base_model, *, random_state
):
    """Run one shared scaffold-group 80/20 split across all endpoints.

    Results and prediction columns follow the notebook's original format.
    The split is made only within the supplied dataset; use ``train_df``
    from the outer hold-out split for baseline experiments.

    Returns
    -------
    results_df : pandas.DataFrame
    predictions_df : pandas.DataFrame
    models : dict
    """
    results = []
    prediction_rows = []
    models = {}

    valid_df = valid_df.copy()
    valid_df["scaffold"] = valid_df["canonical_smiles"].apply(make_scaffold)

    gss = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    global_train_idx, global_test_idx = next(
        gss.split(valid_df, groups=valid_df["scaffold"])
    )

    global_train_index = valid_df.index[global_train_idx]
    global_test_index = valid_df.index[global_test_idx]

    for target in target_cols:
        mask = valid_df[target].notna()
        task_df = valid_df.loc[mask].copy()
        y_task = task_df[target].astype(int).values

        if len(np.unique(y_task)) < 2:
            print(f"SKIP: {target} (only one class)")
            continue

        task_train_mask = task_df.index.isin(global_train_index)
        task_test_mask = task_df.index.isin(global_test_index)

        if task_train_mask.sum() == 0 or task_test_mask.sum() == 0:
            print(f"SKIP: {target} (empty train/test after scaffold split)")
            continue

        X_task = X_all[mask.values]
        X_train = X_task[task_train_mask]
        X_test = X_task[task_test_mask]
        y_train = y_task[task_train_mask]
        y_test = y_task[task_test_mask]

        test_indices = task_df.index[task_test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"SKIP: {target} (train/test has only one class)")
            continue

        model = clone(base_model)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "target": target,
                "n_samples": len(y_task),
                "train_n": len(y_train),
                "test_n": len(y_test),
                "positive_ratio": y_task.mean(),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
            }
        )

        pred_df = pd.DataFrame(
            {
                "target": target,
                "row_index": test_indices,
                "y_true": y_test,
                "y_prob": y_prob,
            }
        )
        pred_df["canonical_smiles"] = task_df.loc[
            test_indices, "canonical_smiles"
        ].values
        pred_df["scaffold"] = task_df.loc[test_indices, "scaffold"].values
        prediction_rows.append(pred_df)

        models[target] = model

    results_df = pd.DataFrame(results)
    predictions_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return results_df, predictions_df, models
