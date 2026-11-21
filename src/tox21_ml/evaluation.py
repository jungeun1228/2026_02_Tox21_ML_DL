"""Final Tox21 Random Forest evaluation on the outer scaffold hold-out split.

This preserves the evaluation logic of the original notebook and does not
read or write checkpoint files. The caller controls which test data is used.

Source: Tox21_Traditional_ML_v2_2026_11_19_seed_setup.ipynb, code cell 122.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_on_holdout(
    train_df,
    test_df,
    X_train_all,
    X_test_all,
    target_cols,
    best_params_by_target,
    *,
    random_state,
    rf_n_jobs=-1,
):
    if isinstance(X_train_all, pd.DataFrame):
        X_train_all = X_train_all.to_numpy()
    if isinstance(X_test_all, pd.DataFrame):
        X_test_all = X_test_all.to_numpy()

    rows = []
    fitted_models = {}

    for target in target_cols:
        if target not in best_params_by_target:
            print(f"SKIP {target}: no tuned params")
            continue

        # train mask
        tr_mask = train_df[target].notna()
        te_mask = test_df[target].notna()

        X_tr = X_train_all[tr_mask.values]
        y_tr = train_df.loc[tr_mask, target].astype(int).values
        X_te = X_test_all[te_mask.values]
        y_te = test_df.loc[te_mask, target].astype(int).values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"SKIP {target}: one class only")
            continue

        params = best_params_by_target[target]
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=rf_n_jobs,
            **params
        )
        model.fit(X_tr, y_tr)
        fitted_models[target] = model

        y_prob = model.predict_proba(X_te)[:, 1]
        roc = roc_auc_score(y_te, y_prob)
        pr = average_precision_score(y_te, y_prob)

        rows.append({
            "target": target,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "test_positive_ratio": y_te.mean(),
            "test_roc_auc": roc,
            "test_pr_auc": pr,
            "best_params": json.dumps(params, ensure_ascii=False),
        })

        print(f"{target}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    holdout_df = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    return holdout_df, fitted_models
