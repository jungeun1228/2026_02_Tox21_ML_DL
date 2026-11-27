"""Development-only scaffold CV experiments for the Tox21 multi-task MLP."""

from time import perf_counter

import numpy as np
import pandas as pd

from .preprocessing import prepare_fold_data
from .splitting import fold_diagnostics, iter_scaffold_cv_indices
from .training import fit_tox21_mlp


def run_multitask_cv(   # 전체 fold에 대해 CV를 수행하는 함수 (기존 모듈을 조합한것)
    train_df,
    X_train_all,
    target_cols,
    fold_df,
    config,
    *,
    verbose=True,
    on_fold_end=None,
): 
    """Train one configuration on the fixed scaffold CV assignments.

    Each fold fits its own scaler, class weights and model. Only development
    data are accepted by design. The validation fold selects the best epoch
    and supplies the CV score; these scores are for model selection, not
    independent final performance estimation.

    All endpoint/fold pairs must have both classes in train and validation.
    Each endpoint is averaged equally over folds, then equally over endpoints.
    Standard deviations are sample SD (ddof=1), not standard errors.
    The same configured training seed is reused for each independently fit model.
    Optional on_fold_end(fold_row, n_splits) runs after each validated fold.
    Exceptions from the callback propagate, allowing early trial termination.
    """
    target_cols = list(target_cols)
    config = dict(config)
    if config.get("selection_metric", "roc_auc") != "roc_auc":
        raise ValueError("This CV workflow selects epochs by ROC-AUC.")
    config.setdefault("selection_metric", "roc_auc")
    config.setdefault("seed", 42)
    config.setdefault("verbose", False)
    if not train_df.index.equals(X_train_all.index):
        raise ValueError("Development molecules and features must be aligned.")
    indices = list(iter_scaffold_cv_indices(fold_df))
    n_splits = len(indices)
    diagnostics = fold_diagnostics(
        train_df, target_cols, fold_df, n_splits=n_splits,
    )
    if not diagnostics["roc_auc_valid"].all():
        raise ValueError("Every endpoint must have both classes in every train/validation fold.")

    fold_rows, task_rows, histories, task_histories = [], [], [], []
    for fold_id, (tr_idx, val_idx) in enumerate(indices):
        start = perf_counter()
        if verbose:
            print(f"Fold {fold_id + 1}/{n_splits} (fold_id={fold_id})")
        X_tr, X_val, y_tr, y_val, _ = prepare_fold_data(
            train_df, X_train_all, target_cols, tr_idx, val_idx,
        )
        model, history, per_task_history, best = fit_tox21_mlp(
            X_tr, y_tr, X_val, y_val, task_names=target_cols, **config,
        )
        best_tasks = per_task_history.loc[
            per_task_history["epoch"].eq(best["best_epoch"])
        ].copy()
        if (best_tasks["task"].tolist() != target_cols
                or best["best_n_valid_tasks"] != len(target_cols)
                or not best_tasks["status"].eq("ok").all()
                or not np.isfinite(best_tasks[["roc_auc", "ap"]].to_numpy()).all()):
            raise ValueError(f"Fold {fold_id} has incomplete best-epoch metrics.")
        if not np.allclose(
            best_tasks[["roc_auc", "ap"]].mean().to_numpy(),
            [best["best_val_roc_auc"], best["best_val_ap"]],
        ):
            raise ValueError(f"Fold {fold_id} best-epoch metrics disagree.")

        fold_rows.append({
            "fold_id": fold_id, "n_train": len(tr_idx), "n_val": len(val_idx),
            **best, "elapsed_seconds": perf_counter() - start,
        })
        task_rows.append(best_tasks.assign(fold_id=fold_id))
        histories.append(history.assign(fold_id=fold_id))
        task_histories.append(per_task_history.assign(fold_id=fold_id))
        if verbose:
            print(f"  Best epoch={best['best_epoch']} | "
                  f"ROC-AUC={best['best_val_roc_auc']:.4f} | "
                  f"AP={best['best_val_ap']:.4f}")
        # Keep only metrics and histories, not five trained models/scalers.
        del model, X_tr, X_val, y_tr, y_val, _

        if on_fold_end is not None:
            on_fold_end(dict(fold_rows[-1]), n_splits)

    fold_df_result = pd.DataFrame(fold_rows)
    per_task_fold_df = pd.concat(task_rows, ignore_index=True)
    per_task_cv_df = (
        per_task_fold_df.groupby("task", sort=False)
        .agg(
            n_valid_folds=("roc_auc", "count"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            positive_ratio_mean=("positive_ratio", "mean"),
        )
        .reindex(target_cols).reset_index()
    )
    if not per_task_cv_df["n_valid_folds"].eq(n_splits).all():
        raise ValueError("Incomplete CV endpoint coverage.")
    summary = {
        "n_folds": n_splits,
        "n_tasks": len(target_cols),
        "cv_macro_roc_auc": float(per_task_cv_df["roc_auc_mean"].mean()),
        "cv_macro_ap": float(per_task_cv_df["ap_mean"].mean()),
        "fold_macro_roc_auc_std": float(fold_df_result["best_val_roc_auc"].std()),
        "fold_macro_ap_std": float(fold_df_result["best_val_ap"].std()),
        "cv_macro_ap_baseline": float(per_task_cv_df["positive_ratio_mean"].mean()),
        "best_epoch_median": float(fold_df_result["best_epoch"].median()),
        "elapsed_seconds": float(fold_df_result["elapsed_seconds"].sum()),
    }
    return {
        "config": config,
        "summary": summary,
        "fold_df": fold_df_result,
        "per_task_fold_df": per_task_fold_df,
        "per_task_cv_df": per_task_cv_df,
        "history_df": pd.concat(histories, ignore_index=True),
        "per_task_history_df": pd.concat(task_histories, ignore_index=True),
    }
