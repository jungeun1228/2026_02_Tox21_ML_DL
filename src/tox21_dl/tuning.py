"""Optuna objective for development-only Tox21 multi-task scaffold CV."""

from copy import deepcopy
import math
import json
from pathlib import Path

import optuna

from .experiments import run_multitask_cv


def suggest_hyperparameters(trial, search_space, fixed_config):
    """Sample seven hyperparameters and build the MLP training config.

    Subsequent hidden dimensions halve, with a minimum width of 64.
    A pos_weight_cap of 1 disables weighting; None leaves it uncapped.
    The fixed configuration is not modified.
    """
    depth = trial.suggest_categorical("hidden_depth", search_space["hidden_depth"]) # .suggest_categorical: 주어진 후보 중 하나를 선택하는 메서드
    width = trial.suggest_categorical("first_hidden_dim", search_space["first_hidden_dim"])
    dropout = trial.suggest_float("dropout", *search_space["dropout"])
    weight_decay = trial.suggest_float("weight_decay", *search_space["weight_decay"], log=True)
    lr = trial.suggest_float("lr", *search_space["lr"], log=True)
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])
    pos_weight_cap = trial.suggest_categorical("pos_weight_cap", search_space["pos_weight_cap"])

    return {
        **fixed_config,
        "hidden_dims": tuple(max(64, width // (2 ** i)) for i in range(depth)),
        "dropout": dropout,
        "weight_decay": weight_decay,
        "lr": lr,
        "batch_size": batch_size,
        "pos_weight_cap": pos_weight_cap,
    }


def make_objective(
    train_df,
    X_train_all,
    target_cols,
    fold_df,
    search_space,
    fixed_config,
    *,
    verbose=False,
    enable_pruning=False,
    trial_results_dir=None,
):
    """Return an Optuna objective using the existing fixed development CV.

    Completed trials must finish every fold. When enabled, pruning is checked
    once after the first three of five folds, using their mean macro ROC-AUC.
    Partial fold metrics are recorded separately from complete CV metrics.
    Optionally export detailed tables for completed trials to a dedicated
    per-study directory. No holdout or model weights are used.
    """
    search_space = deepcopy(search_space)
    fixed_config = deepcopy(fixed_config)
    target_cols = list(target_cols)
    
    if enable_pruning and set(fold_df["fold_id"].unique()) != set(range(5)):
        raise ValueError(
            "Fold-level pruning requires the fixed five-fold CV."
        )

    if trial_results_dir is not None:
        trial_results_dir = Path(trial_results_dir)
    if fixed_config.get("selection_metric", "roc_auc") != "roc_auc":
        raise ValueError("The Optuna objective requires ROC-AUC epoch selection.")

    def objective(trial):
        config = suggest_hyperparameters(trial, search_space, fixed_config)
        # Store JSON-compatible configuration for later SQLite persistence.
        trial.set_user_attr("config", {**config, "hidden_dims": list(config["hidden_dims"])})
        # Store intermediate results from completed folds.
        partial_auc, partial_ap, partial_epochs = [], [], []

        def on_fold_end(row, n_splits): # Callback - to record fold-level metrics and optionally prune the trial

            # Verify that folds are evaluated in the expected order.
            if row["fold_id"] != len(partial_auc):
                raise ValueError("Unexpected CV fold order.")

            partial_auc.append(
                float(row["best_val_roc_auc"])
            )

            partial_ap.append(
                float(row["best_val_ap"])
            )

            partial_epochs.append(
                int(row["best_epoch"])
            )

            n_done = len(partial_auc)

            # Record intermediate fold-level results.
            trial.set_user_attr(
                "n_completed_folds",
                n_done,
            )

            trial.set_user_attr(
                "partial_fold_roc_auc",
                list(partial_auc),
            )

            trial.set_user_attr(
                "partial_fold_ap",
                list(partial_ap),
            )

            trial.set_user_attr(
                "partial_fold_best_epochs",
                list(partial_epochs),
            )

            # A single checkpoint avoids comparing different fold prefixes.
            if enable_pruning and n_done == 3 and n_done < n_splits:

                score = sum(partial_auc) / n_done

                trial.report(
                    score,
                    step=n_done,
                )

                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned after {n_done}/{n_splits} folds: "
                        f"ROC-AUC={score:.6f}"
                    )
                
        result = run_multitask_cv(
            train_df=train_df, X_train_all=X_train_all, target_cols=target_cols,
            fold_df=fold_df, config=config, verbose=verbose, on_fold_end=on_fold_end,
        )
        summary = result["summary"]
        score = float(summary["cv_macro_roc_auc"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid CV macro ROC-AUC.")

        for key in (
            "cv_macro_ap", "fold_macro_roc_auc_std", "fold_macro_ap_std",
            "cv_macro_ap_baseline", "best_epoch_median", "elapsed_seconds",
        ):
            value = float(summary[key])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite CV summary: {key}")
            trial.set_user_attr(key, value)
        trial.set_user_attr("n_folds", int(summary["n_folds"]))
        trial.set_user_attr("n_tasks", int(summary["n_tasks"]))
        folds = result["fold_df"].sort_values("fold_id")
        trial.set_user_attr("fold_best_epochs", folds["best_epoch"].astype(int).tolist())
        trial.set_user_attr("fold_roc_auc", folds["best_val_roc_auc"].astype(float).tolist())
        trial.set_user_attr("fold_ap", folds["best_val_ap"].astype(float).tolist())
        # Export detailed CV results for completed trials.
        if trial_results_dir is not None:

            output_dir = (
                trial_results_dir / f"trial_{trial.number:05d}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for key in (
                "fold_df",
                "per_task_fold_df",
                "per_task_cv_df",
                "history_df",
                "per_task_history_df",
            ):
                result[key].to_csv(
                    output_dir / f"{key}.csv",
                    index=False,
                )

            with (output_dir / "config.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    trial.user_attrs["config"],
                    f,
                    indent=4,
                    allow_nan=False,
                )

            with (output_dir / "summary.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    summary,
                    f,
                    indent=4,
                    allow_nan=False,
                )
        return score

    return objective


def create_fold_pruner(enable_pruning=False):
    """Use a conservative lower-quartile rule at the three-fold checkpoint."""

    if not enable_pruning:
        return optuna.pruners.NopPruner()

    return optuna.pruners.PercentilePruner(
        percentile=25.0,
        n_startup_trials=10,
        n_warmup_steps=3,
        interval_steps=1,
        n_min_trials=5,
    )


def open_sqlite_study(db_path, study_name, study_spec, *, sampler_seed=42, enable_pruning=False):
    """Create or reopen a sequential TPE study with a checked experiment spec.

    Pruning is optional and its settings are included in the experiment spec.
    Study history is persistent; sampler RNG
    state and in-progress neural network checkpoints are not persisted.
    Use one process at a time for this local SQLite workflow.
    """
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize tuples to JSON lists, and reject non-finite metadata.
    spec = json.loads(json.dumps({
        "experiment": study_spec,

        "sampler": {
            "name": "TPE",
            "seed": sampler_seed,
            "n_startup_trials": 10,
        },

        "pruner": (
            {
                "name": "PercentilePruner",
                "percentile": 25.0,
                "n_startup_trials": 10,
                "n_warmup_steps": 3,
                "interval_steps": 1,
                "n_min_trials": 5,
                "report_after_folds": 3,
            }
            if enable_pruning
            else "none"
        ),

    }, allow_nan=False))

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{path.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=10),
        pruner=create_fold_pruner(enable_pruning),
        load_if_exists=True,    # 실행되던거 계속 이어서 실행
    )
    if list(study.directions) != [optuna.study.StudyDirection.MAXIMIZE]:
        raise ValueError("Existing study direction does not match maximize.")
    key = "experiment_spec_v1"
    if key not in study.user_attrs:
        if study.get_trials(deepcopy=False):
            raise ValueError("Existing trials have no registered experiment spec.")
        study.set_user_attr(key, spec)
    elif study.user_attrs[key] != spec:
        raise ValueError("Existing study specification does not match. Use a new study name.")
    return study


def export_study_results(study, output_dir):    # EXPORT: study results (SQLite) to CSV and JSON
    """Export a trial table and, when available, the best completed trial."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "trials.csv", index=False)
    complete = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if complete:
        best = study.best_trial
        payload = {
            "study_name": study.study_name,
            "trial_number": best.number,
            "objective_value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        }
        with (output_dir / "best_trial.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, allow_nan=False)
    return trials_df
