"""Random Forest hyperparameter tuning for Tox21.

This module preserves the original notebook's StratifiedGroupKFold,
ROC-AUC-based parameter selection, and CSV/pickle/JSON checkpoints.

Source: Tox21_Traditional_ML_v2_2026_11_19_seed_setup.ipynb, code cell 116.
"""

# Inner CV for model selection (RF tuning) - based on ROC-AUC (PR-AUC: only tracking) - refined: checkpoint save, verbose included (RF 내부 트리 생성만 병렬화 / 파라미터 조합 탐색은 직렬 / 타깃 12개도 직렬)
# to find best params
import os
import json
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score


def tune_multitarget_random_forest_on_train_cv(
    train_df,
    X_train_all,
    target_cols,
    param_grid,
    n_splits=5,
    *,
    random_state,
    checkpoint_dir="rf_tuning_checkpoints",
    resume=True,
    verbose=1,
    rf_n_jobs=-1
):
    """
    Multi-target RF tuning with:
    - manual CV over ParameterGrid
    - selection metric: ROC-AUC (fixed)
    - tracking metric: PR-AUC (logged but not used for selection)
    - checkpoint save/load
    - resume from previous run
    - verbose progress logs

    Parameters
    ----------
    resume : bool
        True면 checkpoint를 읽어서 이미 끝난 target은 건너뜀
    verbose : int
        0 = silent
        1 = target 단위 로그
        2 = param 단위 로그
        3 = fold 단위 로그
    rf_n_jobs : int
        RandomForestClassifier 내부 병렬화 설정
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    results_csv_path = os.path.join(checkpoint_dir, "rf_tuning_results_partial.csv")
    params_pkl_path = os.path.join(checkpoint_dir, "rf_best_params_by_target.pkl")
    done_targets_json_path = os.path.join(checkpoint_dir, "rf_done_targets.json")

    # 입력 X 정리
    if isinstance(X_train_all, pd.DataFrame):
        X_train_all = X_train_all.to_numpy()

    # checkpoint 로드
    if resume and os.path.exists(results_csv_path):
        existing_results_df = pd.read_csv(results_csv_path)
    else:
        existing_results_df = pd.DataFrame()

    if resume and os.path.exists(params_pkl_path):
        with open(params_pkl_path, "rb") as f:
            best_params_by_target = pickle.load(f)
    else:
        best_params_by_target = {}

    if resume and os.path.exists(done_targets_json_path):
        with open(done_targets_json_path, "r", encoding="utf-8") as f:
            done_targets = set(json.load(f))
    else:
        done_targets = set()

    results = []
    if not existing_results_df.empty:
        results = existing_results_df.to_dict(orient="records")

    total_targets = len(target_cols)
    param_list = list(ParameterGrid(param_grid))
    total_param_combinations = len(param_list)

    if verbose >= 1:
        print("=" * 80)
        print("RF tuning started (selection metric: ROC-AUC)")
        print(f"Total targets: {total_targets}")
        print(f"Param combinations per target: {total_param_combinations}")
        print(f"CV folds: {n_splits}")
        print(f"Resume mode: {resume}")
        print(f"Already completed targets: {len(done_targets)}")
        print("=" * 80)

    for target_idx, target in enumerate(target_cols, start=1):
        if target in done_targets:
            if verbose >= 1:
                print(f"[{target_idx}/{total_targets}] SKIP already completed target: {target}")
            continue

        target_start_time = time.time()

        if verbose >= 1:
            print("\n" + "-" * 80)
            print(f"[{target_idx}/{total_targets}] Start target: {target}")

        mask = train_df[target].notna()
        task_df = train_df.loc[mask].copy()
        y_task = task_df[target].astype(int).values

        if len(task_df) == 0:
            print(f"SKIP: {target} (no samples)")
            done_targets.add(target)
            _save_rf_checkpoints(results, best_params_by_target, done_targets,
                                 results_csv_path, params_pkl_path, done_targets_json_path)
            continue

        if len(np.unique(y_task)) < 2:
            print(f"SKIP: {target} (only one class)")
            done_targets.add(target)
            _save_rf_checkpoints(results, best_params_by_target, done_targets,
                                 results_csv_path, params_pkl_path, done_targets_json_path)
            continue

        X_task = X_train_all[mask]
        groups = task_df["scaffold"].values

        if pd.Series(groups).nunique() < n_splits:
            print(f"SKIP: {target} (unique scaffolds < n_splits)")
            done_targets.add(target)
            _save_rf_checkpoints(results, best_params_by_target, done_targets,
                                 results_csv_path, params_pkl_path, done_targets_json_path)
            continue

        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        best_score = -np.inf
        best_params = None
        best_row = None

        for param_idx, params in enumerate(param_list, start=1):
            param_start_time = time.time()

            if verbose >= 2:
                print(f"\n  [Target {target}] Param {param_idx}/{total_param_combinations}: {params}")

            fold_scores = []
            valid_fold_count = 0

            for fold_idx, (tr_idx, val_idx) in enumerate(
                sgkf.split(X_task, y_task, groups=groups), start=1
            ):
                fold_start_time = time.time()

                X_tr = X_task[tr_idx]
                X_val = X_task[val_idx]
                y_tr = y_task[tr_idx]
                y_val = y_task[val_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                    if verbose >= 3:
                        print(f"    Fold {fold_idx}/{n_splits}: skipped (only one class)")
                    continue

                model = RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=rf_n_jobs,
                    **params
                )

                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_val)[:, 1]

                fold_result = {
                    "roc_auc": roc_auc_score(y_val, y_prob),
                    "pr_auc": average_precision_score(y_val, y_prob),
                }
                fold_scores.append(fold_result)
                valid_fold_count += 1

                if verbose >= 3:
                    fold_elapsed = time.time() - fold_start_time
                    print(
                        f"    Fold {fold_idx}/{n_splits} done "
                        f"(roc_auc={fold_result['roc_auc']:.4f}, "
                        f"pr_auc={fold_result['pr_auc']:.4f}, "
                        f"time={fold_elapsed:.1f}s)"
                    )

            if len(fold_scores) == 0:
                if verbose >= 2:
                    print("    No valid fold scores. Skip this param.")
                continue

            fold_df = pd.DataFrame(fold_scores)

            mean_roc_auc = fold_df["roc_auc"].mean()
            mean_pr_auc = fold_df["pr_auc"].mean()
            std_roc_auc = fold_df["roc_auc"].std()
            std_pr_auc = fold_df["pr_auc"].std()

            # selection metric은 ROC-AUC로 고정
            score = mean_roc_auc

            if verbose >= 2:
                param_elapsed = time.time() - param_start_time
                print(
                    f"    Param summary: valid_folds={valid_fold_count}, "
                    f"mean_roc_auc={mean_roc_auc:.4f} (±{std_roc_auc:.4f}), "
                    f"mean_pr_auc={mean_pr_auc:.4f} (±{std_pr_auc:.4f}), "
                    f"time={param_elapsed/60:.2f} min"
                )

            if score > best_score:
                best_score = score
                best_params = params
                best_row = {
                    "target": target,
                    "n_samples": len(y_task),
                    "positive_ratio": y_task.mean(),
                    "cv_roc_auc_mean": mean_roc_auc,
                    "cv_roc_auc_std": std_roc_auc,
                    "cv_pr_auc_mean": mean_pr_auc,
                    "cv_pr_auc_std": std_pr_auc,
                    "selection_metric": "roc_auc",
                    "selection_score": score,
                    "best_params": json.dumps(params, ensure_ascii=False)
                }

                if verbose >= 2:
                    print(f"    >>> New best ROC-AUC for {target}: {best_score:.4f}")

        if best_params is None:
            print(f"SKIP: {target} (no valid CV result)")
            done_targets.add(target)
            _save_rf_checkpoints(results, best_params_by_target, done_targets,
                                 results_csv_path, params_pkl_path, done_targets_json_path)
            continue

        results = [r for r in results if r["target"] != target]
        results.append(best_row)
        best_params_by_target[target] = best_params
        done_targets.add(target)

        _save_rf_checkpoints(
            results,
            best_params_by_target,
            done_targets,
            results_csv_path,
            params_pkl_path,
            done_targets_json_path
        )

        target_elapsed = time.time() - target_start_time

        if verbose >= 1:
            print(f"[DONE] target={target}")
            print(f"       best_roc_auc={best_score:.4f}")
            print(f"       best_params={best_params}")
            print(f"       elapsed={target_elapsed/60:.2f} min")
            print(f"       checkpoint saved to: {checkpoint_dir}")

    final_results_df = pd.DataFrame(results)

    if not final_results_df.empty:
        final_results_df = final_results_df.sort_values(
            by=["selection_score", "target"],
            ascending=[False, True]
        ).reset_index(drop=True)

    if verbose >= 1:
        print("\n" + "=" * 80)
        print("RF tuning finished")
        print(f"Completed targets: {len(done_targets)}/{total_targets}")
        print(f"Results saved in: {checkpoint_dir}")
        print("=" * 80)

    return final_results_df, best_params_by_target


def _save_rf_checkpoints(
    results,
    best_params_by_target,
    done_targets,
    results_csv_path,
    params_pkl_path,
    done_targets_json_path
):
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)

    with open(params_pkl_path, "wb") as f:
        pickle.dump(best_params_by_target, f)

    with open(done_targets_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(done_targets)), f, ensure_ascii=False, indent=2)
