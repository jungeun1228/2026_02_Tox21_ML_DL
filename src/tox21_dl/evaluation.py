"""Prediction and multi-task evaluation for the Tox21 MLP."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()     # 이 함수에 대해서 gradient 계산을 완전히 끄는 설정 (predict에서는 학습 안하는데도, gradient를 자동계산해버리면 메모리낭비, 속도 저하라서)
def predict(model, loader, device):
    """Return sigmoid scores, targets and masks in DataLoader iteration order.

    Evaluation loaders should use shuffle=False and drop_last=False.
    Targets may contain zero placeholders at masked positions.
    """
    model.eval()    # Dropout을 비활성화하고 BatchNorm이 학습 중에 기록한 통계값을 사용하도록 함
    all_probs, all_y, all_mask = [], [], []

    for batch in loader:
        x = batch["x"].to(device)
        logits = model(x)
        y = batch["y"]
        mask = batch["mask"]
        if logits.ndim != 2 or logits.shape != y.shape or logits.shape != mask.shape:
            raise ValueError("Logits, targets and mask must have the same 2-D shape.")
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits.")

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(y.cpu().numpy())
        all_mask.append(mask.cpu().numpy())

    if not all_probs:
        raise ValueError("Evaluation loader is empty.")

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_mask, axis=0),
    )


def compute_multitask_metrics(probs, y_true, mask, task_names):
    """Compute endpoint metrics and unweighted macro averages.

    Both ROC-AUC and AP are excluded (NaN) for endpoints without both
    observed classes, matching the original notebook's coverage policy.
    An all-invalid evaluation returns NaN macro scores, not zero.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    mask = np.asarray(mask)
    if probs.ndim != 2 or probs.shape != y_true.shape or probs.shape != mask.shape:
        raise ValueError("probs, y_true and mask must have the same 2-D shape.")
    if 0 in probs.shape:
        raise ValueError("Evaluation arrays must be nonempty.")
    task_names = list(task_names)
    if len(task_names) != probs.shape[1] or len(set(task_names)) != len(task_names):
        raise ValueError("task_names must contain one unique name per endpoint.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only 0 or 1.")
    valid = mask.astype(bool)
    if not np.all((y_true[valid] == 0) | (y_true[valid] == 1)):
        raise ValueError("Unmasked labels must be 0 or 1.")
    if not np.all(np.isfinite(probs[valid]) & (probs[valid] >= 0) & (probs[valid] <= 1)):
        raise ValueError("Unmasked predictions must be finite scores in [0, 1].")

    rows = []
    for t, name in enumerate(task_names):
        y_t = y_true[valid[:, t], t]
        p_t = probs[valid[:, t], t]
        n_pos = int((y_t == 1).sum())
        n_neg = int((y_t == 0).sum())
        n_valid = len(y_t)

        roc_auc = ap = np.nan
        if n_valid == 0:
            status = "no_valid_labels"
        elif n_pos == 0:
            status = "no_positive_labels"
        elif n_neg == 0:
            status = "no_negative_labels"
        else:
            status = "ok"
            roc_auc = roc_auc_score(y_t, p_t)
            ap = average_precision_score(y_t, p_t)

        rows.append({
            "task": name,
            "roc_auc": roc_auc,
            "ap": ap,
            "n_valid": n_valid,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_missing": len(y_true) - n_valid,
            "positive_ratio": n_pos / n_valid if n_valid else np.nan,
            "status": status,
        })

    df = pd.DataFrame(rows)
    return {
        "mean_roc_auc": float(df["roc_auc"].mean()),
        "mean_ap": float(df["ap"].mean()),
        "n_valid_tasks": int(df["status"].eq("ok").sum()),
        "n_tasks": len(task_names),
        "per_task_df": df,
    }


def evaluate_multitask(model, loader, device, task_names):  # combine predict() and compute_multitask_metrics() for convenience
    """Predict and evaluate one partition using the same endpoint order."""
    probs, y_true, mask = predict(model, loader, device)
    return compute_multitask_metrics(probs, y_true, mask, task_names)
