"""Development-only scaffold-balanced cross-validation for multi-task Tox21.

One scaffold is assigned to exactly one validation fold. A reproducible
multi-start greedy heuristic balances molecule counts and per-endpoint positive,
negative and missing-label counts. It does not resample molecules or labels.
Only NumPy and pandas are required.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _validate_input(train_df, target_cols, n_splits):
    target_cols = list(target_cols)
    if not target_cols or len(set(target_cols)) != len(target_cols):
        raise ValueError("target_cols must contain unique endpoint names.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    required = ["canonical_smiles", "scaffold", *target_cols]
    if not train_df.columns.is_unique or not set(required).issubset(train_df.columns):
        raise ValueError("Missing required columns or duplicate column names.")
    if not train_df.index.is_unique:
        raise ValueError("train_df index must be unique.")
    if train_df["canonical_smiles"].isna().any() or not train_df["canonical_smiles"].is_unique:
        raise ValueError("Canonical SMILES must be present and unique.")
    if train_df["scaffold"].isna().any():
        raise ValueError("Missing scaffold. Empty-string scaffolds are valid; NaN is not.")
    if not train_df["scaffold"].map(lambda s: isinstance(s, str)).all():
        raise ValueError("Scaffolds must be strings.")
    y = train_df[target_cols].to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isnan(y) | (y == 0) | (y == 1)):
        raise ValueError("Endpoint labels must be 0, 1 or NaN.")
    groups, inverse = np.unique(train_df["scaffold"].to_numpy(dtype=str), return_inverse=True)
    if len(groups) < n_splits:
        raise ValueError("Fewer distinct scaffolds than folds.")
    return target_cols, y, groups, inverse


def make_multitask_scaffold_folds(
    train_df: pd.DataFrame,
    target_cols,
    *,
    n_splits: int = 5,
    random_state: int,
    n_restarts: int = 16,
) -> pd.DataFrame:
    """Assign development molecules to common validation folds (0..K-1).

    Parameters
    ----------
    train_df : outer development partition only, with canonical_smiles,
        scaffold, and binary endpoint columns (missing labels allowed).
    target_cols : ordered endpoint names.
    n_splits : number of validation folds.
    random_state : seed for deterministic greedy restarts and tie-breaking.
    n_restarts : number of greedy candidate assignments.

    Returns
    -------
    DataFrame with canonical_smiles, scaffold, fold_id; exactly the original
    train_df index and order. Empty-string scaffolds form one indivisible group.

    Notes
    -----
    This is a heuristic, not exact stratification or a feasibility solver.
    Class coverage is prioritized over a weighted balance loss. All groups
    remain intact even when class coverage or equal fold sizes are impossible.
    Inspect fold_diagnostics before using the folds for model selection.
    """
    target_cols, y, groups, inverse = _validate_input(train_df, target_cols, n_splits)
    if not isinstance(n_restarts, int) or n_restarts < 1:
        raise ValueError("n_restarts must be an integer >= 1.")
    t = len(target_cols)
    g = len(groups)
    # One row per scaffold: size | positive counts | negative counts | missing counts.
    molecule_stats = np.column_stack([
        np.ones(len(y), dtype=int), y == 1, y == 0, np.isnan(y)
    ]).astype(np.int64)
    stats = np.zeros((g, 1 + 3 * t), dtype=np.int64)
    np.add.at(stats, inverse, molecule_stats)
    totals = stats.sum(axis=0)
    expected = totals / n_splits
    # Each block has its own weight so 12 endpoints do not swamp fold size.
    # Relative squared deviations give rare classes a meaningful contribution.
    weights = np.r_[1., np.full(t, 1. / t), np.full(t, 1. / t), np.full(t, .25 / t)]
    weights = np.divide(weights, np.maximum(expected, 1.) ** 2)
    weights[totals == 0] = 0.
    classes = slice(1, 1 + 2 * t)
    observed_classes = totals[classes] > 0

    def row_loss(counts):
        return np.sum(weights * (counts - expected) ** 2, axis=-1)

    def coverage(counts):
        return np.sum((counts[..., classes] > 0) & observed_classes, axis=-1)

    def key(counts):
        return (-int(coverage(counts).sum()), float(row_loss(counts).sum()))

    # Size and largest share of any endpoint class prioritize hard-to-place groups.
    class_share = stats[:, classes] / np.maximum(totals[classes], 1)
    priority = stats[:, 0] / totals[0] + class_share.max(axis=1)
    rng = np.random.default_rng(random_state)
    best_key = (float("inf"), float("inf"))
    best_assignment = None
    best_counts = None
    for restart in range(n_restarts):
        jitter = np.ones(g) if restart == 0 else rng.uniform(.75, 1.25, size=g)
        order = np.lexsort((rng.random(g), -(priority * jitter)))
        counts = np.zeros((n_splits, stats.shape[1]), dtype=np.int64)
        assignment = np.full(g, -1, dtype=int)
        group_counts = np.zeros(n_splits, dtype=int)
        for step, group in enumerate(order):
            v = stats[group]
            after = counts + v
            gain = coverage(after) - coverage(counts)
            delta = row_loss(after) - row_loss(counts)
            allowed = np.arange(n_splits)
            # Reserve one group per empty fold when necessary.
            empty = np.flatnonzero(group_counts == 0)
            if g - step == len(empty):
                allowed = empty
            winners = allowed[gain[allowed] == gain[allowed].max()]
            winners = winners[np.isclose(delta[winners], delta[winners].min(), rtol=0, atol=1e-12)]
            fold = int(rng.choice(winners))
            assignment[group] = fold
            counts[fold] += v
            group_counts[fold] += 1
        candidate = key(counts)
        if candidate < best_key:
            best_key, best_assignment, best_counts = candidate, assignment.copy(), counts.copy()

    # Local improvement: move a whole scaffold only if the global objective
    # improves and its original fold remains nonempty. No class is split.
    assignment, counts = best_assignment, best_counts
    group_counts = np.bincount(assignment, minlength=n_splits)
    for _ in range(3):
        changed = False
        for group in rng.permutation(g):
            src = int(assignment[group])
            if group_counts[src] <= 1:
                continue
            v = stats[group]
            old_loss, old_cov = row_loss(counts), coverage(counts)
            src_after = counts[src] - v
            dest_after = counts + v
            cov_delta = coverage(src_after) - old_cov[src] + coverage(dest_after) - old_cov
            loss_delta = row_loss(src_after) - old_loss[src] + row_loss(dest_after) - old_loss
            allowed = np.flatnonzero(np.arange(n_splits) != src)
            wins = allowed[cov_delta[allowed] == cov_delta[allowed].max()]
            dest = int(wins[np.argmin(loss_delta[wins])])
            if cov_delta[dest] > 0 or (cov_delta[dest] == 0 and loss_delta[dest] < -1e-10):
                counts[src] -= v
                counts[dest] += v
                group_counts[src] -= 1
                group_counts[dest] += 1
                assignment[group] = dest
                changed = True
        if not changed:
            break

    fold_df = train_df[["canonical_smiles", "scaffold"]].copy()
    fold_df["fold_id"] = assignment[inverse]
    _validate_folds(fold_df, train_df, n_splits)
    diagnostics = fold_diagnostics(train_df, target_cols, fold_df, n_splits=n_splits)
    bad = diagnostics.loc[~diagnostics["roc_auc_valid"], ["fold_id", "target"]]
    if not bad.empty:
        warnings.warn(
            f"{len(bad)} fold/endpoint combinations lack both classes in train or validation. "
            "Inspect fold_diagnostics; full class coverage may be infeasible.",
            RuntimeWarning, stacklevel=2,
        )
    return fold_df


def _validate_folds(fold_df, train_df, n_splits):
    if not fold_df.index.equals(train_df.index):
        raise ValueError("fold_df and train_df indices/order differ.")
    for col in ("canonical_smiles", "scaffold"):
        if not fold_df[col].equals(train_df[col]):
            raise ValueError(f"fold_df and train_df disagree on {col}.")
    ids = fold_df["fold_id"]
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids):
        raise ValueError("fold_id must contain integers without missing values.")
    if set(ids) != set(range(n_splits)):
        raise ValueError("Every fold ID 0..n_splits-1 must occur.")
    if fold_df.groupby("scaffold", sort=False)["fold_id"].nunique().max() != 1:
        raise ValueError("A scaffold occurs in multiple validation folds.")


def fold_diagnostics(train_df, target_cols, fold_df, *, n_splits=5):
    """One row per validation fold and endpoint; includes train class coverage."""
    target_cols, y, _, _ = _validate_input(train_df, target_cols, n_splits)
    _validate_folds(fold_df, train_df, n_splits)
    ids = fold_df["fold_id"].to_numpy()
    rows = []
    for f in range(n_splits):
        val = ids == f
        vy, ty = y[val], y[~val]
        for j, target in enumerate(target_cols):
            vp, vn = int((vy[:, j] == 1).sum()), int((vy[:, j] == 0).sum())
            tp, tn = int((ty[:, j] == 1).sum()), int((ty[:, j] == 0).sum())
            rows.append({
                "fold_id": f, "target": target,
                "n_val_molecules": int(val.sum()),
                "n_val_scaffolds": fold_df.loc[val, "scaffold"].nunique(),
                "val_positive": vp, "val_negative": vn,
                "val_missing": int(np.isnan(vy[:, j]).sum()),
                "val_positive_ratio": vp / (vp + vn) if vp + vn else np.nan,
                "train_positive": tp, "train_negative": tn,
                "roc_auc_valid": bool(min(vp, vn, tp, tn) > 0),
            })
    return pd.DataFrame(rows)


def iter_scaffold_cv_indices(fold_df):
    """Yield (train_positions, val_positions) for .iloc / NumPy indexing.

    fold_df must remain in exactly the same row order as train_df and X_train_all.
    """
    ids = fold_df["fold_id"].to_numpy()
    if len(ids) == 0 or not pd.api.types.is_integer_dtype(ids) or set(ids) != set(range(len(np.unique(ids)))) or len(np.unique(ids)) < 2:
        raise ValueError("Expected consecutive integer fold IDs starting at 0.")
    for fold in np.unique(ids):
        yield np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
