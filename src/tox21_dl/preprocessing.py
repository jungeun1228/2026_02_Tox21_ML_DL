"""Fold-specific preprocessing for the Tox21 multi-task MLP.

Morgan bits remain unchanged. Fit descriptor scaling on training data only.
Labels remain 0, 1 or NaN until the PyTorch dataset applies masking.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tox21_ml.features import DESCRIPTOR_COLS

N_MORGAN_BITS = 2048
FEATURE_COLS = [f"fp_{i}" for i in range(N_MORGAN_BITS)] + list(DESCRIPTOR_COLS)


def _validate_feature_columns(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")
    if list(X.columns) != FEATURE_COLS:
        raise ValueError("Expected 2048 Morgan bits followed by the 11 RDKit descriptors.")
    if not X.index.is_unique or X.empty:
        raise ValueError("Features must be nonempty and have a unique row index.")


def fit_descriptor_scaler(X_train):
    """Fit a new scaler on the training partition's descriptors only."""
    _validate_feature_columns(X_train)
    descriptors = X_train[DESCRIPTOR_COLS]
    if not np.isfinite(descriptors.to_numpy(dtype=np.float64)).all():
        raise ValueError("Training descriptors contain non-finite values.")
    return StandardScaler().fit(descriptors)


def transform_features(X, scaler):
    """Apply a fitted scaler to descriptors and return float32 features.

    Do not refit the scaler or modify the original DataFrame.
    """
    _validate_feature_columns(X)
    # Copy before replacing the descriptor columns; preserve raw features.
    X_array = X.to_numpy(dtype=np.float32, copy=True)
    fingerprints = X_array[:, :N_MORGAN_BITS]
    if not np.all((fingerprints == 0) | (fingerprints == 1)):
        raise ValueError("Morgan fingerprint values must be 0 or 1.")
    descriptors = X[DESCRIPTOR_COLS]
    if not np.isfinite(descriptors.to_numpy(dtype=np.float64)).all():
        raise ValueError("Descriptors contain non-finite values.")
    X_array[:, N_MORGAN_BITS:] = scaler.transform(descriptors)
    if not np.isfinite(X_array).all():
        raise ValueError("Transformed features contain non-finite values.")
    return X_array


def prepare_fold_data(train_df, X_train_all, target_cols, tr_idx, val_idx):
    """Prepare one CV fold from raw, aligned development DataFrames.

    tr_idx and val_idx are positional indices from iter_scaffold_cv_indices.
    They must form a disjoint, complete partition of the development data.

    Returns
    -------
    X_tr, X_val, y_tr, y_val, scaler
        Float32 arrays (missing labels retained) and the fold-specific scaler.
    """
    _validate_feature_columns(X_train_all)
    if not train_df.index.equals(X_train_all.index):
        raise ValueError("Molecule and feature row indices/order must match.")
    target_cols = list(target_cols)
    if (not target_cols or len(set(target_cols)) != len(target_cols)
            or not train_df.columns.is_unique
            or not set(target_cols).issubset(train_df.columns)):
        raise ValueError("Expected unique endpoint columns in train_df.")

    tr_idx = np.asarray(tr_idx)
    val_idx = np.asarray(val_idx)
    for idx in (tr_idx, val_idx):
        if idx.ndim != 1 or idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("Fold indices must be nonempty 1-D integer arrays.")
    if not np.array_equal(
        np.sort(np.concatenate([tr_idx, val_idx])), np.arange(len(train_df))
    ):
        raise ValueError("Fold indices must cover development rows exactly once.")

    y = train_df[target_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    if not np.all(np.isnan(y) | (y == 0) | (y == 1)):
        raise ValueError("Endpoint labels must be 0, 1 or NaN.")

    X_tr_raw = X_train_all.iloc[tr_idx]
    X_val_raw = X_train_all.iloc[val_idx]

    scaler = fit_descriptor_scaler(X_tr_raw)
    X_tr = transform_features(X_tr_raw, scaler)
    X_val = transform_features(X_val_raw, scaler)

    return X_tr, X_val, y[tr_idx], y[val_idx], scaler
