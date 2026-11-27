"""Data splitting utilities for the Tox21 ML project.

The outer scaffold hold-out must be created before any model comparison or tuning.
The held-out test set is reserved for final evaluation.
"""

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold


def greedy_scaffold_indices(
    scaffold_groups,
    *,
    test_size: float = 0.2,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional train/test indices with disjoint scaffold groups.

    Assign whole groups in decreasing size order to the partition with the
    largest remaining molecule-count deficit. The seed breaks size ties only.
    Target proportions are approximate; endpoint labels are not used.
    Empty-string scaffolds are retained as a single group.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    groups = pd.Series(scaffold_groups).reset_index(drop=True)
    if groups.isna().any():
        raise ValueError("Missing scaffold IDs; check invalid SMILES before splitting.")

    items = list(groups.groupby(groups, sort=True).indices.items())
    if len(items) < 2:
        raise ValueError("At least two distinct scaffold groups are required.")

    rng = np.random.RandomState(random_state)
    items.sort(key=lambda item: (len(item[1]), rng.rand()), reverse=True)

    n_total = len(groups)
    target_train = (1 - test_size) * n_total
    target_test = test_size * n_total
    train_idx, test_idx = [], []

    for i, (_, idxs) in enumerate(items):
        # Reserve the last group for an empty partition if necessary.
        if i == len(items) - 1 and not train_idx:
            train_idx.extend(idxs)
        elif i == len(items) - 1 and not test_idx:
            test_idx.extend(idxs)
        elif target_train - len(train_idx) >= target_test - len(test_idx):
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    train_idx = np.asarray(sorted(train_idx), dtype=int)
    test_idx = np.asarray(sorted(test_idx), dtype=int)

    assert len(train_idx) and len(test_idx)
    assert len(train_idx) + len(test_idx) == n_total
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])

    return train_idx, test_idx


def scaffold_holdout_split(
    valid_df: pd.DataFrame,
    X_df: pd.DataFrame,
    *,
    random_state: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split molecules into disjoint scaffold groups, retaining feature-row alignment.

    Parameters
    ----------
    valid_df : pd.DataFrame
        Molecule-level data containing ``canonical_smiles`` and endpoint labels.
    X_df : pd.DataFrame
        Feature matrix with exactly the same row index and order as ``valid_df``.
    random_state : int
        Explicit seed supplied by the calling notebook or script.
    test_size : float
        Target fraction of molecules in the test set; the actual fraction
        may differ because scaffold groups cannot be divided.

    Returns
    -------
    train_df, test_df, X_train_all, X_test_all : tuple[pd.DataFrame, ...]
        The molecule DataFrames include a ``scaffold`` column.
    """
    if not X_df.index.equals(valid_df.index):
        raise ValueError("X_df and valid_df must have identical row indices and order.")

    # Preserve the original Bemis-Murcko scaffold definition.
    scaffold_groups = valid_df["canonical_smiles"].apply(
        lambda smiles: MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    )
    train_idx, test_idx = greedy_scaffold_indices(
        scaffold_groups,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = valid_df.iloc[train_idx].copy()
    test_df = valid_df.iloc[test_idx].copy()
    train_df["scaffold"] = scaffold_groups.iloc[train_idx].values
    test_df["scaffold"] = scaffold_groups.iloc[test_idx].values
    X_train_all = X_df.iloc[train_idx].copy()
    X_test_all = X_df.iloc[test_idx].copy()

    assert X_train_all.index.equals(train_df.index)
    assert X_test_all.index.equals(test_df.index)
    assert set(train_df["scaffold"]).isdisjoint(test_df["scaffold"])

    return train_df, test_df, X_train_all, X_test_all


def load_scaffold_holdout_split(valid_df, X_df, split_csv, *, verify_scaffolds=True):
    """Apply an existing molecule-level split, preserving the input row order.

    The empty-string Murcko scaffold is a real group, not a missing value.
    Fail rather than silently dropping molecules or generating a new split.
    """
    if not X_df.index.equals(valid_df.index) or not valid_df.index.is_unique:
        raise ValueError("Features and molecules must have identical, unique row indices.")
    keys = valid_df["canonical_smiles"]
    if keys.isna().any() or keys.eq("").any() or not keys.is_unique:
        raise ValueError("canonical_smiles must be nonempty and unique.")
    manifest = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    required = ["canonical_smiles", "scaffold", "split"]
    if not set(required).issubset(manifest.columns):
        raise ValueError(f"Split CSV must contain {required}.")
    if manifest.canonical_smiles.eq("").any() or not manifest.canonical_smiles.is_unique:
        raise ValueError("Duplicate or empty canonical_smiles in split CSV.")
    if set(manifest["split"]) != {"train", "test"}:
        raise ValueError("Split CSV must contain only train/test, with both nonempty.")
    expected, actual = set(keys), set(manifest.canonical_smiles)
    if expected != actual:
        raise ValueError(f"Split/data molecule mismatch: {len(expected-actual)} missing, "
                         f"{len(actual-expected)} extra in split CSV.")
    aligned = manifest.set_index("canonical_smiles").loc[keys.to_list()]
    scaffolds = aligned["scaffold"].to_numpy()
    if verify_scaffolds:
        computed = keys.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s)).to_numpy()
        if not np.array_equal(scaffolds, computed):
            raise ValueError("Stored scaffold IDs differ from recomputed Murcko scaffolds.")
    molecules = valid_df.copy()
    molecules["scaffold"] = scaffolds
    tr = np.flatnonzero(aligned["split"].to_numpy() == "train")
    te = np.flatnonzero(aligned["split"].to_numpy() == "test")
    train_df, test_df = molecules.iloc[tr].copy(), molecules.iloc[te].copy()
    if not set(train_df.scaffold).isdisjoint(test_df.scaffold):
        raise ValueError("Scaffold overlap between development and holdout.")
    return train_df, test_df, X_df.iloc[tr].copy(), X_df.iloc[te].copy()