"""Data splitting utilities for the Tox21 ML project.

The outer scaffold hold-out must be created before any model comparison or tuning.
The held-out test set is reserved for final evaluation.
"""

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


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
        Fraction of *scaffold groups* requested for the test set. It does not
        guarantee the same fraction of molecules in the test set.

    Returns
    -------
    train_df, test_df, X_train_all, X_test_all : tuple[pd.DataFrame, ...]
        The molecule DataFrames include a ``scaffold`` column.
    """
    if not X_df.index.equals(valid_df.index):
        raise ValueError("X_df and valid_df must have identical row indices and order.")

    # Preserve the original notebook's scaffold calculation and split settings.
    scaffold_groups = valid_df["canonical_smiles"].apply(
        lambda smiles: MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    )
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        ).split(valid_df, groups=scaffold_groups)
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
