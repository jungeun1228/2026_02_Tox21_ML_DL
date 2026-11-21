"""Molecular feature engineering for the Tox21 traditional-ML project.

This module keeps the original notebook's Morgan fingerprint and RDKit
11-descriptor definitions. No random seed is needed: these operations are
deterministic for a given SMILES and RDKit version.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
    rdmolops,
)

# Descriptor column names are reused for validation and feature-matrix creation.
DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "HeavyAtomCount", "FormalCharge",
]

# Keep the notebook's original Morgan parameters (radius=2, 2048 bits).
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_morgan_fp(smiles):
    """Convert one SMILES string to a Morgan fingerprint, or return None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_gen.GetFingerprintAsNumPy(mol)


def build_morgan_fp_to_df(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Return a DataFrame with `morgan_fp` and `is_valid` columns."""
    df = df.copy()
    fps = []
    is_valid_list = []

    for smi in df[smiles_col]:
        fp = smiles_to_morgan_fp(smi)
        fps.append(fp)
        is_valid_list.append(fp is not None)

    df["morgan_fp"] = fps
    df["is_valid"] = is_valid_list
    return df


def calc_basic_descriptors(smiles):
    """Compute the same eleven RDKit descriptors as the original notebook."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "FormalCharge": rdmolops.GetFormalCharge(mol),
    }


def add_basic_descriptors_to_df(df_w_fp: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit descriptor columns and the `desc_valid` indicator."""
    desc_df = df_w_fp["canonical_smiles"].apply(calc_basic_descriptors).apply(pd.Series)
    df_w_fp_desc = pd.concat([df_w_fp, desc_df], axis=1)
    df_w_fp_desc["desc_valid"] = df_w_fp_desc[DESCRIPTOR_COLS].notna().all(axis=1)
    return df_w_fp_desc


def build_feature_matrix(df_w_fp_desc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid molecules and their 2048-bit + 11-descriptor feature matrix."""
    valid_df = df_w_fp_desc[
        df_w_fp_desc["is_valid"] & df_w_fp_desc["desc_valid"]
    ].copy()

    fp_array = np.stack(valid_df["morgan_fp"].values)
    fp_df = pd.DataFrame(
        fp_array,
        columns=[f"fp_{i}" for i in range(fp_array.shape[1])],
        index=valid_df.index,
    )
    desc_df = valid_df[DESCRIPTOR_COLS]
    X_df = pd.concat([fp_df, desc_df], axis=1)
    return valid_df, X_df
