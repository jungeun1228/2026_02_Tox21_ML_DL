
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def load_sdf(sdf_path: str | Path) -> pd.DataFrame:
    """Load an SDF file and extract canonical SMILES and properties."""

    sdf_path = Path(sdf_path)

    if not sdf_path.is_file():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(str(sdf_path))
    rows = []

    for i, mol in enumerate(supplier):
        if mol is None:
            rows.append({
                "idx": i,
                "status": "parse_failed"
            })
            continue

        try:
            props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            row = {
                "idx": i,
                "canonical_smiles": canonical_smiles,
                "status": "ok",
            }

            row.update(props)
            rows.append(row)

        except Exception as e:
            rows.append({
                "idx": i,
                "status": f"failed: {str(e)}"
            })

    df = pd.DataFrame(rows)
    return df



def duplicate_combine(series):
    """Combine duplicate endpoint labels, treating conflicts as NaN."""

    vals = pd.to_numeric(
        series, errors="coerce"
    ).dropna().unique()

    if len(vals) == 0:
        return np.nan
    if len(vals) == 1:
        return int(vals[0])
    else:
        return np.nan


def group_and_filter_tox21(df_ok: pd.DataFrame):
    """Group duplicate SMILES and filter rows without endpoint labels."""

    # Metadata: keep values from the first occurrence
    meta_column = [
        "idx", "canonical_smiles", "Formula", "FW", "DSSTox_CID"
    ]

    # Toxicity endpoints
    endpoint_column = df_ok.columns[6:]

    # Build aggregation rules
    agg_dict = {col: "first" for col in meta_column}

    for col in endpoint_column:
        agg_dict[col] = duplicate_combine

    # Group compounds by canonical SMILES
    df_grouped = df_ok.groupby(
        "canonical_smiles", as_index=False
    ).agg(agg_dict)

    # Remove compounds with no available endpoint labels
    df_filtered = df_grouped.dropna(
        subset=endpoint_column,
        how="all"
    )

    return df_grouped, df_filtered, endpoint_column
