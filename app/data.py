"""Load and validate the public UCI credit-default dataset.

Only financial and repayment-history columns are retained. The source record ID
and demographic attributes are excluded before the local cache is written.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
TARGET_COLUMN = "default_payment_next_month"
DATASET_ID = 350
DEFAULT_DATA_PATH = Path("data/raw/credit_default_selected.csv")

# UCI's repository describes the columns by both their original names and X1-X23.
_SOURCE_COLUMN_ALIASES = {
    "LIMIT_BAL": ("LIMIT_BAL", "X1"),
    "PAY_0": ("PAY_0", "X6"),
    "PAY_2": ("PAY_2", "X7"),
    "PAY_3": ("PAY_3", "X8"),
    "PAY_4": ("PAY_4", "X9"),
    "PAY_5": ("PAY_5", "X10"),
    "PAY_6": ("PAY_6", "X11"),
    "BILL_AMT1": ("BILL_AMT1", "X12"),
    "BILL_AMT2": ("BILL_AMT2", "X13"),
    "BILL_AMT3": ("BILL_AMT3", "X14"),
    "BILL_AMT4": ("BILL_AMT4", "X15"),
    "BILL_AMT5": ("BILL_AMT5", "X16"),
    "BILL_AMT6": ("BILL_AMT6", "X17"),
    "PAY_AMT1": ("PAY_AMT1", "X18"),
    "PAY_AMT2": ("PAY_AMT2", "X19"),
    "PAY_AMT3": ("PAY_AMT3", "X20"),
    "PAY_AMT4": ("PAY_AMT4", "X21"),
    "PAY_AMT5": ("PAY_AMT5", "X22"),
    "PAY_AMT6": ("PAY_AMT6", "X23"),
}


def _normalized_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _select_features(source: pd.DataFrame) -> pd.DataFrame:
    source_by_name = {_normalized_name(column): column for column in source.columns}
    selected = {}
    for canonical_name in FEATURE_COLUMNS:
        aliases = _SOURCE_COLUMN_ALIASES[canonical_name]
        source_name = next(
            (source_by_name[_normalized_name(alias)] for alias in aliases
             if _normalized_name(alias) in source_by_name),
            None,
        )
        if source_name is None:
            raise ValueError(
                f"UCI dataset is missing {canonical_name}; checked aliases {aliases}."
            )
        selected[canonical_name] = pd.to_numeric(source[source_name], errors="raise")
    return pd.DataFrame(selected)


def _validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = set(expected_columns) - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    cleaned = dataset[expected_columns].copy()
    if cleaned.isna().any().any():
        raise ValueError("Dataset contains missing values; resolve them before training.")

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise").astype(int)
    labels = set(cleaned[TARGET_COLUMN].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Expected both binary target labels 0 and 1; found {labels}.")

    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    return cleaned


def _fetch_from_uci() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies before downloading the UCI dataset."
        ) from exc

    try:
        repository_dataset = fetch_ucirepo(id=DATASET_ID)
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch UCI dataset 350. Check network access or provide a cached "
            "CSV using CREDIT_DATA_PATH."
        ) from exc

    features = _select_features(repository_dataset.data.features)
    targets = repository_dataset.data.targets
    if targets is None or targets.shape[1] != 1:
        raise ValueError("Expected exactly one target column from UCI dataset 350.")
    features[TARGET_COLUMN] = targets.iloc[:, 0].to_numpy()
    return _validate_dataset(features)


def load_credit_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the selected dataset from a local cache or fetch it once from UCI.

    The cache contains only the 19 chosen financial/history features and target;
    the ID and demographic columns are never written to it.
    """
    path = Path(data_path or os.getenv("CREDIT_DATA_PATH", DEFAULT_DATA_PATH))
    if path.exists():
        return _validate_dataset(pd.read_csv(path))

    dataset = _fetch_from_uci()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return dataset
