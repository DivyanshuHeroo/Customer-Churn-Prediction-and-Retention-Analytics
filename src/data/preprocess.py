"""
Data cleaning and train/validation/test splitting.

Cleaning is deliberately *modelling-agnostic* here (fixing dtypes, the famous
blank ``TotalCharges`` rows, mapping the target to 0/1). All
learned transformations (scaling, encoding, imputation) live inside the
scikit-learn ``Pipeline`` in ``src/models`` so there is **zero train/test
leakage** — a detail interviewers consistently probe for.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, load_config, resolve_path, ensure_dir

logger = get_logger("preprocess")

# Columns that are categorical (object) vs numeric in the Telco schema.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_SENIOR = "SeniorCitizen"  # already 0/1 in the raw data


def clean_data(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Apply deterministic, leakage-free cleaning to the raw dataframe."""
    config = config or load_config()
    target = config["data"]["target_column"]
    id_col = config["data"]["id_column"]

    df = df.copy()

    # 1) TotalCharges arrives as text with blanks for tenure==0 customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        # A brand-new customer (tenure 0) has not been charged a total yet;
        # 0 is the correct, interpretable fill rather than a mean.
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        logger.info("Filled %d missing TotalCharges (tenure==0 customers).", n_missing)

    # 2) Map the target to a clean 0/1 integer.
    if df[target].dtype == object:
        df[target] = (df[target].str.strip().str.lower() == "yes").astype(int)

    # 3) Ensure SeniorCitizen is a clean integer flag.
    df[BINARY_SENIOR] = df[BINARY_SENIOR].astype(int)

    # 4) Drop exact duplicate rows if any slipped in.
    before = len(df)
    df = df.drop_duplicates(subset=[id_col]) if id_col in df.columns else df.drop_duplicates()
    if len(df) != before:
        logger.info("Removed %d duplicate rows.", before - len(df))

    logger.info("Cleaned data: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def get_feature_lists(df: pd.DataFrame, config: dict | None = None) -> Tuple[list, list]:
    """Return (numeric_features, categorical_features) present in the frame."""
    config = config or load_config()
    target = config["data"]["target_column"]
    id_col = config["data"]["id_column"]

    drop = {target, id_col}
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    # SeniorCitizen is numeric 0/1 but semantically categorical; treat as numeric
    # is fine for trees/linear models, so we keep it with numeric here.
    if BINARY_SENIOR in df.columns and BINARY_SENIOR not in numeric:
        numeric = numeric + [BINARY_SENIOR]
    categorical = [
        c for c in df.columns
        if c not in drop and c not in numeric and df[c].dtype == object
    ]
    return numeric, categorical


def split_data(
    df: pd.DataFrame, config: dict | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test with stratification on the target.

    Validation is carved out of the training portion so the test set stays
    untouched until final evaluation.
    """
    config = config or load_config()
    target = config["data"]["target_column"]
    seed = config["project"]["random_seed"]
    test_size = config["split"]["test_size"]
    val_size = config["split"]["validation_size"]
    stratify = config["split"]["stratify"]

    strat = df[target] if stratify else None
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )

    strat_tv = train_val[target] if stratify else None
    # val_size is expressed as a fraction of the *original* data.
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, stratify=strat_tv
    )

    logger.info(
        "Split -> train: %d | val: %d | test: %d (churn rate %.1f%% / %.1f%% / %.1f%%)",
        len(train), len(val), len(test),
        100 * train[target].mean(), 100 * val[target].mean(), 100 * test[target].mean(),
    )
    return train, val, test


def build_processed_dataset(config: dict | None = None) -> pd.DataFrame:
    """Load raw data, clean it, persist the cleaned frame, and return it."""
    from src.data.make_dataset import get_raw_data

    config = config or load_config()
    raw = get_raw_data(config)
    cleaned = clean_data(raw, config)

    out_path = resolve_path(config["paths"]["processed_data"])
    ensure_dir(out_path.parent)
    cleaned.to_csv(out_path, index=False)
    logger.info("Saved processed data to %s", out_path)
    return cleaned


if __name__ == "__main__":
    build_processed_dataset()
