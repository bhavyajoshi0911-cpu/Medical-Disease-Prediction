"""Leakage-conscious preprocessing for the Indian liver patient task."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Dataset"
NUMERIC_FEATURES = (
    "Age",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Proteins",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
)
CATEGORICAL_FEATURES = ("Gender",)
ORDERED_FEATURES = ("Age", "Gender", *[feature for feature in NUMERIC_FEATURES if feature != "Age"])
EXPECTED_SOURCE_COLUMNS = frozenset((*ORDERED_FEATURES, TARGET_COLUMN))
TARGET_MAPPING = {1: 1, 2: 0}


def encode_liver_target(values: pd.Series) -> pd.Series:
    """Map the source liver-disease target into a binary positive/negative label."""
    unexpected = set(pd.Series(values).dropna().unique()) - TARGET_MAPPING.keys()
    if unexpected or values.isna().any():
        raise ValueError(f"Unexpected or missing liver target labels: {unexpected}")
    return values.map(TARGET_MAPPING).astype("int8").rename("liver_disease")


def load_liver_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the ILPD CSV and convert predictors plus a binary target."""
    frame = pd.read_csv(csv_path, keep_default_na=False, na_values=["?"], low_memory=False)
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")

    actual_columns = frozenset(frame.columns)
    missing_columns = EXPECTED_SOURCE_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_SOURCE_COLUMNS
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Unexpected dataset schema: "
            f"missing={sorted(missing_columns)}, "
            f"extra={sorted(unexpected_columns)}"
        )

    target = encode_liver_target(frame[TARGET_COLUMN])
    predictors = frame.loc[:, ORDERED_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(predictors[column], errors="coerce")
        invalid = predictors[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values in {column}: {int(invalid.sum())}")
        predictors[column] = converted

    for column in CATEGORICAL_FEATURES:
        predictors[column] = predictors[column].map(
            lambda value: str(value) if pd.notna(value) else None
        )

    return predictors, target


def build_liver_preprocessor(predictors: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer for numeric and gender features."""
    missing_numeric = set(NUMERIC_FEATURES) - set(predictors.columns)
    if missing_numeric:
        raise ValueError(f"Missing numeric feature columns: {sorted(missing_numeric)}")

    categorical_features = [
        column for column in predictors.columns if column not in NUMERIC_FEATURES
    ]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
