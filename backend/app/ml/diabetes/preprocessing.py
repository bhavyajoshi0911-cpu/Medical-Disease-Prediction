"""Leakage-conscious preprocessing for the UCI diabetes readmission task."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "readmitted"
IDENTIFIER_COLUMNS = ("encounter_id", "patient_nbr")
NUMERIC_FEATURES = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
EXCLUDED_FEATURES = ("weight",)
TARGET_MAPPING = {"NO": 0, ">30": 0, "<30": 1}

SOURCE_FEATURES = (
    "race", "gender", "age", "weight", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "time_in_hospital",
    "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
)
EXPECTED_SOURCE_COLUMNS = frozenset(
    (*IDENTIFIER_COLUMNS, *SOURCE_FEATURES, TARGET_COLUMN)
)


class DiabetesSplit(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


def encode_readmission_target(values: pd.Series) -> pd.Series:
    """Map <30 to 1 and NO/>30 to 0, rejecting unknown source labels."""
    unexpected = set(values.dropna().unique()) - TARGET_MAPPING.keys()
    if unexpected or values.isna().any():
        raise ValueError(f"Unexpected or missing readmitted labels: {unexpected}")
    return values.map(TARGET_MAPPING).astype("int8").rename("readmitted_30d")


def load_diabetes_dataset(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load raw encounters and return predictors, binary target, and patient groups.

    The source's '?' marker becomes missing, while the literal "None" lab-result
    category is preserved. Weight is excluded from predictors because 96.86% of
    its values are missing in this source. Identifier columns are returned only
    through the patient-group series and are never predictor columns.
    """
    frame = pd.read_csv(
        csv_path,
        keep_default_na=False,
        na_values=["?"],
        low_memory=False,
    )
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

    target = encode_readmission_target(frame[TARGET_COLUMN])
    groups = frame["patient_nbr"].astype("string").rename("patient_nbr")
    if groups.isna().any():
        raise ValueError("Patient group identifiers must not be missing")

    excluded = {*IDENTIFIER_COLUMNS, TARGET_COLUMN, *EXCLUDED_FEATURES}
    predictors = frame.loc[:, [column for column in SOURCE_FEATURES if column not in excluded]].copy()
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(predictors[column], errors="coerce")
        invalid = predictors[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values in {column}: {int(invalid.sum())}")
        predictors[column] = converted

    categorical_features = [
        column for column in predictors.columns if column not in NUMERIC_FEATURES
    ]
    for column in categorical_features:
        values = predictors[column]
        predictors[column] = values.map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    return predictors, target, groups


def build_diabetes_preprocessor(
    predictors: pd.DataFrame,
) -> ColumnTransformer:
    """Build an unfitted transformer; callers fit it on training data only."""
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
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
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


def split_by_patient(
    predictors: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    fold: int = 0,
    random_state: int = 42,
) -> DiabetesSplit:
    """Create an approximately stratified holdout with no patient overlap."""
    if not (predictors.index.equals(target.index) and target.index.equals(groups.index)):
        raise ValueError("Predictors, target, and patient groups must have aligned indexes")
    if n_splits < 2 or not 0 <= fold < n_splits:
        raise ValueError("fold must be in [0, n_splits), and n_splits must be at least 2")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_indices, test_indices = next(
        split for index, split in enumerate(splitter.split(predictors, target, groups))
        if index == fold
    )
    groups_train = groups.iloc[train_indices]
    groups_test = groups.iloc[test_indices]
    if not set(groups_train).isdisjoint(groups_test):
        raise RuntimeError("Patient overlap detected between training and test folds")

    return DiabetesSplit(
        predictors.iloc[train_indices],
        predictors.iloc[test_indices],
        target.iloc[train_indices],
        target.iloc[test_indices],
        groups_train,
        groups_test,
    )