from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.app.ml.liver.preprocessing import (
    build_liver_preprocessor,
    encode_liver_target,
    load_liver_dataset,
)


def test_encode_liver_target_and_validation() -> None:
    labels = pd.Series([1, 2, 1])

    encoded = encode_liver_target(labels)

    assert encoded.tolist() == [1, 0, 1]
    assert encoded.name == "liver_disease"
    with pytest.raises(ValueError, match="Unexpected or missing"):
        encode_liver_target(pd.Series([0]))


def test_load_liver_dataset_from_temp_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "liver.csv"
    data = pd.DataFrame(
        {
            "Age": [45, 52, 60],
            "Gender": ["Male", "Female", "Male"],
            "Total_Bilirubin": [1.2, 0.8, 2.5],
            "Direct_Bilirubin": [0.4, 0.2, 1.1],
            "Alkaline_Phosphotase": [120, 180, 260],
            "Alamine_Aminotransferase": [30, 40, 95],
            "Aspartate_Aminotransferase": [25, 35, 80],
            "Total_Proteins": [6.5, 7.0, 6.0],
            "Albumin": [3.8, 4.0, 2.7],
            "Albumin_and_Globulin_Ratio": [1.1, 1.2, 0.8],
            "Dataset": [1, 2, 1],
        }
    )
    data.to_csv(csv_path, index=False)

    predictors, target = load_liver_dataset(csv_path)

    assert list(predictors.columns) == [
        "Age",
        "Gender",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Proteins",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]
    assert target.tolist() == [1, 0, 1]
    assert target.name == "liver_disease"


def test_preprocessor_fits_and_transforms_numeric_and_categorical_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "liver.csv"
    data = pd.DataFrame(
        {
            "Age": [45, 52, 60, 51],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Total_Bilirubin": [1.2, 0.8, 2.5, 1.5],
            "Direct_Bilirubin": [0.4, 0.2, 1.1, 0.6],
            "Alkaline_Phosphotase": [120, 180, 260, 210],
            "Alamine_Aminotransferase": [30, 40, 95, 55],
            "Aspartate_Aminotransferase": [25, 35, 80, 60],
            "Total_Proteins": [6.5, 7.0, 6.0, 6.8],
            "Albumin": [3.8, 4.0, 2.7, 3.5],
            "Albumin_and_Globulin_Ratio": [1.1, 1.2, 0.8, 1.0],
            "Dataset": [1, 2, 1, 2],
        }
    )
    data.to_csv(csv_path, index=False)

    predictors, target = load_liver_dataset(csv_path)
    preprocessor = build_liver_preprocessor(predictors)
    transformed = preprocessor.fit_transform(predictors)

    assert transformed.shape[0] == len(predictors)
    assert transformed.shape[1] > 0
    assert target.shape[0] == len(predictors)
