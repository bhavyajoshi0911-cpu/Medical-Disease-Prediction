import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from backend.app.config import Settings
from backend.app.main import create_app


def make_liver_prediction_app(tmp_path: Path):
    row_count = 120
    ages = [row % 80 + 18 for row in range(row_count)]
    genders = ["Female" if row % 2 else "Male" for row in range(row_count)]
    bilirubin = [((row * 7) % 8) + 0.5 for row in range(row_count)]
    features = pd.DataFrame(
        {
            "Age": ages,
            "Gender": genders,
            "Total_Bilirubin": bilirubin,
        }
    )
    target = pd.Series([int(age > 55 or bilirubin_value > 2.5) for age, bilirubin_value in zip(ages, bilirubin)])
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["Age", "Total_Bilirubin"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Gender"],
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=4,
                    max_depth=2,
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=42,
                    tree_method="hist",
                ),
            ),
        ]
    ).fit(features, target)

    model_path = tmp_path / "liver_model.joblib"
    preprocessor_path = tmp_path / "liver_preprocessor.joblib"
    metadata_path = tmp_path / "liver_metadata.json"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "XGBClassifier",
                "algorithm": "XGBClassifier",
                "model_version": "test",
                "training_date": "2026-10-01T00:00:00+00:00",
                "dataset": {"name": "test", "source": "test", "sha256": "a" * 64},
                "features": ["Age", "Gender", "Total_Bilirubin"],
                "feature_schema": [
                    {
                        "name": "Age",
                        "type": "integer",
                        "required": True,
                        "minimum": 18,
                        "maximum": 100,
                    },
                    {
                        "name": "Gender",
                        "type": "category",
                        "required": True,
                        "nullable": False,
                        "categories": ["Female", "Male"],
                    },
                    {
                        "name": "Total_Bilirubin",
                        "type": "number",
                        "required": True,
                        "minimum": 0,
                        "maximum": 20,
                    },
                ],
                "transformed_feature_count": 4,
                "target": "liver_disease",
                "metrics": {},
                "selection_rationale": "Test fixture only.",
                "clinical_validation": False,
                "educational_use_only": True,
                "artifacts": {
                    "model": model_path.name,
                    "preprocessor": preprocessor_path.name,
                },
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        app_name="MedPredict liver prediction test",
        environment="test",
        database_url="sqlite://",
        cors_origins=(),
        model_metadata_path=tmp_path / "model_metadata.json",
    )
    return create_app(settings)


def test_liver_prediction_endpoint_returns_probability_and_model_contributions(tmp_path: Path) -> None:
    app = make_liver_prediction_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/predict/liver",
            json={"Age": 52, "Gender": "Male", "Total_Bilirubin": 3.2},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["disease"] == "liver"
    assert result["prediction_target"] == "liver_disease"
    assert 0 <= result["probability"] <= 1
    assert result["model_version"] == "test"
    assert result["top_features"]
    assert "not a medical diagnosis" in result["disclaimer"].lower()


def test_liver_prediction_rejects_invalid_category_and_numeric_range(tmp_path: Path) -> None:
    app = make_liver_prediction_app(tmp_path)

    with TestClient(app) as client:
        invalid_category = client.post(
            "/api/predict/liver",
            json={"Age": 52, "Gender": "Unknown", "Total_Bilirubin": 3.2},
        )
        invalid_range = client.post(
            "/api/predict/liver",
            json={"Age": 150, "Gender": "Male", "Total_Bilirubin": 3.2},
        )

    assert invalid_category.status_code == 422
    assert invalid_range.status_code == 422
