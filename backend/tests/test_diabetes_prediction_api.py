import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.schemas.prediction import build_diabetes_request_model


def make_prediction_app(tmp_path: Path):
    row_count = 120
    hours = [row % 14 + 1 for row in range(row_count)]
    genders = ["Female" if row % 2 else "Male" for row in range(row_count)]
    features = pd.DataFrame({"time_in_hospital": hours, "gender": genders})
    target = pd.Series(
        [int(hour >= 8 or gender == "Male") for hour, gender in zip(hours, genders)]
    )
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
                ["time_in_hospital"],
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["gender"],
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

    model_path = tmp_path / "diabetes_model.joblib"
    preprocessor_path = tmp_path / "diabetes_preprocessor.joblib"
    metadata_path = tmp_path / "model_metadata.json"
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
                "features": ["time_in_hospital", "gender"],
                "feature_schema": [
                    {
                        "name": "time_in_hospital",
                        "type": "integer",
                        "required": True,
                        "minimum": 1,
                        "maximum": 14,
                    },
                    {
                        "name": "gender",
                        "type": "category",
                        "required": True,
                        "nullable": False,
                        "categories": ["Female", "Male"],
                    },
                ],
                "transformed_feature_count": 3,
                "target": "readmitted_30d",
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
        app_name="MedPredict prediction test",
        environment="test",
        database_url="sqlite://",
        cors_origins=(),
        model_metadata_path=metadata_path,
    )
    return create_app(settings)


def test_dynamic_request_schema_preserves_hyphenated_field_aliases() -> None:
    request_schema = build_diabetes_request_model(
        [
            {"name": "time_in_hospital", "type": "integer", "minimum": 1, "maximum": 14},
            {
                "name": "glyburide-metformin",
                "type": "category",
                "nullable": True,
                "categories": ["No", "Steady"],
            },
        ]
    )

    payload = request_schema.model_validate(
        {"time_in_hospital": 5, "glyburide-metformin": "Steady"}
    )

    assert payload.model_dump(by_alias=True) == {
        "time_in_hospital": 5,
        "glyburide-metformin": "Steady",
    }


def test_prediction_returns_probability_and_model_contributions(tmp_path: Path) -> None:
    app = make_prediction_app(tmp_path)
    request_schema = app.openapi()["paths"]["/api/predict/diabetes"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    request_properties = app.openapi()["components"]["schemas"][request_schema]["properties"]

    with TestClient(app) as client:
        response = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 5, "gender": "Female"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["disease"] == "diabetes"
    assert result["prediction_target"] == "readmitted_30d"
    assert 0 <= result["probability"] <= 1
    assert result["model_version"] == "test"
    assert result["top_features"]
    assert set(request_properties) == {"time_in_hospital", "gender"}
    assert "not a medical diagnosis" in result["disclaimer"].lower()


def test_prediction_rejects_invalid_category_and_numeric_range(tmp_path: Path) -> None:
    app = make_prediction_app(tmp_path)

    with TestClient(app) as client:
        invalid_category = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 5, "gender": "Unknown"},
        )
        invalid_range = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 15, "gender": "Female"},
        )
        extra_field = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 5, "gender": "Female", "extra": 1},
        )

    assert invalid_category.status_code == 422
    assert invalid_range.status_code == 422
    assert extra_field.status_code == 422


def test_prediction_history_persists_metadata_only_and_supports_delete(
    tmp_path: Path,
) -> None:
    app = make_prediction_app(tmp_path)

    with TestClient(app) as client:
        prediction = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 5, "gender": "Female"},
        )
        prediction_id = prediction.json()["prediction_id"]
        listing = client.get("/api/history")
        detail = client.get(f"/api/history/{prediction_id}")
        deleted = client.delete(f"/api/history/{prediction_id}")
        after_delete = client.get("/api/history")
        columns = {
            column["name"]
            for column in inspect(app.state.engine).get_columns("predictions")
        }

    assert prediction.status_code == 200
    assert prediction.json()["created_at"]
    assert listing.json()["total"] == 1
    assert listing.json()["items"][0]["probability"] == prediction.json()["probability"]
    assert detail.json()["id"] == prediction_id
    assert deleted.status_code == 204
    assert after_delete.json()["total"] == 0
    assert {"disease", "prediction_target", "probability", "model_name", "model_version", "created_at"}.issubset(columns)
    assert not {"gender", "time_in_hospital"}.intersection(columns)


def test_prediction_returns_safe_unavailable_message_when_artifact_is_missing(
    tmp_path: Path,
) -> None:
    app = make_prediction_app(tmp_path)
    (tmp_path / "diabetes_model.joblib").unlink()

    with TestClient(app) as client:
        response = client.post(
            "/api/predict/diabetes",
            json={"time_in_hospital": 5, "gender": "Female"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Unable to generate the prediction right now. Please try again."
    )
    assert "Traceback" not in response.text