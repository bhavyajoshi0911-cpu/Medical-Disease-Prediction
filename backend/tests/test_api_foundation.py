import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def make_settings(tmp_path: Path) -> Settings:
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "XGBClassifier",
                "algorithm": "XGBClassifier",
                "model_version": "1.0",
                "training_date": "2026-10-01T00:00:00+00:00",
                "dataset": {"name": "Test dataset", "source": "https://example.invalid", "sha256": "a" * 64},
                "features": ["age", "time_in_hospital"],
                "feature_schema": [
                    {
                        "name": "age",
                        "type": "category",
                        "required": True,
                        "nullable": False,
                        "categories": ["[50,60)"],
                    },
                    {
                        "name": "time_in_hospital",
                        "type": "integer",
                        "required": True,
                        "minimum": 1,
                        "maximum": 14,
                    },
                ],
                "transformed_feature_count": 4,
                "target": "readmitted_30d",
                "metrics": {"cv_pr_auc": {"mean": 0.2}},
                "selection_rationale": "Test metadata only.",
                "clinical_validation": False,
                "educational_use_only": True,
                "artifacts": {
                    "model": "diabetes_model.joblib",
                    "preprocessor": "diabetes_preprocessor.joblib",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "diabetes_model.joblib").write_bytes(b"test artifact")
    (tmp_path / "diabetes_preprocessor.joblib").write_bytes(b"test preprocessor")
    (tmp_path / "heart_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "planned_heart_model",
                "algorithm": "Heart disease prototype",
                "model_version": "0.1",
                "training_date": "2026-10-05T00:00:00+00:00",
                "dataset": {"name": "Heart disease demo", "source": "planned", "sha256": "b" * 64},
                "features": ["age", "cholesterol"],
                "feature_schema": [
                    {
                        "name": "age",
                        "type": "integer",
                        "required": True,
                        "minimum": 18,
                        "maximum": 100,
                    },
                    {
                        "name": "cholesterol",
                        "type": "integer",
                        "required": True,
                        "minimum": 100,
                        "maximum": 500,
                    },
                ],
                "transformed_feature_count": 2,
                "target": "heart_disease",
                "metrics": {},
                "selection_rationale": "Planned heart disease module placeholder.",
                "clinical_validation": False,
                "educational_use_only": True,
                "artifacts": {
                    "model": "heart_model.joblib",
                    "preprocessor": "heart_preprocessor.joblib",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "liver_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "planned_liver_model",
                "algorithm": "Liver disease prototype",
                "model_version": "0.1",
                "training_date": "2026-10-05T00:00:00+00:00",
                "dataset": {"name": "Liver disease demo", "source": "planned", "sha256": "c" * 64},
                "features": ["age", "bilirubin"],
                "feature_schema": [
                    {
                        "name": "age",
                        "type": "integer",
                        "required": True,
                        "minimum": 18,
                        "maximum": 100,
                    },
                    {
                        "name": "bilirubin",
                        "type": "number",
                        "required": True,
                        "minimum": 0,
                        "maximum": 20,
                    },
                ],
                "transformed_feature_count": 2,
                "target": "liver_disease",
                "metrics": {},
                "selection_rationale": "Planned liver disease module placeholder.",
                "clinical_validation": False,
                "educational_use_only": True,
                "artifacts": {
                    "model": "liver_model.joblib",
                    "preprocessor": "liver_preprocessor.joblib",
                },
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        app_name="MedPredict test",
        environment="test",
        database_url="sqlite://",
        cors_origins=("http://localhost:5173",),
        model_metadata_path=metadata_path,
    )


def test_health_checks_database(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "MedPredict test",
        "database": "ok",
    }


def test_model_endpoints_return_saved_schema_and_metadata(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/models")
        diabetes_response = client.get("/api/models/diabetes")
        heart_response = client.get("/api/models/heart")
        liver_response = client.get("/api/models/liver")

    assert response.status_code == 200
    assert len(response.json()["models"]) == 3
    model = diabetes_response.json()
    assert model["available"] is True
    assert model["features"] == ["age", "time_in_hospital"]
    assert model["target"] == "readmitted_30d"
    assert model["clinical_validation"] is False
    assert heart_response.status_code == 200
    assert heart_response.json()["disease"] == "heart"
    assert heart_response.json()["available"] is False
    assert liver_response.status_code == 200
    assert liver_response.json()["disease"] == "liver"
    assert liver_response.json()["available"] is False


def test_missing_model_artifact_is_reported_as_unavailable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    (tmp_path / "diabetes_model.joblib").unlink()
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/models/diabetes")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_heart_model_is_exposed_as_a_planned_module(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    heart_metadata = tmp_path / "heart_metadata.json"
    heart_metadata.write_text(
        json.dumps(
            {
                "model_name": "planned_heart_model",
                "algorithm": "Heart disease prototype",
                "model_version": "0.1",
                "training_date": "2026-10-05T00:00:00+00:00",
                "dataset": {"name": "Heart disease demo", "source": "planned", "sha256": "b" * 64},
                "features": ["age", "cholesterol"],
                "feature_schema": [
                    {
                        "name": "age",
                        "type": "integer",
                        "required": True,
                        "minimum": 18,
                        "maximum": 100,
                    },
                    {
                        "name": "cholesterol",
                        "type": "integer",
                        "required": True,
                        "minimum": 100,
                        "maximum": 500,
                    },
                ],
                "transformed_feature_count": 2,
                "target": "heart_disease",
                "metrics": {},
                "selection_rationale": "Planned heart disease module placeholder.",
                "clinical_validation": False,
                "educational_use_only": True,
                "artifacts": {
                    "model": "heart_model.joblib",
                    "preprocessor": "heart_preprocessor.joblib",
                },
            }
        ),
        encoding="utf-8",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/models/heart")

    assert response.status_code == 200
    assert response.json()["disease"] == "heart"
    assert response.json()["available"] is False