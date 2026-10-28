"""Save the selected diabetes pipeline, preprocessor, and provenance metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from backend.app.ml.diabetes.preprocessing import NUMERIC_FEATURES

DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"
DATASET_URL = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
PREDICTION_TARGET = "readmitted_30d: 1 for <30; 0 for >30 or NO"


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Metadata value is not JSON serializable: {type(value).__name__}")


def _feature_schema(preprocessor: object) -> list[dict[str, object]]:
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    categorical_columns = next(
        columns
        for name, _, columns in preprocessor.transformers_
        if name == "categorical"
    )
    categorical_options = {
        column: [str(value) for value in categories if str(value) != "__missing__"]
        for column, categories in zip(categorical_columns, encoder.categories_)
    }
    raw_categories = {
        column: {str(value) for value in categories}
        for column, categories in zip(categorical_columns, encoder.categories_)
    }

    schema: list[dict[str, object]] = []
    for feature in preprocessor.feature_names_in_:
        name = str(feature)
        if name in NUMERIC_FEATURES:
            bounds: dict[str, int] = {"minimum": 1 if name == "time_in_hospital" else 0}
            if name == "time_in_hospital":
                bounds["maximum"] = 14
            schema.append({"name": name, "type": "integer", "required": True, **bounds})
        else:
            categories = categorical_options[name]
            nullable = "__missing__" in raw_categories[name]
            schema.append(
                {
                    "name": name,
                    "type": "category",
                    "required": not nullable,
                    "nullable": nullable,
                    "categories": categories,
                }
            )
    return schema


def save_diabetes_artifacts(
    model: Pipeline,
    model_directory: str | Path,
    *,
    model_version: str,
    source_sha256: str,
    metrics: Mapping[str, object],
    selection_rationale: str,
) -> dict[str, object]:
    """Serialize a fitted disease pipeline and its preprocessor separately.

    The fitted pipeline remains the authoritative prediction artifact; the
    separate preprocessor file is provided for inspection and metadata parity.
    Load joblib files only from trusted sources because joblib uses pickle.
    """
    if not isinstance(model, Pipeline) or not {"preprocessor", "classifier"}.issubset(model.named_steps):
        raise ValueError("Expected a fitted pipeline with preprocessor and classifier steps")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    if not source_sha256.strip():
        raise ValueError("source_sha256 must not be empty")
    if not selection_rationale.strip():
        raise ValueError("selection_rationale must not be empty")

    check_is_fitted(model.named_steps["preprocessor"])
    check_is_fitted(model.named_steps["classifier"])
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if 1 not in classifier.classes_:
        raise ValueError("Selected classifier must support positive class 1")

    destination = Path(model_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "diabetes_model.joblib"
    preprocessor_path = destination / "diabetes_preprocessor.joblib"
    metadata_path = destination / "model_metadata.json"

    normalized_metrics = json.loads(json.dumps(dict(metrics), default=_json_default))
    metadata: dict[str, object] = {
        "model_name": classifier.__class__.__name__,
        "algorithm": classifier.__class__.__name__,
        "model_version": model_version,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_URL,
            "sha256": source_sha256,
        },
        "features": [str(feature) for feature in preprocessor.feature_names_in_],
        "feature_schema": _feature_schema(preprocessor),
        "transformed_feature_count": len(preprocessor.get_feature_names_out()),
        "target": PREDICTION_TARGET,
        "metrics": normalized_metrics,
        "selection_rationale": selection_rationale,
        "artifacts": {
            "model": model_path.name,
            "preprocessor": preprocessor_path.name,
        },
        "clinical_validation": False,
        "educational_use_only": True,
    }

    joblib.dump(model, model_path, compress=3)
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata