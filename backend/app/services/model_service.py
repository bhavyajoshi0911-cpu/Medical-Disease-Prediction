"""Read model availability and public metadata without loading model binaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.schemas.models import ModelInfo


class ModelMetadataUnavailable(Exception):
    """Raised when local model metadata is missing or malformed."""


class UnsupportedDisease(Exception):
    """Raised when a disease module has not been added to the registry."""


class ModelService:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path

    def _metadata_path_for_disease(self, disease: str) -> Path:
        if disease == "diabetes":
            return self.metadata_path
        if disease == "heart":
            return self.metadata_path.parent / "heart_metadata.json"
        if disease == "liver":
            return self.metadata_path.parent / "liver_metadata.json"
        raise UnsupportedDisease(disease)

    def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = [self.get_model("diabetes")]
        for disease in ("heart", "liver"):
            metadata_path = self.metadata_path.parent / f"{disease}_metadata.json"
            if metadata_path.is_file():
                models.append(self.get_model(disease))
        return models

    def get_model(self, disease: str) -> ModelInfo:
        metadata_path = self._metadata_path_for_disease(disease)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelMetadataUnavailable from error

        artifact_names = metadata.get("artifacts", {})
        model_name = artifact_names.get("model", "")
        preprocessor_name = artifact_names.get("preprocessor", "")
        safe_names = all(
            name and Path(name).name == name
            for name in (model_name, preprocessor_name)
        )
        available = safe_names and all(
            (metadata_path.parent / name).is_file()
            for name in (model_name, preprocessor_name)
        )

        try:
            return ModelInfo(
                disease=disease,
                available=available,
                model_name=metadata["model_name"],
                algorithm=metadata["algorithm"],
                model_version=metadata["model_version"],
                training_date=metadata["training_date"],
                dataset=metadata["dataset"],
                features=metadata["features"],
                feature_schema=metadata["feature_schema"],
                transformed_feature_count=metadata["transformed_feature_count"],
                target=metadata["target"],
                metrics=metadata["metrics"],
                selection_rationale=metadata["selection_rationale"],
                clinical_validation=metadata["clinical_validation"],
                educational_use_only=metadata["educational_use_only"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelMetadataUnavailable from error