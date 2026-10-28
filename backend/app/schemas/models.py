from typing import Any

from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease: str
    available: bool
    model_name: str
    algorithm: str
    model_version: str
    training_date: str
    dataset: dict[str, Any]
    features: list[str]
    feature_schema: list[dict[str, Any]]
    transformed_feature_count: int
    target: str
    metrics: dict[str, Any]
    selection_rationale: str
    clinical_validation: bool
    educational_use_only: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]