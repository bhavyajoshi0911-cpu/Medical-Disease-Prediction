"""Prediction schemas, including a request model generated from metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    prediction_id: int
    created_at: datetime
    prediction: Literal[0, 1]
    disease: str
    prediction_target: str
    probability: float = Field(ge=0, le=1)
    model: str
    model_version: str
    top_features: list[FeatureContribution]
    disclaimer: str


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    disease: str
    prediction_target: str
    probability: float
    model_name: str
    model_version: str
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: list[PredictionHistoryItem]
    total: int
    limit: int
    offset: int


def build_diabetes_request_model(feature_schema: list[dict[str, Any]]) -> type[BaseModel]:
    """Build a strict request schema from the saved fitted-model feature spec."""
    fields: dict[str, tuple[Any, Any]] = {}
    seen_names: set[str] = set()
    for feature in feature_schema:
        source_name = str(feature["name"])
        field_name = source_name.replace("-", "_")
        if field_name in seen_names:
            raise ValueError("Feature names collide after request-field normalization")
        seen_names.add(field_name)

        description = f"Dataset feature: {source_name}."
        if feature["type"] in {"integer", "number"}:
            constraints: dict[str, Any] = {"strict": True, "alias": source_name}
            if "minimum" in feature:
                constraints["ge"] = feature["minimum"]
            if "maximum" in feature:
                constraints["le"] = feature["maximum"]
            field_type = int if feature["type"] == "integer" else float
            fields[field_name] = (field_type, Field(..., description=description, **constraints))
            continue

        if feature["type"] != "category":
            raise ValueError(f"Unsupported input type for {source_name}")
        categories = tuple(str(value) for value in feature.get("categories", []))
        category_type: Any = Literal.__getitem__(categories) if categories else str
        nullable = bool(feature.get("nullable", False))
        field_type = category_type | None if nullable else category_type
        default = None if nullable else ...
        description += " Value must match a category observed during model training."
        fields[field_name] = (
            field_type,
            Field(default, alias=source_name, description=description),
        )

    if not fields:
        raise ValueError("Model metadata contains no input features")
    return create_model(
        "DiabetesPredictionRequest",
        __config__=ConfigDict(extra="forbid", populate_by_name=False),
        **fields,
    )