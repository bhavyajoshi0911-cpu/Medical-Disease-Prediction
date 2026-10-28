"""Load the selected model and produce a prediction without storing input features."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from backend.app.db.models import PredictionRecord
from backend.app.ml.diabetes.explanation import explain_pipeline, top_local_contributions
from backend.app.schemas.prediction import (
    FeatureContribution,
    PredictionResponse,
    build_diabetes_request_model,
)

logger = logging.getLogger(__name__)
DISCLAIMER = (
    "Educational machine-learning output only. This is not a medical diagnosis "
    "or medical advice. Consult a qualified healthcare professional."
)


class PredictionUnavailable(Exception):
    """Raised when the saved model cannot produce a prediction."""


class PredictionService:
    def __init__(self, metadata_path: Path, *, disease: str) -> None:
        self.metadata_path = metadata_path
        self.disease = disease
        self.metadata: dict[str, Any] = {}
        self.request_model: type[BaseModel] | None = None
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.request_model = build_diabetes_request_model(self.metadata["feature_schema"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            logger.warning("Prediction metadata is unavailable for %s: %s", metadata_path, error)
            self.metadata = {}
            self.request_model = None
        self._model: Pipeline | None = None
        self._model_lock = threading.Lock()

    def _load_model(self) -> Pipeline:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            artifact_name = self.metadata.get("artifacts", {}).get("model", "")
            if not artifact_name or Path(artifact_name).name != artifact_name:
                raise PredictionUnavailable
            model_path = self.metadata_path.parent / artifact_name
            try:
                model = joblib.load(model_path)
                if not isinstance(model, Pipeline):
                    raise TypeError("Saved estimator is not an sklearn pipeline")
                preprocessor = model.named_steps["preprocessor"]
                actual_features = [str(name) for name in preprocessor.feature_names_in_]
                if actual_features != self.metadata["features"]:
                    raise ValueError("Saved model schema does not match metadata")
                if 1 not in model.named_steps["classifier"].classes_:
                    raise ValueError("Saved model does not contain positive class 1")
            except Exception as error:
                logger.exception("Unable to load the diabetes model artifact")
                raise PredictionUnavailable from error
            self._model = model
            return model

    def predict(self, payload: BaseModel, session: Session) -> PredictionResponse:
        try:
            model = self._load_model()
            values = payload.model_dump(by_alias=True)
            features = pd.DataFrame([values]).reindex(columns=self.metadata["features"])
            class_labels = list(model.named_steps["classifier"].classes_)
            positive_index = class_labels.index(1)
            probability = float(model.predict_proba(features)[0, positive_index])
            prediction = int(model.predict(features)[0])
            explanation = explain_pipeline(
                model,
                features,
                features,
                background_size=1,
            )
            contributions = top_local_contributions(explanation, max_features=5)
            top_features = [
                FeatureContribution(
                    feature=str(row.feature),
                    contribution=float(row.contribution),
                    direction=str(row.direction),
                )
                for row in contributions.itertuples(index=False)
            ]
            record = PredictionRecord(
                disease=self.disease,
                prediction_target=self.metadata["target"],
                probability=probability,
                model_name=self.metadata["algorithm"],
                model_version=self.metadata["model_version"],
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            return PredictionResponse(
                prediction_id=record.id,
                created_at=record.created_at,
                prediction=prediction,
                disease=self.disease,
                prediction_target=self.metadata["target"],
                probability=probability,
                model=self.metadata["algorithm"],
                model_version=self.metadata["model_version"],
                top_features=top_features,
                disclaimer=DISCLAIMER,
            )
        except PredictionUnavailable:
            raise
        except Exception as error:
            logger.exception("Unable to generate a %s prediction", self.disease)
            raise PredictionUnavailable from error