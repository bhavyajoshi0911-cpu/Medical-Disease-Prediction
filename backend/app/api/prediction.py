from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database import get_database_session
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.prediction_service import PredictionService, PredictionUnavailable


def register_prediction_route(
    router: APIRouter,
    prediction_service: PredictionService,
    *,
    disease: str,
    path: str,
) -> None:
    request_model = prediction_service.request_model
    if request_model is None:
        return

    def predict_model(
        payload: Any,
        session: Session = Depends(get_database_session),
    ) -> PredictionResponse:
        try:
            return prediction_service.predict(payload, session)
        except PredictionUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail="Unable to generate the prediction right now. Please try again.",
            ) from error

    predict_model.__name__ = f"predict_{disease}"
    predict_model.__annotations__ = {
        "payload": request_model,
        "session": Session,
        "return": PredictionResponse,
    }
    router.add_api_route(
        path,
        predict_model,
        methods=["POST"],
        response_model=PredictionResponse,
        tags=["predictions"],
    )


def register_diabetes_prediction_route(
    router: APIRouter,
    prediction_service: PredictionService,
) -> None:
    register_prediction_route(router, prediction_service, disease="diabetes", path="/predict/diabetes")


def register_liver_prediction_route(
    router: APIRouter,
    prediction_service: PredictionService,
) -> None:
    register_prediction_route(router, prediction_service, disease="liver", path="/predict/liver")


def register_heart_prediction_route(
    router: APIRouter,
    prediction_service: PredictionService,
) -> None:
    register_prediction_route(router, prediction_service, disease="heart", path="/predict/heart")
