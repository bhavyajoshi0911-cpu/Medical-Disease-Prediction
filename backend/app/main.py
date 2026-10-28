"""FastAPI application factory for MedPredict."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from backend.app.api import health, history, models
from backend.app.api.prediction import register_diabetes_prediction_route, register_heart_prediction_route, register_liver_prediction_route
from backend.app.config import Settings, get_settings
from backend.app.database import Base, create_database_engine
from backend.app.db import models as database_models
from backend.app.services.model_service import ModelService
from backend.app.services.prediction_service import PredictionService


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or get_settings()
    engine = create_database_engine(config.database_url)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        Base.metadata.create_all(bind=engine)
        yield
        engine.dispose()

    application = FastAPI(
        title=config.app_name,
        description="Educational dataset-specific readmission prediction platform.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = config
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.model_service = ModelService(config.model_metadata_path)
    diabetes_prediction_service = PredictionService(config.model_metadata_path, disease="diabetes")
    heart_prediction_service = PredictionService(config.model_metadata_path.parent / "heart_metadata.json", disease="heart")
    liver_prediction_service = PredictionService(config.model_metadata_path.parent / "liver_metadata.json", disease="liver")
    application.state.prediction_service = diabetes_prediction_service
    application.state.heart_prediction_service = heart_prediction_service
    application.state.liver_prediction_service = liver_prediction_service

    if config.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization"],
        )

    application.include_router(health.router, prefix="/api", tags=["health"])
    application.include_router(models.router, prefix="/api")
    application.include_router(history.router, prefix="/api")
    prediction_router = APIRouter(prefix="/api")
    register_diabetes_prediction_route(prediction_router, diabetes_prediction_service)
    register_heart_prediction_route(prediction_router, heart_prediction_service)
    register_liver_prediction_route(prediction_router, liver_prediction_service)
    application.include_router(prediction_router)
    return application


app = create_app()