from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas.models import ModelInfo, ModelListResponse
from backend.app.services.model_service import (
    ModelMetadataUnavailable,
    ModelService,
    UnsupportedDisease,
)

router = APIRouter(prefix="/models", tags=["models"])


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


@router.get("", response_model=ModelListResponse)
def list_models(request: Request) -> ModelListResponse:
    try:
        return ModelListResponse(models=_service(request).list_models())
    except ModelMetadataUnavailable as error:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable") from error


@router.get("/{disease}", response_model=ModelInfo)
def get_model(disease: str, request: Request) -> ModelInfo:
    try:
        return _service(request).get_model(disease)
    except UnsupportedDisease as error:
        raise HTTPException(status_code=404, detail="Disease model is not available") from error
    except ModelMetadataUnavailable as error:
        raise HTTPException(status_code=503, detail="Model metadata is unavailable") from error