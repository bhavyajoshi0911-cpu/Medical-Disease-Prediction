from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database import get_database_session
from backend.app.db.models import PredictionRecord
from backend.app.schemas.prediction import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
)

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=PredictionHistoryResponse)
def list_history(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_database_session),
) -> PredictionHistoryResponse:
    total = session.scalar(select(func.count()).select_from(PredictionRecord)) or 0
    records = session.scalars(
        select(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return PredictionHistoryResponse(
        items=[PredictionHistoryItem.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{prediction_id}", response_model=PredictionHistoryItem)
def get_history_item(
    prediction_id: int,
    session: Session = Depends(get_database_session),
) -> PredictionHistoryItem:
    record = session.get(PredictionRecord, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction history item not found")
    return PredictionHistoryItem.model_validate(record)


@router.delete("/{prediction_id}", status_code=204)
def delete_history_item(
    prediction_id: int,
    session: Session = Depends(get_database_session),
) -> Response:
    record = session.get(PredictionRecord, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction history item not found")
    session.delete(record)
    return Response(status_code=204)