"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    environment: str
    database_url: str
    cors_origins: tuple[str, ...]
    model_metadata_path: Path


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    metadata_path = Path(
        os.getenv(
            "MODEL_METADATA_PATH",
            str(PROJECT_ROOT / "backend" / "trained_models" / "model_metadata.json"),
        )
    ).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    return Settings(
        app_name=os.getenv("APP_NAME", "MedPredict"),
        environment=os.getenv("ENVIRONMENT", "development"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./medpredict.db"),
        cors_origins=tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip()),
        model_metadata_path=metadata_path,
    )