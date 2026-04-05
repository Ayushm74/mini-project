from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Reads GEE_PROJECT_ID from environment or backend/.env
    gee_project_id: str = Field(default="", alias="GEE_PROJECT_ID")
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


settings = Settings()
