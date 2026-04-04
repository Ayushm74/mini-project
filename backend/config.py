from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    gee_project_id: str = Field(default="", alias="GEE_PROJECT_ID")
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


settings = Settings()
