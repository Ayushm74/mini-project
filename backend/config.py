from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gee_project_id: str = ""
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


settings = Settings()
