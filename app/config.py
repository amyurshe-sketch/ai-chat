from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration pulled from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    yandex_api_key: Optional[str] = Field(default=None, alias="YANDEX_API_KEY")
    yandex_folder_id: Optional[str] = Field(default=None, alias="YANDEX_FOLDER_ID")
    yandex_model: str = Field(default="yandexgpt-lite", alias="YANDEX_MODEL")
    yandex_temperature: float = Field(default=0.3, alias="YANDEX_TEMPERATURE")
    yandex_max_tokens: int = Field(default=800, alias="YANDEX_MAX_TOKENS")
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")

    @property
    def model_uri(self) -> str:
        folder = self.yandex_folder_id or "missing-folder"
        return f"gpt://{folder}/{self.yandex_model}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
