from functools import lru_cache
from typing import List, Optional

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
    yandex_system_prompt: Optional[str] = Field(default=None, alias="YANDEX_SYSTEM_PROMPT")
    yandex_stream: bool = Field(default=True, alias="YANDEX_STREAM")
    yandex_model_uri: Optional[str] = Field(default=None, alias="YANDEX_MODEL_URI")
    yandex_kb_id: Optional[str] = Field(default=None, alias="YANDEX_KB_ID")
    yandex_temperature: float = Field(default=0.3, alias="YANDEX_TEMPERATURE")
    yandex_max_tokens: int = Field(default=800, alias="YANDEX_MAX_TOKENS")
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")
    ai_agent_secret: Optional[str] = Field(default=None, alias="AI_AGENT_SECRET")
    rate_limit_requests: int = Field(default=120, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_window_sec: float = Field(default=60.0, alias="RATE_LIMIT_WINDOW_SEC")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    yandex_assistant_url: str = Field(
        default="https://rest-assistant.api.cloud.yandex.net/v1/responses",
        alias="YANDEX_ASSISTANT_URL",
    )
    yandex_agent_id: Optional[str] = Field(default=None, alias="YANDEX_AGENT_ID")
    vector_store_ids_raw: Optional[str] = Field(
        default=None, alias="YANDEX_VECTOR_STORE_IDS"
    )

    @property
    def model_uri(self) -> str:
        if self.yandex_model_uri:
            return self.yandex_model_uri
        folder = self.yandex_folder_id or "missing-folder"
        return f"gpt://{folder}/{self.yandex_model}"

    @property
    def memory_enabled(self) -> bool:
        return bool(self.vector_store_ids)

    @property
    def vector_store_ids(self) -> List[str]:
        if not self.vector_store_ids_raw:
            return []
        return [item.strip() for item in self.vector_store_ids_raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
