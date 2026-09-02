from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stt_model: str = "large-v3-turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    stt_keys_file: str = "keys.json"
    stt_max_upload_mb: int = 25
    stt_host: str = "127.0.0.1"
    stt_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
