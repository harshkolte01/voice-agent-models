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
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "auto"
    rerank_max_docs: int = 64
    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "auto"
    embed_max_texts: int = 64
    embed_max_length: int = 8192


@lru_cache
def get_settings() -> Settings:
    return Settings()
