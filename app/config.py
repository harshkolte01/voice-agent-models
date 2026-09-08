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
    sravaani_enabled: bool = True
    sravaani_model: str = "ARTPARK-IISc/SraVaani-1.0"
    sravaani_device: str = "auto"
    sravaani_load_on_startup: bool = False
    tts_enabled: bool = True
    tts_model: str = "rumik-ai/rumik-oss-1"
    tts_device: str = "auto"
    tts_load_on_startup: bool = False
    tts_max_chars: int = 2000
    tts_default_speaker: str = "Ira"
    hf_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
