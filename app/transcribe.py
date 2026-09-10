from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from app.config import Settings


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    language_probability: float = 0.0
    model: str = ""


def public_model_id(model_name: str) -> str:
    if model_name.startswith("whisper-"):
        return model_name
    return f"whisper-{model_name}"


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported STT_DEVICE: {device}")
    return requested


def resolve_compute_type(compute_type: str, device: str) -> str:
    requested = compute_type.strip().lower()
    if requested == "auto":
        return "float16" if device == "cuda" else "int8"
    return requested


def normalize_stt_model(requested: str | None, whisper_public_id: str) -> str:
    if not requested or not requested.strip():
        return whisper_public_id
    key = requested.strip().lower()
    aliases = {
        "whisper": whisper_public_id,
        "large-v3-turbo": whisper_public_id,
        whisper_public_id.lower(): whisper_public_id,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown STT model '{requested}'. Use '{whisper_public_id}'.")


def _load_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.whisper_public_id = public_model_id(model_name)
        self._lock = threading.Lock()
        self.model = _load_whisper_model(
            self.model_name,
            self.device,
            self.compute_type,
        )
        self.loaded = True

    @classmethod
    def from_settings(cls, settings: Settings) -> Transcriber:
        return cls(
            model_name=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )

    def listed_stt_models(self) -> list[str]:
        return [self.whisper_public_id]

    def transcribe_sync(
        self,
        audio_path: str,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        chosen = normalize_stt_model(model, self.whisper_public_id)
        kwargs = {}
        if language:
            kwargs["language"] = language

        with self._lock:
            segments, info = self.model.transcribe(audio_path, **kwargs)
            text = "".join(segment.text for segment in segments).strip()

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        detected = getattr(info, "language", None) or language or "unknown"
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        return TranscriptionResult(
            text=text,
            language=detected,
            duration=duration,
            language_probability=round(language_probability, 4),
            model=chosen,
        )

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        return await asyncio.to_thread(
            self.transcribe_sync,
            audio_path,
            language,
            model,
        )
