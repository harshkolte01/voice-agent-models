from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.rerank import resolve_rerank_device
from app.speech_text import prepare_speech_text


KOKORO_PUBLIC_ID = "kokoro-82m"
KOKORO_SAMPLE_RATE = 24000
PACES = ("slow", "fast", "steady")
PACE_LOOKUP = {name.lower(): name for name in PACES}
PACE_LOOKUP.update({f"{name.lower()} pace": name for name in PACES})
VOICE_ALIASES = {
    "Ira": "af_heart",
    "Aisha": "af_bella",
    "Siya": "af_sarah",
    "Zoya": "af_nicole",
}
ALIAS_LOOKUP = {name.lower(): name for name in VOICE_ALIASES}
KOKORO_VOICE_RE = re.compile(r"^[a-z][fm]_[a-z0-9]+$", re.IGNORECASE)
KOKORO_LANG_CODES = frozenset("abefhijpz")
KOKORO_PACE_SPEED = {
    "slow": 0.85,
    "steady": 1.0,
    "fast": 1.2,
}
logger = logging.getLogger("stt.tts")


@dataclass
class SpeechResult:
    wav_bytes: bytes
    speaker: str
    prompt: str
    model: str
    device: str


def normalize_tts_model(requested: str | None) -> str:
    if not requested or not requested.strip():
        return KOKORO_PUBLIC_ID
    key = requested.strip().lower()
    aliases = {
        "kokoro": KOKORO_PUBLIC_ID,
        "kokoro-82m": KOKORO_PUBLIC_ID,
        "kokoro_82m": KOKORO_PUBLIC_ID,
        "hexgrad/kokoro-82m": KOKORO_PUBLIC_ID,
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown TTS model '{requested}'. Use '{KOKORO_PUBLIC_ID}'.")


def kokoro_speed_for_pace(pace: str | None) -> float:
    if value := (pace or "").strip():
        canonical = PACE_LOOKUP.get(value.lower())
        if canonical is not None:
            return KOKORO_PACE_SPEED[canonical]
        raise ValueError(f"pace must be one of: {', '.join(PACES)}")
    return 1.0


def lang_code_for_voice(voice: str, fallback: str = "a") -> str:
    if voice:
        code = voice[0].lower()
        if code in KOKORO_LANG_CODES:
            return code
    fallback_code = (fallback or "a").strip().lower()[:1] or "a"
    if fallback_code in KOKORO_LANG_CODES:
        return fallback_code
    return "a"


def normalize_kokoro_voice(
    value: str | None,
    default_voice: str = "af_heart",
) -> str:
    raw = (value or "").strip()
    if not raw:
        return default_voice
    alias = ALIAS_LOOKUP.get(raw.lower())
    if alias is not None:
        return VOICE_ALIASES[alias]
    if KOKORO_VOICE_RE.fullmatch(raw):
        return raw.lower()
    short_names = ", ".join(VOICE_ALIASES)
    raise ValueError(
        "speaker must be a Kokoro voice such as af_heart, am_liam, bf_emma, "
        f"or a short name ({short_names})"
    )


def _configure_espeak() -> None:
    """Point phonemizer at espeak-ng so OOD words are not skipped (robotic gaps)."""
    try:
        import espeakng_loader

        os.environ.setdefault(
            "PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path()
        )
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        data_path = espeakng_loader.get_data_path()
        if hasattr(EspeakWrapper, "set_data_path"):
            EspeakWrapper.set_data_path(data_path)
        else:
            EspeakWrapper.data_path = data_path
        return
    except Exception:
        pass
    dll = Path(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
    if dll.exists():
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(dll))


def _pipeline_audio(item):
    audio = getattr(item, "audio", None)
    if audio is not None:
        return audio
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        return item[2]
    getter = getattr(item, "__getitem__", None)
    if callable(getter):
        try:
            return item[2]
        except Exception:
            pass
    return item


def _to_float32_audio(audio):
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    samples = np.asarray(audio, dtype=np.float32)
    return samples.reshape(-1)


def pcm_wav_bytes(audio, sample_rate: int = KOKORO_SAMPLE_RATE) -> bytes:
    import numpy as np

    samples = _to_float32_audio(audio)
    pcm_i16 = np.clip(samples * 32767.0, -32767, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())
    return buffer.getvalue()


class KokoroEngine:
    def __init__(
        self,
        device: str,
        max_chars: int,
        default_voice: str,
        default_lang: str,
    ):
        self.public_id = KOKORO_PUBLIC_ID
        self.public_ids = [KOKORO_PUBLIC_ID]
        self.model_name = KOKORO_PUBLIC_ID
        self.device = resolve_rerank_device(device)
        self.max_chars = max_chars
        self.default_voice = normalize_kokoro_voice(default_voice, default_voice)
        self.default_lang = lang_code_for_voice(self.default_voice, default_lang)
        self.enabled = True
        self.loaded = False
        self._lock = threading.RLock()
        self._pipelines: dict[str, object] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> KokoroEngine:
        return cls(
            device=settings.tts_device,
            max_chars=settings.tts_max_chars,
            default_voice=settings.tts_kokoro_voice,
            default_lang=settings.tts_kokoro_lang,
        )

    def ensure_loaded(self) -> None:
        with self._lock:
            self._pipeline(self.default_lang)
            try:
                list(
                    self._pipelines[self.default_lang](
                        "Ready.",
                        voice=self.default_voice,
                        speed=1,
                    )
                )
            except Exception as exc:
                logger.warning("Kokoro warmup skipped: %s", exc)

    def _pipeline(self, lang_code: str):
        pipeline = self._pipelines.get(lang_code)
        if pipeline is not None:
            return pipeline
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS is not installed. pip install 'kokoro>=0.9.4'"
            ) from exc
        _configure_espeak()
        try:
            pipeline = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
                device=self.device,
            )
        except TypeError:
            pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        self._pipelines[lang_code] = pipeline
        self.loaded = True
        return pipeline

    def synthesize_sync(
        self,
        text: str,
        speaker: str | None = None,
        pace: str | None = None,
        model: str | None = None,
    ) -> SpeechResult:
        import numpy as np

        normalize_tts_model(model)
        raw = text.strip()
        if not raw:
            raise ValueError("input must not be empty")
        if len(raw) > self.max_chars:
            raise ValueError(f"input must be at most {self.max_chars} characters")
        spoken = prepare_speech_text(raw)
        if not spoken:
            raise ValueError("input has no speakable text after removing Markdown")
        if spoken != raw:
            logger.debug("TTS sanitized %d chars -> %d chars", len(raw), len(spoken))
        voice = normalize_kokoro_voice(speaker, self.default_voice)
        lang_code = lang_code_for_voice(voice, self.default_lang)
        speed = kokoro_speed_for_pace(pace)
        with self._lock:
            pipeline = self._pipeline(lang_code)
            chunks = []
            for item in pipeline(spoken, voice=voice, speed=speed):
                audio = _pipeline_audio(item)
                if audio is None:
                    continue
                chunk = _to_float32_audio(audio)
                if chunk.size:
                    chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Kokoro produced no audio")
        wav_bytes = pcm_wav_bytes(np.concatenate(chunks), KOKORO_SAMPLE_RATE)
        return SpeechResult(
            wav_bytes=wav_bytes,
            speaker=voice,
            prompt=spoken,
            model=KOKORO_PUBLIC_ID,
            device=self.device,
        )

    async def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        pace: str | None = None,
        model: str | None = None,
    ) -> SpeechResult:
        return await asyncio.to_thread(
            self.synthesize_sync,
            text,
            speaker,
            pace,
            model,
        )


class TtsEngine(KokoroEngine):
    @classmethod
    def from_settings(cls, settings: Settings) -> TtsEngine:
        engine = cls(
            device=settings.tts_device,
            max_chars=settings.tts_max_chars,
            default_voice=settings.tts_kokoro_voice,
            default_lang=settings.tts_kokoro_lang,
        )
        engine.ensure_loaded()
        return engine
