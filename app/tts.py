from __future__ import annotations

import asyncio
import importlib.util
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.gpu_slot import ExclusiveCudaSlot
from app.rerank import resolve_rerank_device


RUMIK_PUBLIC_ID = "rumik-oss-1"
SPEAKERS = ("Ira", "Aisha", "Siya", "Zoya")
TONES = ("happy", "sad", "angry", "excited", "professional")
ACCENTS = (
    "Hindi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Bengali",
    "Punjabi",
    "Indian English",
)
PACES = ("slow", "fast", "steady")
DESCRIPTION_RE = re.compile(r'^\s*<description="', re.IGNORECASE)
SPEAKER_LOOKUP = {name.lower(): name for name in SPEAKERS}
TONE_LOOKUP = {name.lower(): name for name in TONES}
ACCENT_LOOKUP = {name.lower(): name for name in ACCENTS}
ACCENT_LOOKUP.update({f"{name.lower()} accent": name for name in ACCENTS})
PACE_LOOKUP = {name.lower(): name for name in PACES}
PACE_LOOKUP.update({f"{name.lower()} pace": name for name in PACES})


@dataclass
class SpeechResult:
    wav_bytes: bytes
    speaker: str
    prompt: str


def normalize_speaker(value: str) -> str:
    canonical = SPEAKER_LOOKUP.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"speaker must be one of: {', '.join(SPEAKERS)}")
    return canonical


def normalize_tone(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    canonical = TONE_LOOKUP.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"tone must be one of: {', '.join(TONES)}")
    return canonical


def normalize_accent(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    canonical = ACCENT_LOOKUP.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"accent must be one of: {', '.join(ACCENTS)}")
    return canonical


def normalize_pace(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    canonical = PACE_LOOKUP.get(value.strip().lower())
    if canonical is None:
        raise ValueError(f"pace must be one of: {', '.join(PACES)}")
    return canonical


def build_tts_prompt(
    text: str,
    tone: str | None = None,
    accent: str | None = None,
    pace: str | None = None,
) -> str:
    stripped = text.strip()
    if DESCRIPTION_RE.match(stripped):
        return stripped
    tone = normalize_tone(tone)
    accent = normalize_accent(accent)
    pace = normalize_pace(pace)
    parts: list[str] = []
    if tone:
        parts.append(tone)
    if accent:
        parts.append(f"{accent} accent")
    if pace:
        parts.append(f"{pace} pace")
    if not parts:
        return stripped
    return f'<description="{", ".join(parts)}"> {stripped}'


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_rumik_module(repo_id: str):
    from huggingface_hub import snapshot_download

    source = Path(repo_id)
    root = source if source.exists() else Path(
        snapshot_download(
            repo_id,
            ignore_patterns=["assets/*", "samples/*", "benchmarks/*"],
        )
    )
    spec = importlib.util.spec_from_file_location(
        "rumik_oss_checkpoint_server",
        root / "server.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load rumik-oss server.py from {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, root


class TtsEngine:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_chars: int,
        default_speaker: str,
        gpu_slot: ExclusiveCudaSlot | None = None,
    ):
        self.model_name = model_name
        self.public_id = RUMIK_PUBLIC_ID
        self.device = resolve_rerank_device(device)
        self.max_chars = max_chars
        self.default_speaker = normalize_speaker(default_speaker)
        self.gpu_slot = gpu_slot
        self.enabled = True
        self.loaded = False
        self._lock = threading.RLock()
        self._engine = None
        self._module = None
        self._on_cuda = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        gpu_slot: ExclusiveCudaSlot | None = None,
    ) -> TtsEngine:
        return cls(
            model_name=settings.tts_model,
            device=settings.tts_device,
            max_chars=settings.tts_max_chars,
            default_speaker=settings.tts_default_speaker,
            gpu_slot=gpu_slot,
        )

    def _move_to(self, device: str) -> None:
        if self._engine is None:
            return
        self._engine.model.to(device)
        self._engine.mimi.to(device)
        self._engine.device = device
        if hasattr(self._engine, "allowed_ids"):
            self._engine.allowed_ids = self._engine.allowed_ids.to(device)
        self._on_cuda = device.startswith("cuda")

    def _release_cuda(self) -> None:
        if self._engine is not None and self._on_cuda:
            self._move_to("cpu")
            _empty_cuda_cache()

    def _acquire(self) -> None:
        if self._engine is None:
            module, root = _load_rumik_module(self.model_name)
            self._module = module
            self._engine = module.TinyAya(str(root), self.device)
            self.loaded = True
            self._on_cuda = self.device == "cuda"
            return
        if self.device == "cuda" and not self._on_cuda:
            self._move_to("cuda")

    def ensure_loaded(self) -> None:
        with self._inference_guard():
            pass

    def _inference_guard(self):
        if self.device == "cuda" and self.gpu_slot is not None:
            return self.gpu_slot.hold("rumik", self._acquire, self._release_cuda)
        from contextlib import contextmanager

        @contextmanager
        def _local():
            with self._lock:
                if not self.loaded:
                    self._acquire()
                yield

        return _local()

    def synthesize_sync(
        self,
        text: str,
        speaker: str | None = None,
        tone: str | None = None,
        accent: str | None = None,
        pace: str | None = None,
        temperature: float = 0.8,
        top_k: int = 30,
        max_new_tokens: int = 2048,
    ) -> SpeechResult:
        stripped = text.strip()
        if not stripped:
            raise ValueError("input must not be empty")
        if len(stripped) > self.max_chars:
            raise ValueError(f"input must be at most {self.max_chars} characters")
        chosen_speaker = normalize_speaker(speaker or self.default_speaker)
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= top_k <= 2048:
            raise ValueError("top_k must be between 0 and 2048")
        if not 8 <= max_new_tokens <= 3072:
            raise ValueError("max_new_tokens must be between 8 and 3072")
        prompt = build_tts_prompt(stripped, tone=tone, accent=accent, pace=pace)

        with self._inference_guard():
            assert self._engine is not None and self._module is not None
            request = self._module.SpeechRequest(
                input=prompt,
                speaker=chosen_speaker,
                temperature=float(temperature),
                top_k=int(top_k),
                max_new_tokens=int(max_new_tokens),
            )
            wav_bytes = self._engine.synthesize(request)
        return SpeechResult(
            wav_bytes=wav_bytes,
            speaker=chosen_speaker,
            prompt=prompt,
        )

    async def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        tone: str | None = None,
        accent: str | None = None,
        pace: str | None = None,
        temperature: float = 0.8,
        top_k: int = 30,
        max_new_tokens: int = 2048,
    ) -> SpeechResult:
        return await asyncio.to_thread(
            self.synthesize_sync,
            text,
            speaker,
            tone,
            accent,
            pace,
            temperature,
            top_k,
            max_new_tokens,
        )
