from __future__ import annotations

import threading
from pathlib import Path

from app.config import Settings
from app.gpu_slot import ExclusiveCudaSlot
from app.rerank import resolve_rerank_device
from app.transcribe import TranscriptionResult


SRAVAANI_PUBLIC_ID = "sravaani-1.0"


def audio_duration_seconds(audio_path: str) -> float:
    try:
        from faster_whisper.audio import decode_audio

        samples = decode_audio(audio_path, sampling_rate=16000)
        return float(len(samples) / 16000.0)
    except Exception:
        return 0.0


def _hypothesis_text(hyps) -> str:
    if hyps is None:
        return ""
    first = hyps[0] if isinstance(hyps, (list, tuple)) else hyps
    if isinstance(first, str):
        return first.strip()
    text = getattr(first, "text", None)
    if text is None and isinstance(first, dict):
        text = first.get("text")
    return str(text or first).strip()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class SraVaaniEngine:
    def __init__(
        self,
        model_name: str,
        device: str,
        token: str | None = None,
        gpu_slot: ExclusiveCudaSlot | None = None,
    ):
        self.model_name = model_name
        self.device = resolve_rerank_device(device)
        self.token = token
        self.gpu_slot = gpu_slot
        self._lock = threading.RLock()
        self.model = None
        self.loaded = False
        self._on_cuda = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        gpu_slot: ExclusiveCudaSlot | None = None,
    ) -> SraVaaniEngine:
        return cls(
            model_name=settings.sravaani_model,
            device=settings.sravaani_device,
            token=settings.hf_token,
            gpu_slot=gpu_slot,
        )

    def _load(self) -> None:
        import torch
        from transformers import AutoModel

        kwargs: dict = {"trust_remote_code": True}
        if self.token:
            kwargs["token"] = self.token
        model = AutoModel.from_pretrained(self.model_name, **kwargs)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            model = model.to(device=self.device, dtype=dtype)
        except TypeError:
            model = model.to(self.device)
        self.model = model.eval()
        self.loaded = True
        self._on_cuda = self.device == "cuda"

    def _release_cuda(self) -> None:
        if self.model is not None and self._on_cuda:
            self.model.to("cpu")
            self._on_cuda = False
            _empty_cuda_cache()

    def _acquire_cuda(self) -> None:
        if self.model is None:
            self._load()
            return
        if self.device == "cuda" and not self._on_cuda:
            self.model.to(self.device)
            self._on_cuda = True

    def ensure_loaded(self) -> None:
        with self._inference_guard():
            pass

    def _inference_guard(self):
        if self.device == "cuda" and self.gpu_slot is not None:
            return self.gpu_slot.hold("sravaani", self._acquire_cuda, self._release_cuda)
        from contextlib import contextmanager

        @contextmanager
        def _local():
            with self._lock:
                if not self.loaded:
                    self._load()
                yield

        return _local()

    def transcribe_sync(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        path = str(Path(audio_path))
        with self._inference_guard():
            assert self.model is not None
            hyps = None
            if language:
                try:
                    hyps = self.model.transcribe(
                        path,
                        language=language,
                        return_hypotheses=True,
                    )
                except TypeError:
                    hyps = None
            if hyps is None:
                try:
                    hyps = self.model.transcribe(path, return_hypotheses=True)
                except TypeError:
                    hyps = self.model.transcribe(path)

        text = _hypothesis_text(hyps)
        duration = audio_duration_seconds(path)
        return TranscriptionResult(
            text=text,
            language=language or "unknown",
            duration=duration,
            language_probability=0.0,
            model=SRAVAANI_PUBLIC_ID,
        )
