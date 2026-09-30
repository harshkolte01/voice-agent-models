from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

WHISPER_PUBLIC_ID = "whisper-large-v3-turbo"
WHISPER_LOAD_ID = "large-v3-turbo"
PARAKEET_PUBLIC_ID = "parakeet-unified-en-0.6b"
PARAKEET_LOAD_ID = "nvidia/parakeet-unified-en-0.6b"

WHISPER_ALIASES = frozenset(
    {
        "whisper",
        "large-v3-turbo",
        "whisper-large-v3-turbo",
    }
)
PARAKEET_ALIASES = frozenset(
    {
        "parakeet",
        "parakeet-unified-en-0.6b",
        "nvidia/parakeet-unified-en-0.6b",
    }
)


@dataclass(frozen=True)
class SttSpec:
    backend: str  # "whisper" | "parakeet"
    load_id: str
    public_id: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    language_probability: float = 0.0
    model: str = ""


def resolve_stt_spec(model_name: str) -> SttSpec:
    key = (model_name or "").strip().lower()
    if not key:
        raise ValueError("STT_MODEL must not be empty")
    if key in WHISPER_ALIASES:
        return SttSpec(
            backend="whisper",
            load_id=WHISPER_LOAD_ID,
            public_id=WHISPER_PUBLIC_ID,
        )
    if key in PARAKEET_ALIASES:
        return SttSpec(
            backend="parakeet",
            load_id=PARAKEET_LOAD_ID,
            public_id=PARAKEET_PUBLIC_ID,
        )
    raise ValueError(
        f"Unknown STT_MODEL '{model_name}'. "
        f"Use 'large-v3-turbo' (Whisper) or 'parakeet' (Parakeet Unified EN 0.6B)."
    )


def public_model_id(model_name: str) -> str:
    return resolve_stt_spec(model_name).public_id


def normalize_stt_model(requested: str | None, loaded_public_id: str) -> str:
    if not requested or not requested.strip():
        return loaded_public_id
    try:
        requested_spec = resolve_stt_spec(requested)
    except ValueError as exc:
        raise ValueError(
            f"Unknown STT model '{requested}'. Use '{loaded_public_id}'."
        ) from exc
    if requested_spec.public_id != loaded_public_id:
        raise ValueError(
            f"Unknown STT model '{requested}'. Use '{loaded_public_id}'."
        )
    return loaded_public_id


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str, backend: str = "whisper") -> str:
    requested = device.strip().lower()
    if requested == "auto":
        if backend == "parakeet":
            return "cuda" if torch_cuda_available() else "cpu"
        return "cuda" if cuda_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported STT_DEVICE: {device}")
    return requested


def resolve_compute_type(compute_type: str, device: str) -> str:
    requested = compute_type.strip().lower()
    if requested == "auto":
        return "float16" if device == "cuda" else "int8"
    return requested


def _load_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _load_parakeet_model(model_name: str, device: str):
    from omegaconf import OmegaConf, open_dict
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    # NeMo RNNT transcribe path reads cfg.validation_ds; some HF checkpoints leave it null.
    with open_dict(model.cfg):
        if getattr(model.cfg, "validation_ds", None) is None:
            model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})
        elif "use_start_end_token" not in model.cfg.validation_ds:
            model.cfg.validation_ds.use_start_end_token = False
    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    return model


def _split_wav_chunks(wav_path: str, chunk_seconds: float = 25.0) -> list[str]:
    """Split long wav into temp chunk files (NeMo recommends ~5–25s fragments)."""
    with wave.open(wav_path, "rb") as handle:
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        rate = handle.getframerate()
        total_frames = handle.getnframes()
        frames_per_chunk = max(1, int(rate * chunk_seconds))
        if total_frames <= frames_per_chunk:
            return [wav_path]

        paths: list[str] = []
        while True:
            data = handle.readframes(frames_per_chunk)
            if not data:
                break
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(out_path, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(sampwidth)
                out.setframerate(rate)
                out.writeframes(data)
            paths.append(out_path)
        return paths


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        return 0.0
    return float(frames) / float(rate)


def _write_pcm16_mono_wav(path: str, samples, sample_rate: int = 16000) -> None:
    import array

    clipped = [max(-32768, min(32767, int(x))) for x in samples]
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.array("h", clipped).tobytes())


def _convert_with_torchaudio(audio_path: str, out_path: str) -> None:
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(sample_rate) != 16000:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16000)
    samples = (waveform.squeeze(0).clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    _write_pcm16_mono_wav(out_path, samples.tolist(), 16000)


def _ensure_mono_16k_wav(audio_path: str) -> tuple[str, bool]:
    """Return (path, is_temp). Prefer ffmpeg; fall back to torchaudio for WAV/loadable audio."""
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(audio_path, "rb") as handle:
                channels = handle.getnchannels()
                rate = handle.getframerate()
                sampwidth = handle.getsampwidth()
            if channels == 1 and rate == 16000 and sampwidth == 2:
                return audio_path, False
        except wave.Error:
            pass

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return out_path, True
    except FileNotFoundError:
        try:
            _convert_with_torchaudio(audio_path, out_path)
            return out_path, True
        except Exception as exc:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise RuntimeError(
                "ffmpeg is required on PATH for Parakeet STT "
                "(or torchaudio must be able to load the audio)."
            ) from exc
    except subprocess.CalledProcessError as exc:
        try:
            # ffmpeg failed (bad input / codecs); try torchaudio before giving up
            _convert_with_torchaudio(audio_path, out_path)
            return out_path, True
        except Exception:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"ffmpeg failed to convert audio: {detail}") from exc


def _parakeet_text(output) -> str:
    if not output:
        return ""
    first = output[0]
    text = getattr(first, "text", None)
    if text is None and isinstance(first, dict):
        text = first.get("text")
    if text is None:
        text = str(first)
    return str(text).strip()


class Transcriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
    ):
        self.spec = resolve_stt_spec(model_name)
        self.backend = self.spec.backend
        self.model_name = self.spec.load_id
        self.public_id = self.spec.public_id
        # Back-compat for callers/tests that still read whisper_public_id
        self.whisper_public_id = self.public_id
        self.device = resolve_device(device, backend=self.backend)
        self.compute_type = (
            resolve_compute_type(compute_type, self.device)
            if self.backend == "whisper"
            else "n/a"
        )
        self._lock = threading.Lock()
        if self.backend == "whisper":
            self.model = _load_whisper_model(
                self.model_name,
                self.device,
                self.compute_type,
            )
        else:
            self.model = _load_parakeet_model(self.model_name, self.device)
        self.loaded = True

    @classmethod
    def from_settings(cls, settings: Settings) -> Transcriber:
        return cls(
            model_name=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )

    def listed_stt_models(self) -> list[str]:
        return [self.public_id]

    def _transcribe_whisper(
        self,
        audio_path: str,
        language: str | None,
        chosen: str,
    ) -> TranscriptionResult:
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

    def _transcribe_parakeet(
        self,
        audio_path: str,
        chosen: str,
    ) -> TranscriptionResult:
        wav_path, is_temp = _ensure_mono_16k_wav(audio_path)
        chunk_paths: list[str] = []
        try:
            chunk_paths = _split_wav_chunks(wav_path, chunk_seconds=25.0)
            texts: list[str] = []
            with self._lock:
                for chunk in chunk_paths:
                    output = self.model.transcribe([chunk], batch_size=1)
                    piece = _parakeet_text(output)
                    if piece:
                        texts.append(piece)
            text = " ".join(texts).strip()
            duration = _wav_duration_seconds(wav_path)
        finally:
            for chunk in chunk_paths:
                if chunk != wav_path:
                    try:
                        os.unlink(chunk)
                    except OSError:
                        pass
            if is_temp:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        return TranscriptionResult(
            text=text,
            language="en",
            duration=duration,
            language_probability=1.0,
            model=chosen,
        )

    def transcribe_sync(
        self,
        audio_path: str,
        language: str | None = None,
        model: str | None = None,
    ) -> TranscriptionResult:
        chosen = normalize_stt_model(model, self.public_id)
        if self.backend == "whisper":
            return self._transcribe_whisper(audio_path, language, chosen)
        return self._transcribe_parakeet(audio_path, chosen)

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
