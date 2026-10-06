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
from app.stt_text import normalize_stt_text

WHISPER_PUBLIC_ID = "whisper-large-v3-turbo"
WHISPER_LOAD_ID = "large-v3-turbo"
PARAKEET_PUBLIC_ID = "parakeet-unified-en-0.6b"
PARAKEET_LOAD_ID = "nvidia/parakeet-unified-en-0.6b"
PHONON_PUBLIC_ID = "phonon-2"
PHONON_LOAD_ID = "FermionResearch/Phonon-2"

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
PHONON_ALIASES = frozenset(
    {
        "phonon",
        "phonon-2",
        "phonon2",
        "fermionresearch/phonon-2",
    }
)


@dataclass(frozen=True)
class SttSpec:
    backend: str  # "whisper" | "parakeet" | "phonon"
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
    if key in PHONON_ALIASES:
        return SttSpec(
            backend="phonon",
            load_id=PHONON_LOAD_ID,
            public_id=PHONON_PUBLIC_ID,
        )
    raise ValueError(
        f"Unknown STT_MODEL '{model_name}'. "
        f"Use 'large-v3-turbo' (Whisper), 'parakeet' (Parakeet Unified EN 0.6B), "
        f"or 'phonon-2' (Fermion Phonon-2)."
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
        if backend in {"parakeet", "phonon"}:
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


def _bind_phonon_cuda(speech):
    """Move official Phonon-2 dense torch graph onto CUDA.

    Fermion's Windows/Linux pip engine is CPU (packed C kernels). We load
    fp32 weights and run the same ParakeetForTDT decode on GPU.
    """
    import types

    import numpy as np
    import torch
    from fermion._speech.engine_phonon2_cpu import (
        HOP,
        LOG_GUARD,
        N_FFT,
        PREEMPH,
        WIN,
    )

    speech._cenc = None
    speech._ctdt = None
    speech._bf16 = False
    speech.model = speech.model.to("cuda")
    speech.model.eval()
    speech._window = speech._window.to("cuda")
    speech._melf = speech._melf.to("cuda")
    device = torch.device("cuda")

    def _log_mel(self, wave):
        x = torch.as_tensor(np.asarray(wave, dtype=np.float32), device=device)[None]
        x = torch.cat([x[:, :1], x[:, 1:] - PREEMPH * x[:, :-1]], dim=1)
        st = torch.stft(
            x,
            N_FFT,
            hop_length=HOP,
            win_length=WIN,
            window=self._window,
            return_complex=True,
            pad_mode="constant",
        )
        mag = torch.view_as_real(st)
        mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
        mel = torch.log(self._melf @ mag + LOG_GUARD).permute(0, 2, 1)
        n = mel.shape[1]
        mean = mel.mean(1, keepdim=True)
        var = ((mel - mean) ** 2).sum(1) / (n - 1)
        mel = (mel - mean) / (torch.sqrt(var).unsqueeze(1) + 1e-5)
        return mel, torch.ones((1, n), dtype=torch.long, device=device)

    def _decode_single(self, audio, repetition_penalty: float):
        m = self.model
        cfg = m.config
        blank = cfg.blank_token_id
        vocab_size = cfg.vocab_size
        with torch.inference_mode():
            feats, am = self._log_mel(audio)
            enc = m.encoder(input_features=feats, attention_mask=am).last_hidden_state
            t_len = int(m._get_subsampling_output_length(am.sum(-1))[0])
            encp = m.encoder_projector(enc)[0]
            step = 0
            last = blank
            ids: list[int] = []
            frames: list[int] = []
            durs: list[int] = []
            nsym = 0
            it = 0
            h = torch.zeros(
                m.decoder.lstm.num_layers,
                1,
                m.decoder.lstm.hidden_size,
                device=device,
            )
            c = torch.zeros_like(h)
            while step < t_len and it < self._max_sym * t_len + 16:
                emb = m.decoder.embedding(torch.tensor([[last]], device=device))
                out, (h2, c2) = m.decoder.lstm(emb, (h, c))
                dec = m.decoder.decoder_projector(out[0])
                logits = m.joint.head(m.joint.activation(encp[step][None] + dec))[0]
                tok = int(logits[:vocab_size].argmax())
                dur = int(self._durations[int(logits[vocab_size:].argmax())])
                it += 1
                if tok == blank and dur == 0:
                    dur = 1
                if tok != blank:
                    ids.append(tok)
                    frames.append(step)
                    durs.append(dur)
                    last = tok
                    h, c = h2, c2
                if dur == 0:
                    nsym += 1
                    if nsym >= self._max_sym:
                        dur = 1
                        nsym = 0
                else:
                    nsym = 0
                step += dur
        return self._finish(ids, frames, durs, t_len)

    speech._log_mel = types.MethodType(_log_mel, speech)
    speech._decode_single = types.MethodType(_decode_single, speech)
    speech.decode = dict(getattr(speech, "decode", {}) or {})
    speech.decode["device"] = "cuda"
    return speech


def _load_phonon_model(model_name: str, device: str):
    """Load official FermionResearch/Phonon-2 via fermion-research."""
    try:
        import fermion
    except ImportError as exc:
        raise RuntimeError(
            "Phonon-2 requires fermion-research. Install with: "
            "pip install fermion-research --no-deps "
            "&& pip install soundfile scipy zstandard"
        ) from exc
    # Dense fp32 torch graph (movable to CUDA). Packed C kernels stay on CPU.
    os.environ["FERMION_P2_CPU"] = "fp32"
    os.environ["FERMION_P2_CPU_TDT"] = "off"
    os.environ["FERMION_P2_CPU_ENC"] = "off"
    try:
        speech = fermion.load_speech("phonon-2")
    except SystemExit as exc:
        raise RuntimeError(str(exc) or f"failed to load {model_name}") from exc
    if device == "cuda" and torch_cuda_available():
        return _bind_phonon_cuda(speech)
    return speech


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
                "ffmpeg is required on PATH for Parakeet/Phonon STT "
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


def _phonon_result(output, wav_path: str) -> tuple[str, float]:
    """fermion SpeechModel.transcribe returns (text, decode_s, audio_s)."""
    text = ""
    duration = _wav_duration_seconds(wav_path)
    if isinstance(output, (tuple, list)) and output:
        text = str(output[0] or "").strip()
        if len(output) > 2 and output[2] is not None:
            try:
                duration = float(output[2])
            except (TypeError, ValueError):
                pass
    elif output is not None:
        text = str(getattr(output, "text", output) or "").strip()
    return text, duration


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
        elif self.backend == "parakeet":
            self.model = _load_parakeet_model(self.model_name, self.device)
        else:
            self.model = _load_phonon_model(self.model_name, self.device)
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

    def _transcribe_phonon(
        self,
        audio_path: str,
        chosen: str,
    ) -> TranscriptionResult:
        wav_path, is_temp = _ensure_mono_16k_wav(audio_path)
        try:
            with self._lock:
                output = self.model.transcribe(wav_path)
            text, duration = _phonon_result(output, wav_path)
        finally:
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
            result = self._transcribe_whisper(audio_path, language, chosen)
        elif self.backend == "parakeet":
            result = self._transcribe_parakeet(audio_path, chosen)
        else:
            result = self._transcribe_phonon(audio_path, chosen)
        result.text = normalize_stt_text(result.text)
        return result

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
