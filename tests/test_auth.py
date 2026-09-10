from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.auth import parse_bearer
from app.keys import (
    create_key,
    find_valid_key,
    hash_key,
    key_prefix,
    load_keys,
    save_keys,
    slugify,
)
from app.embed import resolve_embed_device
from app.rerank import resolve_rerank_device
from app.transcribe import public_model_id, resolve_compute_type, resolve_device


def test_hash_key_is_sha256() -> None:
    plaintext = "stt_live_example"
    assert hash_key(plaintext) == hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def test_create_key_stores_hash_not_plaintext(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.json"
    plaintext, record = create_key(keys_file, name="Developer A")

    assert plaintext.startswith("stt_live_")
    assert plaintext not in keys_file.read_text(encoding="utf-8")
    assert record.hash == hash_key(plaintext)
    assert record.prefix == key_prefix(plaintext)
    assert record.id == "developer-a"
    assert record.revoked is False
    stored = load_keys(keys_file)
    assert len(stored) == 1
    assert stored[0].hash == record.hash


def test_create_key_duplicate_id_raises(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.json"
    create_key(keys_file, name="dev-a")
    with pytest.raises(ValueError, match="already exists"):
        create_key(keys_file, name="dev-a")


def test_find_valid_key_accepts_matching_key(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.json"
    plaintext, _ = create_key(keys_file, name="dev-a")
    found = find_valid_key(keys_file, plaintext)
    assert found is not None
    assert found.id == "dev-a"


def test_find_valid_key_rejects_unknown_and_revoked(tmp_path: Path) -> None:
    keys_file = tmp_path / "keys.json"
    plaintext, record = create_key(keys_file, name="dev-a")
    assert find_valid_key(keys_file, "stt_live_not-a-real-key") is None

    record.revoked = True
    save_keys(keys_file, [record])
    assert find_valid_key(keys_file, plaintext) is None


def test_parse_bearer_rejects_missing_and_malformed() -> None:
    with pytest.raises(HTTPException) as missing:
        parse_bearer(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as malformed:
        parse_bearer("Token abc")
    assert malformed.value.status_code == 401

    assert parse_bearer("Bearer stt_live_ok") == "stt_live_ok"


def test_slugify() -> None:
    assert slugify("Developer A") == "developer-a"
    assert slugify("***") == "key"


def test_resolve_device_auto_uses_cuda_when_available() -> None:
    with patch("app.transcribe.cuda_available", return_value=True):
        assert resolve_device("auto") == "cuda"
    with patch("app.transcribe.cuda_available", return_value=False):
        assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_rerank_device_auto_uses_cuda_when_available() -> None:
    with patch("app.rerank.torch_cuda_available", return_value=True):
        assert resolve_rerank_device("auto") == "cuda"
    with patch("app.rerank.torch_cuda_available", return_value=False):
        assert resolve_rerank_device("auto") == "cpu"
    assert resolve_rerank_device("cpu") == "cpu"


def test_resolve_embed_device_auto_uses_cuda_when_available() -> None:
    with patch("app.embed.torch_cuda_available", return_value=True):
        assert resolve_embed_device("auto") == "cuda"
    with patch("app.embed.torch_cuda_available", return_value=False):
        assert resolve_embed_device("auto") == "cpu"
    assert resolve_embed_device("cpu") == "cpu"


def test_resolve_compute_type_auto() -> None:
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"


def test_public_model_id() -> None:
    assert public_model_id("large-v3-turbo") == "whisper-large-v3-turbo"
    assert public_model_id("whisper-large-v3-turbo") == "whisper-large-v3-turbo"


def test_normalize_stt_model() -> None:
    from app.transcribe import normalize_stt_model

    whisper = "whisper-large-v3-turbo"
    assert normalize_stt_model(None, whisper) == whisper
    assert normalize_stt_model("whisper", whisper) == whisper
    assert normalize_stt_model("large-v3-turbo", whisper) == whisper
    with pytest.raises(ValueError, match="Unknown STT model"):
        normalize_stt_model("sravaani-1.0", whisper)
    with pytest.raises(ValueError, match="Unknown STT model"):
        normalize_stt_model("omni-7b", whisper)


def test_normalize_tts_model() -> None:
    from app.tts import KOKORO_PUBLIC_ID, normalize_tts_model

    assert normalize_tts_model(None) == KOKORO_PUBLIC_ID
    assert normalize_tts_model("kokoro") == KOKORO_PUBLIC_ID
    assert normalize_tts_model("hexgrad/Kokoro-82M") == KOKORO_PUBLIC_ID
    with pytest.raises(ValueError, match="Unknown TTS model"):
        normalize_tts_model("rumik-oss-1")
    with pytest.raises(ValueError, match="Unknown TTS model"):
        normalize_tts_model("xtts")


def test_normalize_kokoro_voice() -> None:
    from app.tts import kokoro_speed_for_pace, lang_code_for_voice, normalize_kokoro_voice

    assert normalize_kokoro_voice(None) == "af_heart"
    assert normalize_kokoro_voice("Ira") == "af_heart"
    assert normalize_kokoro_voice("Zoya") == "af_nicole"
    assert normalize_kokoro_voice("am_liam") == "am_liam"
    assert lang_code_for_voice("bf_emma") == "b"
    assert kokoro_speed_for_pace("fast") == 1.2
    with pytest.raises(ValueError, match="Kokoro voice"):
        normalize_kokoro_voice("NotAVoice")


def test_speech_request_aliases() -> None:
    from app.schemas import SpeechRequest

    body = SpeechRequest.model_validate({"text": "hello", "voice": "Zoya"})
    assert body.input == "hello"
    assert body.speaker == "Zoya"


def test_kokoro_engine_mocked_synth() -> None:
    import numpy as np

    from app.tts import KokoroEngine

    engine = KokoroEngine(
        device="cpu",
        max_chars=2000,
        default_voice="af_heart",
        default_lang="a",
    )

    def fake_pipeline(_lang: str):
        def generate(text, voice, speed):
            yield ("gs", "ps", np.full(8, 0.1, dtype=np.float32))

        return generate

    engine._pipeline = fake_pipeline  # type: ignore[method-assign]
    result = engine.synthesize_sync("hello", speaker="af_heart", pace="steady")
    assert result.model == "kokoro-82m"
    assert result.speaker == "af_heart"
    assert result.wav_bytes[:4] == b"RIFF"
    assert result.device == "cpu"


def test_pipeline_audio_uses_audio_attribute() -> None:
    from types import SimpleNamespace

    import numpy as np

    from app.tts import _pipeline_audio, _to_float32_audio

    item = SimpleNamespace(graphemes="hi", phonemes="hˈaɪ", audio=np.ones(4, dtype=np.float32))
    chunk = _to_float32_audio(_pipeline_audio(item))
    assert chunk.shape == (4,)
