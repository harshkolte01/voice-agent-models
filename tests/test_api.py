from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.keys import create_key, save_keys, load_keys
from app.embed import EmbeddingResult
from app.rerank import RankedDocument
from app.transcribe import TranscriptionResult
from app.tts import SpeechResult


@pytest.fixture
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "keys.json"
    path.write_text('{"keys": []}\n', encoding="utf-8")
    monkeypatch.setenv("STT_KEYS_FILE", str(path))
    monkeypatch.setenv("STT_DEVICE", "cpu")
    monkeypatch.setenv("STT_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("STT_MODEL", "large-v3-turbo")
    monkeypatch.setenv("STT_MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setenv("RERANK_DEVICE", "cpu")
    monkeypatch.setenv("RERANK_MAX_DOCS", "64")
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBED_DEVICE", "cpu")
    monkeypatch.setenv("EMBED_MAX_TEXTS", "64")
    monkeypatch.setenv("EMBED_MAX_LENGTH", "8192")
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def dummy_transcriber() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "large-v3-turbo"
    dummy.sravaani_loaded = False
    dummy.listed_stt_models.return_value = [
        "whisper-large-v3-turbo",
        "sravaani-1.0",
    ]

    async def fake_transcribe(
        audio_path: str,
        language: str | None = None,
        model: str | None = None,
    ):
        from app.transcribe import normalize_stt_model

        used = normalize_stt_model(model, "whisper-large-v3-turbo")
        return TranscriptionResult(
            text="hello world",
            language=language or "en",
            duration=2.74,
            language_probability=0.98,
            model=used,
        )

    dummy.transcribe.side_effect = fake_transcribe
    return dummy


@pytest.fixture
def dummy_reranker() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "BAAI/bge-reranker-v2-m3"

    async def fake_rank(
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ):
        ranked = [
            RankedDocument(index=index, document=doc, score=float(index))
            for index, doc in enumerate(documents)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    dummy.rank.side_effect = fake_rank
    return dummy


@pytest.fixture
def dummy_embedder() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "BAAI/bge-m3"

    async def fake_encode(texts: list[str]):
        embeddings = []
        for index, _text in enumerate(texts):
            vector = [0.0] * 4
            vector[0] = 1.0 if index == 0 else 0.0
            vector[1] = 1.0 if index != 0 else 0.0
            embeddings.append(vector)
        return EmbeddingResult(embeddings=embeddings, dim=4)

    dummy.encode.side_effect = fake_encode
    return dummy


@pytest.fixture
def dummy_tts() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = False
    dummy.enabled = True
    dummy.device = "cpu"
    dummy.model_name = "rumik-ai/rumik-oss-1"
    dummy.public_id = "rumik-oss-1"

    async def fake_synthesize(
        text: str,
        speaker: str | None = None,
        tone: str | None = None,
        accent: str | None = None,
        pace: str | None = None,
        temperature: float = 0.8,
        top_k: int = 30,
        max_new_tokens: int = 2048,
    ):
        from app.tts import build_tts_prompt, normalize_speaker

        chosen = normalize_speaker(speaker or "Ira")
        prompt = build_tts_prompt(text, tone=tone, accent=accent, pace=pace)
        return SpeechResult(wav_bytes=b"RIFFWAV", speaker=chosen, prompt=prompt)

    dummy.synthesize.side_effect = fake_synthesize
    return dummy


@pytest.fixture
def client(
    keys_file: Path,
    dummy_transcriber: MagicMock,
    dummy_reranker: MagicMock,
    dummy_embedder: MagicMock,
    dummy_tts: MagicMock,
):
    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
        patch("app.main.Embedder.from_settings", return_value=dummy_embedder),
        patch("app.main.TtsEngine.from_settings", return_value=dummy_tts),
    ):
        from app.main import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def api_key(keys_file: Path) -> str:
    plaintext, _ = create_key(keys_file, name="test")
    return plaintext


def auth_header(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def test_health_unauthenticated(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["reranker_loaded"] is True
    assert body["embedder_loaded"] is True
    assert body["sravaani_loaded"] is False
    assert body["tts_loaded"] is False
    assert body["device"] == "cpu"


def test_access_log_line_ist_key_and_cf_ip(
    client: TestClient,
    api_key: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="stt.access"):
        response = client.post(
            "/v1/embeddings",
            headers={
                **auth_header(api_key),
                "cf-connecting-ip": "49.36.11.20",
                "cf-ipcountry": "IN",
            },
            json={"input": ["hello"]},
        )
    assert response.status_code == 200
    matching = [
        record.message
        for record in caplog.records
        if "POST /v1/embeddings" in record.message
    ]
    assert matching
    line = matching[-1]
    assert " IST  " in line
    assert "key=test" in line
    assert "ip=49.36.11.20" in line
    assert " IN  POST /v1/embeddings  200  " in line
    assert line.endswith("ms")


def test_models_requires_api_key(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth_header("stt_live_nope")).status_code == 401


def test_models_rejects_revoked_key(client: TestClient, keys_file: Path, api_key: str) -> None:
    records = load_keys(keys_file)
    records[0].revoked = True
    save_keys(keys_file, records)
    response = client.get("/v1/models", headers=auth_header(api_key))
    assert response.status_code == 401


def test_models_ok(client: TestClient, api_key: str) -> None:
    response = client.get("/v1/models", headers=auth_header(api_key))
    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"id": "whisper-large-v3-turbo", "type": "stt"},
            {"id": "sravaani-1.0", "type": "stt"},
            {"id": "BAAI/bge-reranker-v2-m3", "type": "rerank"},
            {"id": "BAAI/bge-m3", "type": "embedding"},
            {"id": "rumik-oss-1", "type": "tts"},
        ]
    }


def test_transcribe_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"fake-audio", "audio/wav")},
    )
    assert response.status_code == 401


def test_transcribe_ok(client: TestClient, api_key: str, dummy_transcriber: MagicMock) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_header(api_key),
        files={"file": ("test.wav", b"fake-audio", "audio/wav")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["language"] == "en"
    assert body["language_probability"] == 0.98
    assert body["duration"] == 2.74
    assert isinstance(body["processing_ms"], int)
    assert body["device"] == "cpu"
    assert body["model"] == "whisper-large-v3-turbo"
    assert "rtf" in body
    dummy_transcriber.transcribe.assert_called()


def test_transcribe_rejects_unsupported_format(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_header(api_key),
        files={"file": ("notes.txt", b"not-audio", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_transcribe_rejects_empty_file(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_header(api_key),
        files={"file": ("test.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty audio file"


def test_transcribe_rejects_oversize(
    keys_file: Path,
    dummy_transcriber: MagicMock,
    dummy_reranker: MagicMock,
    dummy_embedder: MagicMock,
    dummy_tts: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="tiny-limit")

    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
        patch("app.main.Embedder.from_settings", return_value=dummy_embedder),
        patch("app.main.TtsEngine.from_settings", return_value=dummy_tts),
    ):
        from app.main import app

        with TestClient(app) as test_client:
            response = test_client.post(
                "/v1/audio/transcriptions",
                headers=auth_header(plaintext),
                files={"file": ("test.wav", b"fake-audio", "audio/wav")},
            )

    assert response.status_code == 413
    dummy_transcriber.transcribe.assert_not_called()


def test_rerank_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/v1/rerank",
        json={"query": "meeting notes", "documents": ["doc a", "doc b"]},
    )
    assert response.status_code == 401


def test_rerank_ok(client: TestClient, api_key: str, dummy_reranker: MagicMock) -> None:
    response = client.post(
        "/v1/rerank",
        headers=auth_header(api_key),
        json={"query": "meeting notes", "documents": ["doc a", "doc b"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert isinstance(body["processing_ms"], int)
    assert body["results"] == [
        {"index": 1, "score": 1.0, "document": "doc b"},
        {"index": 0, "score": 0.0, "document": "doc a"},
    ]
    dummy_reranker.rank.assert_called()


def test_rerank_top_k(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/rerank",
        headers=auth_header(api_key),
        json={
            "query": "meeting notes",
            "documents": ["doc a", "doc b", "doc c"],
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["index"] == 2
    assert body["results"][0]["document"] == "doc c"


def test_rerank_rejects_empty_query(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/rerank",
        headers=auth_header(api_key),
        json={"query": "   ", "documents": ["doc a"]},
    )
    assert response.status_code == 422


def test_rerank_rejects_empty_documents(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/rerank",
        headers=auth_header(api_key),
        json={"query": "meeting notes", "documents": []},
    )
    assert response.status_code == 422


def test_rerank_rejects_blank_document(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/rerank",
        headers=auth_header(api_key),
        json={"query": "meeting notes", "documents": ["doc a", "  "]},
    )
    assert response.status_code == 422


def test_rerank_rejects_over_max_docs(
    keys_file: Path,
    dummy_transcriber: MagicMock,
    dummy_reranker: MagicMock,
    dummy_embedder: MagicMock,
    dummy_tts: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANK_MAX_DOCS", "1")
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="rerank-limit")

    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
        patch("app.main.Embedder.from_settings", return_value=dummy_embedder),
        patch("app.main.TtsEngine.from_settings", return_value=dummy_tts),
    ):
        from app.main import app

        with TestClient(app) as test_client:
            response = test_client.post(
                "/v1/rerank",
                headers=auth_header(plaintext),
                json={"query": "meeting notes", "documents": ["doc a", "doc b"]},
            )

    assert response.status_code == 400
    assert "Too many documents" in response.json()["detail"]
    dummy_reranker.rank.assert_not_called()


def test_embed_requires_api_key(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": ["hello"]})
    assert response.status_code == 401


def test_embed_ok(client: TestClient, api_key: str, dummy_embedder: MagicMock) -> None:
    response = client.post(
        "/v1/embeddings",
        headers=auth_header(api_key),
        json={"input": ["query", "document"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert body["model"] == "BAAI/bge-m3"
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 4
    assert isinstance(body["processing_ms"], int)
    dummy_embedder.encode.assert_called()


def test_embed_accepts_single_string(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/embeddings",
        headers=auth_header(api_key),
        json={"input": "hello world"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["embeddings"]) == 1
    assert body["dim"] == 4


def test_embed_rejects_empty_input(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/embeddings",
        headers=auth_header(api_key),
        json={"input": []},
    )
    assert response.status_code == 422


def test_embed_rejects_blank_text(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/embeddings",
        headers=auth_header(api_key),
        json={"input": ["hello", "  "]},
    )
    assert response.status_code == 422


def test_embed_rejects_over_max_texts(
    keys_file: Path,
    dummy_transcriber: MagicMock,
    dummy_reranker: MagicMock,
    dummy_embedder: MagicMock,
    dummy_tts: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_MAX_TEXTS", "1")
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="embed-limit")

    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
        patch("app.main.Embedder.from_settings", return_value=dummy_embedder),
        patch("app.main.TtsEngine.from_settings", return_value=dummy_tts),
    ):
        from app.main import app

        with TestClient(app) as test_client:
            response = test_client.post(
                "/v1/embeddings",
                headers=auth_header(plaintext),
                json={"input": ["one", "two"]},
            )

    assert response.status_code == 400
    assert "Too many texts" in response.json()["detail"]
    dummy_embedder.encode.assert_not_called()


def test_transcribe_sravaani_model(
    client: TestClient,
    api_key: str,
    dummy_transcriber: MagicMock,
) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_header(api_key),
        files={"file": ("test.wav", b"fake-audio", "audio/wav")},
        data={"language": "hi", "model": "sravaani-1.0"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "sravaani-1.0"
    kwargs = dummy_transcriber.transcribe.call_args.kwargs
    assert kwargs["model"] == "sravaani-1.0"


def test_transcribe_unknown_model(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        headers=auth_header(api_key),
        files={"file": ("test.wav", b"fake-audio", "audio/wav")},
        data={"model": "nope"},
    )
    assert response.status_code == 400
    assert "Unknown STT model" in response.json()["detail"]


def test_speech_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello"},
    )
    assert response.status_code == 401


def test_speech_ok(client: TestClient, api_key: str, dummy_tts: MagicMock) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers=auth_header(api_key),
        json={
            "input": "Namaste",
            "speaker": "Ira",
            "tone": "happy",
            "accent": "Hindi",
            "pace": "steady",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFFWAV"
    assert response.headers["x-model"] == "rumik-oss-1"
    assert response.headers["x-speaker"] == "Ira"
    dummy_tts.synthesize.assert_called()


def test_speech_voice_alias(client: TestClient, api_key: str, dummy_tts: MagicMock) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers=auth_header(api_key),
        json={"text": "hello", "voice": "Zoya"},
    )
    assert response.status_code == 200
    kwargs = dummy_tts.synthesize.call_args.kwargs
    assert kwargs["speaker"] == "Zoya"
    assert kwargs["text"] == "hello"


def test_speech_rejects_bad_speaker(client: TestClient, api_key: str) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers=auth_header(api_key),
        json={"input": "hello", "speaker": "NotAVoice"},
    )
    assert response.status_code == 400
    assert "speaker must be one of" in response.json()["detail"]
