from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.keys import create_key, save_keys, load_keys
from app.rerank import RankedDocument
from app.transcribe import TranscriptionResult


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
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def dummy_transcriber() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "large-v3-turbo"

    async def fake_transcribe(audio_path: str, language: str | None = None):
        return TranscriptionResult(
            text="hello world",
            language=language or "en",
            duration=2.74,
            language_probability=0.98,
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
def client(keys_file: Path, dummy_transcriber: MagicMock, dummy_reranker: MagicMock):
    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
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
    assert body["device"] == "cpu"


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
            {"id": "BAAI/bge-reranker-v2-m3", "type": "rerank"},
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="tiny-limit")

    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANK_MAX_DOCS", "1")
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="rerank-limit")

    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
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
