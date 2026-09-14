from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.access_log import should_store
from app.config import get_settings
from app.keys import create_key
from app.ops_auth import COOKIE_NAME
from app.request_store import RequestEvent, store
from app.system_stats import clear_cache
from app.embed import EmbeddingResult
from app.rerank import RankedDocument
from app.transcribe import TranscriptionResult
from app.tts import SpeechResult

OPS_TOKEN = "ops-secret-token"


def test_should_store_skips_ops_health_and_docs() -> None:
    assert should_store("/v1/embeddings") is True
    assert should_store("/v1/audio/transcriptions") is True
    assert should_store("/ops") is False
    assert should_store("/ops/api/requests") is False
    assert should_store("/health") is False
    assert should_store("/docs") is False
    assert should_store("/openapi.json") is False


def test_store_filters_and_newest_first() -> None:
    store.clear()
    store.add(
        RequestEvent(
            id="a",
            ts_ist="2026-09-14 10:00:00 IST",
            key_id="dev-a",
            method="POST",
            path="/v1/embeddings",
            status=200,
            duration_ms=12,
            ip="1.1.1.1",
            country="IN",
            processing_ms=8,
            model="BAAI/bge-m3",
            device="cuda",
        )
    )
    store.add(
        RequestEvent(
            id="b",
            ts_ist="2026-09-14 10:00:01 IST",
            key_id="dev-b",
            method="POST",
            path="/v1/rerank",
            status=401,
            duration_ms=900,
            ip="2.2.2.2",
            country="US",
        )
    )
    newest = store.list()
    assert [item["id"] for item in newest] == ["b", "a"]
    assert store.list(key="dev-a")[0]["id"] == "a"
    assert store.list(path="/v1/rerank")[0]["id"] == "b"
    assert store.list(status="4xx")[0]["id"] == "b"
    assert store.list(min_ms=500)[0]["id"] == "b"
    store.clear()


@pytest.fixture
def keys_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "keys.json"
    path.write_text('{"keys": []}\n', encoding="utf-8")
    monkeypatch.setenv("STT_KEYS_FILE", str(path))
    monkeypatch.setenv("STT_DEVICE", "cpu")
    monkeypatch.setenv("STT_OPS_TOKEN", OPS_TOKEN)
    monkeypatch.setenv("STT_OPS_STORE_SIZE", "200")
    get_settings.cache_clear()
    store.clear()
    clear_cache()
    yield path
    store.clear()
    clear_cache()
    get_settings.cache_clear()


@pytest.fixture
def dummy_transcriber() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "large-v3-turbo"
    dummy.listed_stt_models.return_value = ["whisper-large-v3-turbo"]

    async def fake_transcribe(*_args, **_kwargs):
        return TranscriptionResult(
            text="hello",
            language="en",
            duration=1.0,
            language_probability=0.9,
            model="whisper-large-v3-turbo",
        )

    dummy.transcribe.side_effect = fake_transcribe
    return dummy


@pytest.fixture
def dummy_reranker() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "BAAI/bge-reranker-v2-m3"

    async def fake_rank(query: str, documents: list[str], top_k: int | None = None):
        return [RankedDocument(index=0, document=documents[0], score=1.0)]

    dummy.rank.side_effect = fake_rank
    return dummy


@pytest.fixture
def dummy_embedder() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.model_name = "BAAI/bge-m3"

    async def fake_encode(texts: list[str]):
        return EmbeddingResult(embeddings=[[0.1, 0.2] for _ in texts], dim=2)

    dummy.encode.side_effect = fake_encode
    return dummy


@pytest.fixture
def dummy_tts() -> MagicMock:
    dummy = MagicMock()
    dummy.loaded = True
    dummy.device = "cpu"
    dummy.public_id = "kokoro-82m"

    async def fake_synthesize(**_kwargs):
        return SpeechResult(
            wav_bytes=b"RIFFWAV",
            speaker="af_heart",
            prompt="hi",
            model="kokoro-82m",
            device="cpu",
        )

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


def test_ops_disabled_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_transcriber: MagicMock,
    dummy_reranker: MagicMock,
    dummy_embedder: MagicMock,
    dummy_tts: MagicMock,
) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text('{"keys": []}\n', encoding="utf-8")
    monkeypatch.setenv("STT_KEYS_FILE", str(keys_path))
    monkeypatch.setenv("STT_OPS_TOKEN", "")
    get_settings.cache_clear()
    with (
        patch("app.main.Transcriber.from_settings", return_value=dummy_transcriber),
        patch("app.main.Reranker.from_settings", return_value=dummy_reranker),
        patch("app.main.Embedder.from_settings", return_value=dummy_embedder),
        patch("app.main.TtsEngine.from_settings", return_value=dummy_tts),
    ):
        from app.main import app

        with TestClient(app) as test_client:
            response = test_client.get("/ops")
    assert response.status_code == 404
    assert "Dashboard is off" in response.text
    get_settings.cache_clear()


def test_ops_login_required(client: TestClient) -> None:
    response = client.get("/ops")
    assert response.status_code == 200
    assert "ops token" in response.text
    assert client.get("/ops/api/requests").status_code == 401
    assert client.get("/ops/api/system").status_code == 401


def test_ops_rejects_wrong_token(client: TestClient) -> None:
    response = client.post("/ops/login", data={"token": "nope"})
    assert response.status_code == 401
    assert "Invalid ops token" in response.text


def test_ops_login_cookie_and_dashboard(client: TestClient) -> None:
    login = client.post("/ops/login", data={"token": OPS_TOKEN}, follow_redirects=False)
    assert login.status_code == 303
    assert login.headers["location"] == "/ops"
    assert COOKIE_NAME in login.cookies
    page = client.get("/ops")
    assert page.status_code == 200
    assert "live IST" in page.text
    system = client.get("/ops/api/system")
    assert system.status_code == 200
    body = system.json()
    assert body["stt"]["loaded"] is True
    assert body["embed"]["device"] == "cpu"


def test_ops_accepts_bearer_token(client: TestClient) -> None:
    response = client.get(
        "/ops/api/system",
        headers={"Authorization": f"Bearer {OPS_TOKEN}"},
    )
    assert response.status_code == 200


def test_ops_cloudflare_login_sets_secure_cookie(client: TestClient) -> None:
    response = client.post(
        "/ops/login",
        data={"token": OPS_TOKEN},
        headers={
            "x-forwarded-proto": "https",
            "cf-visitor": '{"scheme":"https"}',
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "secure" in response.headers.get("set-cookie", "").lower()


def test_ops_records_request_not_dashboard_poll(
    client: TestClient,
    keys_file: Path,
) -> None:
    plaintext, _ = create_key(keys_file, name="dash-key")
    client.post("/ops/login", data={"token": OPS_TOKEN})
    embed = client.post(
        "/v1/embeddings",
        headers={
            "Authorization": f"Bearer {plaintext}",
            "cf-connecting-ip": "49.36.11.20",
            "cf-ipcountry": "IN",
        },
        json={"input": ["hello"]},
    )
    assert embed.status_code == 200
    client.get("/ops")
    client.get("/ops/api/requests")
    client.get("/health")
    listed = client.get("/ops/api/requests")
    assert listed.status_code == 200
    rows = listed.json()["requests"]
    assert len(rows) == 1
    row = rows[0]
    assert row["key_id"] == "dash-key"
    assert row["path"] == "/v1/embeddings"
    assert row["status"] == 200
    assert row["ip"] == "49.36.11.20"
    assert row["country"] == "IN"
    assert row["model"] == "BAAI/bge-m3"
    assert row["device"] == "cpu"
    assert isinstance(row["duration_ms"], int)
    assert row["processing_ms"] == int(embed.headers["x-processing-ms"])
    assert plaintext not in listed.text
    assert "Authorization" not in listed.text
