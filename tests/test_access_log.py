from __future__ import annotations

from pathlib import Path

from starlette.requests import Request

from app.access_log import client_country, client_ip, key_label, now_ist
from app.config import get_settings
from app.keys import create_key


def _request(headers: dict[str, str], client: tuple[str, int] | None = ("127.0.0.1", 9)) -> Request:
    header_list = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/embeddings",
        "raw_path": b"/v1/embeddings",
        "query_string": b"",
        "headers": header_list,
        "client": client,
        "server": ("127.0.0.1", 8000),
    }
    return Request(scope)


def test_now_ist_suffix() -> None:
    stamp = now_ist()
    assert stamp.endswith(" IST")
    assert len(stamp) == len("2026-09-08 16:41:12 IST")


def test_client_ip_prefers_cloudflare_header() -> None:
    request = _request(
        {
            "cf-connecting-ip": "49.36.11.20",
            "x-forwarded-for": "1.1.1.1, 2.2.2.2",
        }
    )
    assert client_ip(request) == "49.36.11.20"
    assert client_country(_request({"cf-ipcountry": "in"})) == "IN"


def test_client_ip_falls_back_to_forwarded_then_peer() -> None:
    assert client_ip(_request({"x-forwarded-for": "8.8.8.8, 9.9.9.9"})) == "8.8.8.8"
    assert client_ip(_request({}, client=("10.0.0.5", 80))) == "10.0.0.5"


def test_key_label_valid_and_missing(tmp_path: Path, monkeypatch) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text('{"keys": []}\n', encoding="utf-8")
    monkeypatch.setenv("STT_KEYS_FILE", str(keys_file))
    get_settings.cache_clear()
    plaintext, _ = create_key(keys_file, name="test")
    try:
        assert key_label(_request({})) == "-"
        assert key_label(_request({"authorization": f"Bearer {plaintext}"})) == "test"
        assert key_label(_request({"authorization": "Bearer stt_live_nope"})) == "invalid"
    finally:
        get_settings.cache_clear()
