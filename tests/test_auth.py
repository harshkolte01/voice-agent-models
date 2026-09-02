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


def test_resolve_compute_type_auto() -> None:
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"


def test_public_model_id() -> None:
    assert public_model_id("large-v3-turbo") == "whisper-large-v3-turbo"
    assert public_model_id("whisper-large-v3-turbo") == "whisper-large-v3-turbo"
