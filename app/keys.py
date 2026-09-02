from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

KEY_PREFIX = "stt_live_"
PREFIX_LENGTH = 16


@dataclass
class ApiKeyRecord:
    id: str
    name: str
    prefix: str
    hash: str
    created_at: str
    revoked: bool = False


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_plaintext_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def key_prefix(plaintext: str) -> str:
    return plaintext[:PREFIX_LENGTH]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "key"


def load_keys(path: Path) -> list[ApiKeyRecord]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in raw.get("keys", []):
        records.append(
            ApiKeyRecord(
                id=item["id"],
                name=item["name"],
                prefix=item["prefix"],
                hash=item["hash"],
                created_at=item["created_at"],
                revoked=bool(item.get("revoked", False)),
            )
        )
    return records


def save_keys(path: Path, keys: list[ApiKeyRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"keys": [asdict(key) for key in keys]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_key(
    path: Path,
    name: str,
    key_id: str | None = None,
) -> tuple[str, ApiKeyRecord]:
    keys = load_keys(path)
    resolved_id = key_id or slugify(name)
    if any(existing.id == resolved_id for existing in keys):
        raise ValueError(f"API key id already exists: {resolved_id}")

    plaintext = generate_plaintext_key()
    record = ApiKeyRecord(
        id=resolved_id,
        name=name,
        prefix=key_prefix(plaintext),
        hash=hash_key(plaintext),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        revoked=False,
    )
    keys.append(record)
    save_keys(path, keys)
    return plaintext, record


def find_valid_key(path: Path, plaintext: str) -> ApiKeyRecord | None:
    incoming_hash = hash_key(plaintext)
    for record in load_keys(path):
        if record.revoked:
            continue
        if len(record.hash) != len(incoming_hash):
            continue
        if hmac.compare_digest(record.hash, incoming_hash):
            return record
    return None
