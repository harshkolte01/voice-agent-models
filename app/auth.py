from pathlib import Path

from fastapi import Header, HTTPException

from app.config import get_settings
from app.keys import ApiKeyRecord, find_valid_key


def parse_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Invalid API key")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token.strip()


def require_api_key(
    authorization: str | None = Header(default=None),
) -> ApiKeyRecord:
    token = parse_bearer(authorization)
    settings = get_settings()
    record = find_valid_key(Path(settings.stt_keys_file), token)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return record
