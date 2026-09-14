from __future__ import annotations

import hashlib
import hmac
import json

from fastapi import HTTPException, Request, Response

from app.config import get_settings

COOKIE_NAME = "stt_ops"
COOKIE_SALT = b"stt-ops-session-v1"


def ops_token_configured() -> bool:
    return bool(get_settings().stt_ops_token.strip())


def session_cookie_value(token: str) -> str:
    return hmac.new(token.encode("utf-8"), COOKIE_SALT, hashlib.sha256).hexdigest()


def _digest_equal(left: str, right: str) -> bool:
    if not left or not right or len(left) != len(right):
        return False
    return hmac.compare_digest(left, right)


def token_matches(candidate: str) -> bool:
    expected = get_settings().stt_ops_token.strip()
    return _digest_equal(candidate, expected)


def is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    proto = request.headers.get("x-forwarded-proto", "")
    if proto.split(",")[0].strip().lower() == "https":
        return True
    visitor = request.headers.get("cf-visitor", "")
    if visitor:
        try:
            parsed = json.loads(visitor)
            if str(parsed.get("scheme", "")).lower() == "https":
                return True
        except json.JSONDecodeError:
            if "https" in visitor.lower():
                return True
    return False


def is_ops_authorized(request: Request) -> bool:
    settings = get_settings()
    token = settings.stt_ops_token.strip()
    if not token:
        return False
    cookie = request.cookies.get(COOKIE_NAME)
    expected = session_cookie_value(token)
    if cookie and _digest_equal(cookie, expected):
        return True
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and token_matches(value.strip()):
            return True
    return False


def require_ops(request: Request) -> None:
    if not ops_token_configured():
        raise HTTPException(status_code=404, detail="Ops dashboard is disabled")
    if not is_ops_authorized(request):
        raise HTTPException(status_code=401, detail="Invalid ops token")


def set_ops_cookie(response: Response, request: Request) -> None:
    token = get_settings().stt_ops_token.strip()
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_cookie_value(token),
        httponly=True,
        samesite="lax",
        secure=is_https(request),
        max_age=60 * 60 * 24 * 14,
        path="/ops",
    )


def clear_ops_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/ops")
