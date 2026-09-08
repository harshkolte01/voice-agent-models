from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from app.config import get_settings
from app.keys import find_valid_key

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("stt.access")


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip and cf_ip.strip():
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and forwarded.strip():
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "-"


def client_country(request: Request) -> str:
    country = request.headers.get("cf-ipcountry")
    if country and country.strip() and country.strip().upper() != "XX":
        return country.strip().upper()
    return "-"


def key_label(request: Request) -> str:
    authorization = request.headers.get("authorization")
    if not authorization:
        return "-"
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return "-"
    settings = get_settings()
    record = find_valid_key(Path(settings.stt_keys_file), token.strip())
    if record is None:
        return "invalid"
    return record.id


def configure_access_log() -> None:
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True

    log = logging.getLogger("stt.access")
    log.setLevel(logging.INFO)
    log.propagate = True
    if not any(isinstance(handler, logging.StreamHandler) for handler in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        path = request.url.path
        logger.info(
            "%s  key=%s  ip=%s  %s  %s %s  %s  %sms",
            now_ist(),
            key_label(request),
            client_ip(request),
            client_country(request),
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )
        return response
