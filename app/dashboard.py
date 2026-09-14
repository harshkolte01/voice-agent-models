from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.ops_auth import (
    clear_ops_cookie,
    is_ops_authorized,
    ops_token_configured,
    require_ops,
    set_ops_cookie,
    token_matches,
)
from app.request_store import store
from app.system_stats import snapshot

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
NO_STORE = {"Cache-Control": "no-store"}
router = APIRouter(tags=["ops"])


def _page(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        name,
        context,
        status_code=status_code,
        headers=NO_STORE,
    )


@router.get("/ops", response_class=HTMLResponse)
async def ops_home(request: Request):
    if not ops_token_configured():
        return _page(request, "ops_disabled.html", status_code=404)
    if not is_ops_authorized(request):
        return _page(request, "ops_login.html", error=None)
    return _page(request, "ops.html")


@router.post("/ops/login")
async def ops_login(request: Request, token: str = Form(...)):
    if not ops_token_configured():
        return _page(request, "ops_disabled.html", status_code=404)
    if not token_matches(token.strip()):
        return _page(
            request,
            "ops_login.html",
            status_code=401,
            error="Invalid ops token",
        )
    response = RedirectResponse(url="/ops", status_code=303, headers=NO_STORE)
    set_ops_cookie(response, request)
    return response


@router.post("/ops/logout")
async def ops_logout():
    response = RedirectResponse(url="/ops", status_code=303, headers=NO_STORE)
    clear_ops_cookie(response)
    return response


@router.get("/ops/api/requests")
async def ops_requests(
    request: Request,
    key: str | None = None,
    path: str | None = None,
    status: str | None = None,
    min_ms: int | None = None,
    limit: int = 250,
):
    require_ops(request)
    capped = max(1, min(limit, 1000))
    events = store.list(
        key=key or None,
        path=path or None,
        status=status or None,
        min_ms=min_ms,
        limit=capped,
    )
    return {"requests": events, "keys": store.keys_seen()}


@router.get("/ops/api/system")
async def ops_system(request: Request):
    require_ops(request)
    return snapshot(request.app)
