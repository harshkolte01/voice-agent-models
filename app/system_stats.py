from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI

_CACHE: tuple[float, dict[str, Any]] | None = None
_TTL_SECONDS = 2.0


def snapshot(app: FastAPI) -> dict[str, Any]:
    global _CACHE
    now = time.monotonic()
    if _CACHE is not None and now - _CACHE[0] < _TTL_SECONDS:
        return _CACHE[1]
    data = collect(app)
    _CACHE = (now, data)
    return data


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


def collect(app: FastAPI) -> dict[str, Any]:
    transcriber = getattr(app.state, "transcriber", None)
    reranker = getattr(app.state, "reranker", None)
    embedder = getattr(app.state, "embedder", None)
    tts = getattr(app.state, "tts", None)
    payload: dict[str, Any] = {
        "status": "ok",
        "stt": {
            "loaded": bool(transcriber and transcriber.loaded),
            "device": getattr(transcriber, "device", "unknown"),
            "model": getattr(transcriber, "model_name", None),
        },
        "rerank": {
            "loaded": bool(reranker and reranker.loaded),
            "device": getattr(reranker, "device", "unknown"),
            "model": getattr(reranker, "model_name", None),
        },
        "embed": {
            "loaded": bool(embedder and embedder.loaded),
            "device": getattr(embedder, "device", "unknown"),
            "model": getattr(embedder, "model_name", None),
        },
        "tts": {
            "loaded": bool(tts and getattr(tts, "loaded", False)),
            "device": getattr(tts, "device", None) if tts is not None else None,
            "model": getattr(tts, "public_id", None) if tts is not None else None,
        },
        "cpu_percent": None,
        "ram": None,
        "gpu": None,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload["cpu_percent"] = psutil.cpu_percent(interval=None)
        payload["ram"] = {
            "used_gb": round(memory.used / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2),
            "percent": memory.percent,
        }
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            payload["gpu"] = {
                "name": torch.cuda.get_device_name(index),
                "allocated_gb": round(torch.cuda.memory_allocated(index) / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved(index) / (1024**3), 2),
                "total_gb": round(props.total_memory / (1024**3), 2),
            }
    except Exception:
        pass
    return payload
