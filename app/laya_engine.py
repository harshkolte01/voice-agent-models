from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.rerank import torch_cuda_available

PUBLIC_ID = "laya-router"
ALLOWED_MODELS = frozenset({"english", "multilingual", "typed-decisions"})
DEFAULT_PRELOAD = ("english", "multilingual", "typed-decisions")
DEFAULT_MODEL = "typed-decisions"


@dataclass
class DecideResult:
    answers: dict[str, Any]
    routing: dict[str, Any] | None
    model: str
    device: str
    checkpoint: str | None = None


def resolve_laya_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "auto":
        return "cuda" if torch_cuda_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported LAYA_DEVICE: {device}")
    return requested


def parse_laya_preload(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_PRELOAD)
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip().lower() for item in value]
    else:
        parts = [part.strip().lower() for part in str(value).split(",")]
    names = [part for part in parts if part]
    if not names:
        return list(DEFAULT_PRELOAD)
    unknown = [name for name in names if name not in ALLOWED_MODELS]
    if unknown:
        allowed = ", ".join(sorted(ALLOWED_MODELS))
        raise ValueError(
            f"Unsupported LAYA_PRELOAD entry: {', '.join(unknown)}. Allowed: {allowed}"
        )
    # preserve order, drop duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def normalize_laya_model(model: str | None) -> str | None:
    if model is None:
        return None
    normalized = model.strip().lower()
    if not normalized or normalized in {"auto", "router"}:
        return None
    if normalized not in ALLOWED_MODELS:
        allowed = ", ".join(sorted(ALLOWED_MODELS))
        raise ValueError(f"Unsupported Laya model. Allowed: {allowed}")
    return normalized


def resolve_default_laya_model(value: str | None) -> str | None:
    if value is None:
        return DEFAULT_MODEL
    return normalize_laya_model(value)


def _build_router(device: str, preload: list[str]):
    os.environ.setdefault("USE_TF", "0")
    from app.hf_compat import disable_broken_torchvision, force_hf_hub_copy_cache

    disable_broken_torchvision()
    force_hf_hub_copy_cache()
    from laya import Router

    return Router(preload=preload, device=device)


class LayaEngine:
    def __init__(
        self,
        device: str,
        preload: list[str] | None = None,
        default_model: str | None = DEFAULT_MODEL,
    ):
        self.device = resolve_laya_device(device)
        self.preload = parse_laya_preload(preload)
        self.default_model = resolve_default_laya_model(default_model)
        self.model_name = PUBLIC_ID
        self.public_id = PUBLIC_ID
        self._lock = threading.Lock()
        self._router = _build_router(self.device, self.preload)
        self.loaded = True

    @classmethod
    def from_settings(cls, settings: Settings) -> LayaEngine:
        return cls(
            device=settings.laya_device,
            preload=settings.laya_preload,
            default_model=settings.laya_default_model,
        )

    def predict_sync(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Any],
        model: str | None = None,
    ) -> DecideResult:
        override = normalize_laya_model(model)
        if override is None:
            override = self.default_model
        with self._lock:
            if override is None:
                result = self._router.predict(state, questions)
            else:
                result = self._router.predict(state, questions, model=override)

        if not isinstance(result, dict):
            raise RuntimeError("Laya returned a non-dict result")

        answers = result.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("Laya result missing answers")

        routing = result.get("routing")
        if routing is not None and not isinstance(routing, dict):
            routing = None

        checkpoint = override
        if checkpoint is None and isinstance(routing, dict):
            routed = routing.get("model")
            if isinstance(routed, str):
                checkpoint = routed

        return DecideResult(
            answers=answers,
            routing=routing,
            model=self.public_id,
            device=self.device,
            checkpoint=checkpoint,
        )

    async def predict(
        self,
        state: str | dict[str, Any],
        questions: dict[str, Any],
        model: str | None = None,
    ) -> DecideResult:
        return await asyncio.to_thread(self.predict_sync, state, questions, model)
