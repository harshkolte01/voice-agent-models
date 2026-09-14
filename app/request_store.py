from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestEvent:
    id: str
    ts_ist: str
    key_id: str
    method: str
    path: str
    status: int
    duration_ms: int
    ip: str
    country: str
    processing_ms: int | None = None
    model: str | None = None
    device: str | None = None
    speaker: str | None = None
    request_bytes: int | None = None


class RequestStore:
    def __init__(self, maxlen: int = 2000) -> None:
        self._lock = threading.Lock()
        self._events: deque[RequestEvent] = deque(maxlen=maxlen)

    def configure(self, maxlen: int) -> None:
        size = max(50, maxlen)
        with self._lock:
            items = list(self._events)
            self._events = deque(items[-size:], maxlen=size)

    def add(self, event: RequestEvent) -> None:
        with self._lock:
            self._events.append(event)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def list(
        self,
        *,
        key: str | None = None,
        path: str | None = None,
        status: str | None = None,
        min_ms: int | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._events)
        items.reverse()
        filtered: list[dict[str, Any]] = []
        for event in items:
            if key and event.key_id != key:
                continue
            if path and path not in event.path:
                continue
            if status:
                if len(status) == 3 and status.endswith("xx") and status[0].isdigit():
                    if event.status // 100 != int(status[0]):
                        continue
                elif str(event.status) != status:
                    continue
            if min_ms is not None and event.duration_ms < min_ms:
                continue
            filtered.append(asdict(event))
            if len(filtered) >= limit:
                break
        return filtered

    def keys_seen(self) -> list[str]:
        seen: list[str] = []
        with self._lock:
            for event in reversed(self._events):
                if event.key_id not in seen:
                    seen.append(event.key_id)
        return seen


store = RequestStore()
