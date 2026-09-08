from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class ExclusiveCudaSlot:
    """Keep only one heavy CUDA model resident (SraVaani or rumik-oss)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: str | None = None
        self._release: Callable[[], None] | None = None

    def activate(
        self,
        name: str,
        acquire: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        with self._lock:
            if self._active != name:
                if self._release is not None:
                    self._release()
                self._active = None
                self._release = None
                acquire()
                self._active = name
                self._release = release
                return
            acquire()
            self._release = release

    @contextmanager
    def hold(
        self,
        name: str,
        acquire: Callable[[], None],
        release: Callable[[], None],
    ) -> Iterator[None]:
        with self._lock:
            self.activate(name, acquire, release)
            yield
