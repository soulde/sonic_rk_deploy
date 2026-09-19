"""Non-blocking latest-state mailbox for a visualization worker."""

from __future__ import annotations

import threading

import numpy as np


class LatestState:
    """Keep one copied snapshot; publishing never waits on a renderer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: np.ndarray | None = None

    def publish(self, state: np.ndarray) -> None:
        snapshot = np.asarray(state, dtype=np.float64).copy()
        with self._lock:
            self._state = snapshot

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._state is None else self._state.copy()
