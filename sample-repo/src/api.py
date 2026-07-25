"""TaskFlow — the public façade. Wires a PriorityQueue + Store + WorkerPool together.

External callers import only this module (AGENTS.md convention 6).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .queue import PriorityQueue, Task
from .store import Store, MISS
from .worker import Handler, WorkerPool


class TaskFlow:
    def __init__(self, handlers: Dict[str, Handler], capacity: int = 128,
                 default_ttl: float = 30.0, pool_size: int = 4,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.now = now
        self.queue = PriorityQueue()
        self.store = Store(capacity=capacity, default_ttl=default_ttl, now=now)
        self.pool = WorkerPool(self.queue, self.store, handlers, size=pool_size, now=now)

    def submit(self, kind: str, payload: Any, priority: int = 0,
               dedupe_key: Optional[str] = None) -> Optional[Any]:
        """Submit a task. If a fresh result already exists under ``dedupe_key``, return it
        immediately without enqueueing (the fast path the store makes possible)."""
        if dedupe_key is not None:
            cached = self.store.get(dedupe_key)
            if cached is not MISS:
                return cached
        self.queue.enqueue(Task(kind=kind, payload=payload, priority=priority,
                                dedupe_key=dedupe_key))
        return None

    def run(self, limit: int = 10_000) -> int:
        """Drain the queue via a single worker's drain loop; returns tasks processed."""
        return self.pool.workers[0].drain(limit)

    def result(self, dedupe_key: str) -> Any:
        return self.store.get(dedupe_key)
