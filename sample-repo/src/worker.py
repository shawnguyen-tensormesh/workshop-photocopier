"""Workers pull tasks and dispatch handlers, using the store for dedupe (I2/I3).

The dedupe path consults the store *before* executing (AGENTS.md convention 3): the store's
TTL is the single source of truth for "have I computed this recently". If the store returns
a fresh cached result, the handler is skipped; otherwise it runs and the result is cached.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .queue import PriorityQueue, Task
from .store import Store, MISS

Handler = Callable[[Any], Any]


class Worker:
    def __init__(self, queue: PriorityQueue, store: Store,
                 handlers: Dict[str, Handler], now: Callable[[], float] = time.monotonic) -> None:
        self.queue = queue
        self.store = store
        self.handlers = handlers
        self.now = now

    def _dedupe_lookup(self, task: Task) -> Any:
        if task.dedupe_key is None:
            return MISS
        return self.store.get(task.dedupe_key)

    def run_one(self) -> Optional[Any]:
        """Pull and process a single task. Returns the result (cached or computed), or
        None if the queue is empty."""
        task = self.queue.dequeue()
        if task is None:
            return None

        cached = self._dedupe_lookup(task)
        if cached is not MISS:
            return cached  # I2: duplicate within the window reads the cached result

        handler = self.handlers.get(task.kind)
        if handler is None:
            raise KeyError(f"no handler for kind={task.kind!r}")

        if task.dedupe_key is not None:
            self.store.set(task.dedupe_key, None, status="in_flight")
        try:
            result = handler(task.payload)
            status = "result"
        except Exception as exc:  # errors are values at the queue boundary (convention 5)
            result = {"error": repr(exc)}
            status = "failed"
        if task.dedupe_key is not None:
            self.store.set(task.dedupe_key, result, status=status)
        return result

    def drain(self, limit: int = 10_000) -> int:
        n = 0
        while n < limit and self.queue:
            self.run_one()
            n += 1
        return n


class WorkerPool:
    def __init__(self, queue: PriorityQueue, store: Store, handlers: Dict[str, Handler],
                 size: int = 4, now: Callable[[], float] = time.monotonic) -> None:
        self.workers = [Worker(queue, store, handlers, now) for _ in range(size)]

    def run_round(self) -> int:
        processed = 0
        for w in self.workers:
            if w.run_one() is not None:
                processed += 1
        return processed
