"""Priority queue with stable FIFO-within-priority ordering (invariant I5).

Higher ``priority`` dequeues first; equal priority dequeues in insertion order. Standard
library only. A monotonic sequence counter breaks ties so the heap stays stable even when
priorities collide.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    kind: str
    payload: Any
    priority: int = 0
    dedupe_key: Optional[str] = None
    id: int = field(default=-1)


class PriorityQueue:
    def __init__(self) -> None:
        self._heap: list = []
        self._seq = itertools.count()
        self._ids = itertools.count(1)

    def enqueue(self, task: Task) -> Task:
        if task.id < 0:
            task.id = next(self._ids)
        # negate priority for a min-heap so higher priority comes out first;
        # seq preserves FIFO order among equal priorities (I5).
        heapq.heappush(self._heap, (-task.priority, next(self._seq), task))
        return task

    def dequeue(self) -> Optional[Task]:
        if not self._heap:
            return None
        _, _, task = heapq.heappop(self._heap)
        return task

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
