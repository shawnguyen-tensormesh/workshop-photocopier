"""Tiered key/value store with per-entry TTL and LRU-by-last-access eviction.

Authoritative for dedupe (see AGENTS.md). Standard library only. The clock is injected
via ``now`` so tests can drive expiry deterministically; nothing here may call
``time.monotonic`` directly.

Invariants this module owns:
  I1 — TTL is checked on read: ``get`` of an entry older than its TTL returns MISS.
  I4 — eviction is LRU by last *access* (get and set both count as an access).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class _Miss:
    """Singleton miss sentinel, distinct from a stored ``None`` value."""
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<MISS>"


MISS = _Miss()


@dataclass
class Entry:
    value: Any
    write_ts: float          # when the value was written (basis for TTL / I1)
    last_access: float       # when the value was last read or written (basis for LRU / I4)
    ttl: float               # seconds; <= 0 means "never expires"
    status: str = "result"   # "result" | "failed" | "in_flight"


@dataclass
class Store:
    capacity: int = 128
    default_ttl: float = 30.0
    now: Callable[[], float] = time.monotonic
    _data: Dict[str, Entry] = field(default_factory=dict)

    # ---- expiry -----------------------------------------------------------------
    def _expired(self, e: Entry, at: float) -> bool:
        """True if entry ``e`` is past its TTL at time ``at`` (TTL measured from write)."""
        if e.ttl <= 0:
            return False
        return (at - e.write_ts) >= e.ttl

    # ---- core API ---------------------------------------------------------------
    def get(self, key: str) -> Any:
        """Return the stored value, or MISS if absent or expired.

        Reading counts as an access for LRU purposes (I4).
        """
        e = self._data.get(key)
        if e is None:
            return MISS
        now = self.now()
        # NOTE: refresh last access so a hot entry stays hot for LRU.
        e.last_access = now
        return e.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            status: str = "result") -> None:
        """Insert/replace ``key``. Evicts the LRU entry if over capacity."""
        now = self.now()
        self._data[key] = Entry(
            value=value, write_ts=now, last_access=now,
            ttl=self.default_ttl if ttl is None else ttl, status=status,
        )
        self._evict_if_needed()

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def status_of(self, key: str) -> Optional[str]:
        e = self._data.get(key)
        if e is None or self._expired(e, self.now()):
            return None
        return e.status

    # ---- maintenance ------------------------------------------------------------
    def _evict_if_needed(self) -> None:
        """Drop the least-recently-accessed entries until within capacity (I4).

        Expired entries are dropped first (cheap, and keeps LRU honest).
        """
        if self.capacity <= 0:
            return
        now = self.now()
        for k in [k for k, e in self._data.items() if self._expired(e, now)]:
            del self._data[k]
        while len(self._data) > self.capacity:
            lru_key = min(self._data, key=lambda k: self._data[k].last_access)
            del self._data[lru_key]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
