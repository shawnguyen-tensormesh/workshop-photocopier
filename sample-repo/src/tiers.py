"""Two-tier cache: a fast in-process L1 (a `Store`) backed by a slower, larger L2.

This is the module that gives `taskflow` its name-brand feature: results that fall out of
the hot L1 (by TTL or LRU) are not lost — they demote to L2, and a later read that misses
L1 is *promoted* back up from L2 instead of recomputing. L2 is deliberately modelled as a
plain, capacity-bounded dict here (a stand-in for disk/NVMe or a shared network store);
the point is the promotion/demotion policy, not the backing medium.

Invariants this module owns:
  T1 — read-through: a key present in L2 but not L1 is served (promoted to L1) on get.
  T2 — write-back on eviction: an entry evicted from L1 is written to L2 (not dropped),
       unless it was evicted *because* it expired (expired entries are dead, not demoted).
  T3 — L2 respects its own capacity with LRU-by-access, independent of L1.

Clock is injected (AGENTS.md convention 2). Standard library only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .store import Store, MISS, Entry


@dataclass
class _L2Slot:
    value: Any
    write_ts: float
    last_access: float
    ttl: float
    status: str


@dataclass
class TieredStore:
    """L1 (hot, small) over L2 (warm, large). Public surface mirrors `Store`."""
    l1_capacity: int = 64
    l2_capacity: int = 4096
    default_ttl: float = 30.0
    now: Callable[[], float] = time.monotonic
    l1: Store = field(init=False)
    _l2: Dict[str, _L2Slot] = field(default_factory=dict)
    _stats: Dict[str, int] = field(default_factory=lambda: {"l1_hit": 0, "l2_hit": 0, "miss": 0})

    def __post_init__(self) -> None:
        # L1 delegates its own I1/I4 behavior; we hook eviction via a wrapper below.
        self.l1 = Store(capacity=self.l1_capacity, default_ttl=self.default_ttl, now=self.now)

    # ---- expiry (shared basis with Store) --------------------------------------
    def _l2_expired(self, s: _L2Slot, at: float) -> bool:
        if s.ttl <= 0:
            return False
        return (at - s.write_ts) >= s.ttl

    # ---- core API --------------------------------------------------------------
    def get(self, key: str) -> Any:
        v = self.l1.get(key)
        if v is not MISS:
            self._stats["l1_hit"] += 1
            return v
        # L1 miss: try to promote from L2 (T1)
        s = self._l2.get(key)
        now = self.now()
        if s is None or self._l2_expired(s, now):
            if s is not None:
                del self._l2[key]          # expired in L2: dead, not promotable
            self._stats["miss"] += 1
            return MISS
        # promote: copy back into L1 preserving the ORIGINAL write_ts so TTL keeps counting
        self._stats["l2_hit"] += 1
        remaining = s.ttl - (now - s.write_ts) if s.ttl > 0 else 0.0
        self.l1.set(key, s.value, ttl=remaining if s.ttl > 0 else 0.0, status=s.status)
        s.last_access = now
        return s.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            status: str = "result") -> None:
        # write-through into L1; capture any L1 eviction and write it back to L2 (T2)
        evicted = self._set_l1_capturing_eviction(key, value, ttl, status)
        for ek, ee in evicted:
            self._demote(ek, ee)

    def _set_l1_capturing_eviction(self, key, value, ttl, status) -> list:
        before = dict(self.l1._data)
        self.l1.set(key, value, ttl=ttl, status=status)
        after = self.l1._data
        # entries present before and gone after (excluding the key we just wrote) were evicted
        evicted = []
        for k, e in before.items():
            if k != key and k not in after:
                evicted.append((k, e))
        return evicted

    def _demote(self, key: str, e: Entry) -> None:
        now = self.now()
        # T2: do not demote entries that were evicted because they expired
        if e.ttl > 0 and (now - e.write_ts) >= e.ttl:
            return
        self._l2[key] = _L2Slot(value=e.value, write_ts=e.write_ts,
                                last_access=now, ttl=e.ttl, status=e.status)
        self._evict_l2_if_needed()

    def _evict_l2_if_needed(self) -> None:
        if self.l2_capacity <= 0:
            return
        now = self.now()
        for k in [k for k, s in self._l2.items() if self._l2_expired(s, now)]:
            del self._l2[k]
        while len(self._l2) > self.l2_capacity:           # T3: LRU by access
            lru = min(self._l2, key=lambda k: self._l2[k].last_access)
            del self._l2[lru]

    def delete(self, key: str) -> None:
        self.l1.delete(key)
        self._l2.pop(key, None)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __len__(self) -> int:
        return len(set(self.l1._data) | set(self._l2))
