"""Store invariants I1 (TTL on read) and I4 (LRU by last access).

Uses a FakeClock so expiry is deterministic. These tests are the reproduction the agent
starts from: I1 currently FAILS because ``Store.get`` does not check expiry.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.store import Store, MISS  # noqa: E402


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_i1_ttl_checked_on_read():
    clk = FakeClock()
    s = Store(capacity=8, default_ttl=10.0, now=clk)
    s.set("k", "v")            # written at t=0, ttl=10
    assert s.get("k") == "v"   # fresh
    clk.advance(15)            # now past the TTL window
    assert s.get("k") is MISS, "I1: get() of an expired entry must MISS, not resurrect it"


def test_i1_boundary_is_inclusive():
    clk = FakeClock()
    s = Store(capacity=8, default_ttl=10.0, now=clk)
    s.set("k", "v")
    clk.advance(10)            # exactly at TTL -> expired (>= per _expired)
    assert s.get("k") is MISS


def test_i4_lru_by_last_access():
    clk = FakeClock()
    s = Store(capacity=2, default_ttl=100.0, now=clk)
    s.set("a", 1); clk.advance(1)
    s.set("b", 2); clk.advance(1)
    assert s.get("a") == 1     # touch a -> a is now most-recently accessed
    clk.advance(1)
    s.set("c", 3)              # over capacity -> evict LRU, which is b (a was just read)
    assert "a" in s and "c" in s and "b" not in s
