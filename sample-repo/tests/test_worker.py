"""Worker dedupe invariants I2 (reuse within window) and I3 (re-run after window).

I3 currently FAILS as a *consequence* of the store I1 bug: because get() never expires the
cached result, a duplicate task arriving after the TTL still reads the stale result and the
handler does not re-run.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.queue import PriorityQueue, Task  # noqa: E402
from src.store import Store                # noqa: E402
from src.worker import Worker              # noqa: E402


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _setup(clk, ttl=10.0):
    calls = {"n": 0}

    def handler(payload):
        calls["n"] += 1
        return {"doubled": payload * 2, "call": calls["n"]}

    q = PriorityQueue()
    s = Store(capacity=8, default_ttl=ttl, now=clk)
    w = Worker(q, s, {"compute": handler}, now=clk)
    return q, s, w, calls


def test_i2_dedupe_within_window():
    clk = FakeClock()
    q, s, w, calls = _setup(clk)
    q.enqueue(Task("compute", 21, dedupe_key="job-1"))
    r1 = w.run_one()
    clk.advance(3)  # within the 10s window
    q.enqueue(Task("compute", 21, dedupe_key="job-1"))
    r2 = w.run_one()
    assert r1 == r2
    assert calls["n"] == 1, "I2: duplicate within the window must not re-run the handler"


def test_i3_rerun_after_window():
    clk = FakeClock()
    q, s, w, calls = _setup(clk)
    q.enqueue(Task("compute", 21, dedupe_key="job-1"))
    w.run_one()
    clk.advance(15)  # past the 10s window -> cache legitimately expired
    q.enqueue(Task("compute", 21, dedupe_key="job-1"))
    w.run_one()
    assert calls["n"] == 2, "I3: after the TTL window, the handler must run again"
