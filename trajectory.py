"""The canonical, FROZEN coding-agent trajectory over sample-repo/.

`build_standing_prefix(salt)` is the repository context the agent carries every turn
(AGENTS.md + architecture + tree), with a per-attendee salt header so each seat is a
distinct context. `TURNS` is a fixed sequence of (user, assistant) exchanges that walk an
agent through finding and fixing the store TTL bug. Because both are fixed data, replaying
them sends identical tokens every run — so any latency difference is caching, nothing else.

The user turns inline real file contents / test output (that is what makes the context
grow, turn over turn, the way an agent's does). Assistant turns are short canned replies —
we time prefill (TTFT), not generation.
"""
import os

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample-repo")


def _read(rel):
    with open(os.path.join(_ROOT, rel), "r") as f:
        return f.read()


def _tree():
    return (
        "taskflow/\n"
        "  AGENTS.md\n"
        "  docs/ARCHITECTURE.md\n"
        "  src/{__init__,store,queue,worker,api}.py\n"
        "  tests/{test_store,test_worker}.py\n"
        "  pytest.ini\n"
    )


def build_standing_prefix(salt):
    """System / standing context carried into every turn."""
    return (
        f"<<workshop-seat: {salt}>>  (unique per attendee; do not share context across seats)\n\n"
        "You are a senior coding agent working in the `taskflow` repository. The repository "
        "guide and architecture below are your standing context; follow the conventions and "
        "reason about failures against the stated invariants.\n\n"
        "===== FILE: AGENTS.md =====\n" + _read("AGENTS.md") + "\n"
        "===== FILE: docs/ARCHITECTURE.md =====\n" + _read("docs/ARCHITECTURE.md") + "\n"
        "===== FILE: docs/DESIGN.md =====\n" + _read("docs/DESIGN.md") + "\n"
        "===== REPO TREE =====\n" + _tree()
    )


_PYTEST_RED = """$ python -m pytest -q
F.F..F                                                                    [100%]
=================================== FAILURES ===================================
tests/test_store.py::test_i1_ttl_checked_on_read
    assert s.get("k") is MISS, "I1: get() of an expired entry must MISS, not resurrect it"
E   AssertionError: assert 'v' is <MISS>
tests/test_store.py::test_i1_boundary_is_inclusive
E   AssertionError: assert 'v' is <MISS>
tests/test_worker.py::test_i3_rerun_after_window
    assert calls["n"] == 2, "I3: after the TTL window, the handler must run again"
E   AssertionError: 1 == 2
=========================== short test summary info ============================
FAILED tests/test_store.py::test_i1_ttl_checked_on_read
FAILED tests/test_store.py::test_i1_boundary_is_inclusive
FAILED tests/test_worker.py::test_i3_rerun_after_window
3 failed, 2 passed in 0.02s
"""

_PYTEST_GREEN = """$ python -m pytest -q
.....                                                                     [100%]
5 passed in 0.02s
"""

_FIX = """--- a/src/store.py
+++ b/src/store.py
@@ def get(self, key: str) -> Any:
         e = self._data.get(key)
         if e is None:
             return MISS
         now = self.now()
+        if self._expired(e, now):
+            # I1: expiry is lazy — checked on read; drop and report a miss.
+            del self._data[key]
+            return MISS
         e.last_access = now
         return e.value
"""

TURNS = [
    {
        "user": "The CI for `taskflow` is red. Your standing context (AGENTS.md + architecture) "
                "is above. Here is the failing run:\n\n" + _PYTEST_RED +
                "\nDiagnose the root cause. Start from the module the failures point at.",
        "assistant": "Three failures, two invariants: I1 (TTL checked on read) in test_store, and "
                     "I3 (re-run after the window) in test_worker. Per AGENTS.md, cache/expiry "
                     "symptoms start in src/store.py, and I3 is likely a downstream consequence of "
                     "I1. I'll read src/store.py against I1 first.",
    },
    {
        "user": "Here is src/store.py:\n\n===== FILE: src/store.py =====\n" + _read("src/store.py"),
        "assistant": "Found it. `_expired()` is correct and is used by `set`/`_evict_if_needed`, but "
                     "`get()` never calls it — it refreshes `last_access` and returns the value "
                     "unconditionally. So an entry past its TTL is still served: a direct I1 "
                     "violation. Expiry must be checked lazily on read.",
    },
    {
        "user": "Confirm the dedupe consequence. Here is src/worker.py:\n\n"
                "===== FILE: src/worker.py =====\n" + _read("src/worker.py"),
        "assistant": "Confirmed. `_dedupe_lookup` calls `store.get(dedupe_key)`; since get() never "
                     "expires, a duplicate task after the TTL still gets a non-MISS cached result, "
                     "so the handler is skipped — exactly the I3 failure. Worker logic is correct; "
                     "the bug is entirely in store.get().",
    },
    {
        "user": "Here are the tests that encode the expected behavior:\n\n"
                "===== FILE: tests/test_store.py =====\n" + _read("tests/test_store.py") +
                "\n===== FILE: tests/test_worker.py =====\n" + _read("tests/test_worker.py"),
        "assistant": "The tests match the invariants: I1 expects MISS at and past the TTL boundary "
                     "(inclusive), I3 expects the handler to run twice across the window. Both follow "
                     "once get() checks expiry. I2 and I4 already pass and must stay passing.",
    },
    {
        "user": "Propose the minimal fix to src/store.py. Keep the diff tight and matched to the "
                "surrounding style; do not touch worker.py.",
        "assistant": "Add a lazy expiry check at the top of get(), reusing the existing _expired() "
                     "helper, before refreshing last_access:\n\n" + _FIX +
                     "\nThis satisfies I1 directly and I3 as a consequence, and leaves LRU (I4) "
                     "untouched because expired entries are simply dropped on read.",
    },
    {
        "user": "Applied. Here is the re-run:\n\n" + _PYTEST_GREEN +
                "\nConfirm nothing else regressed.",
        "assistant": "All five green: I1 (both cases), I3 now pass; I2 (reuse within window) and I4 "
                     "(LRU by last access) still pass. The fix is scoped to get() and consistent with "
                     "convention 2 (uses the injected clock via self.now()).",
    },
    {
        "user": "While you're here, verify src/queue.py still holds I5 (priority then FIFO):\n\n"
                "===== FILE: src/queue.py =====\n" + _read("src/queue.py"),
        "assistant": "I5 holds: the heap key is (-priority, seq, task); the monotonic `seq` breaks "
                     "ties in insertion order, so higher priority dequeues first and equal priorities "
                     "are FIFO. No change needed.",
    },
    {
        "user": "Sanity-check the tier layer against D4 (promotion must preserve the write "
                "timestamp) now that get() expires correctly. Here is src/tiers.py:\n\n"
                "===== FILE: src/tiers.py =====\n" + _read("src/tiers.py"),
        "assistant": "tiers.py is consistent with the fix. Its own get() already checks L2 expiry "
                     "and promotes with the ORIGINAL write_ts (remaining = ttl - (now - write_ts)), so "
                     "an entry can't live forever by bouncing tiers (D4). And _demote refuses to write "
                     "back an entry evicted because it expired (T2). The L1 fix composes cleanly: L1 "
                     "now expires on read, so a stale value never gets a chance to demote.",
    },
    {
        "user": "And confirm src/api.py respects convention 6 (public façade only):\n\n"
                "===== FILE: src/api.py =====\n" + _read("src/api.py"),
        "assistant": "api.py is the only module wiring queue+store+pool and exposes submit/run/result; "
                     "it imports the internals but external callers only need TaskFlow. Convention 6 "
                     "holds. The submit() fast-path also benefits from the fix — it now correctly "
                     "misses on an expired dedupe_key and re-enqueues.",
    },
    {
        "user": "Write the root-cause summary for the PR description.",
        "assistant": "Root cause: Store.get() never enforced TTL — it refreshed last_access and "
                     "returned the value without calling _expired(), so entries were served past "
                     "their window (I1), which in turn skipped legitimate re-execution of deduped "
                     "tasks after expiry (I3). Fix: check _expired() lazily at the top of get(), drop "
                     "the entry, and return MISS. One-file, 4-line change; all five invariants green.",
    },
    {
        "user": "Any follow-up risks or tests worth adding before we merge?",
        "assistant": "Two: (1) a concurrency test that a second duplicate arriving during in_flight "
                     "coalesces rather than re-running; (2) a property test that get() never returns a "
                     "value whose age >= ttl across random advance sequences. Neither blocks this fix; "
                     "both harden the invariant. Recommend merge with issue links for the two.",
    },
]
