# taskflow architecture

```
producers --enqueue--> PriorityQueue --pull--> WorkerPool --dispatch--> handler(kind)
                                                    |                        |
                                                    v                        v
                                                 Store  <----results/markers-+
                                              (TTL + LRU)
```

## Data flow

1. A caller uses `TaskFlow.submit(kind, payload, priority=0, dedupe_key=None)`.
2. `submit` builds a `Task`, and — if `dedupe_key` is set — first asks the `Store` whether
   a fresh result already exists under that key.
   - **hit** (fresh entry): the cached result is returned immediately; nothing is enqueued.
   - **miss** (absent or expired): the task is enqueued.
3. A `Worker` pulls the highest-priority task, looks up the handler for `task.kind`, and:
   - writes an `in_flight` marker to the store under `dedupe_key` (so a concurrent duplicate
     coalesces onto the same computation),
   - runs the handler,
   - writes the `result` (or `failed`) under `dedupe_key` with the store's default TTL,
   - clears the `in_flight` marker.
4. Subsequent duplicates within the TTL window read the cached `result` (invariant I2);
   after the window they miss and re-execute (invariant I3).

## Invariants by module

| Module | Holds |
|---|---|
| `store.py` | I1 (TTL checked on read), I4 (LRU by last access) |
| `worker.py` | I2/I3 (dedupe uses the store, honoring its TTL) |
| `queue.py` | I5 (priority then FIFO) |
| `api.py` | wiring only; holds no invariant of its own |

## Clock discipline

Every time-dependent component receives `now: Callable[[], float]`. The production default
is `time.monotonic`. Tests inject a `FakeClock` they advance by hand, which is why nothing
in `src/` may call `time.monotonic()` directly — see AGENTS.md convention 2. A TTL bug that
"only shows up in production" is almost always a component reading the wall clock instead of
the injected `now`, or comparing against the wrong timestamp (write time vs. access time).

## Failure semantics

Handler exceptions are caught in the worker and recorded as a `failed` result (a value,
not an exception) under the dedupe key, with the same TTL as a success. The pool keeps
running. Callers distinguish `failed` from `result` by the entry's `status` field.
