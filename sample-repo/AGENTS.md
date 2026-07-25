# taskflow — agent guide

`taskflow` is a small, dependency-free, in-process task queue with a pluggable, tiered
cache store. It exists to demonstrate a realistic-but-readable codebase: enough structure
that an agent must read several files to reason about a failure, small enough to hold in
one session's context.

You are a coding agent working in this repository. Read this guide fully before acting —
it is the standing context you carry into every turn.

## What the system does

Producers enqueue **tasks** (a `Task` is an id, a payload, a priority, and an optional
`dedupe_key`). A pool of **workers** pulls tasks in priority order and executes a
registered **handler** for the task's `kind`. Results (and in-flight markers) are written
to a **store** — a tiered key/value cache with per-entry TTL and LRU eviction — so that a
duplicate task (same `dedupe_key`) observed within the TTL window is served from cache
instead of re-executed. That "compute once, reuse within the window" property is the whole
point of the store, and it is what the tests exercise most heavily.

## Module map (read in this order when debugging)

- `src/store.py` — `Store`, the tiered cache. `get`/`set`/`delete`, TTL expiry, LRU
  eviction when `capacity` is exceeded. **Most correctness bugs live here**; it is the
  first file to read for any cache/dedupe/expiry symptom.
- `src/queue.py` — `PriorityQueue` and `Task`. Stable ordering: higher priority first,
  ties broken by insertion order (FIFO within a priority). No external deps.
- `src/worker.py` — `Worker` and `WorkerPool`. The pull loop, handler dispatch, and the
  dedupe path that consults the store before executing.
- `src/tiers.py` — `TieredStore`, a fast L1 (`Store`) over a larger, slower L2. Results
  that fall out of L1 (TTL or LRU) demote to L2 and are promoted back on a later read
  instead of being recomputed. Holds invariants T1–T3. Read this when a "recomputed
  something we should have had warm" symptom survives a correct `store.py`.
- `src/api.py` — `TaskFlow`, the thin façade wiring a queue + store + pool together. This
  is the only module external callers should import.
- `docs/ARCHITECTURE.md` — the data-flow diagram and the invariants each module must hold.
- `docs/DESIGN.md` — the design decisions and their rationale (why lazy expiry, why
  LRU-by-access, why the tier promotion preserves the original write timestamp).

## Conventions (follow these exactly)

1. **No third-party dependencies in `src/`.** Standard library only. Tests may use `pytest`.
2. **Time is injected, never read directly.** Every module that needs the clock takes a
   `now: Callable[[], float]` (defaults to `time.monotonic`). Tests pass a fake clock;
   code that calls `time.monotonic()` directly is a bug because it cannot be tested
   deterministically.
3. **The store is authoritative for dedupe.** Workers must not keep their own copy of
   "have I seen this key" — they ask the store. A stale or wrong TTL therefore shows up as
   either duplicate execution (TTL too short / entry missing) or stale results (TTL too
   long / expiry not checked).
4. **Eviction is LRU by last access**, not last write. `get` counts as an access; `set`
   counts as an access. An entry that is read stays hot.
5. **Errors are values at the queue boundary.** Handlers may raise; the worker catches,
   records a `failed` result in the store, and moves on. The pool never dies on a handler
   error.
6. **Public API is `src/api.py` only.** `store`, `queue`, `worker` are internal; do not
   import them across module boundaries except through `TaskFlow`.
7. **Every behavioral change ships with a test** in `tests/`, using the fake clock where
   time matters.

## Invariants (the tests assert these)

- **I1 — TTL is checked on read.** `get(k)` for an entry whose age exceeds its TTL returns
  the miss sentinel and does not resurrect the entry. Expiry is lazy (checked on access),
  not background-swept.
- **I2 — dedupe within the window.** Two tasks with the same `dedupe_key`, the second
  arriving before the first result's TTL elapses, execute the handler exactly once; the
  second reads the cached result.
- **I3 — dedupe after the window.** If the second task arrives after the TTL elapses, the
  handler runs again (the cache legitimately expired).
- **I4 — LRU capacity.** With `capacity=N`, after `N+1` distinct keys are written, exactly
  one entry is evicted, and it is the least-recently-*accessed* one.
- **I5 — priority + FIFO.** Higher priority dequeues first; equal priority dequeues in
  insertion order.

## How to work here

- Reproduce first: `python -m pytest -q` (or the specific test named in the failure).
- When a cache/dedupe/expiry test fails, read `src/store.py` **against invariant I1–I4**
  before changing anything. The failure message names the invariant.
- Keep diffs minimal and matched to the surrounding style. Do not refactor unrelated code.
- After a fix, re-run the full suite; a fix that breaks another invariant is not a fix.

## Glossary

- **miss sentinel** — the singleton `MISS` returned by `store.get` on absence/expiry;
  distinct from a stored `None` value.
- **dedupe_key** — optional caller-supplied key; when set, results are cached under it.
- **window** — the TTL interval during which a cached result satisfies a duplicate task.
