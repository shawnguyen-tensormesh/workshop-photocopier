# taskflow — design decisions and rationale

This document records *why* the cache behaves the way it does. It is standing context: an
agent debugging `taskflow` should be able to justify a change against these decisions, and
a change that contradicts one of them needs an explicit note in its PR.

## D1 — Lazy expiry, checked on read (not a background sweep)

TTL is enforced the moment an entry is read (`get`), not by a timer that walks the store.
We chose lazy expiry for three reasons:

1. **Determinism under test.** With an injected clock, a lazy check is a pure function of
   "what time is it when you ask." A background sweeper would need a thread, a scheduler,
   or a manual `tick()` — all of which leak timing into tests. Convention 2 (inject the
   clock) exists precisely so expiry is testable without wall-clock sleeps.
2. **No work for entries nobody reads.** A sweeper spends CPU expiring entries that would
   never be looked at again. Lazy expiry pays only on access, and the access was going to
   happen anyway.
3. **Correctness is local.** With lazy expiry, the single rule "an entry whose age since
   *write* is ≥ its TTL is dead" lives in one predicate (`_expired`) called from every read
   path. If any read path forgets to call it, that path serves stale data — which is exactly
   the class of bug invariant I1 guards against. (The most common regression here is a `get`
   that refreshes access time and returns the value without consulting `_expired`.)

The boundary is **inclusive**: age *equal to* the TTL is expired (`>=`, not `>`). A caller
that set `ttl=10` means "good for up to, not including, 10 seconds after write." Tests assert
the inclusive boundary so the semantics can't silently drift.

## D2 — Age is measured from write; LRU is measured from access

Two different clocks for two different questions, and conflating them is a classic bug:

- **TTL / freshness** is measured from `write_ts`. Reading an entry does **not** extend its
  life — a value written 20 s ago with a 10 s TTL is stale no matter how often you read it.
- **LRU / hotness** is measured from `last_access`. Reading an entry **does** keep it hot for
  eviction purposes — a frequently-read entry survives capacity pressure.

So `get` updates `last_access` (for LRU) but must judge freshness against `write_ts` (for
TTL). A `get` that computed age from `last_access` would make hot entries immortal; a `get`
that never updated `last_access` would evict entries that are actually in heavy use. The two
timestamps on `Entry` exist to keep these separable.

## D3 — Eviction drops expired entries first, then LRU

`_evict_if_needed` first removes anything already expired (cheap, and it keeps the LRU
choice honest — you should never evict a live entry to make room while a dead one lingers),
then evicts least-recently-accessed live entries until within capacity. Expired-first also
means capacity is measured in *live* entries, which is what callers reason about.

## D4 — The tier preserves the original write timestamp on promotion

When `TieredStore` promotes an entry from L2 back into L1, it must keep the **original**
`write_ts`, not stamp "now". Otherwise demote→promote would reset the TTL and an entry could
live forever by bouncing between tiers — the multi-tier version of the D2 mistake. The
promotion computes the remaining TTL (`ttl - (now - write_ts)`) and installs the entry in L1
with that remainder, so total lifetime is conserved across tiers (invariant T1 with T2).

Correspondingly, an entry evicted from L1 **because it expired** is *not* demoted to L2 — it
is dead, and demoting it would resurrect stale data one tier down (invariant T2). Only live
entries (evicted by LRU pressure) demote.

## D5 — Errors are values; the pool never dies

A handler exception is caught and stored as a `failed` result under the dedupe key with the
same TTL as a success. This makes failures cacheable (a flapping dependency doesn't get
hammered once per duplicate) and keeps the worker loop alive. Callers distinguish `failed`
from `result` via the entry's `status`. The trade-off — a transient failure is remembered
for the TTL window — is accepted deliberately; shorten the TTL for handlers whose failures
are expected to be transient.

## D6 — Dedupe is the store's job, not the worker's

Workers never keep a private "seen" set; they ask the store. This is what makes the TTL the
single knob that controls reuse. It also means every dedupe bug is ultimately a store bug,
which is why the module read-order in AGENTS.md starts at `store.py` for any dedupe symptom.

## Non-goals

- **Persistence across process restarts.** L2 is an in-process dict here; a real deployment
  swaps it for disk/NVMe or a shared store, but the promotion/demotion policy is unchanged.
- **Cross-process coordination.** A single process owns the store. Distributing it is out of
  scope for this repo (and would change the dedupe/in-flight story materially).
- **Exact-once execution under crash.** In-flight markers coalesce concurrent duplicates but
  do not survive a crash mid-handler; at-least-once is the contract.
