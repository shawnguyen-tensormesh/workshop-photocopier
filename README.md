# Your GPU Is a Very Expensive Photocopier

> **👉 In the live session?** Open **[WORKSHOP.md](WORKSHOP.md)** and the live demo at
> **https://tm-photocopier-rag.fly.dev** (or scan the QR). That's the guided hands-on:
> *explore → measure it yourself → take it home.* No install — a phone works; a laptop lets you
> run [`workshop_lab.py`](workshop_lab.py).
>
> The `agent.py` harness below is the **advanced take-home** — reproduce the same effect against
> your own vLLM endpoints.

A hands-on workshop repo: run a real coding-agent workload against a vanilla vLLM endpoint,
watch a per-request readout of where the time goes, then point the **same** harness at a
Tensormesh-backed endpoint — same model, same GPUs — and watch the recompute disappear.

> The only thing that changes between the two runs is one environment variable: `ENDPOINT`.

## 60-second quickstart

> **This is the take-home.** It runs against **your own** two endpoints — we don't host a public one
> (an open inference endpoint would be a free-for-all). For the **live, in-session** hands-on you don't
> need any of this: just open **https://tm-photocopier-rag.fly.dev** and follow [WORKSHOP.md](WORKSHOP.md).
> To run `agent.py` yourself, stand up the two endpoints first (see *Reproduce it on your own cluster* below).

```bash
pip install -r requirements.txt          # just `requests`-free: stdlib + optional transformers

# Run the frozen agent trajectory against YOUR vanilla vLLM endpoint:
export ENDPOINT=<your-vanilla-vLLM-endpoint>/v1
python agent.py replay --salt seat-42

# Now flip the ONE variable to YOUR Tensormesh-backed endpoint and re-run — same tokens:
export ENDPOINT=<your-tensormesh-endpoint>/v1
python agent.py replay --salt seat-42
```

Each turn prints:

```
turn  8 · ctx   8,955 tok · TTFT 4.83s · decode 0.21s · [cache: 0% prompt reused]
```

On the vanilla endpoint under room load, your context has been evicted — every turn
re-reads (prefills) the whole history: **TTFT climbs**. On the Tensormesh endpoint the same
context is served from a tiered KV cache that outlives any single pod: **TTFT stays flat**.

## What's here

| Path | What |
|---|---|
| `agent.py` | the harness — `replay` (frozen, measured) and `live` (free chat, un-measured) modes, HUD, `--salt`, `ENDPOINT`/`MODEL` env |
| `trajectory.py` | the **frozen** canonical coding-agent trajectory (fixed tokens ⇒ any latency delta is caching, nothing else) |
| `sample-repo/` | the context-engineered codebase the agent works in (`taskflow`, a tiered task queue with a real TTL bug the tests catch) |
| `rag/rag.py` | the RAG / CacheBlend workload — a hot-set corpus + query mix for the receipts act |

### Why prefill, not generation

An agent resends its whole history every turn, so the expensive part is *re-reading*
(prefilling) the accumulated context — not generating the next tokens. A KV cache lets you
skip the re-read. `replay` mode times exactly that (it generates only a few tokens per turn
on purpose), which is why the HUD headline is **TTFT**.

### Salting (why your run is *your* run)

`--salt seat-42` injects a unique header into the standing repository context, so every
attendee is a distinct ~9k-token context. No accidental sharing across seats — the cache
either has *your* salt's KV or it doesn't.

## Reproduce it on your own cluster (15 minutes)

The endpoints above are a vanilla vLLM engine and a Tensormesh-backed engine (vLLM + the
Tensormesh Operator's tiered KV cache) serving the **same** model. To stand up the
Tensormesh side yourself, follow the operator quickstart at **docs.tensormesh.ai**, then:

1. Serve any model on two engines — one plain, one with the operator's cache tier wired in.
2. Point `ENDPOINT` at each in turn and run `python agent.py replay`.
3. Drive enough distinct sessions (salts) that the working set exceeds one engine's GPU KV
   capacity — that's when the tier earns its keep (if everything fits in GPU, you won't see
   a difference; that's not a failure, it means your working set is too small to test it).

## Honest caveats

- The numbers you'll see are **regime-dependent**. Caching wins when there is reuse and the
  working set exceeds GPU memory; it is roughly neutral when everything already fits in the
  GPU's own prefix cache. We show you the regime, not a flat multiplier.
- `live` mode is **not** measured — it's for poking at the model. The demo numbers come from
  `replay` (frozen tokens) only.

## License / use

Workshop material. The `taskflow` sample code is illustrative — a deliberately small,
readable codebase with one real bug, not production code.
