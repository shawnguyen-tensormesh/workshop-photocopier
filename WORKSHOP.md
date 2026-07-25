# Your GPU Is a Very Expensive Photocopier — hands-on workshop

**KCD × OpenInfra Days Vietnam 2026 · Shaw Nguyen (Tensormesh)**

A ~45-minute hands-on session. You'll poke a **live** two-engine LLM demo from your own phone or
laptop, **measure** the difference yourself, and leave knowing how to get the same win in your own
stack.

> **The metaphor:** an LLM without a KV cache is a photocopier that re-reads every page of the
> original *from scratch* before making each copy. The reading (prefill) is the expensive part.
> Tensormesh remembers what it already read.

---

## What you'll be able to do by the end

1. Explain **why LLM serving cost is dominated by *prefill*** — recomputing the prompt's KV.
2. Explain **why prefix caching alone isn't enough** for RAG (different passages, different order → cache miss).
3. Explain **what KV reuse (LMCache CacheBlend) does** — reuse KV across queries, positions, and engines.
4. **Measure** cache reuse and time-to-first-token yourself, and read the numbers honestly.
5. Reason about **the regime** — when reuse wins big, and when it doesn't.
6. Know **how to add LMCache to your own vLLM** deployment.

---

## Before you start

- **Everyone (phone is fine):** open the live demo → **https://tm-photocopier-rag.fly.dev** (or scan the QR on screen).
- **For the measurement lab (laptop):** either `curl` (built in) or **Python 3** (built in). No installs.
- Grab this repo's [`workshop_lab.py`](workshop_lab.py) if you want the Python version.

---

## The 60-second mental model

An LLM answers in two phases:

- **Prefill** — read the whole prompt, build a *KV cache* for every token. Cost grows with prompt length. **This is the photocopier reading the pages.**
- **Decode** — generate the answer one token at a time from that KV.

In **RAG** (retrieval-augmented generation) the prompt is *huge*: a system prompt + dozens of retrieved passages + your question — often ~16k tokens. Every request re-reads all of it.

- **Prefix caching** (built into vLLM) reuses KV only for an **exact shared prefix**. But RAG pulls **different** passages in a **different order** every query, so past the system prompt there's little to reuse — the copier still re-reads almost everything.
- **Tensormesh (LMCache CacheBlend)** stores the KV for each *chunk* and **reuses it wherever it appears** — different query, different position, even a *different engine*. Skip the re-read → the first token lands far sooner.

---

## Part 1 — Explore (everyone, ~10 min)

Open **https://tm-photocopier-rag.fly.dev**. Two panes: **Plain vLLM** (recompute) vs **Tensormesh CacheBlend** (reuse).

**Task 1 — Reuse vs recompute.**
Tap any preset question. Watch both stopwatches.
- Plain vLLM time-to-first-token: `______ s`
- Tensormesh time-to-first-token: `______ s`
- How many × sooner did Tensormesh answer? `______ ×`

**Task 2 — Predict, then check.**
You tap the **same** question again.
- *Predict:* does **Plain vLLM** get faster the second time? `yes / no`
- Now try it. What happened, and **why**? (Hint: does plain vLLM keep anything between requests?)

**Task 3 — The fresh engine (sharing).**
Hit **"Ask a fresh engine (B)"**. This is a *brand-new* engine that has never seen these passages.
- Is it slow (cold) or fast (warm)? `______`
- *Why can it be warm on arrival?* (Hint: where does the cache live — inside one GPU, or shared?)

> **The point:** a per-GPU cache is a *pet* — it dies with the pod and helps only itself. A shared
> Tensormesh cache tier is *cattle* — it outlives any pod and every engine draws from it.

---

## Part 2 — Measure it yourself (laptop, ~10 min)

Don't take the on-screen numbers on faith — pull them yourself. Each request reports its
time-to-first-token and **what fraction of the prompt was served from cache**.

> **No endpoint to configure.** The lab and the curl below point at the **live demo you already
> opened** (`tm-photocopier-rag.fly.dev`) — just run them. (Standing up your *own* endpoints is
> Part 3 / `agent.py`.)

**Option A — Python (recommended, one file, no installs):**
```bash
# grab the one-file lab (stdlib only; the demo URL is baked in):
curl -O https://raw.githubusercontent.com/shawnguyen-tensormesh/workshop-photocopier/main/workshop_lab.py
python3 workshop_lab.py "How do I start a return and get a full refund?"
```
Example output:
```
engine                     first token    from cache
----------------------------------------------------
vanilla (plain vLLM)           1.05 s           0 %
fleet (Tensormesh)             0.21 s         100 %
----------------------------------------------------
  -> Tensormesh reached the first token 4.9x sooner.
```

**Option B — curl (no Python):**
```bash
curl -sN "https://tm-photocopier-rag.fly.dev/chat?arm=fleet&q=How%20do%20I%20start%20a%20return%20and%20get%20a%20full%20refund%3F" \
| sed -n 's/^data: //p' \
| python3 -c 'import sys,json
for l in sys.stdin:
 d=json.loads(l).get("done") if l.strip() else None
 if d: print("first token %.2fs  from cache %d%%" % (d["ttft_s"], round(d["reuse_pct"]*100)))'
```
Swap `arm=fleet` for `arm=vanilla` to compare. (Use a question from the **preset list** — those are
pre-warmed into the cache, so you see the reuse; a brand-new phrasing is the *Task 5* case below.)

> **What happens when the whole room taps at once?** The gap gets *bigger*, not smaller. The fleet
> keeps reusing, so it stays ~0.3s no matter the load; plain vLLM has to recompute every request and its
> single engine queues. Measured as a clean ramp — first-token at 1 / 10 / 30 concurrent taps:
> **fleet 0.20s → 0.29s → 0.31s** (flat), **vanilla 1.1s → 7s → ~15s** (up to ~25s at the tail). Every
> person still saw the win. So don't be alarmed if the vanilla pane grinds under load — that's the point.
> The only time the fleet also slows: if everyone asks a *brand-new* question at once (nothing to reuse).

**Task 4 — Same preset, three times.**
Run the lab with one of the **preset questions** (the ones on the site) three times. Is the Tensormesh
win **stable** across runs? Note the `from cache` % — a warmed preset reads ~100% from cache (vanilla:
0%), and the ~5× holds run to run.

**Task 5 — Try to make plain vLLM win.**
Now ask something **not** in the preset list — your own phrasing. Its passages haven't been pre-warmed,
so the fleet has little to reuse (`from cache` drops) and the gap shrinks toward 1×. That's the honest
lesson: **reuse only helps when there's something to reuse.** (More in Part 3.)

---

## Part 3 — Take it home (~5 min)

### Add LMCache to your own vLLM

[LMCache](https://github.com/LMCache/LMCache) plugs a KV cache tier into vLLM — offload to CPU/SSD
and reuse across requests and instances. Conceptually:

1. Run vLLM with the **LMCache KV connector** enabled (a `--kv-transfer-config` on the engine).
2. Point it at an **LMCache server / tier** (CPU RAM, local SSD, or a shared network tier) sized to
   your working set.
3. For non-prefix / RAG reuse, enable **CacheBlend** so chunks are reused regardless of position.

Start here: **https://docs.lmcache.ai** · vLLM integration + connector config, backends, and the
CacheBlend guide. Tensormesh packages and tunes this into a production data-plane appliance.

### The honest regime (so you know when it pays off)

Reuse wins when you'd otherwise **re-prefill the same context** — long shared prompts, RAG over a
fixed corpus, multi-turn chat, agents re-reading the same files. The win **shrinks** when prompts
are short or genuinely unique every time (little to reuse), and a shared tier's real payoff is
serving **working sets larger than GPU memory** and **surviving/​sharing across pods**. Measure your
own workload — don't assume a flat multiplier.

---

## Facilitator run-of-show (45 min)

| min | segment | who |
|---|---|---|
| 0–8 | Hook + the photocopier metaphor; prefill vs decode; why RAG re-reads | present |
| 8–15 | Live demo: reuse vs recompute, the repeat, the fresh engine (Acts 1–3) | present |
| 15–25 | **Part 1 — Explore** (attendees on phones; walk the room) | attendees |
| 25–37 | **Part 2 — Measure it yourself** (laptops; help stragglers) | attendees |
| 37–45 | **Part 3** — debrief Task 5 + the honest regime + how to add LMCache + Q&A | present |

**Facilitator notes**
- The demo is **paced** — one query at a time gives the cleanest single-number read; encourage taking turns.
- **Under a full room it holds up — and looks better:** measured as a ramp, the fleet stayed flat
  (~0.2→0.3s at 1/10/30 concurrent, 100% reuse) while plain vLLM climbed smoothly (~1s→7s→~15s, tail ~25s);
  every person still saw the win. Frame the slow vanilla pane as the point ("vanilla is drowning"). It only
  breaks down if everyone asks a brand-new, unwarmed question at the same moment.
- `from cache %` is the honest signal to point at, more than the raw speedup.
- If someone asks "is the baseline fair?": yes — the room's working set exceeds vanilla's GPU cache,
  so it keeps evicting and recomputing the retrieved passages, while the shared tier holds them all.
  Prefix caching only reuses an *exact* prefix, so RAG (different passages, reordered) still re-reads —
  that deeper story is on the slides.
- The control endpoints are locked on the public deploy; attendees can only read/query.

---

## Resources

- **Live demo:** https://tm-photocopier-rag.fly.dev
- **This repo (take-home):** https://github.com/shawnguyen-tensormesh/workshop-photocopier — incl. the lab script [`workshop_lab.py`](workshop_lab.py)
- **LMCache:** https://github.com/LMCache/LMCache · https://docs.lmcache.ai
- **vLLM:** https://github.com/vllm-project/vllm
- **Tensormesh:** https://tensormesh.ai
