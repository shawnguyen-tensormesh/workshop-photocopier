# Workshop webapp — "Your GPU Is a Very Expensive Photocopier"

A dual-pane RAG doc-chat (plain vLLM vs Tensormesh CacheBlend) with a live **$-saved scoreboard**,
built on **stock LlamaIndex**. The point of the demo is also the point of the product: *reuse beats
rebuild*. We don't reimplement RAG — we point a standard framework at the Tensormesh engine and it
gets faster with **one URL change**.

```
webapp/
  rag_app.py        stock LlamaIndex (TokenTextSplitter + BM25 + OpenAILike) + 2 Tensormesh shims
  server.py         FastAPI: /, /ingest, /chat (SSE), /scoreboard, /gate, /healthz
  static/index.html dual-pane chat + live scoreboard (self-contained)
  corpus/           the shared demo knowledge base (multi-chunk docs + generated catalog)
  gen_corpus.py     generates a LARGE catalog (~319 chunks) so the working set exceeds GPU KV
  Dockerfile        runtime image (tokenizer baked; runs air-gapped)
  requirements.txt
```

## Why LlamaIndex, and what's *not* stock

To the framework, the engine is just an OpenAI-compatible endpoint (`OpenAILike`), so the speedup is
invisible — that's the demo. Only **two** things are Tensormesh-specific, because no RAG framework
provides them:

1. **256-aligned chunking.** `TokenTextSplitter` uses the served model's *own* tokenizer with `chunk_overlap=0`,
   so a RAG chunk is a whole number of blend chunks (`chunkSize=256`). Misalignment is exactly what
   made CacheBlend look like "1×"; alignment is what makes it fire.
2. **Index-time KV precompute.** After ingest we send each chunk through the fleet engine once, so its
   KV is cached in the blend tier *before* the first question ("cache the chunks, then blend").

Retrieved chunks are joined with the blend separator (`" # # "`) so each is an independently-reusable
segment, behind a **stable system prefix** (APC catches the prefix; CacheBlend catches the shifting
chunks). `assert_blend_fires()` / `GET /gate` refuse to call a run green unless
`external_kv_transfer > 0` — a silent misalignment can't masquerade as a working demo.

## Config (env)

| var | default | notes |
|---|---|---|
| `MODEL` | `NousResearch/Meta-Llama-3.1-8B-Instruct` | served name + tokenizer |
| `ARM_FLEET` | `http://bench-workshop-b1` | CacheBlend engine (reuse) |
| `ARM_VANILLA` | `http://bench-workshop-base` | plain vLLM engine (recompute) |
| `CHUNK_TOK` | `512` | RAG chunk = 2 blend chunks; must be ≥ `blend_min_tokens` (256) |
| `TOP_K` | `32` | retrieved chunks per query (32×512 ≈ 16k context) |
| `BLEND_SEP` | `" # # "` | must equal the server's `blend_special_str` |
| `DOLLARS_PER_GPU_HR` | `2.50` | the one agreed number for the $ meter |
| `PREFILL_TOK_S` | `13000` | tok/s/GPU for the GPU-seconds math |

## Run locally (against the two engines)

```bash
pip install -r requirements.txt
export ARM_FLEET=http://<b1-svc>  ARM_VANILLA=http://<base-svc>  HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface
uvicorn server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## The demo (use case)

**Headline — the room queries a shared knowledge base, over multiple turns, under concurrent load.**
Plain vLLM recomputes the same retrieved passages every turn and every user (the photocopier); the
fleet reuses them. The scoreboard shows tokens reused → GPU-seconds avoided → **$ saved**, and the
TTFT gap widens as more people join — because the win lives in the *concurrent, memory-pressured,
shared-cache* regime (the paced single-query win is ~5×, and plain vLLM only falls further behind as
the room piles on).

**Corpus size matters (measured 2026-07-20).** The external-KV scoreboard only moves when the working
set exceeds GPU KV (~87 chunks at the 3.3GB demo cap). A tiny corpus is fully caught by vLLM's own
prefix cache (APC) → blend tier idle → scoreboard reads 0. `gen_corpus.py` builds a ~319-chunk catalog
(~3.7× GPU KV); a 40-query sweep then hit the blend tier on 26/40 queries (35k tokens reused). Scale
the catalog to your GPU-KV cap with `python gen_corpus.py`.

## Pre-demo gate

```bash
curl "http://localhost:8000/gate"     # {"blend_fired": true, "external_kv_delta": <n>, ...}
```
If `blend_fired` is false: check chunk alignment (`CHUNK_TOK` multiple of 256, right tokenizer),
that the fleet cache is warm/**dedicated** (never share a churned blend server across fleets — that
silently corrupts reuse), and that `BLEND_SEP` matches the server's `blend_special_str`.

## Deploy

Build the image (`Dockerfile`) and run it alongside the two engines; point `ARM_FLEET` / `ARM_VANILLA`
at their service URLs. For a public route, put the app behind any reverse proxy / tunnel to the
engines. This repo is the website only.
