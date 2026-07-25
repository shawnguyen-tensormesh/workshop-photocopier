#!/usr/bin/env python3
"""
RAG core — STOCK LlamaIndex (ingest + 256-aligned chunking + BM25 retrieval + generation),
plus the two things no framework provides, which are where Tensormesh's value lives:

  1. 256-token-ALIGNED chunking  — the TokenTextSplitter uses the served model's OWN tokenizer (Llama-3.1-8B here), so a
     RAG chunk is a whole number of blend chunks (chunkSize=256). Misalignment is what made
     CacheBlend look like "1x"; alignment is what makes it fire.
  2. index-time KV PRECOMPUTE     — after ingest we send each chunk through the fleet engine
     once, so its KV is cached in the blend tier BEFORE the first real question. ("cache the
     chunks, then blend the requests.")

To the framework, our engine is just an OpenAI-compatible endpoint (OpenAILike). The speedup
is invisible to LlamaIndex — that's the whole point: reuse your RAG stack, change one URL.

Blend fires only when retrieved chunks are delimited by the blend separator (" # # ") so each
is an independently-reusable segment, with a STABLE system prefix (APC catches that; blend
catches the shifting chunks). We assert external_kv_transfer > 0 so a silent misalignment can't
pass as a green demo.
"""
from __future__ import annotations
import json, logging, os, time, threading
import httpx

logger = logging.getLogger("rag_app")

from llama_index.core import Settings
from llama_index.core.schema import Document, TextNode, MetadataMode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever

# ---- config (env) --------------------------------------------------------------------------
MODEL        = os.environ.get("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
ARM_FLEET    = os.environ.get("ARM_FLEET",   "http://bench-workshop-b1")     # CacheBlend arm
ARM_VANILLA  = os.environ.get("ARM_VANILLA", "http://bench-workshop-base")   # plain vLLM arm
ARM_B2       = os.environ.get("ARM_B2",      "http://bench-workshop-b2")     # 2nd fleet engine (Act 3: cross-engine sharing)
BLEND_SEP    = os.environ.get("BLEND_SEP", " # # ")   # must equal the server's blend_special_str
CHUNK_TOK    = int(os.environ.get("CHUNK_TOK", "512"))   # RAG chunk = 2 blend chunks; >= blend_min_tokens(256)
TOP_K        = int(os.environ.get("TOP_K", "32"))   # 32 x 512-tok chunks = ~16k context (the reliable reuse regime for this demo)
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "1024"))  # room for the answer (512 overflowed on open-ended Qs -> empty)
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "")  # reasoning models only; leave empty for Llama/most models
SYSTEM_PROMPT = ("You are a helpful assistant. Answer the user's question using ONLY the "
                 "passages provided. If the answer is not in the passages, say you don't know.")

# ---- served-model tokenizer so chunking is token-exact & 256-aligned ----------------------------
_hf_tok = None
def _tokenizer():
    global _hf_tok
    if _hf_tok is None:
        from transformers import AutoTokenizer
        _hf_tok = AutoTokenizer.from_pretrained(MODEL)   # HF_HUB_OFFLINE=1 -> loads from local cache
    return _hf_tok

def _splitter() -> TokenTextSplitter:
    # chunk_overlap=0 on purpose: overlap shifts token boundaries and breaks blend alignment.
    return TokenTextSplitter(chunk_size=CHUNK_TOK, chunk_overlap=0, tokenizer=_tokenizer().encode)

def _llm(arm_url: str, max_tokens: int = MAX_TOKENS) -> OpenAILike:
    # reasoning_effort (reasoning models only): shorter analysis phase -> reaches the final answer within
    # budget and cuts the "silent while reasoning" gap that looked like a hang.
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    return OpenAILike(model=MODEL, api_base=arm_url.rstrip("/") + "/v1", api_key="none",
                      is_chat_model=True, temperature=0.0, max_tokens=max_tokens, timeout=600,
                      additional_kwargs=extra)


# ---- pre-tokenized, chunk-aligned prompt path (so CacheBlend fires on the interactive query) ----
# CacheBlend reuses a chunk's KV only if the chunk's token segment is byte-identical wherever it
# lands. Text-chat lets the tokenizer re-merge at the " # # " seams -> boundaries drift -> ~33%
# reuse. Sending explicit token IDs, with each chunk PADDED to a 256-multiple (the blend chunk
# size), keeps every segment aligned -> ~75% reuse (measured, Llama-3.1-8B). Store (precompute)
# and query MUST use the identical representation, so both go through _chunk_ids().
_SEP_IDS = None
def _sep_ids() -> list[int]:
    global _SEP_IDS
    if _SEP_IDS is None:
        _SEP_IDS = _tokenizer().encode(BLEND_SEP)[1:]   # drop BOS
    return _SEP_IDS

_FILLER = None
def _chunk_ids(text: str) -> list[int]:
    """Chunk text -> token IDs (no BOS), padded up to a 256-multiple so the segment is blend-aligned."""
    global _FILLER
    ids = _tokenizer().encode(text)[1:]
    if _FILLER is None:
        _FILLER = _tokenizer().encode(" filler")[-1]
    r = len(ids) % 256
    if r:
        ids = ids + [_FILLER] * (256 - r)
    return ids

def _stream_ids(url: str, ids: list[int], max_tokens: int):
    """Stream /v1/completions on a raw prompt-token-id list; yield text deltas."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    with httpx.stream("POST", url.rstrip("/") + "/v1/completions", json=body, timeout=600) as r:
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                t = json.loads(data)["choices"][0].get("text", "")
            except Exception:
                t = ""
            if t:
                yield t


class RagSession:
    """One attendee / one corpus. Per-session so uploads stay isolated (no cross-tenant reuse)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.nodes: list[TextNode] = []
        self.retriever: BM25Retriever | None = None
        self.warm = 0  # chunks confirmed warmed into the tier (precompute progress)
        self._lock = threading.Lock()

    # ---- PHASE 1: ingest + index (+ precompute) --------------------------------------------
    def ingest(self, texts: list[str], precompute: bool = True, background: bool = False) -> dict:
        docs = [Document(text=t) for t in texts if t and t.strip()]
        nodes = _splitter().get_nodes_from_documents(docs)
        with self._lock:
            self.nodes = nodes
            # bm25s errors if top_k > corpus size; clamp to the number of chunks.
            self.retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(TOP_K, len(nodes)))
        warmed = 0
        if precompute:
            # background=True for the startup corpus so the server serves immediately (a large
            # corpus can take minutes to warm); synchronous for uploads so the caller knows it's ready.
            if background:
                threading.Thread(target=self._precompute, daemon=True).start()
            else:
                warmed = self._precompute()
        return {"docs": len(docs), "chunks": len(nodes), "chunk_tokens": CHUNK_TOK,
                "precomputed": warmed, "precompute": "background" if (precompute and background) else "sync" if precompute else "off"}

    def _precompute(self) -> int:
        """Warm each chunk into the blend tier in the SAME chat representation the queries use.
        Precompute and query MUST share the endpoint/format (same chat template) or the chunk
        token sequences differ and fingerprints won't match -> blend won't fire."""
        n = 0
        sys_ids = _tokenizer().encode(SYSTEM_PROMPT); sep = _sep_ids()
        q_ids = _tokenizer().encode("\nQuestion: index\nAnswer:")[1:]
        for node in self.nodes:
            chunk = node.get_content(metadata_mode=MetadataMode.NONE)
            # store each chunk as the SAME padded token segment the query will send -> fingerprints match
            ids = sys_ids + sep + _chunk_ids(chunk) + sep + q_ids
            body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0}
            for attempt in range(3):  # retry: a transient tunnel/forward blip must not leave the tier cold
                try:
                    httpx.post(ARM_FLEET.rstrip("/") + "/v1/completions", json=body, timeout=600)
                    n += 1
                    break
                except Exception:
                    time.sleep(1)
        with self._lock:
            self.warm = n
        logger.info("CB precompute done: warmed %d/%d chunks for session %s", n, len(self.nodes), self.session_id)
        return n

    # ---- PHASE 2: retrieve + assemble (blend-aligned) --------------------------------------
    def retrieve(self, query: str) -> list[str]:
        if not self.retriever:
            return []
        # retrieve() returns NodeWithScore; go through .node for the raw chunk text.
        return [ns.node.get_content(metadata_mode=MetadataMode.NONE) for ns in self.retriever.retrieve(query)]

    def build_prompt_ids(self, history: list[dict], query: str) -> tuple[list[int], int]:
        """Instruct-templated prompt with the retrieved chunks spliced in as PRE-TOKENIZED,
        256-aligned segments. The chat template (special tokens) makes the Instruct model answer
        coherently; the aligned ` # # `-delimited chunk segments let CacheBlend reuse them wherever
        they land. Blend keys on the segments, so the template wrapper doesn't break reuse."""
        tok = _tokenizer(); sep = _sep_ids()
        chunks = self.retrieve(query)
        SENT = "<<<CHUNKS>>>"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:                       # multi-turn: prior turns re-sent -> photocopier waste on vanilla
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": f"Passages:{SENT}\nQuestion: {query}"})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        left, right = text.split(SENT)
        ids = tok.encode(left, add_special_tokens=False)   # template head (incl. special tokens)
        for c in chunks:
            ids += sep + _chunk_ids(c)                     # aligned reusable segments
        ids += sep + tok.encode(right, add_special_tokens=False)  # "\nQuestion:… <assistant>"
        return ids, len(chunks)

    # ---- generation via raw pre-tokenized completions: TTFT + per-query blend reuse --------
    def chat_stream(self, arm: str, history: list[dict], query: str):
        """Yield ('token', text) deltas then ('done', {metrics incl. per-query blend reuse})."""
        arm_url = ARM_FLEET if arm == "fleet" else ARM_VANILLA
        ids, k = self.build_prompt_ids(history, query)
        # Bracket THIS engine's own token-source counters (not the shared blend server, whose global
        # counters the v3b sim loop pollutes → false partial-reuse readings). reused = local prefix
        # cache + external tier; recomputed = local_compute. Verified 2026-07-25: a warmed preset moves
        # reused by exactly the full context (+16384) and recomputed by 0 → a truthful per-query 100%.
        s0 = token_sources(arm_url) if arm == "fleet" else {}
        t0 = time.time(); ttft = None; full = []
        try:
            for t in _stream_ids(arm_url, ids, MAX_TOKENS):
                if ttft is None:
                    ttft = time.time() - t0
                full.append(t); yield ("token", t)
        except Exception:
            pass
        if not "".join(full).strip():           # never show a blank pane: non-streaming fallback
            try:
                r = httpx.post(arm_url.rstrip("/") + "/v1/completions",
                               json={"model": MODEL, "prompt": ids, "max_tokens": MAX_TOKENS,
                                     "temperature": 0.0}, timeout=600).json()
                txt = (r["choices"][0].get("text") or "").strip()
                if txt:
                    if ttft is None:
                        ttft = time.time() - t0
                    full.append(txt); yield ("token", txt)
            except Exception:
                pass
        reused = requested = 0
        if arm == "fleet":
            s1 = token_sources(arm_url)
            reused = int(max(0, (s1["local_cache_hit"] + s1["external_kv_transfer"])
                              - (s0.get("local_cache_hit", 0) + s0.get("external_kv_transfer", 0))))
            recomputed = int(max(0, s1["local_compute"] - s0.get("local_compute", 0)))
            requested = reused + recomputed
            # rare: metric scrape didn't advance (race) — fall back to the assembled context length
            if requested == 0:
                requested = len(ids)
        pct = round(reused / requested, 3) if requested else 0.0
        yield ("done", {"arm": arm, "chunks": k,
                        "ttft_s": round(ttft or (time.time() - t0), 3),
                        "wall_s": round(time.time() - t0, 3),
                        "blend_reused": reused, "blend_requested": requested, "reuse_pct": pct,
                        "text": "".join(full)})

    def fire_load(self, arm: str, i: int, k: int = 16, max_tokens: int = 1) -> float | None:
        """APC-defeating load query: SYS + K chunks in a SHIFTED order (rotates with i) + a
        question. The shifting chunk block near the front breaks vanilla's exact-prefix cache
        (it recomputes the whole block every query); the fleet reuses the chunks via non-prefix
        CacheBlend. Under concurrency this is what makes vanilla saturate while the fleet holds.
        Returns prefill latency (max_tokens=1), or None on error."""
        arm_url = ARM_FLEET if arm == "fleet" else ARM_VANILLA
        with self._lock:
            nodes = self.nodes
        if not nodes:
            return None
        n = len(nodes)
        order = [(i * 3 + j) % n for j in range(min(k, n))]  # shifted per query
        chunks = [nodes[o].get_content(metadata_mode=MetadataMode.NONE) for o in order]
        context = BLEND_SEP + BLEND_SEP.join(chunks) + BLEND_SEP
        msgs = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=f"Passages:{context}\nQuestion {i}: summarize.")]
        t0 = time.time()
        try:
            _llm(arm_url, max_tokens).chat(msgs)
            return time.time() - t0
        except Exception:
            return None


# ---- load driver: REAL-RAG replay over the DEMO corpus. Each request runs the SAME retrieve +
# blend-aligned prompt path an attendee uses, drawn from a LARGE, diverse question space built
# from the corpus's own entities. The point (honest, no artifice): a big space of distinct
# reordered chunk-blocks makes the block working-set exceed the capped GPU-KV, so vanilla's
# exact-prefix cache (APC) thrashes and RECOMPUTES, while blend serves every chunk from the CPU
# tier and REUSES. Driving over the demo corpus also REINFORCES the attendee tier (no eviction),
# unlike the old synthetic-doc driver which churned the corpus out of L1. --------------------
_Q_BANK: "list[str] | None" = None


def _fnv(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def _question_bank(session: "RagSession") -> list[str]:
    """A large, diverse, deterministic question space from the corpus entities (catalog SKUs +
    policy/benefit/engineering topics). Distinct questions retrieve distinct chunk-blocks, which
    is what defeats vanilla's prefix cache honestly (no salt, no forced reorder)."""
    global _Q_BANK
    if _Q_BANK is not None:
        return _Q_BANK
    import re
    names, seen = [], set()
    for node in session.nodes:
        for m in re.findall(r'^#{2,3}\s+(.+)$', node.get_content(metadata_mode=MetadataMode.NONE), re.M):
            nm = m.strip().lstrip('#').strip()
            if nm and nm.lower() not in seen and 3 < len(nm) < 80:
                seen.add(nm.lower()); names.append(nm)
    prod_t = ["What are the specifications of the {}?",
              "How much does the {} cost?",
              "What is the warranty and RMA process for the {}?",
              "What are the power draw and dimensions of the {}?",
              "What is the lead time and stock status for the {}?"]
    topic_qs = [
        "What is the return window and restocking fee for opened items?",
        "How do I start a return and what is required for a full refund?",
        "What is the shipping and delivery policy for expedited orders?",
        "What does the product warranty cover and what voids it?",
        "How much paid time off do full-time employees accrue?",
        "What are the health insurance and retirement benefits?",
        "What is the parental leave policy?",
        "Describe the service topology in the engineering runbook.",
        "What is the deployment and rollback procedure?",
        "How is on-call rotation and incident response handled?",
        "What is the escalation path for a severity-one incident?",
        "What are the customer support hours and response SLAs?",
    ]
    qs = list(topic_qs)
    for nm in names:
        for t in prod_t:
            qs.append(t.format(nm))
    _Q_BANK = sorted(set(qs), key=_fnv)   # deterministic order so warm + measure are reproducible
    logger.info("load question bank: %d questions from %d corpus entities", len(_Q_BANK), len(names))
    return _Q_BANK


def load_warm(session: "RagSession", stop=None) -> int:
    """Gentle SEQUENTIAL warm of the fleet tier before heavy concurrent load (a cold CacheBlend
    engine stalls under concurrent recompute). The demo-session precompute already warms every
    chunk; this also exercises the retrieve+assemble path across the question space."""
    bank = _question_bank(session)
    if not bank:
        return 0
    step = max(1, len(bank) // 60)   # ~60 warm queries spread across the space
    n = 0
    for q in bank[::step]:
        if stop is not None and stop.is_set():
            break
        try:
            ids, _ = session.build_prompt_ids([], q)
            httpx.post(ARM_FLEET.rstrip("/") + "/v1/completions",
                       json={"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0}, timeout=300)
            n += 1
        except Exception:
            pass
    return n


def fire_load_rag(session: "RagSession", arm: str, i: int) -> float | None:
    """One real-RAG load request: pick a question from the diverse bank, run the SAME retrieve +
    blend-aligned prompt path as an attendee, prefill-only (max_tokens=1). Fleet reuses the warm
    chunks (non-prefix blend); vanilla's APC misses the near-unique reordered block -> recompute.
    arm 'b2' = the second fleet engine (v3b) — same shared cache, but NOT the engine live taps use,
    so the meter can be fed there without contending with the demo."""
    arm_url = {"fleet": ARM_FLEET, "vanilla": ARM_VANILLA, "b2": ARM_B2}.get(arm) or ARM_FLEET
    if not arm_url:
        return None
    bank = _question_bank(session)
    if not bank:
        return None
    q = bank[i % len(bank)]
    try:
        ids, _ = session.build_prompt_ids([], q)
    except Exception:
        return None
    t0 = time.time()
    try:
        httpx.post(arm_url.rstrip("/") + "/v1/completions",
                   json={"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0}, timeout=300)
        return time.time() - t0
    except Exception:
        return None


# One-tap questions the room picks from. DELIBERATELY MANY + DIVERSE (>28): the working set of
# distinct ~16k-token contexts must exceed the engine's GPU prefix-cache capacity (~460k tok ≈ 28
# contexts) so plain vLLM keeps EVICTING and recomputing them, while Tensormesh's larger shared L1
# (170 GB) holds them all and reuses. Validated 2026-07-24: paced, ~4-5x lower TTFT, repeat-proof.
DEMO_PRESETS = [
    "What is the return window and restocking fee for opened items?",
    "How do I start a return and get a full refund?",
    "What is the shipping and delivery policy for expedited orders?",
    "What does the product warranty cover, and what voids it?",
    "How do I file a warranty RMA for a failed server?",
    "What is the bulk-order discount policy?",
    "What payment methods does Northwind accept?",
    "How much paid time off do full-time employees accrue?",
    "What are the health insurance and retirement benefits?",
    "What is the parental leave policy?",
    "What onboarding steps are in the employee handbook?",
    "Describe the service topology in the engineering runbook.",
    "What is the deployment and rollback procedure?",
    "How is on-call rotation and incident response handled?",
    "What is the disaster-recovery RTO and RPO?",
    "What are the specifications of the Northwind 2U server?",
    "How much does the Northwind rack PDU cost?",
    "What is the capacity of the Northwind NVMe SSD?",
    "What are the port counts on the Northwind switch?",
    "What is the wattage of the Northwind UPS?",
    "What cable categories does Northwind sell?",
    "What routing features does the Northwind router support?",
]


def _prefill(url, ids):
    try:
        httpx.post(url.rstrip("/") + "/v1/completions",
                   json={"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0}, timeout=300)
        return True
    except Exception:
        return False


def prime_regime(session: "RagSession", stop=None) -> dict:
    """Set up the capacity/eviction regime, run PACED (sequential) for a clean single-number
    read. The presets (~22) alone FIT in vanilla's ~28-context GPU cache, so priming must
    push them OUT: (1) warm every preset into the fleet's big L1 (fleet will reuse); (2) on vanilla,
    ask the presets THEN ~40 more distinct bank questions — 22+40 >> 28, so the presets become the
    oldest and evict. Result: the first tap of ANY preset recomputes on vanilla but reuses on fleet."""
    warmed = 0
    for q in DEMO_PRESETS:                      # 1) warm preset assemblies into fleet L1
        if stop is not None and stop.is_set():
            break
        try:
            ids, _ = session.build_prompt_ids([], q); warmed += _prefill(ARM_FLEET, ids)
        except Exception:
            pass
    # 2) on vanilla: presets first, then extra distinct questions to evict them past the GPU cap
    bank = _question_bank(session)
    extra = [q for q in bank if q not in set(DEMO_PRESETS)][:40]
    filled = 0
    for q in DEMO_PRESETS + extra:
        if stop is not None and stop.is_set():
            break
        try:
            ids, _ = session.build_prompt_ids([], q); filled += _prefill(ARM_VANILLA, ids)
        except Exception:
            pass
    logger.info("prime_regime: warmed %d presets (fleet), filled %d on vanilla (%d presets + %d evictors)",
                warmed, filled, len(DEMO_PRESETS), len(extra))
    return {"warmed": warmed, "vanilla_filled": filled, "presets": len(DEMO_PRESETS), "evictors": len(extra)}


def engine_ttft(arm_url: str):
    """(sum, count) of the engine's own vllm:time_to_first_token_seconds histogram. A windowed
    mean (Δsum/Δcount) from this reflects ALL load hitting the engine — the in-cluster loadgen
    AND webapp chats — not just this process's samples. That's what the scoreboard TTFT tile uses."""
    s = c = 0.0
    got = False
    try:
        r = httpx.get(arm_url.rstrip("/") + "/metrics", timeout=10)
        for ln in r.text.splitlines():
            if ln.startswith("vllm:time_to_first_token_seconds_sum"):
                s += float(ln.rsplit(" ", 1)[1]); got = True
            elif ln.startswith("vllm:time_to_first_token_seconds_count"):
                c += float(ln.rsplit(" ", 1)[1])
    except Exception:
        return None
    return (s, c) if got else None


_served_cache = {"name": None, "at": 0.0}

def served_model(arm_url: str = None):
    """The model id the engine actually serves (from /v1/models) — lets the info panel auto-detect
    a model swap instead of trusting a hardcoded string. Cached 5 min; falls back to MODEL."""
    now = time.time()
    if _served_cache["name"] and now - _served_cache["at"] < 300:
        return _served_cache["name"]
    try:
        r = httpx.get((arm_url or ARM_FLEET).rstrip("/") + "/v1/models", timeout=8)
        mid = (r.json().get("data") or [{}])[0].get("id")
        if mid:
            _served_cache.update(name=mid, at=now)
            return mid
    except Exception:
        pass
    return _served_cache["name"] or MODEL


def token_sources(arm_url: str) -> dict:
    """{local_compute, local_cache_hit, external_kv_transfer} cumulative for an arm — the
    recompute-vs-reuse breakdown engineers care about."""
    out = {"local_compute": 0.0, "local_cache_hit": 0.0, "external_kv_transfer": 0.0}
    try:
        r = httpx.get(arm_url.rstrip("/") + "/metrics", timeout=10)
        for ln in r.text.splitlines():
            if ln.startswith("vllm:prompt_tokens_by_source_total") and "{" in ln:
                lab = ln.split("{", 1)[1].split("}")[0]
                src = [x.split("=")[1].strip('"') for x in lab.split(",") if x.startswith("source=")]
                if src and src[0] in out:
                    out[src[0]] += float(ln.rsplit(" ", 1)[1])
    except Exception:
        return out
    return out


def _external_kv(arm_url: str):
    """vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} — the reuse pulse."""
    try:
        r = httpx.get(arm_url.rstrip("/") + "/metrics", timeout=10)
        total = 0.0; found = False
        for ln in r.text.splitlines():
            if ln.startswith("vllm:prompt_tokens_by_source_total") and 'source="external_kv_transfer"' in ln:
                total += float(ln.rsplit(" ", 1)[1]); found = True
        return total if found else 0.0
    except Exception:
        return None


BLEND_METRICS = os.environ.get("BLEND_METRICS", "")  # e.g. http://127.0.0.1:18002 (blend server :8080 via tunnel)

def blend_lookup():
    """Blend server's OWN self-consistent counters: {hit, requested} tokens. True reuse rate =
    hit/requested (~80%). This is the truthful CacheBlend signal — the engine's prompt_tokens_by_source
    undercounts reuse (blend's reused tokens show as local_compute). Returns {} if unreachable."""
    if not BLEND_METRICS:
        return {}
    out = {}
    try:
        r = httpx.get(BLEND_METRICS.rstrip("/") + "/metrics", timeout=10)
        for ln in r.text.splitlines():
            if ln.startswith("lmcache_blend_lookup_hit_tokens_total "):
                out["hit"] = float(ln.rsplit(" ", 1)[1])
            elif ln.startswith("lmcache_blend_lookup_requested_tokens_total "):
                out["requested"] = float(ln.rsplit(" ", 1)[1])
    except Exception:
        return {}
    return out


def blend_hit_tokens():
    """Total tokens the blend server served FROM CACHE. Returns None if unreachable."""
    return blend_lookup().get("hit")


def cross_engine_probe(session: "RagSession", query: str) -> dict:
    """Act 3 — cross-engine sharing: query a SECOND fleet engine (B) that shares the same blend
    server. B never computed these passages, but engine A / the loadgen already warmed them into
    the shared tier, so B reuses them on arrival. Signal = B's external_kv delta; also returns a
    short answer to prove B is correct. (Engine accounting may undercount blend; we surface both.)"""
    ext0 = _external_kv(ARM_B2); bh0 = blend_hit_tokens()
    ids, _ = session.build_prompt_ids([], query)   # pre-tokenized, blend-aligned (same path as chat_stream)
    ans = ""
    try:
        # PREFILL-ONLY (max_tokens=1): cross-engine reuse is measured on the prefill, so we don't
        # generate an answer — the UI is numbers-only anyway.
        httpx.post(ARM_B2.rstrip("/") + "/v1/completions",
                   json={"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0},
                   timeout=600).json()
    except Exception as e:
        ans = f"(engine B error: {e})"
    time.sleep(1.0)
    ext1 = _external_kv(ARM_B2); bh1 = blend_hit_tokens()
    ext_delta = int(ext1 - ext0) if (ext1 is not None and ext0 is not None) else None
    blend_delta = int(bh1 - bh0) if (bh1 is not None and bh0 is not None) else None
    return {"external_kv_delta": ext_delta, "blend_hit_delta": blend_delta, "answer": ans[:240]}


def assert_blend_fires(session: "RagSession", query: str) -> dict:
    """Pre-demo gate: fire ONE prefill-only fleet query (max_tokens=1 — fast, no long generation)
    and require reuse to fire. The truthful signal is the BLEND SERVER's hit-tokens delta (the engine
    counter undercounts CacheBlend as local_compute); fall back to the fleet external_kv delta."""
    bh0 = blend_hit_tokens(); ext0 = _external_kv(ARM_FLEET)
    ids, _ = session.build_prompt_ids([], query)   # real RAG context, pre-tokenized (same path as chat_stream)
    try:
        httpx.post(ARM_FLEET.rstrip("/") + "/v1/completions",
                   json={"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0},
                   timeout=600)   # prefill only -> fast
    except Exception:
        pass
    time.sleep(1.0)
    bh1 = blend_hit_tokens(); ext1 = _external_kv(ARM_FLEET)
    blend_delta = int(bh1 - bh0) if (bh1 is not None and bh0 is not None) else None
    ext_delta = int(ext1 - ext0) if (ext1 is not None and ext0 is not None) else None
    ok = bool((blend_delta and blend_delta > 0) or (ext_delta and ext_delta > 0))
    return {"blend_fired": ok, "blend_hit_delta": blend_delta, "external_kv_delta": ext_delta,
            "note": "OK" if ok else "NOT firing — check chunk alignment / cache warmth / BLEND_SEP"}
