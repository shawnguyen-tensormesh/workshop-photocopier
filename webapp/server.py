#!/usr/bin/env python3
"""
Workshop website — dual-pane RAG doc-chat (vanilla vLLM vs Tensormesh CacheBlend) + live
$-saved scoreboard. FastAPI thin shell around rag_app (stock LlamaIndex).

  GET  /                      the attendee site (dual-pane chat + scoreboard)
  POST /ingest                {session, texts[]} -> chunk + index + precompute (per-session)
  GET  /chat?session&arm&q    SSE stream of ONE arm's answer (frontend opens two, side by side)
                              history is carried by the client and posted back for multi-turn
  GET  /scoreboard            JSON: cumulative recompute-avoided -> GPU-sec -> $ saved; latest TTFTs
  GET  /gate?q                pre-demo: assert blend fires on the shared corpus (external_kv>0)
  GET  /healthz

Headline use case: the room asks a SHARED knowledge base (webapp/corpus, loaded + precomputed
at startup) over multiple turns, under concurrent load -> vanilla recomputes the same chunks
every turn/user (the "expensive photocopier"); Tensormesh reuses them. Personal hook: POST your
own text to a private session.
"""
import json, os, threading, time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

import rag_app
from rag_app import RagSession

HERE = os.path.dirname(os.path.abspath(__file__))
DOLLARS_PER_GPU_HR = float(os.environ.get("DOLLARS_PER_GPU_HR", "2.50"))
PREFILL_TOK_S = float(os.environ.get("PREFILL_TOK_S", "13000"))   # tok/s/GPU for the GPU-sec math
MAX_RUNS_PER_MIN = int(os.environ.get("MAX_RUNS_PER_MIN", "20"))

# On the PUBLIC fly deploy set PUBLIC_READONLY=1: control endpoints (/load/*, /ingest) 403 so a
# visitor can't drive load against the dev engines or push arbitrary uploads. The read-only demo
# surface (/, /chat, /scoreboard, /presets, /sysinfo, /gate, /crossengine) stays open for the QR.
PUBLIC_READONLY = os.environ.get("PUBLIC_READONLY", "0") == "1"

# Deploy-time facts for the "system & methodology" panel. Defaults track the current live fleet;
# override via env (fly secrets / start.sh) so a config change flows through here instead of the
# static HTML rotting. The genuinely-runtime bits (model, context size, corpus size, cross-engine
# wiring) are pulled live in /sysinfo from rag_app + the demo session, not hardcoded.
SYS = {
    "hardware": os.environ.get("SYS_HW",
        "1× NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120) per engine · tensor-parallel 1"),
    "stack":    os.environ.get("SYS_STACK", "vLLM + LMCache v0.5.2 (nightly 2026-07-21)"),
    "kv_cap":   os.environ.get("SYS_KVCAP", "3.3 GB"),
    "recomp":   os.environ.get("SYS_RECOMP", "0.15"),
    "l1":       os.environ.get("SYS_L1", "170 GB CPU"),
    "blend_chunk": os.environ.get("SYS_BLEND_CHUNK", "256"),
}

app = FastAPI(title="Your GPU Is a Very Expensive Photocopier")

_sessions: dict[str, RagSession] = {}
_lock = threading.Lock()
# reuse is read from the fleet engine's external_kv counter (delta from a lazily-captured
# baseline) so ALL traffic counts (attendee chats + the load driver); TTFTs are recent samples.
_score = {"fleet_ext_base": None, "ttft": {"fleet": [], "vanilla": []},
          # engine-metric windowed-mean TTFT (reflects ALL load, incl. the in-cluster loadgen)
          "ttft_prev": {"fleet": None, "vanilla": None},
          "ttft_avg": {"fleet": None, "vanilla": None},
          # lazily-captured per-arm prompt-token-source baseline so the CAPACITY bars reflect
          # whatever drives the engines (the in-cluster loadgen), not just the webapp's own driver.
          "src_base": {}}
_ratelimit: dict[str, list] = {}

# --- load driver: presenter-controlled background pressure so the demo shows the LOAD regime.
# Fires APC-defeating shifted-chunk queries; under concurrency vanilla saturates (recompute)
# while the fleet holds (reuse) — the gap the dual-pane can't show on a single idle query.
_load = {"running": False, "warming": False, "concurrency": 0, "arms": [], "sent": 0, "errors": 0,
         "i": 0, "started_at": 0.0, "sent_by_arm": {"fleet": 0, "vanilla": 0},
         "src_base": {}}
_load_lock = threading.Lock()
_load_stop = threading.Event()
LOAD_K = int(os.environ.get("LOAD_K", "16"))  # chunks/query (~8k tok) — heavy enough to saturate vanilla


def _record_ttft(arm: str, ttft: float) -> None:
    with _lock:
        tt = _score["ttft"][arm]
        tt.append(round(ttft, 3))
        del tt[:-30]


def _session(sid: str) -> RagSession:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = RagSession(sid)
        return _sessions[sid]


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with _lock:
        q = [t for t in _ratelimit.get(ip, []) if now - t < 60]
        if len(q) >= MAX_RUNS_PER_MIN:
            _ratelimit[ip] = q; return False
        q.append(now); _ratelimit[ip] = q; return True


@app.on_event("startup")
def _load_shared_corpus():
    """Load + precompute the shared demo corpus so the headline A/B works immediately."""
    corpus_dir = os.path.join(HERE, "corpus")
    texts = []
    if os.path.isdir(corpus_dir):
        for fn in sorted(os.listdir(corpus_dir)):
            if fn.endswith((".txt", ".md")):
                texts.append(open(os.path.join(corpus_dir, fn), encoding="utf-8").read())
    if texts:
        # background=True: index synchronously (fast) but warm the KV tier in a thread so the
        # server serves immediately (a large corpus takes minutes to precompute). /gate stays
        # red until warming finishes, then goes green.
        info = _session("demo").ingest(texts, precompute=True, background=True)
        print(f"[startup] shared corpus 'demo' indexed (precompute in background): {info}")
        # pre-build the load-driver question bank from the corpus entities (nodes are set
        # synchronously by ingest; the bank is what the real-RAG load replay cycles through).
        threading.Thread(target=lambda: rag_app._question_bank(_session("demo")), daemon=True).start()

        def _prime():
            # wait for the chunk precompute, then set up the capacity/eviction regime (warm presets
            # into fleet L1 + fill vanilla's finite GPU cache) so the first tap already contrasts.
            demo = _session("demo")
            for _ in range(180):
                if demo.nodes and demo.warm >= len(demo.nodes):
                    break
                time.sleep(2)
            print(f"[startup] priming eviction regime: {rag_app.prime_regime(demo)}")
            time.sleep(8)
            print("[startup] starting gentle sim-user loop")
            _sim_loop()          # runs forever in this daemon thread
        threading.Thread(target=_prime, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.post("/ingest")
async def ingest(req: Request):
    if PUBLIC_READONLY:
        return JSONResponse({"error": "disabled on the public demo"}, status_code=403)
    body = await req.json()
    sid = body.get("session", "demo")
    texts = body.get("texts") or []
    if not texts:
        return JSONResponse({"error": "no texts"}, status_code=400)
    # personal uploads go to a private session; never touch 'demo'
    if sid == "demo":
        return JSONResponse({"error": "use a private session id for uploads"}, status_code=400)
    return JSONResponse(_session(sid).ingest(texts, precompute=True))


# Live attendee-tap counter: the background sim loop yields to it so a presenter's paced query
# never shares the engine with the loop (Llama CacheBlend degrades under concurrency).
_active = {"n": 0, "last": 0.0}   # n = live /chat streams in flight; last = when that last changed
_active_lock = threading.Lock()
def _bump_active(d: int):
    with _active_lock:
        _active["n"] = max(0, _active["n"] + d)
        _active["last"] = time.time()

SIM_COOLDOWN = 2.0    # seconds of quiet after a tap before the sim loop will fire
SIM_PERIOD = 30.0     # one round every ~30s — sequential (never piles up), gentle enough that a
                      # live tap almost never shares the engine, so the per-query win stays pristine


def _sim_idle() -> bool:
    return _active["n"] == 0 and (time.time() - _active["last"]) > SIM_COOLDOWN


def _sim_loop():
    """One gentle 'simulated user' that fires ONE reuse query every ~SIM_PERIOD seconds against the
    SECOND fleet engine (v3b / ARM_B2) — the Act-3 engine, which shares the blend cache but is NOT
    the engine live 'Ask both' taps use (v3 + std). So its reuse still climbs the shared-cache meter
    (blend_hit_tokens is global) while NEVER contending with the demo — taps stay pristine. Sequential
    (waits for each return, never piles up); also yields to any in-flight tap. Runs forever in the
    startup daemon thread. (No vanilla churn — that would hit the live vanilla engine; the startup
    prime already evicts the presets so first-taps win.)"""
    demo = _session("demo")
    while True:
        try:
            if not _sim_idle():
                time.sleep(0.5); continue
            rag_app.fire_load_rag(demo, "b2", _next_i())    # v3b reuse -> meter ticks, zero live contention
            end = time.time() + SIM_PERIOD                  # wait out the period (interruptibly)
            while time.time() < end:
                time.sleep(0.5)
        except Exception:
            time.sleep(2)


@app.get("/chat")
def chat(session: str = "demo", arm: str = "fleet", q: str = "", history: str = "[]"):
    ip = "x"
    if not q:
        return JSONResponse({"error": "empty q"}, status_code=400)
    try:
        hist = json.loads(history)
    except Exception:
        hist = []
    sess = _session(session)

    def gen():
        _bump_active(1)
        try:
            for kind, payload in sess.chat_stream(arm, hist, q):
                if kind == "token":
                    yield "data: " + json.dumps({"t": payload}) + "\n\n"
                else:
                    _record_ttft(arm, payload["ttft_s"])  # reuse tracked via the engine counter
                    yield "data: " + json.dumps({"done": payload}) + "\n\n"
        finally:
            _bump_active(-1)
    return StreamingResponse(gen(), media_type="text/event-stream")


def _pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p / 100 * (len(s) - 1)))], 2)


def _update_engine_ttft():
    """Windowed-mean TTFT per arm from the engine histogram (Δsum/Δcount between polls) — reflects
    ALL load (in-cluster loadgen + webapp), so the tiles show the real gap even when the driver
    runs in-cluster rather than through the webapp."""
    for arm in ("fleet", "vanilla"):
        url = rag_app.ARM_FLEET if arm == "fleet" else rag_app.ARM_VANILLA
        cur = rag_app.engine_ttft(url)
        with _lock:
            prev = _score["ttft_prev"][arm]
            if cur:
                if prev and cur[1] > prev[1]:
                    _score["ttft_avg"][arm] = (cur[0] - prev[0]) / (cur[1] - prev[1])
                _score["ttft_prev"][arm] = cur


def _arm_breakdown(arm: str):
    """Per-arm engineering metrics: throughput (rps, from the load driver), engine windowed-mean
    TTFT, and the recompute-vs-reuse token split since load started."""
    arm_url = rag_app.ARM_FLEET if arm == "fleet" else rag_app.ARM_VANILLA
    elapsed = max(1e-6, time.time() - _load["started_at"]) if _load["started_at"] else 0
    rps = round(_load["sent_by_arm"].get(arm, 0) / elapsed, 2) if elapsed else None
    ttft = _score["ttft_avg"][arm]
    reused = recomputed = None
    if arm == "fleet":
        # FLEET: use the blend server's OWN self-consistent counters (hit / requested). This is the
        # truthful reuse rate (~80%); the engine's prompt_tokens_by_source undercounts CacheBlend
        # (reused tokens show as local_compute), which made an engine-denominated ratio read falsely low.
        bl = rag_app.blend_lookup()
        if bl.get("requested"):
            bb = _score["src_base"].get("_blend")
            if bb is None:
                _score["src_base"]["_blend"] = bb = bl
            hit_d = max(0.0, bl.get("hit", 0) - bb.get("hit", 0))
            req_d = max(0.0, bl.get("requested", 0) - bb.get("requested", 0))
            reused = int(hit_d)
            recomputed = int(max(0.0, req_d - hit_d))
    else:
        cur = rag_app.token_sources(arm_url)
        if cur:
            # vanilla: engine counter is truthful (no LMCache). Lazy baseline so the split reflects
            # ALL traffic since page-load (the in-cluster loadgen drives it).
            base = _score["src_base"].get(arm)
            if base is None:
                _score["src_base"][arm] = base = cur
            recomputed = int(max(0, cur["local_compute"] - base["local_compute"]))
            reused = int(max(0, (cur["external_kv_transfer"] + cur["local_cache_hit"])
                             - (base["external_kv_transfer"] + base["local_cache_hit"])))
    total = (reused or 0) + (recomputed or 0)
    return {"rps": rps, "ttft": round(ttft, 2) if ttft else None,
            "reused_tokens": reused, "recomputed_tokens": recomputed,
            "reuse_pct": round(100 * reused / total, 1) if total else None}


@app.get("/scoreboard")
def scoreboard():
    _update_engine_ttft()  # refresh engine-metric TTFT (hits /metrics; outside the lock)
    # Reuse = tokens the BLEND SERVER served from cache (the truthful CacheBlend signal; the engine
    # counter undercounts it as local_compute), minus a lazily-captured baseline -> reflects ALL
    # traffic (attendee chats + in-cluster loadgen). Fall back to fleet external_kv if blend is unreachable.
    cur = rag_app.blend_hit_tokens()
    if cur is None:
        cur = rag_app._external_kv(rag_app.ARM_FLEET)
    with _lock:
        if _score["fleet_ext_base"] is None and cur is not None:
            _score["fleet_ext_base"] = cur
        base = _score["fleet_ext_base"] or 0
        reuse = int(max(0, (cur if cur is not None else base) - base))
        gpu_sec = reuse / PREFILL_TOK_S if PREFILL_TOK_S else 0.0
        ft, vt = _score["ttft_avg"]["fleet"], _score["ttft_avg"]["vanilla"]
        out = {"reuse_tokens": reuse, "gpu_seconds_saved": round(gpu_sec, 1),
               "dollars_saved": round(gpu_sec / 3600.0 * DOLLARS_PER_GPU_HR, 4),
               "ttft_fleet_s": round(ft, 2) if ft else None,
               "ttft_vanilla_s": round(vt, 2) if vt else None,
               "ttft_ratio": round(vt / ft, 1) if (ft and vt) else None,   # vanilla ÷ tensormesh
               "dollars_per_gpu_hr": DOLLARS_PER_GPU_HR,
               "load": {"running": _load["running"], "warming": _load["warming"],
                        "concurrency": _load["concurrency"], "arms": _load["arms"],
                        "sent": _load["sent"], "errors": _load["errors"]}}
    out["arms"] = {"fleet": _arm_breakdown("fleet"), "vanilla": _arm_breakdown("vanilla")}
    return out


# ---- load driver ---------------------------------------------------------------------------
def _next_i() -> int:
    with _load_lock:
        _load["i"] += 1
        return _load["i"]


def _load_worker(arm: str, session):
    # one arm per worker so each arm's throughput is INDEPENDENT (a faster arm completes more) —
    # firing both arms in one worker would bottleneck the fast arm on the slow one -> equal, bogus rps.
    while not _load_stop.is_set():
        i = _next_i()
        ttft = rag_app.fire_load_rag(session, arm, i)
        with _load_lock:
            if ttft is None:
                _load["errors"] += 1
            else:
                _load["sent"] += 1
                _load["sent_by_arm"][arm] = _load["sent_by_arm"].get(arm, 0) + 1
        if ttft is not None:
            _record_ttft(arm, ttft)


def _load_run(concurrency: int, armlist: list):
    demo = _session("demo")
    # 1. wait for the corpus precompute to finish so the warm+load don't compete on the fleet engine.
    for _ in range(120):
        if _load_stop.is_set():
            return
        if demo.nodes and demo.warm >= len(demo.nodes):
            break
        time.sleep(2)
    # 2. GENTLE sequential warm of the fleet tier over the DEMO corpus — a cold CacheBlend engine
    #    stalls under concurrent recompute, so it must be warm before heavy load.
    rag_app.load_warm(demo, stop=_load_stop)
    if _load_stop.is_set():
        return
    # 3. baseline + spawn the per-arm worker pools.
    with _load_lock:
        _load["warming"] = False
        _load["started_at"] = time.time()
        _load["src_base"] = {a: rag_app.token_sources(rag_app.ARM_FLEET if a == "fleet" else rag_app.ARM_VANILLA)
                             for a in armlist}
    with _lock:
        _score["ttft"] = {"fleet": [], "vanilla": []}
    for arm in armlist:
        for _ in range(concurrency):
            threading.Thread(target=_load_worker, args=(arm, demo), daemon=True).start()


@app.post("/load/start")
def load_start(concurrency: int = 3, arms: str = "both"):
    if PUBLIC_READONLY:
        return JSONResponse({"error": "disabled on the public demo"}, status_code=403)
    # default LOW (3): on one replica per arm, high concurrency saturates both and the CPU cache
    # tier goes throughput-neutral (perf-contract §C.3) -> no honest TTFT gap. Low load keeps
    # vanilla honestly recomputing (0% reuse) as live "room texture" without a bogus race.
    with _load_lock:
        if _load["running"]:
            return {"error": "already running", **{k: _load[k] for k in ("concurrency", "arms")}}
        concurrency = max(1, min(int(concurrency), 32))
        armlist = ["vanilla", "fleet"] if arms == "both" else [arms]
        _load.update(running=True, warming=True, concurrency=concurrency, arms=armlist,
                     sent=0, errors=0, sent_by_arm={"fleet": 0, "vanilla": 0}, started_at=0.0, src_base={})
        _load_stop.clear()
    threading.Thread(target=_load_run, args=(concurrency, armlist), daemon=True).start()
    return {"started": True, "warming": True, "concurrency": concurrency, "arms": armlist, "k_chunks": LOAD_K}


@app.post("/load/stop")
def load_stop():
    if PUBLIC_READONLY:
        return JSONResponse({"error": "disabled on the public demo"}, status_code=403)
    _load_stop.set()
    with _load_lock:
        _load["running"] = False
        _load["warming"] = False
    return {"stopped": True, "sent": _load["sent"], "errors": _load["errors"]}


@app.get("/load/status")
def load_status():
    return dict(_load)


@app.get("/presets")
def presets():
    """The one-tap question set (single source of truth; the frontend renders these as chips).
    Deliberately >28 diverse questions so the room's working set exceeds the GPU cache capacity."""
    return {"presets": rag_app.DEMO_PRESETS}


@app.get("/gate")
def gate(q: str = "Summarize the key points of the documents."):
    """Pre-demo gate: prove the stock-framework path actually lights up CacheBlend."""
    return JSONResponse(rag_app.assert_blend_fires(_session("demo"), q))


@app.get("/crossengine")
def crossengine(q: str = "what is the return window and restocking fee for opened items?"):
    """Act 3: a second fleet engine (B) reuses KV the fleet already warmed — cache on arrival."""
    return JSONResponse(rag_app.cross_engine_probe(_session("demo"), q))


@app.get("/sysinfo")
def sysinfo():
    """What you're looking at — hardware / model / stack / settings / workload. Deploy-time facts
    come from SYS (env-overridable); the runtime bits (served model, context size, corpus size,
    cross-engine wiring) are pulled live so the panel can't drift from the running config."""
    ctx_tok = rag_app.TOP_K * rag_app.CHUNK_TOK
    demo = _sessions.get("demo")
    chunks = len(demo.nodes) if demo else 0
    return {
        "hardware": SYS["hardware"],
        "model": rag_app.served_model(),
        "stack": SYS["stack"],
        "vanilla": "stock vLLM — recomputes the full context for every query (no KV reuse)",
        "tensormesh": (f"+ LMCache CacheBlend via MP connector → shared blend server "
                       f"(L1 = {SYS['l1']}) · recomp_ratio {SYS['recomp']} · chunk {SYS['blend_chunk']} tok"),
        "kv_cap": ("Tensormesh shares one 170 GB CPU-RAM cache tier across the fleet — it holds far more context "
                   "KV than a single engine keeps in GPU memory, so context is reused across queries, pods, and "
                   "engines (blend serves it back)"),
        "workload": (f"RAG over a ~{chunks}-chunk knowledge base · each query assembles ~{round(ctx_tok/1000)}k "
                     f"tokens (top-{rag_app.TOP_K} × {rag_app.CHUNK_TOK}-tok passages, BM25) · prefill-only (mt=1)"),
        "roomload": "paced — one query at a time, each measured cleanly",
        "cross_engine": bool(rag_app.ARM_B2),
    }


@app.get("/healthz")
def healthz():
    demo = _sessions.get("demo")
    return {"ok": True, "sessions": len(_sessions),
            "demo_chunks": len(demo.nodes) if demo else 0,
            "demo_warm": demo.warm if demo else 0}
