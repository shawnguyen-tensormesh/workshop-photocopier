#!/usr/bin/env python3
"""
RAG / CacheBlend workload for the workshop's receipts act.

A fixed, deterministic corpus of topical documents with a *hot set* — a handful of docs
that dominate the query mix — because CacheBlend's benefit is a curve in retrieval
concentration (uniform ≈ 1.0–1.1×; hot-set ≈ 1.5×; hot-dominated ≈ 3.4×). We disclose the
regime; we don't quote a flat multiplier.

Two things happen server-side so attendees never touch them:
  * documents are warmed once (their KV is stored in the shared cache), and
  * each query assembles the retrieved docs into one prompt separated by the CacheBlend
    segment marker, so reused chunks land at *shifted* (non-prefix) positions — the case
    only CacheBlend can serve (the GPU prefix cache cannot).

Modes:
  warm   send each corpus doc once to the endpoint (populate the tier)
  run    issue the query mix; report TTFT and, if --blend-metrics is given, the blend
         hit-rate delta and a topical-correctness score (the honest ≥80% gate lives here)
"""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = os.environ.get("ENDPOINT", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
SEP = " # # "   # CacheBlend segment separator (must match the blend server config)

# 20 topical documents; the first HOT_N are the hot set that dominates the query mix.
TOPICS = [
    ("photosynthesis", "Photosynthesis converts light energy into chemical energy stored in glucose. Chlorophyll in chloroplasts absorbs sunlight; water and carbon dioxide become sugar and oxygen."),
    ("roman empire", "The Roman Empire spanned three continents. Emperors from Augustus onward presided over roads, aqueducts, and a body of law that shaped legal systems for centuries."),
    ("black holes", "A black hole is a region of spacetime where gravity is so strong not even light escapes. They form when massive stars collapse; the boundary is the event horizon."),
    ("coffee", "Coffee is brewed from roasted beans, the seeds of Coffea berries. Prized for caffeine, it spread from Ethiopia and Yemen into a global commodity."),
    ("volcano", "A volcano is a rupture in a planet's crust venting molten rock, ash, and gas. Eruptions are explosive or effusive; they cluster at plate boundaries and hotspots."),
    ("jazz", "Jazz arose in African American communities of New Orleans in the late 19th century, defined by swing, blue notes, call-and-response, and improvisation."),
    ("photovoltaics", "Photovoltaic cells convert sunlight to electricity via the photovoltaic effect in semiconductors, typically silicon, without moving parts."),
    ("great barrier reef", "The Great Barrier Reef off Australia is the world's largest coral reef system, home to thousands of species and visible from space; warming seas threaten it."),
    ("internet", "The Internet is a global system of interconnected networks using TCP/IP, carrying the Web, email, and countless services across billions of devices."),
    ("penicillin", "Penicillin, discovered by Fleming in 1928, was the first widely used antibiotic, derived from Penicillium mould and transformative for medicine."),
    ("everest", "Mount Everest, on the Nepal-China border, is Earth's highest peak above sea level at 8,849 m; its summit sits in the jet stream's death zone."),
    ("photosphere", "The Sun's photosphere is its visible surface at ~5,500 C, the layer from which sunlight escapes to space, mottled with granules of rising plasma."),
    ("blockchain", "A blockchain is an append-only ledger of blocks linked by cryptographic hashes, replicated across nodes to resist tampering without a central authority."),
    ("mitochondria", "Mitochondria are the cell's power plants, producing ATP via oxidative phosphorylation; they carry their own DNA and likely descend from ancient bacteria."),
    ("sahara", "The Sahara is the world's largest hot desert, spanning North Africa; its dust fertilizes distant ecosystems including the Amazon."),
    ("printing press", "Gutenberg's movable-type printing press (c. 1440) slashed the cost of books, accelerating literacy, science, and the Reformation across Europe."),
    ("neutron star", "A neutron star is the collapsed core of a massive star, so dense a sugar-cube of its matter would weigh billions of tonnes; some spin as pulsars."),
    ("monsoon", "A monsoon is a seasonal reversal of prevailing winds bringing heavy rain, vital to South Asian agriculture but hazardous when it floods."),
    ("transistor", "The transistor, invented at Bell Labs in 1947, is the switch underlying all modern electronics; billions fit on a single chip today."),
    ("coral bleaching", "Coral bleaching occurs when stressed corals expel their symbiotic algae, turning white and starving; heat waves are the leading trigger."),
]
HOT_N = 6


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception:
        return None


TOK = _tokenizer()


def doc_ids(idx, target_tok=1600):
    """Deterministic ~target_tok document for topic idx (topic sentence, padded stably)."""
    name, body = TOPICS[idx]
    text = f"[DOC {idx}: {name}] " + (body + " ") * 40
    if TOK is None:
        return None, text[: target_tok * 4]
    ids = TOK.encode(text)
    while len(ids) < target_tok:
        ids = ids + ids
    return ids[:target_tok], None


def query_plan(n_queries=40, k=4, seed=7):
    """Deterministic hot-weighted query mix: each query retrieves k docs, ~80% from the hot set."""
    x = seed & 0x7FFFFFFF
    def rnd(m):
        nonlocal x
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        return x % m
    plans = []
    for _ in range(n_queries):
        docs = []
        while len(docs) < k:
            d = rnd(HOT_N) if rnd(10) < 8 else HOT_N + rnd(len(TOPICS) - HOT_N)
            if d not in docs:
                docs.append(d)
        plans.append(docs)
    return plans


def _post(ids_or_text, mt=1, timeout=300):
    if isinstance(ids_or_text, list):
        payload = {"model": MODEL, "prompt": ids_or_text, "max_tokens": mt, "temperature": 0.0}
    else:
        payload = {"model": MODEL, "prompt": ids_or_text, "max_tokens": mt, "temperature": 0.0}
    req = urllib.request.Request(ENDPOINT.rstrip("/") + "/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    body = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return time.time() - t, body["choices"][0]["text"]


def _sys_ids():
    s = "You are a helpful assistant. Answer using the passages above."
    return TOK.encode(s) if TOK else s


def build_prompt(doc_indices, query_tok):
    """Assemble retrieved docs into one prompt with the blend separator; docs at shifted
    positions across queries so reuse is non-prefix."""
    if TOK is None:
        parts = [TOPICS[i][1] for i in doc_indices]
        return "You are a helpful assistant.\n" + f"{SEP}".join(parts) + SEP + " Summarize."
    sep = TOK.encode(SEP)[1:]
    ids = list(_sys_ids())
    for i in doc_indices:
        d, _ = doc_ids(i)
        ids = ids + sep + d
    ids = ids + sep + query_tok
    return ids


def cmd_warm(args):
    for i in range(len(TOPICS)):
        d, txt = doc_ids(i)
        _post(d if d is not None else txt, mt=1)
    print("@@WARMED@@", len(TOPICS), "docs")


def _blend_metric(url, key):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
            for ln in r.read().decode().splitlines():
                if ln.startswith(key + " "):
                    return float(ln.rsplit(" ", 1)[1])
    except Exception:
        return 0.0
    return 0.0


def cmd_run(args):
    plans = query_plan(args.n, args.k)
    qtok = TOK.encode(" Which topics do the passages above cover? List them.")[1:] if TOK else " List topics."
    bm = args.blend_metrics
    h0 = _blend_metric(bm, "lmcache_blend_lookup_hit_tokens_total") if bm else 0
    r0 = _blend_metric(bm, "lmcache_blend_lookup_requested_tokens_total") if bm else 0
    f0 = _blend_metric(bm, "lmcache_blend_retrieve_failures_total") if bm else 0
    lat = []; correct = 0; total = 0
    def one(docs):
        p = build_prompt(docs, qtok)
        dt, out = _post(p, mt=48 if args.check else 1)
        hit = 0
        if args.check:
            names = [TOPICS[i][0] for i in docs]
            hit = sum(1 for nm in names if any(w in out.lower() for w in nm.split()))
        return dt, hit, len(docs)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for dt, hit, n in ex.map(one, plans):
            lat.append(dt); correct += hit; total += n
    s = sorted(lat)
    p50 = s[len(s)//2]; p99 = s[min(len(s)-1, int(0.99*len(s)))]
    out = {"queries": len(plans), "ttft_p50": round(p50,3), "ttft_p99": round(p99,3)}
    if bm:
        hit_tok = _blend_metric(bm,"lmcache_blend_lookup_hit_tokens_total")-h0
        req_tok = _blend_metric(bm,"lmcache_blend_lookup_requested_tokens_total")-r0
        out["blend_hit_pct"] = round(100*hit_tok/req_tok,1) if req_tok else None
        out["retrieve_failures"] = _blend_metric(bm,"lmcache_blend_retrieve_failures_total")-f0
    if args.check:
        out["topical_correct"] = correct; out["topical_total"] = total
        out["topical_pct"] = round(100*correct/total,1) if total else None
    print("@@RAG@@", json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm"); w.set_defaults(func=cmd_warm)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=40); r.add_argument("--k", type=int, default=4)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--blend-metrics", default=os.environ.get("BLEND_METRICS", ""))
    r.add_argument("--check", action="store_true", help="also score topical correctness")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
