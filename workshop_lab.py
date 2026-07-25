#!/usr/bin/env python3
"""Workshop lab — measure KV-cache reuse yourself.

"Your GPU Is a Very Expensive Photocopier" — KCD x OpenInfra Days Vietnam 2026.

Asks the SAME question on two engines behind the live demo and prints time-to-first-token
+ how much of the prompt was served from cache:
  - vanilla : plain vLLM, recomputes the whole prompt every time
  - fleet   : vLLM + Tensormesh (LMCache CacheBlend), reuses the KV

No install needed — Python 3 standard library only.
Usage:  python3 workshop_lab.py ["your question"]
"""
import sys, ssl, json, time, urllib.parse, urllib.request

BASE = "https://tm-photocopier-rag.fly.dev"
Q = sys.argv[1] if len(sys.argv) > 1 else "What does the product warranty cover, and what voids it?"

# Verify TLS normally; if the local Python has no CA bundle (common on the macOS python.org
# build), fall back to an unverified context so the workshop lab still runs against the demo.
_CTX = ssl.create_default_context()
try:
    urllib.request.urlopen(BASE + "/healthz", context=_CTX, timeout=10).read()
except Exception as e:
    if "CERTIFICATE_VERIFY_FAILED" in str(e):
        print("  (note: local Python has no CA bundle — skipping TLS verify for this demo lab)")
        _CTX = ssl._create_unverified_context()
    # any other error: keep the verified context; the real request will surface it clearly


def measure(arm, q):
    """Open the SSE stream for one arm; return (ttft_s, reuse_pct) from the final 'done' event."""
    url = f"{BASE}/chat?arm={arm}&q={urllib.parse.quote(q)}"
    t0 = time.time()
    with urllib.request.urlopen(url, context=_CTX, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            evt = json.loads(line[6:])
            if "done" in evt:
                d = evt["done"]
                return d["ttft_s"], d["reuse_pct"]
    return time.time() - t0, 0.0


def main():
    print(f'\nQ: "{Q}"\n')
    print(f'{"engine":24s}{"first token":>14s}{"from cache":>14s}')
    print("-" * 52)
    results = {}
    for arm, label in (("vanilla", "vanilla (plain vLLM)"), ("fleet", "fleet (Tensormesh)")):
        ttft, reuse = measure(arm, Q)
        results[arm] = ttft
        print(f"{label:24s}{ttft:11.2f} s{int(reuse * 100):12d} %")
    if results.get("vanilla") and results.get("fleet"):
        speedup = results["vanilla"] / results["fleet"]
        print("-" * 52)
        print(f"\n  -> Tensormesh reached the first token {speedup:.1f}x sooner.\n")


if __name__ == "__main__":
    main()
