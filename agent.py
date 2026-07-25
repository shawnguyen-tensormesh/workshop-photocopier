#!/usr/bin/env python3
"""
Your GPU Is a Very Expensive Photocopier — workshop agent harness.

One file, one dependency-light client. It replays a *frozen* coding-agent trajectory
against an OpenAI-compatible endpoint and prints a per-turn readout of where the time
goes. The only thing that changes between a "vanilla" run and a "Tensormesh" run is the
ENDPOINT environment variable — same model, same tokens, same trajectory.

    export ENDPOINT=http://vanilla.example/v1     # or the Tensormesh-backed endpoint
    python agent.py replay --salt seat-42

Modes
  replay  (default)  Replay the canonical trajectory turn-by-turn. Each turn sends the
                     accumulated context and times the prefill (TTFT). Because the tokens
                     are identical every run, any TTFT difference is caching, nothing else.
  live               Free-form temp=0 chat against the endpoint. Clearly labelled
                     UN-MEASURED — for playing around, not for the numbers.

The HUD per turn:
    turn 4 · ctx 23_512 tok · TTFT 4.83s · decode 0.21s · [cache: 61% prompt reused]

Why prefill, not generation: an agent resends its whole history every turn, so the
expensive part is re-reading (prefilling) the context. That is exactly what a KV cache
lets you skip — and what this readout makes visible.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

DEFAULT_ENDPOINT = os.environ.get("ENDPOINT", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")


# ---------- token counting (best-effort; exact count comes from the server usage) ----------
def _load_tokenizer(model):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception:
        return None


class Counter:
    def __init__(self, model):
        self.tok = _load_tokenizer(model)

    def count(self, text):
        if self.tok is not None:
            try:
                return len(self.tok.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)  # ~4 chars/token fallback


# ---------- HTTP ----------
def _post_stream(endpoint, payload, timeout):
    """POST /chat/completions with stream=True. Returns (ttft_s, decode_s, text, usage)."""
    url = endpoint.rstrip("/") + "/chat/completions"
    data = json.dumps({**payload, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks.append(piece)
    total = time.time() - t0
    if ttft is None:                       # no content streamed (e.g. max_tokens too small)
        ttft = total
    decode = max(0.0, total - ttft)
    return ttft, decode, "".join(chunks), usage


# ---------- context assembly ----------
def standing_prefix(salt):
    """The repository context an agent carries every turn: AGENTS.md + architecture +
    the source modules. Salting injects a unique header so each attendee is a distinct
    context (no accidental prefix sharing across seats)."""
    from trajectory import build_standing_prefix
    return build_standing_prefix(salt)


def canonical_turns():
    from trajectory import TURNS
    return TURNS


# ---------- modes ----------
def run_replay(endpoint, model, salt, max_tokens, timeout, json_out):
    counter = Counter(model)
    system = standing_prefix(salt)
    turns = canonical_turns()
    history = [{"role": "system", "content": system}]
    print(f"# endpoint = {endpoint}")
    print(f"# model    = {model}")
    print(f"# salt     = {salt!r}  (standing prefix {counter.count(system):,} tok)")
    print(f"# replaying {len(turns)} frozen turns — timing prefill (TTFT) per turn\n")
    records = []
    for i, turn in enumerate(turns, 1):
        history.append({"role": "user", "content": turn["user"]})
        ctx_tok = sum(counter.count(m["content"]) for m in history)
        payload = {"model": model, "messages": history,
                   "max_tokens": max_tokens, "temperature": 0.0}
        try:
            ttft, decode, text, usage = _post_stream(endpoint, payload, timeout)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"turn {i}: ERROR {e}")
            break
        prompt_tok = (usage or {}).get("prompt_tokens", ctx_tok)
        cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
        cache_str = ""
        if cached is not None and prompt_tok:
            cache_str = f" · [cache: {100*cached//prompt_tok}% prompt reused]"
        print(f"turn {i:>2} · ctx {prompt_tok:>7,} tok · TTFT {ttft:5.2f}s · decode {decode:4.2f}s{cache_str}")
        records.append({"turn": i, "ctx_tok": prompt_tok, "ttft_s": round(ttft, 3),
                        "decode_s": round(decode, 3), "cached_tokens": cached})
        # freeze the assistant side so the next turn's tokens are identical every run
        history.append({"role": "assistant", "content": turn["assistant"]})
    if records:
        ttfts = [r["ttft_s"] for r in records]
        print(f"\n# total prefill time: {sum(ttfts):.2f}s over {len(records)} turns "
              f"(mean TTFT {sum(ttfts)/len(ttfts):.2f}s)")
    if json_out:
        print("@@JSON@@", json.dumps({"endpoint": endpoint, "salt": salt, "records": records}))
    return records


def run_live(endpoint, model, salt, max_tokens, timeout):
    system = standing_prefix(salt)
    history = [{"role": "system", "content": system}]
    print("### LIVE MODE — UN-MEASURED (temp=0 free chat). Numbers here are NOT the demo numbers.")
    print("### Type a message; Ctrl-D to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        payload = {"model": model, "messages": history, "max_tokens": max_tokens, "temperature": 0.0}
        ttft, decode, text, usage = _post_stream(endpoint, payload, timeout)
        print(f"agent> {text}\n  ({ttft:.2f}s to first token, {decode:.2f}s decode — un-measured)\n")
        history.append({"role": "assistant", "content": text})


def main():
    ap = argparse.ArgumentParser(description="Workshop agent harness (replay/live).")
    ap.add_argument("mode", nargs="?", default="replay", choices=["replay", "live"])
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible base URL (or $ENDPOINT)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="served model name (or $MODEL)")
    ap.add_argument("--salt", default=os.environ.get("SALT", "seat-local"),
                    help="unique per-attendee salt injected into the standing prefix")
    ap.add_argument("--max-tokens", type=int, default=8, help="replay times prefill; a few tokens suffice")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", action="store_true", help="also emit a machine-readable @@JSON@@ line")
    args = ap.parse_args()
    if args.mode == "live":
        run_live(args.endpoint, args.model, args.salt, max(64, args.max_tokens), args.timeout)
    else:
        run_replay(args.endpoint, args.model, args.salt, args.max_tokens, args.timeout, args.json)


if __name__ == "__main__":
    main()
