#!/usr/bin/env python3
"""Single-stream bench for the Flash-Next vLLM lane.

Mirrors the bluespace3 bench protocol: prompt ladder 80K..260K step 30K
(+8K), output cap 1280, single stream, cache-defeating unique prompts.
Metrics: TTFT (s), prefill tok/s (prompt_tokens/ttft), decode tok/s
(completion_tokens-1 / time-after-first-token), ITL avg (ms).
Token counts come from usage (NOT SSE event counts — MTP batches several
tokens per event).

Usage:
  bench-flashnext.py [--url http://127.0.0.1:8090/v1/chat/completions]
                     [--model qwen3.8-flash-next]
                     [--lengths 8000,80000,...,260000] [--out-cap 1280]
                     [--tokenizer /data/models/Qwen3.8-Flash-Next-W4A16-FP8PLE]
Results: ~/logs/flashnext-bench-<ts>.jsonl (one line per length).
"""
import argparse, json, random, sys, time, uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer

PARAS = [
    "The history of computing is a story of abstraction layers, each one hiding the complexity of the layer beneath it while enabling entirely new kinds of work above.",
    "Rivers carve canyons slowly, grain by grain, and the patient observer can measure in a single afternoon the shape of forces that took a million years to express themselves.",
    "A well designed service separates its control plane from its data plane, so that heavy traffic never interferes with the decisions that route it.",
    "Baking bread is chemistry with lunch at the end; the yeast does not know it is being cultivated, and the baker does not fully control the crumb.",
    "The telescope taught us that the universe is larger than our senses suggest, and the microscope taught us the same lesson pointing downward.",
    "In distributed systems, the network is not a function call; it is a rumor that arrives late, out of order, and occasionally not at all.",
    "Chess engines do not dream of victory; they enumerate consequences with a patience that humans find both admirable and slightly insulting.",
    "The best documentation is a conversation with a future reader who has forgotten everything you currently take for granted.",
    "Mountains are made of compressed seashells and ancient sunlight, which is another way of saying that geology is memory at a very slow frame rate.",
    "An optimizer walks a landscape it can only feel through gradients, stepping downhill in the dark with all the confidence of a hiker who has never seen the map.",
    "Coffee houses were the original social networks: a protocol for exchanging packets of gossip, philosophy, and slightly exaggerated shipping news.",
    "The compiler does not care about your intentions, only about the set of transformations that preserve the observable behavior of your program.",
    "Wind turbines and apple trees both harvest a gradient: one of pressure, one of sugar, both the residue of sunlight arriving unevenly.",
    "A queue is a promise that order will be preserved, and every busy system eventually learns which of its promises it can afford to keep.",
    "Maps simplify; that is their job and their danger, and every navigator must decide which details were dropped by someone who never imagined their journey.",
    "The spice trade moved empires because flavor, unlike calories, is a luxury that creates its own demand once tasted.",
    "Debugging is the art of forming a hypothesis about a system you did not build, then arranging for it to betray the truth.",
    "Fungi are the quiet diplomats of the forest, negotiating trade agreements between roots on a timescale no committee could tolerate.",
    "Every startup pitch is a compressed file that decompresses differently in the mind of each investor who opens it.",
    "The music of the spheres turned out to be gravitational waves, a bass line so deep it took a century of instrumentation to hear.",
    "Caching is a bet that the future resembles the past, and the entire economics of computing rests on that bet paying off most of the time.",
    "A lighthouse is a publisher with exactly one message and an audience that reads it only in emergencies.",
    "Tea, taxes, and telegrams each redrew borders in their time; the sequence suggests that logistics wins wars more often than cavalry.",
    "Mathematics is the discipline of writing things so precisely that other people can be convinced against their will.",
]


def build_prompt(tok, target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    paras = PARAS[:]
    rng.shuffle(paras)
    text = "\n\n".join(paras)
    base = tok.encode(text)
    need = target_tokens
    chunks = []
    acc = 0
    while acc < need:
        rng.shuffle(paras)
        p = "\n\n".join(paras)
        ids = tok.encode(p)
        chunks.append(ids)
        acc += len(ids)
    ids = [i for c in chunks for i in c][:need]
    nonce = f"Bench session {uuid.uuid4().hex[:12]}.\n\n"
    body = tok.decode(ids, skip_special_tokens=True)
    return nonce + body


def run_one(client: httpx.Client, url: str, model: str, prompt: str, out_cap: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": out_cap,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    ttft = None
    usage = None
    with client.stream("POST", url, json=payload) as r:
        if r.status_code != 200:
            body = r.read().decode("utf-8", "ignore")[:400]
            raise RuntimeError(f"HTTP {r.status_code}: {body}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            d = ch.get("delta") or {}
            if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                ttft = time.monotonic() - t0
    tend = time.monotonic()
    if usage is None:
        raise RuntimeError("no usage in stream")
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    ttft = ttft or (tend - t0)
    res = {
        "ttft_s": round(ttft, 3),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "prefill_tok_s": round(pt / ttft, 1) if ttft > 0 else None,
        "decode_tok_s": round((ct - 1) / (tend - t0 - ttft), 1) if ct > 1 and (tend - t0) > ttft else None,
        "itl_ms": round((tend - t0 - ttft) * 1000.0 / (ct - 1), 2) if ct > 1 else None,
        "e2e_s": round(tend - t0, 2),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--lengths", default="8000,80000,110000,140000,170000,200000,230000,260000")
    ap.add_argument("--out-cap", type=int, default=1280)
    ap.add_argument("--tokenizer", default="/data/models/Qwen3.8-Flash-Next-W4A16-FP8PLE")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out_path = Path.home() / "logs" / f"flashnext-bench-{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"bench: url={args.url} model={args.model} lengths={lengths} out_cap={args.out_cap}")
    print(f"bench: results -> {out_path}")

    limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
    with httpx.Client(timeout=httpx.Timeout(connect=30, read=args.timeout, write=900, pool=30), limits=limits) as client:
        # warmup (short, cheap)
        try:
            w = run_one(client, args.url, args.model, "warmup ping: answer with one word.", 8)
            print(f"warmup ok: {w['ttft_s']}s")
        except Exception as e:
            print(f"warmup failed (continuing): {e}")
        for L in lengths:
            prompt = build_prompt(tok, L, seed=L)
            ok = False
            for attempt in (1, 2):
                try:
                    t0 = time.time()
                    res = run_one(client, args.url, args.model, prompt, args.out_cap)
                    res["len"] = L
                    res["attempt"] = attempt
                    res["wall"] = time.strftime("%H:%M:%S")
                    with open(out_path, "a") as f:
                        f.write(json.dumps(res) + "\n")
                    print(
                        f"len={L:>7} pt={res['prompt_tokens']:>7} ct={res['completion_tokens']:>5} "
                        f"ttft={res['ttft_s']:>8.2f}s prefill={res['prefill_tok_s']:>8.1f} t/s "
                        f"decode={res['decode_tok_s']:>7.1f} t/s itl={res['itl_ms']}ms e2e={res['e2e_s']}s",
                        flush=True,
                    )
                    ok = True
                    break
                except Exception as e:
                    print(f"len={L} attempt {attempt} FAILED: {e}", flush=True)
                    time.sleep(5)
            if not ok:
                print(f"len={L} GIVING UP", flush=True)

    print("bench done.")


if __name__ == "__main__":
    main()
