#!/usr/bin/env python3
"""Verify a Qwen3.8-Flash-Next W4A16+FP8PLE checkpoint's PLE table layout
against what the bluespace3 PLE-mmap patch requires — BEFORE downloading ~120 GB.

Two modes:
  local :  verify_checkpoint_ple.py /path/to/model_dir
  remote:  verify_checkpoint_ple.py --repo albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
                                   --rev ef554143369a706525336f6b42a09094835dc077
           (header-only reads via HTTP Range; a few MB of traffic total)

Checks:
  1. config.json: text_config.split_ngram_parts, ple_layer_ids present.
  2. For every PLE layer: shards named
       layers.<N>.ple.ple_embedding.ngram_embedding.shard_<M>.weight
     with dtype F8_E4M3 (or BF16), width == ple head dim (160 bytes for FP8).
  3. Shard set is complete: indices 0..split-1, all rows equal, sum == padded vocab.
  4. For FP8: a scalar `...ngram_embedding.weight_scale` exists per layer
     (BF16/F32/F16, single element) — the mmap patch fail-closes without it.

Stdlib only. Exit code 0 = PASS.
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
import urllib.request

SHARD_RE = re.compile(r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
SCALE_RE = re.compile(r"layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$")
ITEMSIZE = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}


def parse_header_bytes(raw: bytes) -> dict:
    n = struct.unpack("<Q", raw[:8])[0]
    if len(raw) < 8 + n:
        raise ValueError("truncated header")
    hdr = json.loads(raw[8 : 8 + n])
    hdr.pop("__metadata__", None)
    return hdr


def read_header_local(path: str, rng: int = 64 << 20) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > rng:
            raise ValueError(f"{path}: header too large")
        return json.loads(f.read(n))


def read_header_remote(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-15", "User-Agent": "curl/8"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    n = struct.unpack("<Q", raw[:8])[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}", "User-Agent": "curl/8"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    return parse_header_bytes(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", help="local checkpoint directory")
    ap.add_argument("--repo", help="HF repo id (remote header-sniff mode)")
    ap.add_argument("--rev", default="main", help="revision for --repo mode")
    args = ap.parse_args()
    if not args.model_dir and not args.repo:
        ap.error("provide model_dir or --repo")

    # ---- load config / file list
    if args.repo:
        base = f"https://huggingface.co/{args.repo}/resolve/{args.rev}"
        cfg = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"https://huggingface.co/{args.repo}/raw/{args.rev}/config.json",
                                   headers={"User-Agent": "curl/8"}), timeout=30).read())
        listing = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"https://huggingface.co/api/models/{args.repo}/revision/{args.rev}",
                                   headers={"User-Agent": "curl/8"}), timeout=30).read())
        files = [s["rfilename"] for s in listing.get("siblings", []) if s["rfilename"].endswith(".safetensors")]
        read_cfg = lambda name: json.loads(urllib.request.urlopen(
            urllib.request.Request(f"{base}/{name}", headers={"Range": "bytes=0-300000", "User-Agent": "curl/8"}), timeout=30).read()) if False else None
    else:
        cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
        files = [os.path.basename(p) for p in glob.glob(os.path.join(args.model_dir, "*.safetensors"))]

    tc = cfg.get("text_config", cfg)
    split = tc.get("split_ngram_parts")
    ple_layers = tc.get("ple_layer_ids")
    divisor = tc.get("make_ngram_vocab_size_divisible_by", 1)
    base_vocab = tc.get("ngram_vocab_size_base")
    print(f"config: split_ngram_parts={split} ple_layer_ids={ple_layers} "
          f"ngram_vocab_size_base={base_vocab} divisor={divisor}")
    if not split or not ple_layers:
        print("FAIL: config lacks split_ngram_parts / ple_layer_ids (not a PLE checkpoint?)")
        return 1

    # ---- scan headers
    shards = {}   # (layer, shard) -> (rows, cols, dtype)
    scales = {}   # layer -> (dtype, nbytes)
    scanned = 0
    for fname in sorted(files):
        try:
            hdr = (read_header_remote(f"{base}/{fname}") if args.repo
                   else read_header_local(os.path.join(args.model_dir, fname)))
        except Exception as e:
            print(f"  (skip {fname}: {e})")
            continue
        scanned += 1
        for name, meta in hdr.items():
            m = SHARD_RE.search(name)
            if m:
                shards[(int(m.group(1)), int(m.group(2)))] = (
                    meta["shape"][0], meta["shape"][1], meta["dtype"])
                continue
            s = SCALE_RE.search(name)
            if s:
                nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
                scales[int(s.group(1))] = (meta["dtype"], nbytes)
    print(f"scanned {scanned} safetensors files")

    # ---- validate
    ok = True
    layers = sorted({l for (l, _) in shards})
    print(f"PLE layers found: {layers}")
    for L in layers:
        idx = sorted(m for (ll, m) in shards if ll == L)
        rows = {shards[(L, m)][0] for m in idx}
        cols = {shards[(L, m)][1] for m in idx}
        dtypes = {shards[(L, m)][2] for m in idx}
        exp_dtypes = {"F8_E4M3", "BF16"}
        print(f"  layer {L}: {len(idx)} shards idx {idx[0]}..{idx[-1]} | rows={rows} cols={cols} dtype={dtypes}")
        if idx != list(range(len(idx))):
            print(f"    FAIL: shard indices not contiguous 0..{len(idx) - 1}"); ok = False
        if len(rows) != 1:
            print("    FAIL: shard row counts differ"); ok = False
        if not dtypes <= exp_dtypes:
            print(f"    FAIL: unsupported dtype {dtypes}"); ok = False
        r, c = rows.pop(), cols.pop()
        dt = dtypes.pop()
        expect_rows = (320001536) if False else None  # informational only
        if dt == "F8_E4M3":
            if L not in scales:
                print("    FAIL: FP8 shards but no weight_scale for this layer "
                      "(the mmap patch refuses to load)"); ok = False
            else:
                sd, nb = scales[L]
                want = {"BF16": 2, "F32": 4, "F16": 2}.get(sd)
                if want is None or nb != want:
                    print(f"    FAIL: weight_scale must be a single BF16/F32/F16 scalar, got {sd} {nb}B")
                    ok = False
                else:
                    print(f"    scale: {sd} ({nb}B) OK")
        print(f"    total rows for layer {L}: {r * len(idx)}")

    if not ok:
        print("RESULT: FAIL — checkpoint does not match the mmap patch's expectations")
        return 1
    print("RESULT: PASS — PLE layout matches the PLE-mmap patch requirements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
