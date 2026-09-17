# Benchmarks — Qwen3.8-Flash-Next W4A16+FP8PLE on 2× CMP 170HX (TP=2, vLLM 0.29.0)

Box: 2× CMP 170HX (64 GB each, Gen2 x16), 235 GB RAM, vLLM 0.29.0 + corrected PLE-mmap
patch set, TP=2, 262,144-token native context, BF16 KV, **no speculative decoding**
(`tp` profile, first light — see `docs/deployment.md`). Client: `scripts/bench-flashnext.py`
(single stream, unique prompts per depth, output cap 1280 for the ladder / 256 for the
sample; token counts from API `usage`).

## Results

### Sample (out-cap 256)

| prompt | prefill t/s | decode t/s | ITL ms |
|---|---|---|---|
| 2.0K | 555 | 37.2 | 26.9 |
| 8.0K | 1,468 | 37.3 | 26.8 |

### Depth ladder (out-cap 1280)

| prompt | TTFT s | prefill t/s | decode t/s |
|---|---|---|---|
| 8K |  |  |  |
| 80K |  |  |  |
| 110K |  |  |  |
| 140K |  |  |  |
| 170K |  |  |  |
| 200K |  |  |  |
| 230K |  |  |  |
| 260K |  |  |  |

(populated after the run; raw JSONL preserved alongside this file as
`flashnext-bench-<ts>.jsonl`.)

## KV cache

- `GPU KV cache size: 1,489,925 tokens` (19.19 GiB KV budget per GPU at
  `--gpu-memory-utilization 0.90`), i.e. **5.68×** at full 262,144-token requests.
- PLE table: 47.68 GiB on disk, 100 % prewarmed into page cache (read once at boot).

## Notes & caveats

- These are **TP=2, no-spec** numbers. The original repo's headline (60–112 tok/s
  decode; 5,500–6,100 t/s prefill) is **PP=2 + MTP spec=4**, a config that requires
  its three unpublished PP patches (see `docs/drift-log.md` § D6). Do not compare
  across parallel modes without saying so.
- Decode here is step-bound: ITL ≈ 26.9 ms → ~37 tok/s ceiling without speculation,
  independent of content. Adding MTP (spec ≤ 4) is the known multiplier
  (their rig: ×1.8–3.4 depending on content predictability).
- Temperatures under load: CMPs at 150 W cap — see per-run notes below.

## Environment fingerprint

- vLLM 0.29.0 (pip), torch 2.13.0, CUDA runtime from wheel (cu13), driver 610.x.
- `VLLM_PLE_MMAP=1 WORKERS=32 CHUNK=2048 PREWARM=1 READAHEAD=256`,
  `-cc.cudagraph_mode=PIECEWISE`, flashinfer sampler disabled, autotune off.
