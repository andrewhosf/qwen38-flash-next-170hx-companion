# Benchmarks — Qwen3.8-Flash-Next W4A16+FP8PLE on 2× CMP 170HX (TP=2, vLLM 0.29.0)

Box: 2× CMP 170HX (64 GB each, Gen2 x16), 235 GB RAM, vLLM 0.29.0 + corrected PLE-mmap
patch set, TP=2, 262,144-token native context, BF16 KV, lazy PLE table from page cache.
Client: `scripts/bench-flashnext.py` — single stream, unique shuffled prompts per depth,
token counts from API `usage`. Measured 2026-09-16.

## A. TP=2, no speculative decoding (out-cap 1280)

| prompt | TTFT | prefill t/s | decode t/s | ITL ms |
|---|---|---|---|---|
| 8K | 4.7 s | 1,725 | 34.1 | 29.4 |
| 80K | 47.7 s | 1,678 | 37.4 | 26.8 |
| 110K | 66.3 s | 1,657 | 37.1 | 27.0 |
| 140K | 85.6 s | 1,635 | 37.0 | 27.0 |
| 170K | 104.9 s | 1,620 | 36.9 | 27.1 |
| 200K | 125.1 s | 1,598 | 37.0 | 27.0 |
| 230K | 146.4 s | 1,569 | 37.0 | 27.0 |
| 260K | 165.5 s | 1,570 | 36.9 | 27.1 |

Prefill holds within **−9 %** from 8K to 260K; decode is step-bound and flat at ~37 t/s.

## B. TP=2 + MTP spec=4 (draft: `runtime/mtp-int4-g32`)

| prompt | TTFT | prefill t/s | decode t/s | note |
|---|---|---|---|---|
| 2K | 3.4 s | 607 | 37.7 | out-cap 256 |
| 8K | 4.9 s | 1,637 | 36.5–37.7 | ≈ no-spec |
| 80K run 1 | 50.1 s | 1,595 | **64.2** | +73 % vs no-spec |
| 80K run 2 | 49.4 s | 1,619 | 34.2 | −8 % (least friendly content) |
| 200K | 128.7 s | 1,553 | **48.9** | +32 % vs no-spec |

Acceptance (cumulative over the sample runs): drafts 1,612; draft tokens 6,448 (=4/turn);
accepted 3,017 → **~47 %**, per-position 71/58/46/38 %.
*Decode speed = step rate × mean acceptance length — it is a content property, not just
hardware. This compact INT4 draft accepts less than the reference BF16/optimized heads
(86–90 % on the 2×3090 stack it was assembled for).*

## KV cache

| config | pool | concurrency @262K |
|---|---|---|
| no spec | 1,489,925 tokens | 5.68× |
| spec=4 | 1,175,464 tokens | 4.48× |

(PLE table: 47.68 GiB on disk, 100 % prewarmed into page cache at boot.)

## Caveats — compare like-for-like

- **SUPERSEDED (2026-09-17):** the "likely headroom from expert-parallel" note below is
  now measured — `EP=1 + --moe-backend humming` gives decode 70.5/94.5/94.3 t/s
  (@8K/22K/100K) and prefill ~3,630 t/s on this same box. See
  [`ep-humming-campaign.md`](ep-humming-campaign.md); that config is the recommended one.

- **Parallel mode matters.** The original repo's headline (prefill 5,500–6,100 t/s;
  decode 60–112 t/s) is **PP=2 + spec=4**, requiring three unpublished PP patches (see
  `docs/drift-log.md` § D6). This run is its `tp` fallback mode: prefill ≈ 1.6–1.7K t/s.
  A TP2+EP comparison of the same model class on the same card model measured
  ~2,300–2,545 t/s prefill (Level1Techs thread) — so there is likely headroom from
  expert-parallel and/or larger scheduler chunks on this box; unverified here.
- Spec-decode numbers are sample-grade (single run per point); treat ±10 % as noise and
  remember section B's spread is real content variance.
- Deeper validation (spec-depth sweeps, EP, `--max-num-batched-tokens` 4096+) is future work.

## Environment fingerprint

- vLLM 0.29.0 (pip), torch 2.13.0 (cu13 wheels), driver 610.x, Ubuntu 24.04, Python 3.12.
- `VLLM_PLE_MMAP=1 WORKERS=32 CHUNK=2048 PREWARM=1 READAHEAD=256`,
  `-cc.cudagraph_mode=PIECEWISE`, `--no-enable-flashinfer-autotune`,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, TP=2, `--max-model-len 262144`, `--max-num-seqs 8`.
- Boot: config→API-ready ≈ 6 min cold-ish (page cache warm from earlier runs; first-ever
  cold boot reads 116 GiB + prewarms 47.7 GiB).
