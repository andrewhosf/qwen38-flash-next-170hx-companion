# EP + humming campaign — the 2.2× decode result (2026-09-17)

Follow-up to `benchmarks.md`. Section A's caveat ("likely headroom from expert-parallel")
is now **measured and closed**: `--enable-expert-parallel --moe-backend humming` is a
2.1–2.5× win on this box and is the config we run in production.

## Method

One-variable-per-bounce A/B against a re-measured baseline, all arms on the same
TP=2 pair (CMP 170HX bus 61/62), same checkpoint (`albucino W4A16-FP8PLE`), same
spec MTP depth-4 unless stated, single-stream ladder 8K/22K/100K (out-cap 1280) via
`scripts/bench-flashnext.py`. Each arm bounced the lane clean (stop → verify VRAM
released → start) and passed health+smoke before benching.

## Results (decode t/s, single stream)

| Arm | Change | 8K | 22K | 100K | prefill |
|---|---|---|---|---|---|
| 0 | baseline (TP=2, MTP spec=4) | 33.7 | 43.1 | 37.4 | ~1,650 |
| 1 | MTP depth 3 | 57.0 | 38.1 | 39.4 | ~1,600 |
| 2 | `--max-num-batched-tokens 8192` | 35.5 | 34.0 | 35.5 | ~2,060 |
| 3 | KV 23.56 GiB/card | 33.9 | 39.9 | 35.0 | ~1,650 |
| **4** | **EP + humming** | **70.5** | **94.5** | **94.3** | **~3,630** |
| 5 | arm 4 + depth 3 | 68.0 | 87.1 | 75.5 | ~3,650 |
| 6 | arm 4 + MNB 8192 | 65.4 | 32.7* | 73.1 | ~3,930 |

\* arm 6's 22K point is an outlier run; its other points run ~7% below arm 4's. Decode
is no better with MNB on top of EP+humming, so the clamp warning (`max_num_scheduled_tokens
= 2048`) is **not the binding constraint once EP+humming is in** — don't stack it.

**Winner: arm 4** — `MODE=tp, EP=1, ALL2ALL=allgather_reducescatter, MOE_BACKEND=humming,
SPEC_TOKENS=4`. Live since 2026-09-17, agent traffic observed at 71–94 t/s.

## Why EP + humming works here

Flash-Next is a big MoE (~512 experts, few active per token). Two things were costing
us at decode:

1. **Plain TP=2 replicates the dispatch.** Both cards compute with both cards' expert
   shards for every token, and the layer-wise allreduce (PYNCCL only — this board has
   no P2P; custom allreduce is disabled) pays on every layer. With **EP**, each card
   owns half the experts and tokens are dispatched to where their experts live
   (allgather/reduce-scatter instead of allreduce over replicated shards). Decode is
   bandwidth-bound on reading activated expert weights at tiny batches; halving the
   per-card expert footprint halves that read. It also frees the replication
   overhead, which is where the prefill doubling comes from.
2. **The default WNA16 MoE kernels (triton→Marlin fallback) are the wrong kernels
   for this box.** The 170HX is an Ampere-class die (sm_80) — no FP8 tensor cores —
   while the checkpoint is W4A16 + FP8 PLE. **humming** JIT-compiles (NVRTC) kernels
   specialized for W4A16 group-quantized MoE on sm_80, using indexed GEMM that matches
   the MoE gather pattern. This is exactly the combo the published runblock shipped
   (`--moe-backend humming --enable-expert-parallel` on validated 2×3090 hardware);
   we initially ran without it and lost 2.2× for nothing.

**Depth-4 spec interacts with the win:** on the slow baseline, the 4th draft token was
dead weight (depth-3 won at short context). On the fast EP+humming stack, draft passes
are cheap relative to target verification, so depth-4 beats depth-3 everywhere (arm 5
lost 8–17%). Optimal spec depth is stack-dependent — re-sweep it whenever the MoE
backend changes.

## Pitfalls found (also in `docs/drift-log.md` D7–D8)

- `humming` JIT needs `libnvrtc-builtins.so.13.0` at worker runtime: export
  `LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cu13/lib`. Without it the
  worker dies at first MoE shape with `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`.
- `use_local_argmax_reduction: true` (in the published runblock) crashes vLLM 0.29.0:
  `Qwen4ExpMTP does not implement get_top_tokens()`. The flag works only in the
  container's pinned dev build. Omit it on released vLLM.

## Environment fingerprint

Same as `benchmarks.md` §Environment, plus: `EP=1`, `MOE_BACKEND=humming`,
`LD_LIBRARY_PATH` (cu13 lib), driver 610.43.02 (pinned — never upgrade on 170HX boxes),
watchdog + arm-driver harness retained on the box (`~/ops/flashnext-serving/`).
