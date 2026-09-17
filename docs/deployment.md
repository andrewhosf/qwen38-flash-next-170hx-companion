# Deployment recipe (as actually run — 2× CMP 170HX, vLLM 0.29.0, 2026-09)

This is the full path that ends in a serving `Qwen3.8-Flash-Next` on two CMP 170HX
cards with the PLE table served from page cache. Reference box: Proxmox-free bare metal,
TR PRO 3945WX, 235 GB RAM, driver 610.43.02 (CUDA 13.x UMD), Ubuntu 24.04, Python 3.12.

> **TL;DR of differences vs the original repo:** run its `tp` mode (TP=2), not `pp1m` —
> upstream vLLM 0.29.0 hard-blocks PP>1 for this model's PLE (see `docs/drift-log.md` § D6)
> and the original repo's three PP patches were never published. Everything else
> (PLE-mmap patch, env knobs, spec constraints) follows the original repo.

## 0. Requirements

- 2× CMP 170HX (GA100 / sm_80, 64 GB each), PCIe Gen2 x16 preferred; both cards on the
  same host bridge is fine. A x4 slot *works* but adds decode latency.
- ≥ ~128 GB RAM (we ran with 235 GB; the 48 GiB FP8 PLE table lives in page cache —
  more RAM = fewer cold-page reads; the original repo ran with 60 GB + NVMe).
- ~130 GB free disk for the checkpoint.
- NVIDIA driver new enough for a CUDA 13 runtime (610.x verified); **no** driver change needed.

## 1. vLLM 0.29.0 + the corrected patch set

```bash
python3 -m venv vllm029-venv
./vllm029-venv/bin/pip install vllm==0.29.0 websockets

VLLM_SP="$PWD/vllm029-venv/lib/python3.12/site-packages" \
  bash install/apply-all.sh
```

`apply-all.sh` = copy `ple_mmap.py` → apply the original `0001`/`0002` →
apply `fix1..fix3` (this repo) → `verify_install.py`. Idempotent, asserts every edit.

## 2. nvcc for flashinfer JIT (host-dependent, common trap)

If your host's system CUDA toolkit is older than ~12.9, the flashinfer JIT fails at engine
startup (`nvcc fatal: Unknown option '--compress-mode=size'`). Point it at the toolkit
bundled in the venv (vLLM's wheels ship one under `site-packages/nvidia/cu13/`):

```bash
VENV_CUDA="$PWD/vllm029-venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$VENV_CUDA"
export PATH="$VENV_CUDA/bin:$PATH"
```

If the flashinfer *sampling* kernels still fail their CCCL header check, disable the
flashinfer sampler — vLLM's native sampler is used instead and is fine:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## 3. Checkpoint

See `docs/checkpoint.md`. Download with `scripts/download-checkpoint.sh` (resumable,
SHA256-verified), or point `MODEL` in the config at your own directory.

## 4. Launch

Config lives in `configs/flashnext-serving.conf.example`; launcher is `scripts/run.sh`.

Key values for the TP=2 first-light profile:

| Setting | Value | Why |
|---|---|---|
| `MODE` | `tp` | PP is gated upstream (D6); TP=2 is the original repo's own fallback mode |
| `DEVICES` | e.g. `2,3` | `CUDA_DEVICE_ORDER=PCI_BUS_ID`; pick your two CMPs |
| `MAX_LEN` | `262144` | native context; add `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` + YaRN overrides for 1M |
| `GPU_UTIL` | `0.90` | weights ≈38 GiB/card at TP2 + KV + activations |
| `SPEC_TOKENS` | `0` | first light without MTP; see below |
| `VLLM_PLE_MMAP` | `1` | the whole point — table from page cache, ~0 VRAM |
| `VLLM_PLE_MMAP_{WORKERS,CHUNK,PREWARM,READAHEAD}` | `32,2048,1,256` | recipe defaults; **never CHUNK=8192** (prefill collapses) |
| `-cc.cudagraph_mode=PIECEWISE` | mandatory | the mmap hashing+gather op cannot run inside a full CUDA graph |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `5400` | weight load + 48 GiB prewarm on cold cache |

### MTP / speculative decoding

- The original repo spec-decodes with **spec=4** (QSA ring-buffer constraint: spec ≥5 asserts;
  KV must be BF16 — fp8 KV is blocked by QSA).
- **TP=2 + MTP is not yet validated in this companion.** The albucino checkpoint ships the
  MTP draft separately (`runtime/mtp-int4-g32/`); vLLM 0.29.0 accepts
  `--speculative-config '{"method":"mtp","num_speculative_tokens":N,"model":"<that dir>"}'`,
  but we ran first light without it. The original repo's PP=2 + MTP benches (60–112 tok/s
  decode) are their PP numbers, not ours.

### systemd

See `configs/flashnext.service.example`. Swap discipline (learned the hard way, generally
good practice for single-owner GPUs):

```bash
# stop every unit that owns the cards, THEN verify, THEN start:
sudo systemctl stop <old-lane> <old-lane>-warmup <old-lane>-kvprime
sleep 10
pgrep -x llama-server; nvidia-smi --query-gpu=index,memory.used --format=csv   # expect ~0
sudo systemctl start flashnext.service
journalctl -u flashnext.service -f
```

## 5. Boot timeline (warm page cache)

| t | phase |
|---|---|
| +0:00 | config resolution; `quantization=inc` (auto-round → INC loader, Marlin kernels) |
| +0:30 | TP ranks up, NCCL (PYNCCL fallback on no-P2P boards) |
| +1:00 | weights stream (page cache: fast; cold SATA: 6–8 min) |
| +3:30 | PLE `load_weights` (shards dropped from stream) → `build_tables` → `PLE mmap: layer 1 attached …` + **prewarm** (reads the full table once) |
| +5:00 | engine ready → KV cache → PIECEWISE graph capture |
| +6:00 | HTTP server up (`/health` 200) |

First request: expect a slower first prefill (kernel autotune + graph capture warmup).

## 6. Verify + bench

```bash
python scripts/smoke-flashnext.py http://127.0.0.1:8000 qwen3.8-flash-next   # health/chat/tool-call
python scripts/bench-flashnext.py                                            # 8K..260K single-stream ladder
```

See `results/benchmarks.md` for what we measured on 2× CMP 170HX.
