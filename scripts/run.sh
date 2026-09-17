#!/bin/bash
# Flash-Next (Qwen3.8-Flash-Next 176B W4A16 + FP8 PLE) serving launcher.
# Pattern follows bluespace3/qwen38-flash-next-170hx scripts/start-flashnext.sh,
# adapted for vLLM 0.29.0 + the corrected patch set (see install/), TP=2 first light.
#
# Config comes from /etc/flashnext-serving.conf (see configs/flashnext-serving.conf.example).
# Env overrides: FIN_VENV (venv bin dir), FIN_VENV_CUDA, FIN_CACHE_ROOT.
set -euo pipefail

CONF=${FIN_CONF:-/etc/flashnext-serving.conf}
[ -r "$CONF" ] || { echo "FATAL: cannot read $CONF" >&2; exit 1; }
# shellcheck disable=SC1090
. "$CONF"

VENV=${FIN_VENV:-/opt/flashnext/venv/bin}

# flashinfer JIT: if the system CUDA toolkit is old, use the venv-bundled one
# (nvcc fatal: Unknown option '--compress-mode=size' otherwise).
VENV_CUDA=${FIN_VENV_CUDA:-$(dirname "$VENV")/lib/python3.12/site-packages/nvidia/cu13}
if [ -x "$VENV_CUDA/bin/nvcc" ]; then
  export CUDA_HOME="$VENV_CUDA"
  export PATH="$VENV_CUDA/bin:$PATH"
  # humming MoE JIT (NVRTC) needs libnvrtc-builtins.so.13.0 at WORKER runtime, not
  # just at build time. Without this the worker dies at the first MoE shape with
  # NVRTC_ERROR_BUILTIN_OPERATION_FAILURE (see docs/drift-log.md § D10).
  export LD_LIBRARY_PATH="$VENV_CUDA/lib:${LD_LIBRARY_PATH:-}"
fi
# Our sampler kernels failed the CCCL header check; the native sampler is fine.
export VLLM_USE_FLASHINFER_SAMPLER=0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$DEVICES"
export TRITON_CACHE_DIR=${FIN_CACHE_ROOT:-/opt/flashnext/cache}/triton
export VLLM_CACHE_ROOT=${FIN_CACHE_ROOT:-/opt/flashnext/cache}/vllm
export TMPDIR=/tmp
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# PLE mmap knobs (recipe: WORKERS=32 CHUNK=2048 PREWARM=1 READAHEAD=256; never CHUNK=8192)
export VLLM_PLE_MMAP=1
export VLLM_PLE_MMAP_WORKERS=32
export VLLM_PLE_MMAP_CHUNK=2048
export VLLM_PLE_MMAP_PREWARM=1
export VLLM_PLE_MMAP_READAHEAD=256
export VLLM_ENGINE_READY_TIMEOUT_S=5400
[ "${ALLOW_LONG:-0}" = "1" ] && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Boot-window gate (2026-09-16): for ~7 minutes after a reboot, freshly-spawned worker
# processes can die rebuilding POSIX semaphores (SemLock._rebuild -> FileNotFoundError in
# /dev/shm) — observed 5x in the reboot drill; a start after the window succeeds cleanly.
# Wait until past the window AND a real CUDA context probe passes on the target devices.
# Mid-day restarts (uptime already past the gate) skip instantly.
if [ "${SKIP_BOOT_GATE:-0}" != "1" ] && [ "$(cut -d. -f1 /proc/uptime)" -lt "${BOOT_GATE_SECONDS:-480}" ]; then
  for _i in $(seq 1 90); do
    if [ "$(cut -d. -f1 /proc/uptime)" -ge "${BOOT_GATE_SECONDS:-480}" ] && \
       "$VENV/python" -c "import torch; [torch.zeros(8, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]" >/dev/null 2>&1; then
      echo "flashnext: boot gate passed (uptime $(cut -d. -f1 /proc/uptime)s)"
      break
    fi
    echo "flashnext: boot gate waiting (uptime $(cut -d. -f1 /proc/uptime)s, target ${BOOT_GATE_SECONDS:-480}s)"
    sleep 10
  done
fi

ARGS=(serve "$MODEL"
  --served-model-name $SERVED_NAMES
  --host "${HOST:-0.0.0.0}" --port "$PORT"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --trust-remote-code
  --no-enable-flashinfer-autotune
  -cc.cudagraph_mode=PIECEWISE
)

case "$MODE" in
  tp)
    ARGS+=(--tensor-parallel-size 2)
    if [ "${EP:-0}" = "1" ]; then
      ARGS+=(--enable-expert-parallel --all2all-backend "${ALL2ALL:-allgather_reducescatter}")
    fi
    if [ -n "${MOE_BACKEND:-}" ]; then ARGS+=(--moe-backend "$MOE_BACKEND"); fi
    ;;
  pp)
    ARGS+=(--pipeline-parallel-size 2)
    # NOTE: PP>1 is blocked by an upstream gate on vLLM 0.29.0 for this model's PLE.
    # It only works with bluespace's three unpublished PP patches. See docs/drift-log.md § D6.
    ;;
  single) ;;
  *) echo "FATAL: unknown MODE=$MODE" >&2; exit 1;;
esac

if [ "${YARN:-0}" = "1" ]; then
  ARGS+=(--hf-overrides "$YARN_OVERRIDES")
fi

if [ "${SPEC_TOKENS:-0}" -gt 0 ]; then
  # spec=4 is the cap under the QSA ring-buffer constraint (spec 5+ asserts).
  if [ -n "${MTP:-}" ]; then
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS},\"model\":\"${MTP}\"}")
  else
    ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
  fi
fi

# Dry-run: print the exact command and exit.
if [ "${FIN_PRINT_ONLY:-0}" = "1" ]; then
  printf 'CMD: %q ' "$VENV/vllm"; printf '%q ' "${ARGS[@]}"; printf '\n'
  exit 0
fi

[ -d "$MODEL" ] || { echo "FATAL: model dir missing: $MODEL" >&2; exit 1; }
if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "FATAL: something is already serving on :${PORT}" >&2
  exit 1
fi

echo "flashnext: mode=$MODE devices=$DEVICES len=$MAX_LEN seq=$MAX_SEQS mem=$GPU_UTIL spec=${SPEC_TOKENS:-0} port=$PORT"
exec "$VENV/vllm" "${ARGS[@]}"
