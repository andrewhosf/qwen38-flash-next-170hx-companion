#!/bin/bash
# apply-all.sh — apply the complete corrected bluespace PLE-mmap patch set to vLLM 0.29.0.
#
# Usage:
#   VLLM_SP=/path/to/site-packages bash apply-all.sh
#   (VLLM_SP may point at the site-packages dir, a source tree containing vllm/, or .../vllm itself)
#
# What it does:
#   1. copies ple_mmap.py into vllm/models/qwen4_exp/nvidia/
#   2. applies the original 0001/0002 patches from bluespace3/qwen38-flash-next-170hx
#   3. applies this repo's three drift fixes (see docs/drift-log.md)
#   4. runs verify_install.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SP="${VLLM_SP:?set VLLM_SP to the site-packages/tree containing vllm/}"
SP="$VLLM_SP"
[ -d "$SP/vllm" ] || SP="$(dirname "$VLLM_SP")"
PY="${PYTHON:-python3}"

echo "== target: $SP =="
[ -d "$SP/vllm/models/qwen4_exp/nvidia" ] || { echo "does not look like a vLLM install: $SP"; exit 1; }

echo "== 1/4 copy ple_mmap.py =="
cp "$HERE/../patches/bluespace-original/ple_mmap.py" "$SP/vllm/models/qwen4_exp/nvidia/ple_mmap.py"

echo "== 2/4 apply original 0001/0002 (bluespace3) =="
if ( cd "$SP" && patch -N -p1 --forward < "$HERE/../patches/bluespace-original/0001-ple-layer-mmap-hook.patch" ); then
  echo "   0001 applied"
else
  echo "   NOTE: 0001 reported issues (already applied, or offsets) — continuing; fix scripts will assert."
fi
if ( cd "$SP" && patch -N -p1 --forward < "$HERE/../patches/bluespace-original/0002-envs-ple-knobs.patch" ); then
  echo "   0002 applied"
else
  echo "   NOTE: 0002 reported issues (already applied, or offsets) — continuing; fix scripts will assert."
fi

echo "== 3/4 apply drift fixes (this repo) =="
VLLM_SP="$SP" "$PY" "$HERE/fix1_ple_drift.py"
VLLM_SP="$SP" "$PY" "$HERE/fix2_ple_drift.py"
VLLM_SP="$SP" "$PY" "$HERE/fix3_model_wire.py"

echo "== 4/4 verify =="
VLLM_SP="$SP" "$PY" "$HERE/verify_install.py"
