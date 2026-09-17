#!/usr/bin/env python3
"""Fix #3: wire `ple_mmap.build_tables` into model.py load paths (bluespace drift).

The patch design (per ple_mmap.build_tables docstring) requires calls from
Qwen4ExpForCausalLM.load_weights and Qwen4ExpForConditionalGeneration.load_weights
after their streamed weight passes — the shipped patch set never includes the
model.py hunks. This adds them + the two required imports.

Idempotent; backup *.pre-fix3-20260916; asserts every anchor.
"""
import os as _os
import shutil
import sys
import py_compile

_root = _os.environ.get("VLLM_SP", "")
if not _root or not _os.path.isdir(_root):
    sys.exit("VLLM_SP must point to the site-packages (or tree) that contains vllm/")
_vllm = _os.path.join(_root, "vllm")
if not _os.path.isdir(_vllm):
    _vllm = _root
M = _os.path.join(_vllm, "models/qwen4_exp/nvidia/model.py")
assert _os.path.exists(M), M
print(f"target: {M}")
CALL = (
    "        loaded = loader.load_weights(weights, mapper=mapper)\n"
    "        if ple_mmap.enabled():\n"
    "            _ple_cfg = get_current_vllm_config()\n"
    "            ple_mmap.build_tables(_ple_cfg.model_config, _ple_cfg.compilation_config)\n"
    "        return loaded\n"
)

src = open(M).read()
orig = src

if "ple_mmap.build_tables" in src:
    print("[skip] model.py already wired")
else:
    # ---- imports ----
    old = "from vllm.config import VllmConfig\n"
    assert src.count(old) == 1, f"config import count {src.count(old)}"
    src = src.replace(old, "from vllm.config import VllmConfig, get_current_vllm_config\n", 1)
    print("[ok] added get_current_vllm_config to vllm.config import")

    old = (
        "from vllm.transformers_utils.configs.qwen4_exp import (\n"
        "    Qwen4ExpTextConfig,\n"
        ")\n"
    )
    assert src.count(old) == 1, f"qwen4exp config import count {src.count(old)}"
    src = src.replace(old, old + "from . import ple_mmap\n", 1)
    print("[ok] added `from . import ple_mmap`")

    # ---- call site A: Qwen4ExpForCausalLM.load_weights (ends before Qwen4ExpProcessingInfo) ----
    a_old = (
        "        return loader.load_weights(weights, mapper=mapper)\n"
        "\n"
        "\n"
        "class Qwen4ExpProcessingInfo"
    )
    assert src.count(a_old) == 1, f"site A count {src.count(a_old)}"
    a_new = (
        CALL
        + "\n"
        + "\n"
        + "class Qwen4ExpProcessingInfo"
    )
    src = src.replace(a_old, a_new, 1)
    print("[ok] wired build_tables into Qwen4ExpForCausalLM.load_weights")

    # ---- call site C: Qwen4ExpForConditionalGeneration.load_weights (before @classmethod get_mamba_state_dtype_from_config) ----
    c_old = (
        "        return loader.load_weights(weights, mapper=mapper)\n"
        "\n"
        "    @classmethod\n"
        "    def get_mamba_state_dtype_from_config("
    )
    assert src.count(c_old) == 1, f"site C count {src.count(c_old)}"
    c_new = (
        CALL
        + "\n"
        "    @classmethod\n"
        "    def get_mamba_state_dtype_from_config("
    )
    src = src.replace(c_old, c_new, 1)
    print("[ok] wired build_tables into Qwen4ExpForConditionalGeneration.load_weights")

if src != orig:
    bak = M + ".pre-fix3-20260916"
    if not _os.path.exists(bak):
        shutil.copy(M, bak)
        print(f"      (backup: {bak})")
    py_compile.compile(M, doraise=True)
    open(M, "w").write(src)
    print("[ok] model.py written + compiled")

# verify
src = open(M).read()
checks = [
    ("get_current_vllm_config" in src, "get_current_vllm_config imported"),
    ("from . import ple_mmap" in src, "ple_mmap imported"),
    (src.count("ple_mmap.build_tables(") == 2, "both load sites wired (2 calls)"),
]
for ok, msg in checks:
    print(("PASS " if ok else "FAIL ") + msg)
if not all(ok for ok, _ in checks):
    print("FIX3-FAIL"); sys.exit(1)
print("FIX3-OK")
