#!/usr/bin/env python3
"""Fix bluespace-ple-mmap patch drift on vLLM 0.29.0 (applied 2026-09-16).

Three edits, each asserted; idempotent; one .pre-fix-20260916 backup per file:
 1. ple_layer.py : def compute_ngram_ids -> def _hash_ngram_ids
 2. ple_layer.py : delete the stale direct_register_custom_op block for
    qwen4_exp_compute_ple_ngram_ids (its functions were removed by patch hunk 11)
 3. compilation.py : _attention_ops entry
    "vllm::qwen4_exp_compute_ple_ngram_ids" -> "vllm::qwen4_exp_ple_mmap_forward"
"""
import os
import shutil
import sys
import py_compile

_root = os.environ.get("VLLM_SP", "")
if not _root or not os.path.isdir(_root):
    sys.exit("VLLM_SP must point to the site-packages (or tree) that contains vllm/")
SP = os.path.join(_root, "vllm")
if not os.path.isdir(SP):
    SP = _root
PLE = f"{SP}/models/qwen4_exp/nvidia/ple_layer.py"
COMP = f"{SP}/config/compilation.py"
assert os.path.exists(PLE), PLE
assert os.path.exists(COMP), COMP
print(f"target: {SP}")

def backup(path):
    bak = path + ".pre-fix-20260916"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
        print(f"      (backup: {bak})")

def write_and_check(path, src):
    py_compile.compile(path, doraise=True)
    open(path, "w").write(src)
    print(f"[ok] wrote {path}")

# ---------- 1: rename the method ----------
src = open(PLE).read()
old = "def compute_ngram_ids("
new = "def _hash_ngram_ids("
if old in src:
    n = src.count(old)
    assert n == 1, f"rename: {n} matches"
    src = src.replace(old, new, 1)
    backup(PLE)
    write_and_check(PLE, src)
    print("[ok] ple_layer: renamed compute_ngram_ids -> _hash_ngram_ids")
else:
    print("[skip] ple_layer: old def name not present (already renamed?)")

# ---------- 2: drop stale old-op registration ----------
src = open(PLE).read()
pat = '''\ndirect_register_custom_op(
    op_name="qwen4_exp_compute_ple_ngram_ids",
    op_func=qwen4_exp_compute_ple_ngram_ids,
    mutates_args=["output"],
    fake_impl=qwen4_exp_compute_ple_ngram_ids_fake,
)\n'''
n = src.count(pat)
if n == 0:
    print("[skip] ple_layer: stale old-op registration not present")
else:
    assert n == 1, f"old registration block: {n} matches"
    src = src.replace(pat, "\n", 1)
    backup(PLE)
    write_and_check(PLE, src)
    print("[ok] ple_layer: deleted stale qwen4_exp_compute_ple_ngram_ids registration")

# ---------- 3: splitting-ops name swap ----------
src = open(COMP).read()
old = '"vllm::qwen4_exp_compute_ple_ngram_ids",'
new = '"vllm::qwen4_exp_ple_mmap_forward",'
if old in src:
    n = src.count(old)
    assert n == 1, f"splitting ops: {n} matches"
    src = src.replace(old, new, 1)
    backup(COMP)
    write_and_check(COMP, src)
    print("[ok] compilation: _attention_ops swapped to qwen4_exp_ple_mmap_forward")
else:
    print("[skip] compilation: old op name not present")

# ---------- verify ----------
ple = open(PLE).read()
comp = open(COMP).read()
checks = [
    ("def _hash_ngram_ids(" in ple, "ple_layer has _hash_ngram_ids def"),
    ("def compute_ngram_ids(" not in ple, "old def name gone"),
    ("qwen4_exp_compute_ple_ngram_ids" not in ple, "no stale op refs in ple_layer"),
    ("qwen4_exp_compute_ple_ngram_ids" not in comp, "no stale op refs in compilation"),
    ('"vllm::qwen4_exp_ple_mmap_forward",' in comp, "mmap op in _attention_ops"),
]
for ok, msg in checks:
    print(("PASS " if ok else "FAIL ") + msg)
if not all(ok for ok, _ in checks):
    print("FIX-FAIL"); sys.exit(1)
print("FIX-OK")
