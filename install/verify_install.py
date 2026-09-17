#!/usr/bin/env python3
"""verify_install.py — post-install checks for the corrected bluespace PLE-mmap set.

    VLLM_SP=/path/to/site-packages python verify_install.py
"""
import os
import sys

_root = os.environ.get("VLLM_SP", "")
if not _root or not os.path.isdir(_root):
    sys.exit("VLLM_SP must point to the site-packages (or tree) that contains vllm/")
_vllm = os.path.join(_root, "vllm")
if not os.path.isdir(_vllm):
    _vllm = _root

def read(p):
    with open(p) as f:
        return f.read()

checks = []
def add(cond, label):
    checks.append((bool(cond), label))

ple = read(os.path.join(_vllm, "models/qwen4_exp/nvidia/ple_layer.py"))
mmap = read(os.path.join(_vllm, "models/qwen4_exp/nvidia/ple_mmap.py"))
comp = read(os.path.join(_vllm, "config/compilation.py"))
model = read(os.path.join(_vllm, "models/qwen4_exp/nvidia/model.py"))
envs = read(os.path.join(_vllm, "envs.py"))

add("def _hash_ngram_ids(" in ple, "ple_layer: _hash_ngram_ids defined")
add("def compute_ngram_ids(" not in ple, "ple_layer: old method name gone")
add("qwen4_exp_compute_ple_ngram_ids" not in ple, "ple_layer: stale op refs gone")
add("from . import ple_mmap" in ple, "ple_layer: imports ple_mmap")
add("embedding = self.ngram_embedding" in ple, "ple_layer: early embedding binding present")
add("import os" in ple and "from typing import cast" in ple and "VocabParallelEmbedding" in ple,
    "ple_layer: os/cast/VocabParallelEmbedding imports present")
add("def build_tables(" in mmap, "ple_mmap: build_tables defined")
add("qwen4_exp_ple_mmap_forward" in mmap, "ple_mmap: mmap op registered")
add("qwen4_exp_compute_ple_ngram_ids" not in comp, "compilation: old op name gone")
add('"vllm::qwen4_exp_ple_mmap_forward",' in comp, "compilation: mmap op in _attention_ops")
add(model.count("ple_mmap.build_tables(") == 2, "model.py: build_tables wired at both load sites")
add("from . import ple_mmap" in model, "model.py: imports ple_mmap")
for knob in ["VLLM_PLE_MMAP", "VLLM_PLE_MMAP_WORKERS", "VLLM_PLE_MMAP_CHUNK",
             "VLLM_PLE_MMAP_PREWARM", "VLLM_PLE_MMAP_READAHEAD"]:
    add(knob in envs, f"envs.py: {knob} registered")

failed = 0
for ok, label in checks:
    print(("PASS " if ok else "FAIL ") + label)
    failed += (not ok)

if failed:
    print(f"\n{failed} check(s) FAILED — re-run install/apply-all.sh and read the output.")
    sys.exit(1)
print("\nAll checks passed. Next: docs/deployment.md (checkpoint, launch).")
