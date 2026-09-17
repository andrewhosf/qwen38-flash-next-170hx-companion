#!/usr/bin/env python3
"""Fix #2: missing imports + missing binding (bluespace patch drift, vLLM 0.29.0).

The shipped 0001 patch's new code references `os`, `cast`, and
`VocabParallelEmbedding`, and its weight_scale intercept relies on an early
`embedding = self.ngram_embedding` binding — none of which the patch adds to
this base (they exist in the author's reference patched file).

Insertions, each exact-string, idempotent, with backups (*.pre-fix2-20260916):
 1. `import os` after `import math`
 2. `from typing import cast` after the collections.abc import
 3. VocabParallelEmbedding import before `from vllm.model_executor.models.utils import AutoWeightsLoader`
 4. `embedding = self.ngram_embedding` after `shard_prefix = "ngram_embedding.shard_"`
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
PLE = _os.path.join(_vllm, "models/qwen4_exp/nvidia/ple_layer.py")
assert _os.path.exists(PLE), PLE
print(f"target: {PLE}")

def backup(path):
    bak = path + ".pre-fix2-20260916"
    if not _os.path.exists(bak):
        shutil.copy(path, bak)
        print(f"      (backup: {bak})")

def insert_after(src, anchor, addition, expect=1):
    n = src.count(anchor)
    assert n == expect, f"anchor found {n}x (expected {expect}): {anchor[:80]!r}"
    return src.replace(anchor, anchor + addition, 1)

src = open(PLE).read()
orig = src

if "\nimport os\n" in src:
    print("[skip] import os present")
else:
    src = insert_after(src, "import math\n", "import os\n")
    print("[ok] added `import os`")

if "from typing import cast" in src:
    print("[skip] typing.cast present")
else:
    src = insert_after(src, "from collections.abc import Iterable, Sequence\n", "from typing import cast\n")
    print("[ok] added `from typing import cast`")

if "vocab_parallel_embedding import" in src:
    print("[skip] VocabParallelEmbedding import present")
else:
    anchor = "from vllm.model_executor.models.utils import AutoWeightsLoader\n"
    assert src.count(anchor) == 1
    addition = (
        "from vllm.model_executor.layers.vocab_parallel_embedding import (\n"
        "    VocabParallelEmbedding,\n"
        ")\n"
    )
    src = src.replace(anchor, addition + anchor, 1)
    print("[ok] added VocabParallelEmbedding import")

anchor = '        shard_prefix = "ngram_embedding.shard_"\n'
if '        embedding = self.ngram_embedding\n' in src:
    print("[skip] early embedding binding present")
else:
    src = insert_after(src, anchor, "        embedding = self.ngram_embedding\n")
    print("[ok] added `embedding = self.ngram_embedding`")

if src != orig:
    backup(PLE)
    py_compile.compile(PLE, doraise=True)
    open(PLE, "w").write(src)
    print("[ok] file written + compiled")
else:
    print("[skip] no changes needed")

# verify
src = open(PLE).read()
checks = [
    ("\nimport os\n" in src, "os imported"),
    ("from typing import cast" in src, "cast imported"),
    ("VocabParallelEmbedding,\n" in src, "VocabParallelEmbedding imported"),
    ("        embedding = self.ngram_embedding\n" in src, "early embedding binding"),
]
for ok, msg in checks:
    print(("PASS " if ok else "FAIL ") + msg)
if not all(ok for ok, _ in checks):
    print("FIX2-FAIL"); sys.exit(1)
print("FIX2-OK")
