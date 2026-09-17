# Drift log — what the original repo's patches don't do, and how each failure looks

Context: installed vLLM **0.29.0** (stable, pip), applied `ple_mmap.py` + `0001-ple-layer-mmap-hook.patch`
+ `0002-envs-ple-knobs.patch` from `bluespace3/qwen38-flash-next-170hx`. All hunks apply cleanly
with only line-offset noise. Then the server fails, in this order, unless the following fixes
are applied. (Their reference file `patches/ple_layer.patched.py` was generated against a
slightly different upstream snapshot — the fixes below restore intent for v0.29.0.)

---

## D1 — stale op registration → `NameError` at import

**Symptom** (first boot, within ~30 s):

```
ERROR ... File ".../vllm/models/qwen4_exp/nvidia/ple_layer.py", line 1234, in <module>
    op_func=qwen4_exp_compute_ple_ngram_ids,
NameError: name 'qwen4_exp_compute_ple_ngram_ids' is not defined
... Model architectures ['Qwen4ExpForConditionalGeneration'] failed to be inspected.
```

**Cause:** patch hunk 11 removes the `qwen4_exp_compute_ple_ngram_ids` + `_fake` function
definitions, but the `direct_register_custom_op(op_name="qwen4_exp_compute_ple_ngram_ids", ...)`
call site that references them is NOT in the patch context (their base had it elsewhere/absent).

**Fix:** delete the stale registration block (keep the `qwen4_exp_ple_short_conv` one).
See `install/fix1_ple_drift.py`.

## D2 — PLE method rename → `NameError` at weight load

**Symptom:** about three minutes into weight streaming:

```
NameError: name 'embedding' is not defined   (misleading; the real drift is the rename)
```

(or, if the binding exists but the method name doesn't: `AttributeError: ... has no attribute '_hash_ngram_ids'`)

**Cause:** the patch's call site + the mmap op call `self._hash_ngram_ids(...)`, but upstream
v0.29.0 names the method `compute_ngram_ids`. The patch never renames it.

**Fix:** rename the def. Body stays v0.29.0's own (their reference body differs in one
buffer-slice line — that's a base-version difference, not a requirement).
See `install/fix1_ple_drift.py`.

## D3 — splitting-ops name → `RuntimeError` at table init (fail-closed guard)

**Symptom** (at model construction, once D1/D2 are fixed):

```
RuntimeError: VLLM_PLE_MMAP=1 requires 'vllm::qwen4_exp_ple_mmap_forward' in
compilation_config.splitting_ops ... Got splitting_ops=[... 'vllm::qwen4_exp_compute_ple_ngram_ids' ...]
```

**Cause:** v0.29.0's default `CompilationConfig._attention_ops` lists the OLD op name;
the patch's own safety check requires the NEW name.

**Fix:** swap the entry in `vllm/config/compilation.py`.
See `install/fix1_ple_drift.py`.

## D4 — missing imports + missing binding in `ple_layer.py`

**Symptom sequence (after D1–D3):**

```
NameError: name 'embedding' is not defined        ← the intercept block runs before the shard branch that used to bind it
NameError: name 'cast' is not defined             ← next, once the weight_scale arrives
NameError: name 'os' is not defined               ← FORCE_FP8 path (non-mmap)
NameError: name 'VocabParallelEmbedding' is not defined  ← stock path (non-mmap)
```

**Cause:** the patch's new code uses `os`, `cast`, `VocabParallelEmbedding`, and an
early `embedding = self.ngram_embedding` binding. Their reference file has all four;
the shipped hunks add none of them to this base.

**Fix:** add the imports + the binding.
See `install/fix2_ple_drift.py`.

## D5 — `build_tables` never called → `RuntimeError` after load

**Symptom** (weights load fully, then):

```
RuntimeError: PLE mmap table not initialized — load_weights ran but build_tables did not
```

**Cause:** `ple_mmap.build_tables()` is designed to be called from
`Qwen4ExpForCausalLM.load_weights` and `Qwen4ExpForConditionalGeneration.load_weights`
(see its docstring). The patch set never includes the `model.py` hunks.

**Fix:** wire the calls (+ `get_current_vllm_config`, `from . import ple_mmap` imports).
See `install/fix3_model_wire.py`.

## D6 — upstream hard gate: PP>1 refused (architectural, not a bug)

**Symptom:**

```
NotImplementedError: Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1
because non-first pipeline ranks do not receive the raw input_ids it needs. Please run with PP=1.
```

**Cause:** vLLM v0.29.0 (`vllm/model_executor/models/config.py::verify_and_update_config`)
blocks PP for this model. The original repo's PP=2 success depended on three patches
(pp_utils input relay / model_runner scatter for draft tokens / mtp.py last-rank embedding)
that were **not published** with the repo.

**Workaround:** use the original repo's own `tp` launcher mode — TP=2, PP=1. (Their `tp`
mode is a first-class mode in their `start-flashnext.sh`; it works with the same patch set.)
PP remains possible only if those three patches surface (or are re-derived).

## D7 — flashinfer JIT vs system nvcc (host-class issue)

**Symptom:** engine dies during "JIT kernel warmup":

```
nvcc fatal : Unknown option '--compress-mode=size'
```

**Cause:** flashinfer's JIT build uses the `nvcc` on PATH; on hosts with an old system
CUDA toolkit (e.g. 12.0) but modern vLLM wheels, the wheel-bundled CUDA toolkit is
`site-packages/nvidia/cu13/` and must be selected explicitly.

**Fix (in the launcher):**

```bash
VENV_CUDA=$VENV/../lib/python3.12/site-packages/nvidia/cu13   # venv-bundled CUDA 13.x
export CUDA_HOME="$VENV_CUDA"
export PATH="$VENV_CUDA/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0   # if the sampler kernels still fail their CCCL check
```

---

If you hit a drift we haven't listed, the most reliable debugging move is: diff the
applied file against `bluespace3/qwen38-flash-next-170hx`'s `patches/ple_layer.patched.py`
(their reference), and treat "reference has it, applied doesn't" as a missing hunk.
