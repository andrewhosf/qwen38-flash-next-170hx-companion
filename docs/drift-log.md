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

## D8 — client payloads with multiple system messages → 400

**Symptom:** a client gets `400` with `"System message must be at the beginning."` on
every call (e.g. Hermes-style agents that send `[system, system, user]`).

**Cause:** the checkpoint's `chat_template.jinja` renders only `messages[0]` as system
and raises on any other system message. llama.cpp (a common previous lane) is lenient
about this; vLLM's stock template is not.

**Fix:** merge all LEADING system messages into one system block —
`scripts/patch_chat_template.py <model_dir>/chat_template.jinja`, then restart the
server to reload the template. (Same class of fix as the DFlash2-fork template patch
for the 27B lane.)

**Also note:** the template accepts `reasoning_effort` ∈ {xhigh (default), medium, low}
and 400s on anything else (e.g. `high`). Match client settings.

---

## D9 — first boot after a reboot: worker spawn dies rebuilding semaphores

**Symptom:** several consecutive service starts fail in the first ~7 minutes after a
reboot. The engine reports `WorkerProc initialization failed due to an exception in a
background process`, and the worker traceback ends in:

```
File ".../multiprocessing/synchronize.py", line 115, in __setstate__
    self._semlock = _multiprocessing.SemLock._rebuild(*state)
FileNotFoundError: [Errno 2] No such file or directory
```

**Cause class:** a semaphore-availability race during early boot — the `/dev/shm`
semaphores a spawned worker must rebuild are gone by the time it starts. (Exact culprit
not isolated; it matches the logind session/IPC-teardown class, disappears once the
system has been up ~8 minutes, and login-session runs are unaffected. Retries inside the
window keep failing; the service crash-loops until past it.)

**Mitigations (all in this repo / its docs):**
1. Boot gate in `scripts/run.sh` — waits until uptime ≥ `BOOT_GATE_SECONDS` (480 s) AND
   a CUDA probe passes on the target devices (mid-day restarts skip it instantly).
2. `loginctl enable-linger <service-user>`.
3. `RemoveIPC=no` in `/etc/systemd/logind.conf`.

With these, two full reboot drills passed (first gated boot, `restarts=0`, smoke OK).

---

## D10 — humming MoE JIT can't find libnvrtc-builtins (worker dies at first MoE shape)

**Symptom:** with `--moe-backend humming`, the engine loads fine, then at the first
inference shape the worker dies:

```
nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
nvrtc_compile: compile failed: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
ERROR ... WorkerProc failed to start.
```

**Cause:** humming JIT-compiles its kernels with NVRTC at runtime. Setting `CUDA_HOME`/
`PATH` to the venv-bundled CUDA (D7) covers `nvcc`, but the worker process resolves
`libnvrtc-builtins.so.13.0` through the **dynamic linker**, which only searches the
system paths (finds the 12.0 builtins, wants 13.0). The file is present at
`site-packages/nvidia/cu13/lib/` but nothing points the linker there.

**Fix (in the launcher, next to the CUDA_HOME export):**

```bash
export LD_LIBRARY_PATH="$VENV_CUDA/lib:${LD_LIBRARY_PATH:-}"
```

(Already in `scripts/run.sh`. Failure mode is unfriendly: the lane crash-loops with
what looks like a model error — check for the nvrtc line before debugging anything else.)

---

## D11 — `use_local_argmax_reduction: true` crashes released vLLM (works in the pinned dev build)

**Symptom:** adding `"use_local_argmax_reduction": true` to the speculative-config
(the published `serve-container.sh` in the checkpoint's runtime snapshot sets it)
fails at model load on pip vLLM 0.29.0:

```
ValueError: use_local_argmax_reduction is enabled but draft model
Qwen4ExpMTP does not implement get_top_tokens().
```

**Cause:** the checkpoint's runtime snapshot was assembled inside container image
`vllm/vllm-openai:qwen38-flash-next@sha256:fc120e...` running `vllm 0.1.dev20073+g8e685d198`
— a dev build where the Qwen4ExpMTP drafter implements `get_top_tokens()`. Released
0.29.0 does not. The flag is a dev-build feature, not a 0.29.0 knob.

**Fix:** omit it. (MTP drafting still works without it; measured decode is unaffected
when it's omitted — the winning arm-4 config does not use it.)

---

If you hit a drift we haven't listed, the most reliable debugging move is: diff the
applied file against `bluespace3/qwen38-flash-next-170hx`'s `patches/ple_layer.patched.py`
(their reference), and treat "reference has it, applied doesn't" as a missing hunk.
