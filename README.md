# qwen38-flash-next-170hx-companion

Companion to [`bluespace3/qwen38-flash-next-170hx`](https://github.com/bluespace3/qwen38-flash-next-170hx) —
**Qwen3.8-Flash-Next 176B on 2× CMP 170HX (64 GB), W4A16 + FP8 PLE, vLLM**.

The original repo ships the clever part (the PLE-mmap patch set + launcher + benchmarks)
but its patch series was exported from a working tree that carried extra edits. If you
start from a clean checkout of vLLM, their `scripts/apply-patches.sh` succeeds, and then
the server fails at import, at weight-load, and at table-init — four separate drifts we
hit and fixed on our own dual-170HX box. This repo is the missing pieces, plus a
dead-reproducible install path, so the next person doesn't have to rediscover them.

## What's here

| Path | What it is |
|---|---|
| `install/` | One-shot installer for a **vLLM 0.29.0** install: applies the original `ple_mmap` patch set **plus** the four reconstructed fixes, with assertions + backups. Also a `verify_install.py`. |
| `patches/` | The corrected patch series as plain unified diffs against the vLLM v0.29.0 tag (superset of `0001`/`0002` from the original repo). |
| `docs/drift-log.md` | Every fix, its root cause, the exact error you'd see without it. |
| `docs/deployment.md` | The full deployment recipe as actually run: checkpoint, env knobs, PP vs TP decision, launch, pitfalls. |
| `docs/checkpoint.md` | The checkpoint situation: the original ModelScope export is gone; the HF equivalent + how to verify its PLE layout against the patch's expectations *before* downloading 120 GB. |
| `scripts/` | Download (resumable + SHA256), launcher `run.sh`, systemd unit, bench + smoke clients, PLE-structure verifier. |
| `results/` | Measured benchmarks on 2× CMP 170HX (see `results/benchmarks.md`). |

## TL;DR status quo (2026-09-16)

- Upstream vLLM (0.29.0) **hard-blocks PP>1** for Qwen4Exp PLE (`NotImplementedError: ... pipeline_parallel_size=1 ...`). The original repo's PP=2 + MTP works only with three PP patches that were **never published**.
- Therefore: run the original repo's **`tp` mode** (TP=2). This companion ships a working `tp` config; PP remains future work unless those patches surface.
- Checkpoint: use `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` at the pinned revision — same composition as the (now unfindable) ModelScope `W4A16-fp8ple` export. Verify with `scripts/verify_checkpoint_ple.py` before committing to the download.
- Client compatibility: patch the chat template to merge multi-system payloads (`scripts/patch_chat_template.py`); `reasoning_effort` accepts `xhigh`/`medium`/`low` only.
- Reboot-safe: boot gate + `linger`/`RemoveIPC=no` (see `docs/drift-log.md` D9); optional NVMe tier shaves the PLE prewarm **100 s → 30 s** (`docs/deployment.md` §8). Two reboot drills passed.
- Four fixes needed for vLLM 0.29.0 (all in `install/`):
  1. rename + re-register the PLE ngram-id op path (`compute_ngram_ids` → `_hash_ngram_ids`, drop the stale op registration);
  2. add `vllm::qwen4_exp_ple_mmap_forward` to the default splitting-ops list;
  3. add the three imports + the early `embedding = self.ngram_embedding` binding the patch's own `load_weights` intercept relies on;
  4. wire `ple_mmap.build_tables(...)` into both `model.py` load paths (the patch designed for these calls but never shipped them).

## Credits & licensing

- Original patch design, launcher, and benchmark methodology: [bluespace3/qwen38-flash-next-170hx](https://github.com/bluespace3/qwen38-flash-next-170hx) (Apache-2.0; license retained in `LICENSES/`).
- W4A16 auto-round + FP8 PLE checkpoint composition: Intel (auto-round), RadixArk (FP8 PLE), packaged by [albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE); base model © Qwen (Qwen Community License).
- This repo's additions: Apache-2.0. No model weights are redistributed here.
