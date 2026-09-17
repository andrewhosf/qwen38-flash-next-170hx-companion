# Checkpoint notes

## The situation

The original repo's instructions say:

> search ModelScope for "Qwen3.8-Flash-Next-W4A16-fp8ple" (AutoRound export)

As of **2026-09-16**, that export is not findable anywhere: ModelScope keyword search
(several spellings), the ModelScope web listing of ALL `Flash-Next` quantizations,
Hugging Face name/full-text search, the GitHub repo's issues/forks, and the public
forum threads that discuss this recipe. Treat it as gone (or never-indexed).

## The verified equivalent

**[`albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)**
— pin revision `ef554143369a706525336f6b42a09094835dc077`.

It is the same *composition class* the patch expects ("auto-round main weights + fp8
ngram table" — the phrase in the patch's own `VLLM_PLE_FORCE_FP8` escape hatch):

| Component | Format | Source |
|---|---|---|
| Backbone | AutoRound W4A16 (INT4 g128 sym, GPTQ-packed) | Intel `/Qwen3.8-Flash-Next-W4A16-AutoRound` |
| n-gram / PLE table | FP8 E4M3 + BF16 global scale, **128 shards × 2,500,012 rows** per layer | RadixArk FP8 table |
| MTP | separate compact INT4 g32 draft under `runtime/mtp-int4-g32/` | albucino |

Verified layout (byte-level, before/after download):

- 86 files / ~120.9 GiB total; 25 safetensors (16 weight shards + 10 `model-plefp8-*` files wait — 15 weight + 10 PLE).
- PLE tensor names: `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<M>.weight`
  — matches the patch's `_SHARD_RE` (`search`, so the `model.` prefix is fine).
- Scale: `...ngram_embedding.weight_scale` present (BF16 scalar) — required or the
  mmap loader fail-closes.
- Full SHA256 set verified against the repo's `SHA256SUMS` after download (86/86 OK).

## Before you download ~120 GB — verify first

```bash
python scripts/verify_checkpoint_ple.py \
  --repo albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --rev  ef554143369a706525336f6b42a09094835dc077
```

This reads only safetensors headers over HTTP Range (a few MB) and checks the PLE
shard set / dtype / scale exactly as `ple_mmap.py` will at load time.

## Download

```bash
bash scripts/download-checkpoint.sh   # resumable, 20 parallel streams, SHA256 verify at the end
```

Measured on a US residential link: ~17 MB/s with `-P 20` (single stream ~1 MB/s per
connection after HF throttling kicks in — more connections is the lever).

## License

The checkpoint's model card inherits the Qwen Community License; Intel/RadixArk/albucino
compositions. Nothing here redistributes weights — you download from HF yourself.
