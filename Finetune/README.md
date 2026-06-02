# EXACT 2026 — Finetune

Self-contained LoRA fine-tune of **[Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)** for the NL → Z3-Python translation task of the EXACT 2026 Track 1 competition.

This folder is everything the training box needs. Ship the whole `Finetune/` to the machine with the GPU and run `./setup.sh` — it installs the env **and downloads the base model** for you. Then `./train.sh`.

> **New to this box? Read [QUICKSTART.md](QUICKSTART.md)** — the 3-command version. The rest of this file is the detailed reference.

## Training data shape (minimal annotation schema)

The trainer reads a JSON list of records. Only **four parallel arrays per record** are required:

```json
[
  {
    "premises-NL":  ["NL premise 1", "NL premise 2", ...],
    "premises-FOL": ["FOL premise 1", "FOL premise 2", ...],
    "questions-NL": ["NL question 1", "NL question 2", ...],
    "questions-FOL":["FOL goal 1",    "FOL goal 2",    ...]
  }
]
```

FOL is the same Unicode/Pythonic syntax used in the EXACT release (`∀x (P(x) → Q(x))` or `ForAll(x, P(x) -> Q(x))`). Empty `""` or missing entries in `questions-FOL` mean "not annotated yet" — those rows train on premise translation only, with a `goal = True` placeholder.

For MCQ, the cleanest annotation is **one (NL, FOL) pair per option** rewritten as a standalone declarative proposition (see `data/annotation_template.json`). The LoRA never sees MCQ structure; each row is one independent (NL, FOL) translation pair.

The original release file (`Logic_Based_Educational_Queries.json`) also works as-is — the loader auto-detects fields. Once you've annotated `questions-FOL` in that file, training picks it up immediately.

After annotation is complete, pass `REQUIRE_GOAL=1 ./train.sh` (or `--require-goal-fol`) to drop any record without an annotated goal so the LoRA sees full supervision only.

A 3-record annotation template is in `data/annotation_template.json` — copy it as a starting point.

## What this trains

The translator's job at inference time is to convert a list of natural-language premises into a Z3 Python program (declarations + a `premises` list + a `goal` Bool). This LoRA supervises the **NL → premise translation** half of that task using the `premises-NL` and `premises-FOL` fields from `Logic_Based_Educational_Queries.json`. The goal generation stays prompt-driven (the few-shot examples teach the model the surrounding template). Empirically 94.8% of training rows produce Z3 programs that exec cleanly in the sandbox.

## Hardware target

- **GPU**: RTX 5070, 12 GB GDDR7, Blackwell (sm_120). Defaults are sized for that envelope.

### What fits on 12 GB (estimated, QLoRA / 4-bit base)

| Item | Size |
| --- | --- |
| Qwen3.5-4B-Base (nf4 4-bit) | ~2.5 GB |
| LoRA adapter (bf16) | ~0.1 GB |
| 8-bit AdamW optimizer state | ~0.2 GB |
| Gradients (bf16) | ~0.1 GB |
| Activations (batch=2, seq=2048, **FA2 + gradient checkpointing**) | ~2–3 GB |
| CUDA workspace + fragmentation | ~1–2 GB |
| **Total** | **~6–8 GB** (≥4 GB margin) |

The defaults (`BATCH=2 GRAD_ACCUM=8`, effective batch 16) fit 12 GB with margin. Two settings are essentially **required** for that to hold:
- **Flash Attention 2** — without it, attention scores are O(seq²) per layer and OOM the 5070 at seq=2048. Default `ATTN=flash_attention_2`. If the wheel fails to build on Blackwell, fall back with `ATTN=sdpa ./train.sh` (still memory-efficient, ~10% slower).
- **Gradient checkpointing** — saves ~2 GB of activations for ~25% compute. Default `GRAD_CKPT=1`; on 12 GB keep it on.

Bigger GPU (16 GB)? `BATCH=4 GRAD_ACCUM=4`. (24 GB+, 4090/A100)? `BATCH=8 GRAD_ACCUM=2 GRAD_CKPT=0`. Still OOMing on 12 GB? `BATCH=1 GRAD_ACCUM=16` and optionally `MAX_SEQ_LEN=1536`.

## Folder layout

```
Finetune/
  README.md                  this file
  requirements.txt           pip deps (torch pinned via setup.sh index)
  setup.sh                   one-shot setup: venv + torch (cu128) + deps + MODEL DOWNLOAD + checks
  train.sh                   one-shot trainer wrapper with 5070 (12 GB) defaults
  .gitignore
  data/
    annotation_ready_merged.json           the annotated EXACT training data (train.sh default)
    annotation_template.json               3-record minimal-schema example to copy from
  finetune/                  Python package
    __init__.py
    types.py                 Record dataclass
    load.py                  EXACT JSON → Record (handles the real schema)
    fol_converter.py         Unicode + Pythonic FOL → Z3 Python DSL
    prompt.py                Few-shot NL → Z3-Python prompt
    train_lora.py            TRL SFTTrainer entrypoint
  tests/
    test_smoke.py            CPU-only sanity checks; run BEFORE training
  artifacts/                 (created at runtime) where the LoRA adapter lands
```

## Setup

Prereq: Linux box (Ubuntu 22.04+ recommended) with NVIDIA driver installed. Confirm `nvidia-smi` works and shows the RTX 5070 with ≥12 GB.

```bash
# 1. Unpack on the training machine, cd in.
chmod +x setup.sh train.sh

# 2. Install everything AND download the base model.
#    Creates .venv, installs torch (cu128 for Blackwell) + all deps, then
#    snapshot-downloads Qwen/Qwen3.5-4B-Base into the HF cache.
./setup.sh
```

`setup.sh` ends with a torch + transformers + bitsandbytes import check, prints the GPU it found, and confirms the model architecture loads. Re-run if you change `requirements.txt`.

Override the model or wheel channel via env vars (no file edits):

```bash
MODEL_ID=Qwen/Qwen3.5-4B-Base ./setup.sh   # default; change only if your repo id differs
CUDA_WHL=cu126 ./setup.sh                   # older driver that can't do CUDA 12.8
HF_TOKEN=hf_xxx ./setup.sh                  # only if the model repo is gated
```

### Driver / CUDA note for Blackwell

The 5070 is sm_120. You need:
- A recent NVIDIA driver (**R570+**) that supports CUDA **12.8**.
- The PyTorch wheels installed by `setup.sh` ship their own CUDA runtime, so a system CUDA install isn't strictly required — but the driver must be new enough.

Check with `nvidia-smi`: the CUDA version printed top-right is the *maximum* the driver supports — it must be ≥12.8 for the default `cu128` wheels.

If your driver is older, run `CUDA_WHL=cu126 ./setup.sh` (or `cu121`). You'll lose some Blackwell-specific kernels but training still works on the bf16 path.

> **transformers version**: Qwen3.5 (`model_type: qwen3_5`) is a new architecture. `requirements.txt` pins `transformers>=4.57`. If model loading still errors with *"Unrecognized model type … qwen3_5"*, install from source: `pip install -U "git+https://github.com/huggingface/transformers"`.

## Verify the env (before burning GPU time)

```bash
source .venv/bin/activate
pytest tests/ -v
```

All four tests run on CPU in under a second. If they pass, your install is correct. If they fail, fix that before kicking off training — `train.sh` won't recover.

## Run the fine-tune

```bash
./train.sh
```

That's it. Wall time on the 5070 for ~720 training rows (Qwen3.5-4B-Base + 4-bit base + LoRA r=32, 3 epochs): roughly **1–2 hours**.

The LoRA adapter is written to `artifacts/translator-lora/` (config + safetensors). Checkpoints land in `artifacts/translator-lora/checkpoint-<step>/`.

### Knobs

All overridable as env vars without editing files:

```bash
EPOCHS=5     ./train.sh                 # train longer
LORA_R=64    ./train.sh                 # bigger adapter
BATCH=4 GRAD_ACCUM=4 ./train.sh         # 16 GB card: bigger batch
BATCH=8 GRAD_ACCUM=2 GRAD_CKPT=0 ./train.sh   # 24 GB+ (4090/A100): bigger batch, no grad-ckpt
BATCH=1 GRAD_ACCUM=16 ./train.sh        # last resort if 12 GB still OOMs
ATTN=sdpa    ./train.sh                 # if flash-attn install failed on Blackwell
MAX_SEQ_LEN=1536 ./train.sh             # tighten if you're getting OOM
LR=1e-4      ./train.sh
MODEL_ID=Qwen/Qwen3.5-4B-Base ./train.sh   # override base model (match setup.sh)
DATA=path/to/other.json ./train.sh
OUT=artifacts/lora-v2    ./train.sh
```

For finer control (max_seq_len, lora_alpha, lora_dropout, warmup_ratio, use_4bit, target modules), edit `TrainConfig` in `finetune/train_lora.py`.

### Watching progress

`trl`'s `SFTTrainer` logs to stdout every 10 steps. To get a richer dashboard, set the env var `WANDB_API_KEY` and edit `args = SFTConfig(... report_to="wandb")` in `train_lora.py:train`.

## Ship the result back

```bash
tar -czf translator-lora.tar.gz -C artifacts translator-lora
ls -lh translator-lora.tar.gz   # typically 100-200 MB for a 4B model + r=32 LoRA
```

scp / rsync that tarball back to your inference box, drop it into `artifacts/translator-lora/`, and the inference pipeline will pick it up via `--lora artifacts/translator-lora`.

## Troubleshooting

- **`CUDA out of memory` during training** — drop `BATCH=1 GRAD_ACCUM=16`. If that still OOMs, lower `max_seq_len` (edit `TrainConfig`) to 1536 or 1024. The dataset's median premise count is small so 2048 is generous already.
- **`Unrecognized model type ... qwen3_5` / `KeyError: 'qwen3_5'`** — your transformers is too old for this architecture. `requirements.txt` pins `transformers>=4.57`; if it persists, install from source: `pip install -U "git+https://github.com/huggingface/transformers"`.
- **`Triton compiler error` / kernel compile failures on Blackwell** — happens with PyTorch + Triton versions that predate sm_120 support. Ensure torch came from the `cu128` channel (re-run `./setup.sh`) and `pip install --upgrade triton`.
- **`AttributeError: module 'torch' has no attribute 'xxx'` after upgrading deps** — usually a torch/transformers version skew. Re-run `setup.sh` to reinstall torch from the cu128 channel before transformers.
- **`OSError: We couldn't connect to 'https://huggingface.co'`** — set `HF_HUB_OFFLINE=0` (it gets stuck on 1 sometimes via stale env files) or `pip install --upgrade huggingface_hub` then `huggingface-cli login` if the model needs a token.
- **Smoke tests pass, but training crashes at step 1 with `bitsandbytes` errors** — some bitsandbytes 0.4x releases lag new CUDA. Upgrade with `pip install -U bitsandbytes` (a recent release is needed for cu128/Blackwell).
- **Training is mysteriously slow** — confirm bf16 is actually being used (look for `mixed_precision: bf16` in the `accelerate` log line). On Blackwell you can also try `--bf16 false --fp8 true` after upgrading transformers + accelerate to versions that support FP8 SFT (verify support before flipping).

## What's NOT in this folder

This folder builds the LoRA only. The downstream inference pipeline (Z3 runner, voter, CoT fallback, eval scorer, CLI) lives in the parent project. Once the LoRA is shipped back, the inference side picks it up automatically via the `--lora` flag.
