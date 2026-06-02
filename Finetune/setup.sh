#!/usr/bin/env bash
# One-shot environment setup on the training machine.
#
# Target: a Linux box (Ubuntu 22.04+/Debian) with an NVIDIA RTX 5070
# (12 GB, Blackwell sm_120) and the NVIDIA driver already installed (check with
# `nvidia-smi`). Blackwell needs CUDA 12.8 wheels — that's what we install.
#
# This script:
#   1. sanity-checks the GPU,
#   2. creates a .venv and installs PyTorch (cu128) + all deps,
#   3. installs Flash-Attention 2 (optional, falls back to sdpa),
#   4. DOWNLOADS the base model used for fine-tuning into the HF cache,
#   5. verifies torch + the model architecture actually load.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Overridable env vars:
#   MODEL_ID   model to download   (default: Qwen/Qwen3.5-4B-Base)
#   CUDA_WHL   torch wheel channel (default: cu128; use cu126/cu121 on older drivers)
#   HF_TOKEN   HF access token, only if the model repo is gated

set -euo pipefail

# Always operate relative to this script's folder so paths resolve no matter
# where it's invoked from.
cd "$(dirname "$0")"

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B-Base}
CUDA_WHL=${CUDA_WHL:-cu128}

# 1. Sanity-check the GPU.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found — install the NVIDIA driver before running this." >&2
    exit 1
fi
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2. System Python: prefer 3.11, else any python3. On minimal Ubuntu the venv
#    and pip modules are a separate apt package — check for them up front so we
#    fail with an actionable message instead of a cryptic ensurepip error.
PYTHON_BIN=$(command -v python3.11 || command -v python3 || true)
if [ -z "$PYTHON_BIN" ]; then
    echo "No python3 on PATH. Install it first, e.g.:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip build-essential git" >&2
    exit 1
fi
echo "Using $PYTHON_BIN ($($PYTHON_BIN --version))"
if ! "$PYTHON_BIN" -c "import ensurepip, venv" >/dev/null 2>&1; then
    echo "Python is missing the venv/pip modules. Install them, e.g.:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip" >&2
    exit 1
fi
# A C compiler is only needed to BUILD flash-attn (optional — the trainer falls
# back to ATTN=sdpa). Warn but don't fail if it's missing.
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "[warn] no C compiler found — flash-attn won't build. Either"
    echo "       'sudo apt-get install -y build-essential' to enable FA2, or just"
    echo "       train with 'ATTN=sdpa ./train.sh' (slightly slower, no build needed)."
fi

# 3. Create venv if absent.
if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Upgrade pip + install torch first (with the right CUDA wheels for Blackwell).
pip install --upgrade pip wheel
# CUDA 12.8 wheels — required for the 5070's sm_120. On an older driver that
# can't do 12.8, re-run with CUDA_WHL=cu126 (or cu121) — you lose some
# Blackwell-specific kernels but training still works on the bf16 path.
echo "== Installing torch from the ${CUDA_WHL} channel =="
pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHL}" "torch>=2.7"

# 5. Install everything else (default PyPI index).
pip install -r requirements.txt

# 6. Flash Attention 2 — installed separately because the wheel build needs to
# see the torch install we just did. If this fails on Blackwell (sm_120) due to
# a kernel-build issue, the trainer falls back gracefully with `ATTN=sdpa ./train.sh`.
pip install flash-attn --no-build-isolation || echo "[warn] flash-attn install failed — train with ATTN=sdpa ./train.sh"

# 7. Download the base model into the HF cache so training runs offline.
#    Set HF_TOKEN beforehand if the repo is gated.
echo "== Downloading model: $MODEL_ID =="
python - <<'PY'
import os
from huggingface_hub import snapshot_download

mid = os.environ["MODEL_ID"]
print(f"Fetching {mid} into the HF cache (first run can take several minutes)...")
path = snapshot_download(repo_id=mid, token=os.environ.get("HF_TOKEN") or None)
print(f"Cached at: {path}")
PY

# 8. Import + architecture check.
python - <<'PY'
import os
import torch
print(f"torch    {torch.__version__}")
print(f"CUDA     {torch.version.cuda}")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU      {torch.cuda.get_device_name(0)}  (sm_{cc[0]}{cc[1]})")
else:
    print("GPU      (none visible to torch)")
import transformers, peft, trl, datasets, bitsandbytes
print(f"transformers {transformers.__version__}  peft {peft.__version__}  trl {trl.__version__}")
print(f"datasets {datasets.__version__}  bitsandbytes {bitsandbytes.__version__}")
try:
    import flash_attn
    print(f"flash-attn {flash_attn.__version__}  OK")
except ImportError:
    print("flash-attn  NOT INSTALLED — pass ATTN=sdpa to train.sh")

# Confirm transformers actually understands this model's architecture.
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained(os.environ["MODEL_ID"], trust_remote_code=True)
    print(f'model    {os.environ["MODEL_ID"]}  (model_type={getattr(cfg, "model_type", "?")})  OK')
except Exception as exc:  # noqa: BLE001
    print(f"[warn] could not load the model config: {exc}")
    print("       If it says the model_type (e.g. 'qwen3_5') is unknown, your")
    print("       transformers is too old for this architecture. Install from source:")
    print('         pip install -U "git+https://github.com/huggingface/transformers"')
PY

echo
echo "Environment ready. Model '$MODEL_ID' is cached. Run ./train.sh to fine-tune."
