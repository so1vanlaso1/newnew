#!/usr/bin/env bash
# One-shot setup for the EXACT 2026 Llama NL->FOL pipeline on Linux + NVIDIA GPU.
#
# Creates a venv, installs PyTorch (CUDA) + all deps, logs in to Hugging Face,
# and downloads BOTH model pieces into the HF cache so the pipeline runs offline:
#   * the GATED base   meta-llama/Llama-3.1-8B-Instruct
#   * the LoRA adapter  fvossel/Llama-3.1-8B-Instruct-nl-to-fol
#
# The base is gated by Meta: (1) request access on its HF page and be approved,
# and (2) provide a token. Export HF_TOKEN before running, or the script runs an
# interactive `huggingface-cli login`.
#
# Usage:
#   export HF_TOKEN=hf_xxx        # optional; else interactive login
#   chmod +x quickstart.sh
#   ./quickstart.sh
#
# Env overrides:
#   VENV_DIR   venv dir                 (default: .venv)
#   CUDA_WHL   torch wheel channel      (default: unset = pip's default CUDA build;
#                                        set cu124 / cu121 to pin, or cpu for CPU-only)
#   BASE_MODEL gated base repo          (default: meta-llama/Llama-3.1-8B-Instruct)
#   ADAPTER    LoRA adapter repo        (default: fvossel/Llama-3.1-8B-Instruct-nl-to-fol)
#   SKIP_MODEL_DOWNLOAD=1   install deps only, don't fetch the ~16 GB base

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=${VENV_DIR:-.venv}
CUDA_WHL=${CUDA_WHL:-}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
ADAPTER=${ADAPTER:-fvossel/Llama-3.1-8B-Instruct-nl-to-fol}

section() { printf '\n== %s ==\n' "$1"; }

# ── 1. GPU sanity check ────────────────────────────────────────────────────
section "GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "[warn] nvidia-smi not found. A CUDA GPU with >=8 GB VRAM is recommended for 4-bit."
    echo "[warn] 4-bit needs CUDA; set CUDA_WHL=cpu only for a slow CPU smoke test."
fi

# ── 2. Find a Python 3.10-3.12 interpreter ─────────────────────────────────
section "Python check"
PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)
        case "$ver" in
            3.10|3.11|3.12) PYTHON_BIN="$cand"; echo "Using $cand (Python $ver)"; break ;;
        esac
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "No Python 3.10-3.12 found. Install one, e.g.:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3-pip" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import ensurepip, venv" >/dev/null 2>&1; then
    echo "Python is missing the venv/pip modules. Install them, e.g.:" >&2
    echo "  sudo apt-get install -y python3-venv python3-pip" >&2
    exit 1
fi

# ── 3. Create / reuse the venv ─────────────────────────────────────────────
section "Virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Created $VENV_DIR"
else
    echo "$VENV_DIR already exists - reusing it"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"

# ── 4. Upgrade pip + install PyTorch ───────────────────────────────────────
section "pip + PyTorch"
"$PY" -m pip install --upgrade pip wheel
if [ -z "$CUDA_WHL" ]; then
    # On Linux, the default PyPI torch wheel already ships a bundled CUDA runtime.
    echo "Installing torch (default CUDA build)..."
    "$PY" -m pip install torch
elif [ "$CUDA_WHL" = "cpu" ]; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing torch from the $CUDA_WHL channel..."
    "$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_WHL"
fi

# ── 5. Install the project + GPU/dev extras ────────────────────────────────
section "Project dependencies"
"$PY" -m pip install -e ".[gpu,dev]"

# ── 6. Hugging Face auth (the base model is GATED) ─────────────────────────
section "Hugging Face login"
# `hf` is the current CLI; `huggingface-cli` is deprecated. Prefer hf, fall back.
hf_login()  { if command -v hf >/dev/null 2>&1; then hf auth login "$@";  else huggingface-cli login "$@";  fi; }
hf_whoami() { if command -v hf >/dev/null 2>&1; then hf auth whoami >/dev/null 2>&1; else huggingface-cli whoami >/dev/null 2>&1; fi; }
if [ -n "${HF_TOKEN:-}" ]; then
    # NOTE: a set HF_TOKEN is passed EXPLICITLY to the download below and overrides
    # any cached login. If it's stale/revoked the download 403s even though a good
    # cached token would have worked — in that case `unset HF_TOKEN` and re-run.
    hf_login --token "$HF_TOKEN" || \
        echo "[warn] CLI login failed; the token is passed directly to the download below"
elif hf_whoami; then
    echo "Already logged in to Hugging Face (cached token) - skipping login."
else
    echo "Not logged in and HF_TOKEN not set. Launching interactive login."
    echo "You need a token from https://huggingface.co/settings/tokens and"
    echo "access granted at https://huggingface.co/$BASE_MODEL"
    hf_login
fi

# ── 7. Download the model (base + adapter) into the HF cache ────────────────
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    section "Downloading model: $BASE_MODEL + adapter"
    BASE_MODEL="$BASE_MODEL" ADAPTER_MODEL="$ADAPTER" "$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

tok = os.environ.get("HF_TOKEN") or None
base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_MODEL"]
print(f"Fetching base {base} (~16 GB, first run takes a while)...")
try:
    p = snapshot_download(repo_id=base, token=tok)
    print(f"  base cached at: {p}")
except Exception as e:
    raise SystemExit(
        f"\nFailed to download {base}: {e}\n\n"
        "The base is gated. If you HAVE been granted access, the usual causes are:\n"
        "  * HF_TOKEN in your shell is stale/revoked (it OVERRIDES your cached login):\n"
        "      unset HF_TOKEN && ./quickstart.sh\n"
        "  * your token lacks gated-repo read scope -> use a classic 'Read' token, or\n"
        "    enable 'Read access to public gated repos' on a fine-grained token.\n"
        f"  * no access yet -> request it at https://huggingface.co/{base}\n"
        "Or skip the gate entirely with the ungated mirror (same weights):\n"
        "  BASE_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct ./quickstart.sh\n"
        "  then run with: --base-model NousResearch/Meta-Llama-3.1-8B-Instruct\n"
    )
print(f"Fetching adapter {adapter}...")
p = snapshot_download(repo_id=adapter, token=tok)
print(f"  adapter cached at: {p}")
print("Model download complete.")
PY
else
    echo "Skipping model download (SKIP_MODEL_DOWNLOAD=1)."
fi

# ── 8. Verify the install ──────────────────────────────────────────────────
section "Verify"
"$PY" - <<'PY'
import torch
print(f"torch    {torch.__version__}   CUDA build {torch.version.cuda}")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU      {torch.cuda.get_device_name(0)}  (sm_{cc[0]}{cc[1]})  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("GPU      (none visible to torch -- 4-bit needs CUDA)")
import transformers, peft, accelerate, bitsandbytes
print(f"transformers {transformers.__version__}  peft {peft.__version__}  "
      f"accelerate {accelerate.__version__}  bitsandbytes {bitsandbytes.__version__}")
import z3
print("z3 + pipeline deps OK")
PY

echo
echo "Environment ready."
echo "Activate it with:  source $VENV_DIR/bin/activate"
echo "Then:  python run_pipeline.py --limit 5 --show-gold --show-fol   (smoke test)"
echo "See RUN_COMMANDS.md for the full command list."
