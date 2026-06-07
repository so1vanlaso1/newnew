<#
.SYNOPSIS
    One-shot setup for the EXACT 2026 Llama NL->FOL pipeline on Windows + NVIDIA GPU.

.DESCRIPTION
    Creates a virtual environment, installs PyTorch (CUDA wheels) + all
    dependencies, logs in to Hugging Face, and downloads BOTH model pieces into
    the HF cache so the pipeline can run offline afterwards:

      * the GATED base   meta-llama/Llama-3.1-8B-Instruct
      * the LoRA adapter  fvossel/Llama-3.1-8B-Instruct-nl-to-fol

    The base is gated by Meta: you must (1) request access on its HF page and be
    approved, and (2) provide an HF token. Set $env:HF_TOKEN before running, or
    the script will run an interactive Hugging Face login.

.PARAMETER CudaWheel
    PyTorch CUDA wheel channel to match your driver. Default cu124.
    Use cu121 on an older driver, or cpu for a (very slow) CPU-only install.

.PARAMETER VenvDir
    Virtual-environment directory. Default .venv

.PARAMETER SkipModelDownload
    Install deps only; do not download the ~16 GB base model.

.EXAMPLE
    $env:HF_TOKEN = "hf_xxx"; .\quickstart.ps1

.EXAMPLE
    .\quickstart.ps1 -CudaWheel cu121
#>
param(
    [string]$CudaWheel = "cu124",
    [string]$VenvDir = ".venv",
    [string]$BaseModel = "meta-llama/Llama-3.1-8B-Instruct",
    [string]$Adapter = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Section($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }

# ── 1. GPU sanity check ────────────────────────────────────────────────────
Section "GPU check"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
} else {
    Write-Warning "nvidia-smi not found. A CUDA GPU with >=8 GB VRAM is recommended for 4-bit."
    Write-Warning "4-bit needs CUDA; pass -CudaWheel cpu only for a slow CPU smoke test."
}

# ── 2. Find a Python 3.10-3.12 interpreter ─────────────────────────────────
Section "Python check"
$pyCmd = $null
$pyArgs = @()
$candidates = @(
    @{ cmd = "python"; args = @() },
    @{ cmd = "py";     args = @("-3.12") },
    @{ cmd = "py";     args = @("-3.11") },
    @{ cmd = "py";     args = @("-3.10") }
)
foreach ($c in $candidates) {
    if (-not (Get-Command $c.cmd -ErrorAction SilentlyContinue)) { continue }
    try {
        $ver = & $c.cmd @($c.args) -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
    } catch { $ver = $null }
    if ($ver -match "^3\.(10|11|12)$") {
        $pyCmd = $c.cmd; $pyArgs = $c.args
        Write-Host "Using '$($c.cmd) $($c.args -join ' ')' (Python $ver)"
        break
    }
}
if (-not $pyCmd) { throw "No Python 3.10-3.12 found. Install Python 3.12 from python.org and re-run." }

# ── 3. Create the venv ─────────────────────────────────────────────────────
Section "Virtual environment"
if (-not (Test-Path $VenvDir)) {
    & $pyCmd @pyArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    Write-Host "Created $VenvDir"
} else {
    Write-Host "$VenvDir already exists - reusing it"
}
$py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

# ── 4. Upgrade pip + install PyTorch (CUDA wheels) ─────────────────────────
Section "pip + PyTorch ($CudaWheel)"
& $py -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
if ($CudaWheel -eq "cpu") {
    & $py -m pip install torch
} else {
    & $py -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaWheel"
}
if ($LASTEXITCODE -ne 0) { throw "torch install failed (try a different -CudaWheel, e.g. cu121)" }

# ── 5. Install the project + GPU/dev extras ────────────────────────────────
Section "Project dependencies"
& $py -m pip install -e ".[gpu,dev]"
if ($LASTEXITCODE -ne 0) { throw "project dependency install failed" }

# ── 6. Hugging Face auth (the base model is GATED) ─────────────────────────
Section "Hugging Face login"
$hfcli = Join-Path $VenvDir "Scripts\huggingface-cli.exe"
if ($env:HF_TOKEN) {
    & $hfcli login --token $env:HF_TOKEN
    if ($LASTEXITCODE -ne 0) { Write-Warning "token login failed; the token is still used directly for the download below" }
} else {
    Write-Host "No `$env:HF_TOKEN set. Launching interactive login."
    Write-Host "You need a token from https://huggingface.co/settings/tokens and"
    Write-Host "access granted at https://huggingface.co/$BaseModel"
    & $hfcli login
}

# ── 7. Download the model (base + adapter) into the HF cache ────────────────
if (-not $SkipModelDownload) {
    Section "Downloading model: $BaseModel + adapter"
    $env:BASE_MODEL = $BaseModel
    $env:ADAPTER_MODEL = $Adapter
    $dl = @'
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
        f"\nFailed to download {base}: {e}\n"
        "The base is GATED. Request access at its HF page and set a valid HF_TOKEN\n"
        "(or run 'huggingface-cli login'), then re-run quickstart.ps1.\n"
    )
print(f"Fetching adapter {adapter}...")
p = snapshot_download(repo_id=adapter, token=tok)
print(f"  adapter cached at: {p}")
print("Model download complete.")
'@
    # Pipe via stdin (not -c) so PowerShell doesn't mangle the embedded quotes.
    $dl | & $py -
    if ($LASTEXITCODE -ne 0) { throw "model download failed (see message above)" }
} else {
    Write-Host "Skipping model download (-SkipModelDownload)."
}

# ── 8. Verify the install ──────────────────────────────────────────────────
Section "Verify"
$verify = @'
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
'@
# Pipe via stdin (not -c) so PowerShell doesn't mangle the embedded quotes.
$verify | & $py -

Write-Host "`nEnvironment ready." -ForegroundColor Green
Write-Host "Activate it with:  .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Then see RUN_COMMANDS.md for how to run the pipeline (start with a smoke test)."
