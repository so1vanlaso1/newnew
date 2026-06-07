# Run commands — Llama NL→FOL pipeline

Every command for running the pipeline, in order. Windows / PowerShell. Run them
from the project root (the folder containing `run_pipeline.py`).

One model does everything: the **fvossel/Llama-3.1-8B-Instruct-nl-to-fol** LoRA
adapter on the gated **meta-llama/Llama-3.1-8B-Instruct** base, loaded in 4-bit.

---

## 0. One-time setup

```powershell
# Installs venv + CUDA torch + deps, logs in to HF, downloads the model.
# Set your HF token first (the base model is gated — request access on its HF page).
$env:HF_TOKEN = "hf_xxxxxxxx"
.\quickstart.ps1
```

See [quickstart.ps1](quickstart.ps1) for options (`-CudaWheel cu121`, `-SkipModelDownload`, …).

## 1. Activate the environment (every new shell)

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Sanity checks (no GPU needed)

```powershell
# Unit tests for the FOL→Z3 chain, bolt-ons, grouping, solver, voter (CPU only).
pytest tests\ -v

# Confirm the dataset loads and the answer-type split looks right.
python -m cli.inspect Logic_Based_Educational_Queries.json
```

## 3. Smoke test the full pipeline (loads the model, 5 rows)

```powershell
python run_pipeline.py --limit 5 --show-gold --show-fol
```

This loads the model once, then runs translate → group → Z3 → vote → CoT and
prints each verdict plus a running accuracy. Detailed traces land in `Result\`.

## 4. Full run

```powershell
# All rows, print gold + accuracy, write predictions JSON.
python run_pipeline.py --show-gold --out Result\predictions.json
```

Outputs (all under `Result\`, sharing one timestamp):

| File | Contents |
| --- | --- |
| `run_llama_<ts>.txt` | per-record stage-by-stage summary + verdict |
| `run_llama_<ts>_model_io.txt` | raw model I/O: sentence→FOL, and name→clusters |
| `run_llama_<ts>_premises.txt` | canonicalized FOL + injected facts + Z3 program |
| `run_llama_<ts>_solver.txt` | verdict, proof core, reasoning, winning program |
| `predictions.json` (with `--out`) | machine-readable predictions |

## 5. Useful variations

```powershell
# Only Yes/No/Uncertain rows, or only MCQ rows:
python run_pipeline.py --only ynu --show-gold
python run_pipeline.py --only mcq --show-gold

# Process a window of rows (skip 100, take 50):
python run_pipeline.py --start 100 --limit 50 --show-gold

# Dump exactly what the model emits per record (for debugging):
python run_pipeline.py --limit 10 --dump-io Result\llama_io.jsonl --show-fol

# Self-consistency (K>1 sampling instead of greedy) — slower, more robust FOL:
python run_pipeline.py --k 5 --show-gold

# Beam search at K=1 (better single-shot FOL than greedy):
python run_pipeline.py --num-beams 5 --show-gold
```

## 6. Faster / leaner runs (ablations)

```powershell
# Symbolic-only: skip predicate grouping AND the CoT fallback (fastest).
python run_pipeline.py --no-group --no-cot --show-gold --show-fol

# Turn off individual repair stages to measure their contribution:
python run_pipeline.py --no-ground-goals  --show-gold   # bolt-on A off
python run_pipeline.py --no-type-facts    --show-gold   # bolt-on B off
python run_pipeline.py --no-deterministic-align --show-gold  # bolt-on D off
python run_pipeline.py --no-mcq-tiebreak  --show-gold   # abstain on MCQ ties
```

## 7. VRAM / precision

```powershell
# Default is 4-bit (~5.5 GB; needs ~8 GB VRAM with headroom).
# If you have lots of VRAM and want full precision:
python run_pipeline.py --precision bf16 --limit 5      # ~16 GB, Ampere+ for bf16
python run_pipeline.py --precision fp16 --limit 5      # ~16 GB, any CUDA GPU

# bnb 4-bit compute dtype (bfloat16 only on Ampere+):
python run_pipeline.py --compute-dtype bfloat16 --limit 5
```

## 8. Gold-FOL diagnostic (no model, no GPU)

Solve the dataset's annotated FOL straight through Z3 — the quickest way to see
the explanations/scoring without loading the model:

```powershell
python -m cli.explain_gold --data Logic_Based_Educational_Queries.json --out Result\gold_explanations.json
python -m cli.eval --data Logic_Based_Educational_Queries.json --pred Result\gold_explanations.json
```

---

### Full option list

```powershell
python run_pipeline.py --help
```
