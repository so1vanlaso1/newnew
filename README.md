# EXACT 2026 Track 1 — Logic QA Pipeline (Llama NL→FOL → Z3)

Neuro-symbolic pipeline for the **EXACT 2026 (IJCNN) — Track 1: Logic-Based
Educational Queries** task. It translates each natural-language premise/question
to **first-order logic** with **one** model, bridges that FOL into a **Z3** program,
solves for entailment, votes, and falls back to chain-of-thought when the symbolic
path is inconclusive.

**The one model:** [`fvossel/Llama-3.1-8B-Instruct-nl-to-fol`](https://huggingface.co/fvossel/Llama-3.1-8B-Instruct-nl-to-fol)
— a LoRA adapter on the gated [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
base, loaded in **4-bit (NF4)**. The same resident model serves three roles; the
last two run on the base chat model with the adapter disabled, so there is no
second model to load:

| Role | Adapter | What it does |
| --- | --- | --- |
| 1. translate | **on** | one NL sentence → one FOL formula (`𝜙=∀x (…)`) |
| 2. group | off | cluster synonymous predicate names (base chat) |
| 3. CoT fallback | off | reason in NL when symbolic fails (base chat) |

## Pipeline

```
NL premises + Q  ──►  Llama+adapter (per sentence → FOL)
                          │
                          ▼
                  predicate grouping (Llama base chat clusters synonym names)
                          │
                          ▼
                  deterministic FOL repair (re-ground goal, assert sort guards,
                  align typo-twin predicate names)  →  assemble Z3 program
                          │
                          ▼
                  safe-exec → Z3 entailment  (MCQ → "Unknown" if no option follows)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   a proof was found              no proof / open-ended
   (high confidence)                      │
            │                             ▼
            │                    CoT fallback (Llama base chat, K=5)
            └──────────────┬──────────────┘
                           ▼
        {answer, explanation, fol, cot, premises, confidence}
```

## Quick start (Windows + NVIDIA GPU)

The base model is **gated**: request access on its
[HF page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), get a
[token](https://huggingface.co/settings/tokens), then:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
.\quickstart.ps1            # venv + CUDA torch + deps + HF login + model download
```

`quickstart.ps1` pins PyTorch CUDA wheels (default `cu124`; pass `-CudaWheel cu121`
for an older driver), installs everything, and downloads both the base and the
adapter into the HF cache. 4-bit needs roughly **8 GB of VRAM** with headroom.

Then run the pipeline — see **[RUN_COMMANDS.md](RUN_COMMANDS.md)** for every command.
The shortest path:

```powershell
.\.venv\Scripts\Activate.ps1
python run_pipeline.py --limit 5 --show-gold --show-fol   # smoke test
python run_pipeline.py --show-gold --out Result\predictions.json   # full run
```

## Project layout

```
run_pipeline.py               THE runner: load model → translate → group →
                              assemble → Z3 → vote → CoT → write Result/*.txt
quickstart.ps1                one-shot Windows installer (deps + HF + model)
RUN_COMMANDS.md               every command to run the pipeline
Logic_Based_Educational_Queries.json   the dataset
src/
  data/load.py                dataset JSON → Record (schema-flexible aliases)
  data/types.py               Record / Translation / SolverVerdict / FinalAnswer
  translator/llama_fol.py     the model: base(4-bit)+adapter; NL→FOL + chat backend,
                              FOL normalization, and FOL→Z3 assembly
  translator/fol_converter.py Unicode/Pythonic FOL → Z3 Python DSL
  translator/fol_repair.py    bolt-ons A/B: re-ground goal, assert free sort guards
  translator/predicate_group.py  bolt-on D: align synonymous predicate names
  translator/parse.py         goal-expression extraction
  translator/repair.py        auto-declare undeclared symbols in a Z3 program
  solver/z3_runner.py         AST-validated safe exec → Z3 → Yes/No/Uncertain/MCQ
  vote.py                     majority vote with conservative confidence
  fallback/cot.py             CoT fallback with self-consistency (supports Unknown)
  explain.py                  unsat-core / CoT → FinalAnswer
  pipeline.py                 end-to-end orchestration with per-stage timing
  eval/score.py               accuracy + per-answer-type breakdown
  cli/eval.py                 score a FinalAnswer-format predictions file
  cli/inspect.py              inspect a dataset file (fields + answer-type histogram)
  cli/explain_gold.py         solve the dataset's GOLD FOL straight through Z3 (no model)
tests/                        pytest CPU tests (no GPU/model download needed)
Finetune/                     (separate) self-contained LoRA training harness — not
                              needed to run the pipeline; kept for reference
```

## How the FOL→Z3 bridge works

The adapter emits one FOL formula per sentence using Unicode operators
(`∀ ∃ ¬ ∧ ∨ → ↔ ⊕`), prefixed with `𝜙=`. The backend peels the marker; then:

1. **normalize** the surface form (`translator.llama_fol.normalize_fol`),
2. **group** synonymous predicate names so a premise's `EligibleForScholarship`
   and the goal's `QualifyForUniversityScholarship` become one relation,
3. **repair** deterministically (`translator.fol_repair`): re-ground a goal the
   model over-generalized into a universal rule, and assert free sort guards
   (`Student(sophia)`) so gated rules can fire,
4. **convert** to a Z3 Python program (`translator.fol_converter`) and **solve**.

Every rename/repair is done by deterministic code; the model only proposes
clusters, so the symbolic guarantees hold. All of these can be ablated with CLI
flags (`--no-group`, `--no-ground-goals`, `--no-type-facts`,
`--no-deterministic-align`, `--no-mcq-tiebreak`, `--no-cot`).

## Testing

```powershell
pytest tests\ -v
```

The tests cover the FOL converter, the FOL→Z3 assembly + repair bolt-ons, the
predicate grouping, the Z3 wrapper, the voter, and the adapter/base compatibility
guard — all on CPU with **no model download**.

## Troubleshooting

- **`OSError: ... meta-llama/Llama-3.1-8B-Instruct is gated`** — request access on
  the model's HF page, then set a valid `$env:HF_TOKEN` (or run
  `huggingface-cli login`) and re-run `quickstart.ps1`.
- **`CUDA out of memory`** — 4-bit needs ~8 GB. Close other GPU apps, or lower
  throughput with `--batch-size 1` (default). Full-precision (`--precision bf16`)
  needs ~16 GB.
- **`bitsandbytes` import / 4-bit fails on Windows** — ensure `bitsandbytes>=0.44`
  and a CUDA-enabled torch (`python -c "import torch; print(torch.cuda.is_available())"`
  must print `True`).
- **`No module named torch`** — you're outside the venv. Run
  `.\.venv\Scripts\Activate.ps1`.
- **Z3 returns `unknown` / wrong on many rows** — inspect what the model emitted
  with `--dump-io Result\llama_io.jsonl --limit 5 --show-fol` and check the FOL.
- **Loader returns 0 records / wrong fields** — run
  `python -m cli.inspect <path>` and update `FIELD_ALIASES` in `src/data/load.py`.
