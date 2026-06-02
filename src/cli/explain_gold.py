"""CLI: run the symbolic solver + explanation over a dataset's GOLD FOL.

Unlike ``cli.run`` (which calls the LLM translator), this path skips the model
entirely and feeds each record's annotated ``premises_fol`` / ``question_fol``
straight through the same FOL converter, Z3 solver, and explanation renderer
the pipeline uses. It is the quickest way to see the new explanations — the
unsat-core citations for Yes/No and the counter-model witness for Uncertain —
across every case in the dataset, with no GPU required.

Example:
    .venv/Scripts/python.exe -m cli.explain_gold \
        --data ../Finetune/data/annotation_ready_merged.json \
        --out ../Finetune/artifacts/gold_explanations.json

Run from the ``src`` directory, or from anywhere — the script puts both ``src``
and ``Finetune`` on the path itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "Finetune"))

from finetune.fol_converter import convert_premises_to_z3py  # noqa: E402
from finetune.load import load_records  # noqa: E402
from solver.z3_runner import run_yes_no_uncertain  # noqa: E402
from data.types import Translation  # noqa: E402
from explain import from_symbolic  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "Finetune" / "data" / "annotation_ready_merged.json"


def _build_code(record) -> str | None:
    """Convert a record's gold FOL into a runnable Z3-Python program, mirroring
    the production tolerance (drop <=25% unparseable premises; bail if the goal
    is not first-order)."""
    if not record.premises_fol or not record.question_fol:
        return None
    setup, premises, goal, skipped = convert_premises_to_z3py(
        record.premises_fol, goal_fol=record.question_fol
    )
    if len(skipped) > max(1, len(record.premises_fol) // 4):
        return None
    if not premises or goal is None:
        return None
    code = "\n".join(setup) + "\npremises = [\n"
    code += ",\n".join(f"    {p}" for p in premises) + "\n]\n"
    code += f"goal = {goal}\n"
    return code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset JSON")
    ap.add_argument("--out", type=Path, required=True, help="where to write explanations JSON")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (0 = all)")
    ap.add_argument("--timeout-ms", type=int, default=5000, help="per-check Z3 timeout")
    args = ap.parse_args()

    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]

    out: dict[str, dict] = {}
    answers = Counter()
    n_witness = n_core = n_unsupported = n_correct = n_graded = 0

    for r in records:
        code = _build_code(r)
        if code is None:
            n_unsupported += 1
            answers["Unsupported"] += 1
            out[r.id] = {"answer": None, "status": "unsupported_non_fol",
                         "groundtruth": r.answer}
            continue

        verdict = run_yes_no_uncertain(code, timeout_ms=args.timeout_ms)
        if verdict.status != "solved" or verdict.answer is None:
            answers[verdict.status] += 1
            out[r.id] = {"answer": None, "status": verdict.status,
                         "error": verdict.error, "groundtruth": r.answer}
            continue

        translation = Translation(code=code, raw_text="<gold-fol>", sample_index=0)
        final = from_symbolic(
            r, verdict.answer, confidence=1.0,
            unsat_core=verdict.unsat_core,
            winning_translation=translation,
            witness=verdict.witness,
        )
        answers[verdict.answer] += 1
        if verdict.unsat_core:
            n_core += 1
        if verdict.witness:
            n_witness += 1
        if r.answer is not None:
            n_graded += 1
            if final.answer == r.answer:
                n_correct += 1

        entry = final.model_dump()
        entry["groundtruth"] = r.answer
        entry["unsat_core"] = verdict.unsat_core
        out[r.id] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"records              : {len(records)}")
    print(f"by answer            : {dict(answers)}")
    print(f"with unsat-core cite : {n_core}")
    print(f"with witness (Uncert): {n_witness}")
    print(f"unsupported (non-FOL): {n_unsupported}")
    if n_graded:
        print(f"accuracy vs gold     : {n_correct}/{n_graded} = {n_correct / n_graded:.3f}")
    print(f"wrote                : {args.out}")


if __name__ == "__main__":
    main()
