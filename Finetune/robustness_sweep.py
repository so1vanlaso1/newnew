"""Full-dataset robustness + determinism sweep.

For every row that yields a runnable first-order program, confirm:
  * the solver returns a definitive status in bounded time (no hang / no timeout),
  * an INDEPENDENT re-solve (fresh context) reproduces the same verdict
    (determinism + soundness of the entailment logic),
  * re-running the solver twice gives the identical verdict (no MBQI-order flake).

Reports any timeouts, slow rows, or mismatches.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

FINETUNE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINETUNE_ROOT.parent
sys.path.insert(0, str(FINETUNE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import z3  # noqa: E402
from finetune.fol_converter import convert_premises_to_z3py  # noqa: E402
from finetune.load import load_records  # noqa: E402
from solver.z3_runner import _exec_program, _make_solver, run_yes_no_uncertain  # noqa: E402

RECORDS = load_records(FINETUNE_ROOT / "data" / "annotation_ready_merged.json")


def build(rec):
    setup, prem, goal, skipped = convert_premises_to_z3py(rec.premises_fol or [], goal_fol=rec.question_fol)
    if not prem or goal is None:
        return None
    code = "\n".join(setup) + "\npremises = [\n" + ",\n".join("    " + p for p in prem) + "\n]\ngoal = " + str(goal) + "\n"
    return code


def independent(code):
    ns = _exec_program(code)
    prem, goal = ns["premises"], ns["goal"]
    ctx = goal.ctx

    def entails(g):
        s = _make_solver(ctx, 5000)
        for p in prem:
            s.add(p)
        s.add(z3.Not(g))
        return s.check() == z3.unsat

    pos, neg = entails(goal), entails(z3.Not(goal))
    if pos and neg:
        return "inconsistent"
    if pos:
        return "Yes"
    if neg:
        return "No"
    return "Uncertain"


def main():
    n_runnable = 0
    timeouts = []
    slow = []
    mismatches = []
    flaky = []
    max_ms = 0.0
    for idx, rec in enumerate(RECORDS):
        try:
            code = build(rec)
        except Exception:
            code = None
        if code is None:
            continue
        n_runnable += 1
        t0 = time.perf_counter()
        v1 = run_yes_no_uncertain(code)
        dt = (time.perf_counter() - t0) * 1000
        max_ms = max(max_ms, dt)
        if dt > 1500:
            slow.append((idx, round(dt)))
        if v1.status == "timeout":
            timeouts.append(idx)
            continue
        if v1.status != "solved":
            continue  # parse_error / inconsistent — definitive, not a hang
        # determinism: re-run
        v2 = run_yes_no_uncertain(code)
        if v2.answer != v1.answer:
            flaky.append((idx, v1.answer, v2.answer))
        # independent soundness
        try:
            indep = independent(code)
            if indep != v1.answer:
                mismatches.append((idx, v1.answer, indep))
        except Exception as e:
            mismatches.append((idx, v1.answer, f"err:{e}"))

    print(f"runnable first-order programs: {n_runnable} / {len(RECORDS)}")
    print(f"timeouts: {len(timeouts)} {timeouts[:20]}")
    print(f"flaky (non-deterministic) verdicts: {len(flaky)} {flaky[:20]}")
    print(f"independent-recheck mismatches: {len(mismatches)} {mismatches[:20]}")
    print(f"slowest solve: {round(max_ms)} ms; rows >1.5s: {len(slow)} {slow[:20]}")


if __name__ == "__main__":
    main()
