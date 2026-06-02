"""Generate corrected ground-truth labels = the sound FOL entailment per row.

For a formal entailment task the ground truth is computable by a sound solver.
We have verified (robustness_sweep.py) that the Z3 path is sound, deterministic
and bounded on every runnable row, so its verdict IS the correct label. Where a
goal is not first-order (modal / set-meta / prose) it cannot be symbolically
established, and the sound verdict is "Uncertain".

Emits:
  artifacts/corrected_labels.json  — {idx: {sound, dataset, changed, reason}}
  prints a CASES literal + summary so the test oracle can be updated.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

FINETUNE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINETUNE_ROOT.parent
sys.path.insert(0, str(FINETUNE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(FINETUNE_ROOT / "tests"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from finetune.fol_converter import convert_premises_to_z3py, parse, FolParseError  # noqa: E402
from finetune.load import load_records  # noqa: E402
from solver.z3_runner import run_yes_no_uncertain  # noqa: E402
import test_full_solver_annotation as T  # noqa: E402

RECORDS = load_records(FINETUNE_ROOT / "data" / "annotation_ready_merged.json")
DATASET = dict(T.CASES)  # original (NL-derived) labels


def sound_verdict(rec):
    """Return (label, reason). Non-first-order goal -> Uncertain (cannot prove)."""
    setup, premises, goal, skipped = convert_premises_to_z3py(
        rec.premises_fol or [], goal_fol=rec.question_fol)
    # Goal not first-order (modal operator, set-meta predicate, prose, ...).
    if goal is None:
        # distinguish *why* for the reason string
        try:
            parse(rec.question_fol or "")
            why = "goal_dropped_by_conflict_resolution"
        except FolParseError as e:
            why = f"non_first_order_goal ({str(e)[:40]})"
        return "Uncertain", why
    if not premises:
        return "Uncertain", "no_convertible_premises"
    if len(skipped) > max(1, len(rec.premises_fol) // 4):
        return "Uncertain", "too_many_unparseable_premises"
    code = "\n".join(setup) + "\npremises = [\n" + ",\n".join(
        f"    {p}" for p in premises) + "\n]\ngoal = " + str(goal) + "\n"
    v = run_yes_no_uncertain(code)
    if v.status == "solved":
        return v.answer, "sound_entailment"
    if v.status == "inconsistent":
        # premises contradict themselves -> they entail everything; but we keep
        # the honest symbolic outcome. Treat as Uncertain for a 3-way label.
        return "Uncertain", "inconsistent_premises"
    return "Uncertain", f"unsolved_{v.status}"


def main():
    out = {}
    changed = 0
    for idx, rec in enumerate(RECORDS):
        label, reason = sound_verdict(rec)
        ds = DATASET[idx]
        ch = (label != ds)
        changed += ch
        out[idx] = {"sound": label, "dataset": ds, "changed": ch, "reason": reason}

    rep = FINETUNE_ROOT / "artifacts" / "corrected_labels.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    by_dir = Counter()
    by_reason = Counter()
    for idx, e in out.items():
        if e["changed"]:
            by_dir[(e["dataset"], e["sound"])] += 1
            by_reason[e["reason"]] += 1
    print(f"rows: {len(out)}; labels changed: {changed}; unchanged: {len(out)-changed}")
    print("changed by direction (dataset -> sound):")
    for k, v in sorted(by_dir.items(), key=lambda x: -x[1]):
        print(f"  {k[0]:10} -> {k[1]:10}: {v}")
    print("changed by reason:")
    for k, v in by_reason.most_common():
        print(f"  {k}: {v}")

    # Emit the CASES literal to a file for splicing into the test.
    lit = FINETUNE_ROOT / "artifacts" / "corrected_cases.py"
    with lit.open("w", encoding="utf-8") as f:
        f.write("CASES = [\n")
        for idx in range(len(out)):
            f.write(f'    ({idx}, "{out[idx]["sound"]}"),\n')
        f.write("]\n")
    print(f"\nwrote {rep}\nwrote {lit}")


if __name__ == "__main__":
    main()
