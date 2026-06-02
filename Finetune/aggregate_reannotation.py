"""Aggregate workflow re-annotation output -> validate -> re-label -> write dataset.

Input: artifacts/reannot/results.json  (the workflow return value:
       {"batches": [{"batch": i, "records": [{record, premises_fol,
        questions:[{row, question_fol}]}]}]})

Steps per row:
  * assemble premises_fol (record-level) + question_fol (row-level)
  * structural validation: 1:1 with premises_nl, parses, no scope leak, the
    full program execs in the Z3 sandbox and yields a BoolRef goal
  * re-derive the sound Yes/No/Uncertain label via the solver
Writes:
  * data/annotation_ready_merged.reannotated.json  (corrected dataset + answer)
  * artifacts/reannot/validation.json              (per-row status + broken list)
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

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from finetune.fol_converter import (  # noqa: E402
    parse, collect_free_vars, convert_premises_to_z3py, FolParseError,
    Quant, Not, BinOp, Cmp, Arith, App,
)
from solver.z3_runner import run_yes_no_uncertain  # noqa: E402

DATA = FINETUNE_ROOT / "data" / "annotation_ready_merged.json"
REANNOT = FINETUNE_ROOT / "artifacts" / "reannot"
RAW = json.loads(DATA.read_text(encoding="utf-8"))


def _bound(node, acc):
    if isinstance(node, Quant):
        acc.update(node.vars); _bound(node.body, acc)
    elif isinstance(node, Not):
        _bound(node.body, acc)
    elif isinstance(node, (BinOp, Cmp, Arith)):
        _bound(node.left, acc); _bound(node.right, acc)
    elif isinstance(node, App):
        for a in node.args:
            _bound(a, acc)


def has_scope_leak(fol: str) -> bool:
    try:
        node = parse(fol)
    except FolParseError:
        return False
    fv = collect_free_vars(node); bv = set(); _bound(node, bv)
    return bool(fv & bv)


def sound_label(premises_fol, question_fol):
    setup, premises, goal, skipped = convert_premises_to_z3py(premises_fol, goal_fol=question_fol)
    if goal is None or not premises or len(skipped) > max(1, len(premises_fol) // 4):
        return "Uncertain", "non_fol_or_unconvertible"
    code = "\n".join(setup) + "\npremises = [\n" + ",\n".join(f"    {p}" for p in premises) + "\n]\ngoal = " + str(goal) + "\n"
    v = run_yes_no_uncertain(code)
    if v.status == "solved":
        return v.answer, "sound"
    return "Uncertain", f"unsolved_{v.status}"


def load_results():
    res = json.loads((REANNOT / "results.json").read_text(encoding="utf-8"))
    by_record = {}
    by_row = {}
    for b in res.get("batches", []):
        if b.get("failed"):
            continue
        for rec in b.get("records", []):
            by_record[rec["record"]] = rec["premises_fol"]
            for q in rec["questions"]:
                by_row[q["row"]] = q["question_fol"]
    return by_record, by_row


def main():
    by_record, by_row = load_results()
    out_rows = []
    report = {"rows": [], "broken": [], "changed_premise_sets": 0, "changed_questions": 0,
              "missing": [], "label_dist": {}}
    seen_records_changed = set()

    for row_idx, r in enumerate(RAW):
        rec = r["_source"]["record"]
        prem_nl = r.get("premises-NL", [])
        new_prem = by_record.get(rec)
        new_q = by_row.get(row_idx)
        if new_prem is None or new_q is None:
            report["missing"].append(row_idx)
            # keep original as fallback
            new_prem = r.get("premises-FOL", [])
            new_q = r.get("question-FOL", "")

        status = {"row": row_idx, "record": rec, "issues": []}
        if len(new_prem) != len(prem_nl):
            status["issues"].append(f"premise_count {len(new_prem)}!={len(prem_nl)}")
        for j, p in enumerate(new_prem):
            try:
                parse(p)
            except FolParseError as e:
                status["issues"].append(f"premise[{j}] parse_error: {str(e)[:40]}")
            if has_scope_leak(p):
                status["issues"].append(f"premise[{j}] scope_leak")
        q_parses = True
        try:
            parse(new_q)
        except FolParseError as e:
            q_parses = False
            status["issues"].append(f"question parse_error: {str(e)[:40]}")
        if q_parses and has_scope_leak(new_q):
            status["issues"].append("question scope_leak")

        label, lreason = sound_label(new_prem, new_q)
        status["label"] = label
        status["label_reason"] = lreason
        if lreason != "sound":
            status["issues"].append(f"label:{lreason}")

        if new_prem != r.get("premises-FOL", []) and rec not in seen_records_changed:
            seen_records_changed.add(rec)
        if new_q != r.get("question-FOL", ""):
            report["changed_questions"] += 1

        if status["issues"]:
            report["broken"].append(status)
        report["rows"].append(status)
        report["label_dist"][label] = report["label_dist"].get(label, 0) + 1

        nr = dict(r)
        nr["premises-FOL"] = new_prem
        nr["question-FOL"] = new_q
        nr["answer"] = label
        if r.get("premises-FOL") != new_prem or r.get("question-FOL") != new_q:
            nr["_fol_reannotated"] = True
        out_rows.append(nr)

    report["changed_premise_sets"] = len(seen_records_changed)
    out_path = FINETUNE_ROOT / "data" / "annotation_ready_merged.reannotated.json"
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (REANNOT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows: {len(out_rows)}")
    print(f"missing from workflow output: {len(report['missing'])} {report['missing'][:15]}")
    print(f"changed premise sets: {report['changed_premise_sets']}/411; changed questions: {report['changed_questions']}")
    print(f"rows with residual issues: {len(report['broken'])}")
    print(f"label distribution: {report['label_dist']}")
    print(f"wrote {out_path.name} + validation.json")


if __name__ == "__main__":
    main()
