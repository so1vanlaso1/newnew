"""Triage NL<->FOL quality across annotation_ready_merged.

Reports scale (distinct premise sets / questions), NL/FOL length alignment, and
structural FOL issues (parse failure, non-first-order, scope leak). Dumps a
sample of NL<->FOL pairs for eyeballing faithfulness.
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

FINETUNE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FINETUNE_ROOT.parent
sys.path.insert(0, str(FINETUNE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from finetune.fol_converter import (  # noqa: E402
    parse, collect_free_vars, FolParseError, Quant, Not, BinOp, Cmp, Arith, App,
)

RAW = json.loads((FINETUNE_ROOT / "data" / "annotation_ready_merged.json").read_text(encoding="utf-8"))


def bound_vars(node, acc):
    if isinstance(node, Quant):
        acc.update(node.vars); bound_vars(node.body, acc)
    elif isinstance(node, Not):
        bound_vars(node.body, acc)
    elif isinstance(node, (BinOp, Cmp, Arith)):
        bound_vars(node.left, acc); bound_vars(node.right, acc)
    elif isinstance(node, App):
        for a in node.args:
            bound_vars(a, acc)


def classify(fol):
    try:
        node = parse(fol)
    except FolParseError as e:
        return "parse_error", str(e)[:50]
    fv = collect_free_vars(node)
    bv = set(); bound_vars(node, bv)
    if fv & bv:
        return "scope_leak", sorted(fv & bv)
    return "ok", None


def main():
    n_rows = len(RAW)
    distinct_records = {r["_source"]["record"] for r in RAW}
    # distinct premise sets (by record) and distinct question-FOL strings
    distinct_qfol = {r.get("question-FOL", "") for r in RAW}
    distinct_premise_sets = {}
    for r in RAW:
        rec = r["_source"]["record"]
        distinct_premise_sets[rec] = tuple(r.get("premises-FOL", []))

    print(f"rows: {n_rows}")
    print(f"distinct source records (premise sets): {len(distinct_records)}")
    print(f"distinct question-FOL strings: {len(distinct_qfol)}")

    # NL/FOL length alignment per row
    misaligned = [i for i, r in enumerate(RAW)
                  if len(r.get('premises-NL', [])) != len(r.get('premises-FOL', []))]
    print(f"rows with premises NL/FOL length MISMATCH: {len(misaligned)} {misaligned[:15]}")

    # Structural triage of all premise FOL (dedup by premise-set) and question FOL
    prem_flags = Counter()
    prem_examples = {}
    seen_prem = set()
    for r in RAW:
        for p in r.get("premises-FOL", []):
            if p in seen_prem:
                continue
            seen_prem.add(p)
            kind, info = classify(p)
            prem_flags[kind] += 1
            if kind != "ok" and kind not in prem_examples:
                prem_examples[kind] = (p, info)
    print(f"\ndistinct premise formulas: {sum(prem_flags.values())}")
    for k, v in prem_flags.most_common():
        print(f"  premise {k}: {v}")

    q_flags = Counter()
    for q in distinct_qfol:
        kind, _ = classify(q)
        q_flags[kind] += 1
    print(f"\ndistinct question formulas: {sum(q_flags.values())}")
    for k, v in q_flags.most_common():
        print(f"  question {k}: {v}")

    # Sample NL<->FOL pairs for eyeballing faithfulness (premises of record 0 + a few questions)
    print("\n=== SAMPLE premise NL<->FOL (record 0) ===")
    r0 = RAW[0]
    for nl, fol in zip(r0["premises-NL"], r0["premises-FOL"]):
        print(f"  NL : {nl}")
        print(f"  FOL: {fol}\n")
    print("=== SAMPLE question NL<->FOL (rows 0,1,2,300,600) ===")
    for i in [0, 1, 2, 300, 600, 900, 1200]:
        if i < n_rows:
            print(f"  [{i}] NL : {RAW[i]['question-NL']}")
            print(f"       FOL: {RAW[i]['question-FOL']}\n")


if __name__ == "__main__":
    main()
