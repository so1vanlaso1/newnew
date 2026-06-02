"""Prepare per-batch input files for the NL->FOL re-annotation workflow.

Groups the 1878 rows by their source record (premises are shared per record),
then writes small per-batch JSON files (BATCH records each) that workflow agents
read directly. Each batch file is self-contained: NL premises + current FOL +
the record's questions (row index, NL, current FOL).
"""
from __future__ import annotations

import json
from pathlib import Path

FINETUNE_ROOT = Path(__file__).resolve().parent
RAW = json.loads((FINETUNE_ROOT / "data" / "annotation_ready_merged.json").read_text(encoding="utf-8"))

BATCH = 4
OUT_DIR = FINETUNE_ROOT / "artifacts" / "reannot"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for old in OUT_DIR.glob("in_*.json"):
    old.unlink()

# Group rows by source record, preserving first-seen order.
records: dict[int, dict] = {}
order: list[int] = []
for row_idx, r in enumerate(RAW):
    rec = r["_source"]["record"]
    if rec not in records:
        records[rec] = {
            "record": rec,
            "premises_nl": r.get("premises-NL", []),
            "premises_fol_current": r.get("premises-FOL", []),
            "questions": [],
        }
        order.append(rec)
    records[rec]["questions"].append({
        "row": row_idx,
        "question_nl": r.get("question-NL", ""),
        "question_fol_current": r.get("question-FOL", ""),
    })

rec_list = [records[r] for r in order]
batches = [rec_list[i:i + BATCH] for i in range(0, len(rec_list), BATCH)]

for bi, batch in enumerate(batches):
    (OUT_DIR / f"in_{bi:03d}.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

# Manifest: how many batches, and the row->record map for reassembly.
manifest = {
    "n_batches": len(batches),
    "n_records": len(rec_list),
    "n_rows": len(RAW),
    "batch_dir": str(OUT_DIR),
    "row_to_record": [r["_source"]["record"] for r in RAW],
}
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

sizes = [len((OUT_DIR / f"in_{bi:03d}.json").read_text(encoding="utf-8")) for bi in range(len(batches))]
print(f"records: {len(rec_list)}  batches: {len(batches)} (BATCH={BATCH})")
print(f"batch file size: avg {sum(sizes)//len(sizes)} bytes, max {max(sizes)} bytes")
print(f"wrote {len(batches)} batch files + manifest to {OUT_DIR}")
