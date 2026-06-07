#!/usr/bin/env python
"""End-to-end neuro-symbolic runner for EXACT 2026 Track 1 — single Llama pipeline.

ONE model does everything: the fvossel/Llama-3.1-8B-Instruct-nl-to-fol LoRA
adapter on top of meta-llama/Llama-3.1-8B-Instruct, loaded in 4-bit. The same
resident model serves three roles (the base chat model is used with the adapter
disabled for the last two):

    row ─► Llama+adapter   (NL premises + question/options → one FOL formula each)
        ─► predicate grouping (Llama base chat: cluster synonymous predicate names)
        ─► deterministic FOL repair + canonicalize → assemble Z3 program
        ─► Z3 entailment solver (one solve per candidate)
        ─► majority vote over the verdicts
        ─► CoT fallback on the Llama base chat model (only if symbolic is inconclusive)
        ─► Yes / No / Uncertain   or   the winning MCQ option

The answer type is decided structurally from the QUESTION alone (lettered options
→ MCQ, else Yes/No/Uncertain). The gold answer is loaded separately and used
*only* for the optional accuracy read-out — it never reaches the model or solver.

Examples
--------
    # Full run (4-bit, predicate grouping on, CoT fallback on):
    python run_pipeline.py --limit 20 --show-gold

    # Faster ablation: skip grouping and CoT (symbolic-only):
    python run_pipeline.py --limit 20 --no-group --no-cot --show-gold --show-fol

    # Inspect what the model emits per record:
    python run_pipeline.py --limit 5 --dump-io llama_io.jsonl --show-fol
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data.load import load_records                       # noqa: E402
from data.types import AnswerType, Record                # noqa: E402
from fallback.cot import CotConfig                        # noqa: E402
from pipeline import PipelineConfig, process_record       # noqa: E402

# ── Defaults (override on the CLI) ──────────────────────────────────────────
DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"
# The single model: the fvossel NL→FOL LoRA on the gated Llama-3.1-8B base.
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol"


# ── Input gating: the pipeline may see premises-NL + question only ─────────
def gate_inputs(r: Record) -> tuple[Record, str | None]:
    """Return (pipeline_record, gold_answer).

    The returned record carries ONLY the natural-language premises, the question,
    and any options parsed out of the question. The gold answer, gold FOL, and
    stored explanation are stripped so they can't influence the verdict; the gold
    answer is handed back separately for scoring/display. Answer type is re-derived
    from question structure (options present → MCQ, else Yes/No/Uncertain)."""
    atype = AnswerType.MCQ if r.options else AnswerType.YES_NO_UNCERTAIN
    gated = Record(
        id=r.id,
        premises_nl=r.premises_nl,
        premises_fol=None,
        question_nl=r.question_nl,
        question_fol=None,
        answer_type=atype,
        answer=None,
        options=r.options,
        raw={},
    )
    return gated, r.answer


class _PrecomputedTranslator:
    """A `Translator`-shaped stand-in returning translations computed up front by
    the runner. `backend` is the live Llama chat backend, so the pipeline can run
    the CoT fallback on the base model (adapter disabled)."""

    def __init__(self, by_id: dict[str, list], backend: object):
        self._by_id = by_id
        self.backend = backend

    def translate(self, record):  # noqa: ANN001
        return self._by_id.get(record.id, [])


def _build_config(args: argparse.Namespace):
    from translator.llama_fol import LlamaFolConfig

    return LlamaFolConfig(
        base_model=str(args.base_model),
        adapter=str(args.adapter),
        precision=args.precision,
        compute_dtype=args.compute_dtype,
        k_samples=args.k,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_len=args.max_input_len,
        temperature=args.temperature,
        top_p=args.top_p,
        dump_io_path=str(args.dump_io) if args.dump_io else None,
        ground_goals=not args.no_ground_goals,
        assert_type_facts=not args.no_type_facts,
        type_guard_min_rules=args.type_guard_min_rules,
    )


def translate_and_group(args: argparse.Namespace, gated: list, backend) -> tuple[dict, dict]:
    """Phase A (translate) → Phase B (group) → Phase C (assemble), all on ONE
    resident model. Returns ({record.id: list[list[Translation]]}, trace)."""
    from translator.llama_fol import LlamaFolTranslator, normalize_fol, assemble_translations
    from translator.predicate_group import apply_canonical, group_relations_debug, safe_canonical_map
    from translator.fol_repair import add_type_facts, ground_goal

    cfg = _build_config(args)
    records = [rec for rec, _gold in gated]
    translator = LlamaFolTranslator(backend, cfg)

    # ── Phase A: Llama+adapter → raw FOL ────────────────────────────────────
    print(f"[llama] phase A: translating {len(records)} records NL->FOL "
          f"(k={cfg.k_samples})...")
    raw = {rec.id: translator.translate_to_fol(rec) for rec in records}

    # ── Phase B: Llama base chat → synonymous predicate-name CLUSTERS ───────
    llm_clusters_by_id: dict[str, list[list[str]]] = {rid: [] for rid in raw}
    model_io: dict[str, dict] = {rid: {} for rid in raw}
    if not args.no_group:
        print("[llama] phase B: grouping predicate names (base chat, adapter off)...")

        def chat(messages: list[dict]) -> str:
            # lora_path=None → adapter disabled → base Llama-3.1-8B-Instruct chat.
            return backend.chat_generate([messages], 1, 0.0, 1.0, 512, None)[0][0]

        for rec in records:
            rfol = raw[rec.id]
            all_fol = [normalize_fol(s) for s in rfol.premises_fol]
            for _label, cands in rfol.goals:
                all_fol += [normalize_fol(s) for s in cands]
            try:
                dbg = group_relations_debug(all_fol, chat)
            except Exception as e:  # never let one record kill the batch
                print(f"[llama] grouping failed for {rec.id}: {e}")
                dbg = {"names": [], "raw_response": f"<error: {e}>", "clusters": [], "mapping": {}}
            llm_clusters_by_id[rec.id] = dbg["clusters"]
            model_io[rec.id] = dbg
            if dbg["clusters"]:
                print(f"[llama]   {rec.id}: proposed {len(dbg['clusters'])} cluster(s)")

    # ── Phase C: canonicalize + assemble (no model) ────────────────────────
    print("[llama] phase C: canonicalizing names + assembling Z3 programs...")
    by_id: dict[str, list] = {}
    trace: dict[str, dict] = {}
    for rec in records:
        rfol = raw[rec.id]
        all_fol = [normalize_fol(s) for s in rfol.premises_fol]
        for _label, cands in rfol.goals:
            all_fol += [normalize_fol(s) for s in cands]
        m = safe_canonical_map(
            all_fol,
            llm_clusters_by_id.get(rec.id, []),
            deterministic=not args.no_deterministic_align,
        )
        prem = [apply_canonical(normalize_fol(s), m) for s in rfol.premises_fol]
        goals = [
            (label, [apply_canonical(normalize_fol(s), m) for s in cands])
            for label, cands in rfol.goals
        ]
        translations = assemble_translations(
            rfol.answer_type, prem, goals, cfg.max_skip_fraction,
            ground_goals=cfg.ground_goals,
            assert_type_facts=cfg.assert_type_facts,
            type_guard_min_rules=cfg.type_guard_min_rules,
        )
        by_id[rec.id] = translations
        # Human-readable view of the same deterministic transforms (bolt-ons A/B).
        grounded = [
            (label, [ground_goal(c, prem) if cfg.ground_goals else c for c in cands])
            for label, cands in goals
        ]
        rep_goal = next((c for _l, cs in grounded for c in cs), None)
        typed_prem = (
            add_type_facts(prem, rep_goal, min_rules=cfg.type_guard_min_rules)
            if cfg.assert_type_facts else prem
        )
        type_facts_added = [p for p in typed_prem if p not in prem]
        trace[rec.id] = {
            "answer_type": rfol.answer_type.value,
            "premises_nl": list(rec.premises_nl),
            "premises_fol": list(rfol.premises_fol),   # raw model output
            "question_nl": rec.question_nl,
            "options": list(rec.options or []),
            "goals": rfol.goals,                       # list[(label, [raw_fol,...])]
            "merged": m,                               # {old_name: canonical_name}
            "canon_premises": prem,
            "canon_goals": goals,
            "grounded_goals": grounded,
            "type_facts_added": type_facts_added,
            "model_io": model_io.get(rec.id, {}),
            "program": (translations[0][0].code if translations and translations[0] else None),
            "n_programs": sum(len(g) for g in translations),
        }
    return by_id, trace


def _result_dir() -> Path:
    d = ROOT / "Result"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_summary(
    args: argparse.Namespace, gated: list, finals: dict, trace: dict,
    elapsed_s: float, n_correct: int, n_scored: int, stamp: str,
) -> Path:
    """Human-readable per-stage summary of this run into Result/."""
    path = _result_dir() / f"run_llama_{stamp}.txt"

    def _wrap(s: str, n: int = 100) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"Run summary — {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"base={args.base_model}  adapter={args.adapter}")
    lines.append(f"precision={args.precision}  k={args.k}  beams={args.num_beams}")
    grouping = "off (--no-group)" if args.no_group else "on (Llama base chat)"
    cot = "off (--no-cot)" if args.no_cot else "on (Llama base chat)"
    lines.append(f"grouping={grouping}   cot_fallback={cot}")
    acc = f"{n_correct}/{n_scored} = {n_correct / n_scored:.1%}" if n_scored else "n/a (no --show-gold)"
    lines.append(f"records={len(gated)}   accuracy={acc}   elapsed={elapsed_s:.1f}s")
    lines.append("=" * 78)

    for i, (rec, gold) in enumerate(gated, 1):
        final = finals.get(rec.id)
        tr = trace.get(rec.id, {})
        kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YNU"
        lines.append("")
        lines.append(f"### [{i}/{len(gated)}] {rec.id}   [{kind}]")
        lines.append(f"Q: {_wrap(rec.question_nl, 120)}")

        if tr:
            lines.append("-- PHASE A: Llama+adapter  NL → FOL --")
            for j, (nl, fol) in enumerate(zip(tr["premises_nl"], tr["premises_fol"])):
                lines.append(f"  P{j}: {_wrap(nl, 80)}")
                lines.append(f"      → {fol}")
            for label, cands in tr["goals"]:
                tag = "goal" if label == "__goal__" else f"option {label!r}"
                lines.append(f"  {tag}:")
                for c in cands:
                    lines.append(f"      → {c}")

            lines.append("-- PHASE B: predicate grouping --")
            merged = tr.get("merged") or {}
            if merged:
                lines.append(f"  merged {len(merged)} name(s):")
                for old, new in sorted(merged.items()):
                    lines.append(f"      {old}  →  {new}")
            else:
                lines.append("  (no synonymous names merged)")

            lines.append("-- PHASE C: canonicalize + solve --")
            lines.append(f"  assembled Z3 programs: {tr.get('n_programs', 0)}")

        if final is not None:
            source = final.debug.get("source", "?")
            verdict = f"  predicted: {final.answer!r}  (confidence {final.confidence:.2f}, via {source})"
            lines.append(verdict if tr else f"-- VERDICT --\n{verdict}")
            if gold is not None:
                ok = _answers_match(final.answer, gold, rec)
                lines.append(f"  gold:      {gold!r}   [{'CORRECT' if ok else 'WRONG'}]")
            if final.explanation:
                lines.append(f"  why: {_wrap(final.explanation, 160)}")
            if final.fol:
                lines.append("  Z3 program (winning):")
                for prog in final.fol:
                    lines.append("      " + prog.replace("\n", "\n      "))
        else:
            lines.append("  (no verdict)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_io(args: argparse.Namespace, gated: list, trace: dict, stamp: str) -> Path:
    """File 1: the raw INPUT/OUTPUT of the model in both roles, per record.
       Role 1 = Llama+adapter (sentence → FOL). Role 2 = Llama base (names → clusters)."""
    from translator.llama_fol import SYSTEM_PROMPT

    path = _result_dir() / f"run_llama_{stamp}_model_io.txt"
    lines = [f"MODEL I/O — {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"base={args.base_model}  adapter={args.adapter}  "
             f"grouping={'off' if args.no_group else 'Llama base chat'}",
             "=" * 78,
             "", f"[system prompt for role 1] {SYSTEM_PROMPT}"]
    for i, (rec, _gold) in enumerate(gated, 1):
        tr = trace.get(rec.id)
        if not tr:
            continue
        lines.append(f"\n### [{i}] {rec.id}")
        lines.append("\n-- ROLE 1: Llama+adapter  (user sentence → FOL) --")
        for j, (nl, fol) in enumerate(zip(tr["premises_nl"], tr["premises_fol"])):
            lines.append(f"  premise[{j}]")
            lines.append(f"    IN : {nl}")
            lines.append(f"    OUT: {fol}")
        for label, cands in tr["goals"]:
            tag = "goal" if label == "__goal__" else f"option {label!r}"
            inp = tr["question_nl"] if label == "__goal__" else label.strip().rstrip(".").strip()
            lines.append(f"  {tag}")
            lines.append(f"    IN : {inp}")
            for c in cands:
                lines.append(f"    OUT: {c}")
        io = tr.get("model_io") or {}
        lines.append("\n-- ROLE 2: Llama base  (predicate-name grouping) --")
        if args.no_group:
            lines.append("  (grouping disabled with --no-group)")
        elif not io:
            lines.append("  (no relation names to group)")
        else:
            lines.append(f"  IN  (relation names): {io.get('names', [])}")
            lines.append(f"  OUT (raw response)  : {(io.get('raw_response') or '').strip()}")
            lines.append(f"  PARSED clusters     : {io.get('clusters', [])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_premises_file(args: argparse.Namespace, gated: list, trace: dict, stamp: str) -> Path:
    """File 2: the COMPLETED (canonicalized) premises + goal and the Z3 program."""
    path = _result_dir() / f"run_llama_{stamp}_premises.txt"
    lines = [f"COMPLETED PREMISES (after grouping/canonicalization) — {datetime.now():%Y-%m-%d %H:%M:%S}",
             "=" * 78]
    for i, (rec, _gold) in enumerate(gated, 1):
        tr = trace.get(rec.id)
        if not tr:
            continue
        lines.append(f"\n### [{i}] {rec.id}   [{tr['answer_type']}]")
        lines.append("premises (final FOL):")
        for j, p in enumerate(tr.get("canon_premises", [])):
            lines.append(f"  [{j}] {p}")
        lines.append("goal(s) (final FOL):")
        for label, cands in tr.get("canon_goals", []):
            tag = "goal" if label == "__goal__" else f"option {label!r}"
            for c in cands:
                lines.append(f"  {tag}: {c}")
        added = tr.get("type_facts_added") or []
        if added:
            lines.append("type facts injected (bolt-on B):")
            for f in added:
                lines.append(f"  + {f}")
        grounded = tr.get("grounded_goals") or []
        if grounded:
            lines.append("re-grounded goal(s) (bolt-on A):")
            for label, cands in grounded:
                tag = "goal" if label == "__goal__" else f"option {label!r}"
                for c in cands:
                    lines.append(f"  {tag}: {c}")
        if tr.get("program"):
            lines.append("assembled Z3 program (sample 0):")
            lines.append("    " + tr["program"].replace("\n", "\n    "))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_solver_file(args: argparse.Namespace, gated: list, finals: dict, trace: dict, stamp: str) -> Path:
    """File 3: how the solver reached each answer — verdict, proof-core premises,
    explanation, and the winning Z3 program."""
    path = _result_dir() / f"run_llama_{stamp}_solver.txt"
    lines = [f"SOLVER TRACE — {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 78]
    for i, (rec, gold) in enumerate(gated, 1):
        final = finals.get(rec.id)
        tr = trace.get(rec.id, {})
        lines.append(f"\n### [{i}] {rec.id}   [{'MCQ' if rec.answer_type == AnswerType.MCQ else 'YNU'}]")
        lines.append(f"question: {rec.question_nl}")
        if final is None:
            lines.append("  (no verdict)")
            continue
        lines.append(f"verdict: {final.answer!r}  (confidence {final.confidence:.2f}, "
                     f"via {final.debug.get('source', '?')})")
        if gold is not None:
            lines.append(f"gold:    {gold!r}   [{'CORRECT' if _answers_match(final.answer, gold, rec) else 'WRONG'}]")
        core = final.premises or []
        canon = tr.get("canon_premises", [])
        if core:
            lines.append("premises used in the proof (unsat core):")
            for idx in core:
                fol = canon[idx] if idx < len(canon) else "?"
                lines.append(f"  [{idx}] {fol}")
        else:
            lines.append("premises used in the proof: (none reported — e.g. Uncertain / Unknown / CoT)")
        if final.explanation:
            lines.append(f"reasoning: {final.explanation}")
        if final.fol:
            lines.append("winning Z3 program:")
            for prog in final.fol:
                lines.append("    " + prog.replace("\n", "\n    "))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── Main loop ────────────────────────────────────────────────────────────
def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so Unicode FOL symbols in
    # printed verdicts don't crash on redirect.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="release JSON to read rows from")
    ap.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                    help="HF repo id or local dir of the GATED Llama-3.1-8B-Instruct base")
    ap.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER,
                    help="HF repo id or local dir of the fvossel NL→FOL LoRA adapter")
    ap.add_argument("--precision", choices=["4bit", "bf16", "fp16"], default="4bit",
                    help="4bit (NF4, ~5.5 GB, default) | bf16 | fp16 (full weights, ~16 GB)")
    ap.add_argument("--compute-dtype", choices=["float16", "bfloat16"], default="float16",
                    help="bnb 4-bit compute / load dtype; float16 is universal, bfloat16 on Ampere+")
    ap.add_argument("--k", type=int, default=1,
                    help="self-consistency samples per translate (1 = deterministic greedy/beam)")
    ap.add_argument("--num-beams", type=int, default=1,
                    help="beam count when k=1 (1 = greedy); raise for better single-shot FOL")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="sentences decoded per generate() call (left-padded)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="max FOL decode length")
    ap.add_argument("--max-input-len", type=int, default=512, help="max input tokens per sentence")
    ap.add_argument("--temperature", type=float, default=0.7, help="(k>1 sampling) temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="(k>1 sampling) nucleus top-p")
    ap.add_argument("--dump-io", type=Path, default=None,
                    help="append each record's NL→FOL trace as JSON lines to this file")
    # Pipeline-shape toggles (all preserve the same downstream behavior by default).
    ap.add_argument("--no-group", action="store_true",
                    help="skip the predicate-name grouping phase (synonyms stay unmerged)")
    ap.add_argument("--no-cot", action="store_true",
                    help="disable the CoT fallback (symbolic-only)")
    ap.add_argument("--no-ground-goals", action="store_true",
                    help="disable bolt-on A (goal re-grounding); ablation")
    ap.add_argument("--no-type-facts", action="store_true",
                    help="disable bolt-on B (free sort-guard assertion); ablation")
    ap.add_argument("--type-guard-min-rules", type=int, default=2,
                    help="min #rules before bolt-on B's sort-guard heuristic fires")
    ap.add_argument("--no-deterministic-align", action="store_true",
                    help="disable bolt-on D's deterministic prefix/edit-distance matching; ablation")
    ap.add_argument("--no-mcq-tiebreak", action="store_true",
                    help="when >1 MCQ option is entailed, abstain ('Unknown') instead of "
                         "picking the smallest-proof option; ablation")
    ap.add_argument("--cot-k", type=int, default=5, help="CoT self-consistency samples")
    ap.add_argument("--wall-clock-budget-s", type=float, default=600.0,
                    help="per-record wall-clock budget (high by default; an 8B model is slow)")
    ap.add_argument("--solver-timeout-ms", type=int, default=5000, help="per Z3 call timeout")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N rows")
    ap.add_argument("--only", choices=["ynu", "mcq", "all"], default="all", help="filter by task type")
    ap.add_argument("--show-gold", action="store_true", help="print gold answer + running accuracy")
    ap.add_argument("--show-fol", action="store_true", help="print the winning Z3 program")
    ap.add_argument("--out", type=Path, default=None, help="optional path to write predictions JSON")
    args = ap.parse_args()

    # ── Load + gate rows ────────────────────────────────────────────────
    records = load_records(args.data)
    gated = [gate_inputs(r) for r in records]
    if args.only == "ynu":
        gated = [g for g in gated if g[0].answer_type == AnswerType.YES_NO_UNCERTAIN]
    elif args.only == "mcq":
        gated = [g for g in gated if g[0].answer_type == AnswerType.MCQ]
    gated = gated[args.start:]
    if args.limit:
        gated = gated[: args.limit]
    print(f"[info] loaded {len(records)} rows from {args.data.name}; processing {len(gated)}")

    # ── Load the single model, then run translate → group → assemble ─────
    from translator.llama_fol import LlamaFolBackend

    backend = LlamaFolBackend(_build_config(args))
    by_id, trace = translate_and_group(args, gated, backend)
    translator = _PrecomputedTranslator(by_id, backend)

    pcfg = PipelineConfig(
        enable_cot_fallback=not args.no_cot,
        cot=CotConfig(k_samples=args.cot_k),
        # A single solved Z3 proof IS a proof → high confidence (thresholds=1).
        # CoT then only fires when symbolic produced nothing.
        vote_high_threshold=1,
        vote_medium_threshold=1,
        wall_clock_budget_s=args.wall_clock_budget_s,
        solver_timeout_ms=args.solver_timeout_ms,
        # Forward-chaining records routinely entail several MCQ options along one
        # chain; pick the most directly supported (smallest proof) unless ablated.
        mcq_tiebreak_smallest_core=not args.no_mcq_tiebreak,
    )

    predictions: dict[str, dict] = {}
    finals: dict[str, object] = {}
    n_correct = 0
    n_scored = 0
    t_start = time.perf_counter()

    for i, (rec, gold) in enumerate(gated, 1):
        final, timings = process_record(rec, translator, pcfg)
        finals[rec.id] = final
        source = final.debug.get("source", "?")
        kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YNU"

        line = (
            f"[{i:>4}/{len(gated)}] {rec.id:<16} {kind} "
            f"=> {final.answer!r}  (conf={final.confidence:.2f}, via={source}, {timings.total_s:.1f}s)"
        )
        if args.show_gold and gold is not None:
            correct = _answers_match(final.answer, gold, rec)
            n_scored += 1
            n_correct += int(correct)
            mark = "OK " if correct else "XX "
            line += f"  | gold={gold!r} {mark}({n_correct}/{n_scored}={n_correct / n_scored:.1%})"
        print(line)

        if args.show_fol and final.fol:
            print("    ---- Z3 program ----")
            for prog in final.fol:
                print("    " + prog.replace("\n", "\n    "))
            print("    --------------------")

        predictions[rec.id] = {
            "answer_type": rec.answer_type.value,
            "predicted": final.answer,
            "confidence": final.confidence,
            "source": source,
            "explanation": final.explanation,
            "gold": gold,
            "elapsed_s": round(timings.total_s, 3),
        }

    dt = time.perf_counter() - t_start
    print(f"\n[done] {len(gated)} rows in {dt:.1f}s ({dt / max(len(gated), 1):.1f}s/row)")
    if args.show_gold and n_scored:
        print(f"[accuracy] {n_correct}/{n_scored} = {n_correct / n_scored:.1%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[wrote] {len(predictions)} predictions -> {args.out}")

    # Drop the run's detailed output files into Result/ (sharing one timestamp).
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[wrote] run summary  -> {write_run_summary(args, gated, finals, trace, dt, n_correct, n_scored, stamp)}")
    print(f"[wrote] model I/O    -> {write_model_io(args, gated, trace, stamp)}")
    print(f"[wrote] completed FOL-> {write_premises_file(args, gated, trace, stamp)}")
    print(f"[wrote] solver trace -> {write_solver_file(args, gated, finals, trace, stamp)}")


def _answers_match(pred: str | None, gold: str, rec: Record) -> bool:
    """Loose equality for the optional accuracy read-out."""
    if pred is None:
        return False
    p = pred.strip().rstrip(".").lower()
    g = gold.strip().rstrip(".").lower()
    if p == g:
        return True
    if rec.answer_type == AnswerType.MCQ and rec.options:
        opts = [o.strip().rstrip(".").lower() for o in rec.options]
        gi = _letter_to_index(g)
        pi = _letter_to_index(p)
        g_text = opts[gi] if gi is not None and gi < len(opts) else g
        p_text = opts[pi] if pi is not None and pi < len(opts) else p
        return g_text == p_text
    return False


def _letter_to_index(s: str) -> int | None:
    s = s.strip().rstrip(").").strip()
    if len(s) == 1 and "a" <= s <= "h":
        return ord(s) - ord("a")
    return None


if __name__ == "__main__":
    main()
