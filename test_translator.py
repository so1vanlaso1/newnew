#!/usr/bin/env python
"""Translator harness: NL premises + question  ->  FOL / Z3 program -> verdict.

Drives the LLM translation stage (no CoT fallback) and then runs the SAME Z3
solve path the real pipeline uses, so you can see not just whether the model
emits a `<z3py>` block but whether that block actually runs and answers the
question. For every row it records, per sample:

  * the gated input (premises-NL + question only),
  * the raw model completion (the thinking block is separated out),
  * the parsed + auto-repaired <z3py> program, or a PARSE-FAIL marker,
  * per-sample parse / exec status,

then a per-record solved verdict (vs gold, if available). The headline reports
parse-rate, executable-rate, and end-to-end solve accuracy — executable-rate is
the metric that actually predicts solvability; a 100% parse rate over
non-executable programs (the old metric) is misleading.

Examples
--------
    # Real translate on the local base model + LoRA (CPU here, slow):
    python test_translator.py --backend hf --limit 2 --k 1 --out fol_eval.txt

    # Plumbing / format check with no model:
    python test_translator.py --backend stub --limit 3 --out fol_eval.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data.load import load_records                       # noqa: E402
from data.types import AnswerType, Record, Translation   # noqa: E402
from pipeline import PipelineConfig, _solve_mcq, _solve_ynu  # noqa: E402
from solver.z3_runner import premises_of                 # noqa: E402
from translator.infer import Translator, TranslatorConfig  # noqa: E402
from translator.parse import extract_goal_expr, parse_translator_output  # noqa: E402
from translator.prompt import build_messages_for_record  # noqa: E402
from translator.repair import repair_program             # noqa: E402
from vote import aggregate                               # noqa: E402

DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"
# Single model for the whole pipeline — a HF repo id (resolved from the HF
# cache), NOT the old local models/Qwen3.5-4B multimodal dir.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B-Base"
DEFAULT_LORA = ROOT / "Finetune" / "Artifact" / "artifacts" / "artifacts" / "translator-lora"


# ── Input gating: translator sees premises-NL + question only ──────────────
def gate_inputs(r: Record) -> Record:
    atype = AnswerType.MCQ if r.options else AnswerType.YES_NO_UNCERTAIN
    return Record(
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


# ── Zero-dependency backend for a format/plumbing check ────────────────────
class StubBackend:
    _CODE = (
        "<think>\nThe project follows from being well-tested, so I model a single\n"
        "predicate over the universe.\n</think>\n"
        "<z3py>\n"
        "U = DeclareSort('U')\n"
        "P = Function('P', U, BoolSort())\n"
        "a = Const('a', U)\n"
        "premises = [Or(P(a), Not(P(a)))]\n"
        "goal = P(a)\n"
        "</z3py>"
    )

    def chat_generate(self, batch_messages, n, temperature, top_p, max_tokens, lora_path):  # noqa: ANN001
        return [[self._CODE for _ in range(n)] for _ in batch_messages]


def render_prompt_input(backend: object, messages: list[dict]) -> str:
    """Render the exact model input when the backend exposes its chat template."""
    render_prompt = getattr(backend, "render_prompt", None)
    if callable(render_prompt):
        return render_prompt(messages)
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )


def build_translator(args: argparse.Namespace) -> Translator:
    lora_path = None if args.no_lora else str(args.lora)
    if lora_path and not Path(lora_path).exists():
        print(f"[warn] LoRA path not found, running base model only: {lora_path}")
        lora_path = None

    tcfg = TranslatorConfig(
        model=str(args.model),
        enable_lora=lora_path is not None,
        lora_path=lora_path,
        k_samples=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        n_fewshot=args.n_fewshot,
    )

    if args.backend == "stub":
        print("[info] backend=stub (no model loaded -- format check only)")
        return Translator(StubBackend(), tcfg)

    # transformers + PEFT (the only native-Windows path)
    import torch

    from translator.infer import TransformersBackend

    has_cuda = torch.cuda.is_available()
    load_4bit = args.precision == "4bit"
    if load_4bit and not has_cuda:
        print("[warn] --precision 4bit needs CUDA; CPU torch detected, using bf16.")
        load_4bit = False
    device_map = "auto" if has_cuda else "cpu"
    if not has_cuda:
        print("[warn] running on CPU -- a 4B model is slow; keep --limit and --k small.")
    mode = "THINKING" if args.think else "NO-THINK (direct answer)"
    print(
        f"[info] backend=transformers  model={tcfg.model}  device={device_map}\n"
        f"       lora={tcfg.lora_path or '(none)'}  precision={'4bit' if load_4bit else 'bf16'}  "
        f"k={tcfg.k_samples}  max_new_tokens={tcfg.max_new_tokens}  mode={mode}"
    )
    stop = None if args.no_stop else ["</z3py>"]
    backend = TransformersBackend(
        tcfg,
        load_4bit=load_4bit,
        device_map=device_map,
        enable_thinking=args.think,
        stop_strings=stop,
    )
    return Translator(backend, tcfg)


# ── Output formatting ──────────────────────────────────────────────────────
def split_think(text: str) -> tuple[str, str]:
    """Separate a `<think>…</think>` block from the answer portion."""
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        thinking = head.replace("<think>", "").strip()
        return thinking, tail.strip()
    return "", text.strip()


def indent(block: str, pad: str = "    ") -> str:
    return "\n".join(pad + ln for ln in block.splitlines())


def _verdict_matches_gold(pred: str | None, gold: str | None, rec: Record) -> bool:
    """Loose match for the read-out: case/period-insensitive, and MCQ letter↔text."""
    if pred is None or gold is None:
        return False
    p = pred.strip().rstrip(".").lower()
    g = gold.strip().rstrip(".").lower()
    if p == g:
        return True
    if rec.answer_type == AnswerType.MCQ and rec.options:
        opts = [o.strip().rstrip(".").lower() for o in rec.options]

        def to_text(s: str) -> str:
            s2 = s.rstrip(").").strip()
            if len(s2) == 1 and "a" <= s2 <= "h":
                i = ord(s2) - ord("a")
                return opts[i] if i < len(opts) else s
            return s

        return to_text(p) == to_text(g)
    return False


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="HF repo id or local dir of the base model")
    ap.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    ap.add_argument("--backend", choices=["hf", "stub"], default="hf")
    ap.add_argument("--no-lora", action="store_true", help="base model only (skip the adapter)")
    ap.add_argument("--precision", choices=["bf16", "4bit"], default="bf16",
                    help="(CUDA only) bf16 full weights | 4bit NF4 via bitsandbytes")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true",
                       help="THINKING mode: model reasons in a <think> block first (default)")
    think.add_argument("--no-think", dest="think", action="store_false",
                       help="NO-THINK mode: pre-fill an empty think block so it emits FOL directly (faster)")
    ap.set_defaults(think=True)
    ap.add_argument("--no-stop", action="store_true",
                    help="do NOT stop at </z3py>; let the model run to max_new_tokens "
                         "(slower, lets you see runaway/over-generation)")
    ap.add_argument("--k", type=int, default=1, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=4000,
                    help="raise if a thinking model never closes its </think> block")
    ap.add_argument("--n-fewshot", type=int, default=2)
    ap.add_argument("--limit", type=int, default=2, help="process first N rows (0 = all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only", choices=["ynu", "mcq", "all"], default="all")
    ap.add_argument("--hide-raw", action="store_true", help="omit the full raw completion")
    ap.add_argument("--out", type=Path, default=ROOT / "translator_eval.txt")
    args = ap.parse_args()

    # Keep the gold answer beside each gated record so we can report not just
    # "did it parse" but "did the solved verdict match gold". Gold never reaches
    # the model — it is only used for the read-out below.
    pairs = [(gate_inputs(r), r.answer) for r in load_records(args.data)]
    if args.only == "ynu":
        pairs = [p for p in pairs if p[0].answer_type == AnswerType.YES_NO_UNCERTAIN]
    elif args.only == "mcq":
        pairs = [p for p in pairs if p[0].answer_type == AnswerType.MCQ]
    pairs = pairs[args.start:]
    if args.limit:
        pairs = pairs[: args.limit]
    records = [p[0] for p in pairs]
    golds = [p[1] for p in pairs]
    print(f"[info] loaded from {args.data.name}; translating {len(records)} rows")

    translator = build_translator(args)
    backend = translator.backend

    pcfg = PipelineConfig()
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("TRANSLATOR EVALUATION  (NL -> FOL / Z3 program -> solved verdict)")
    lines.append(f"model        : {args.model}")
    lines.append(f"lora         : {translator.cfg.lora_path or '(none / base model)'}")
    lines.append(f"k_samples    : {args.k}    n_fewshot: {args.n_fewshot}    "
                 f"max_new_tokens: {args.max_new_tokens}    temp: {args.temperature}")
    lines.append(f"mode         : {'THINKING' if args.think else 'NO-THINK (direct answer)'}")
    lines.append(f"rows         : {len(records)} from {args.data.name}")
    lines.append("=" * 80)

    n_samples = 0
    n_parsed = 0
    n_executable = 0
    n_solved = 0      # records that produced a definite (non-None) verdict
    n_correct = 0     # records whose verdict matched gold
    n_scored = 0      # records with a gold answer to score against

    for ri, rec in enumerate(records, 1):
        gold = golds[ri - 1]
        kind = "MCQ" if rec.answer_type == AnswerType.MCQ else "YES_NO_UNCERTAIN"
        lines.append("")
        lines.append("#" * 80)
        lines.append(f"RECORD {ri}/{len(records)}   id={rec.id}   type={kind}")
        lines.append("#" * 80)
        lines.append("PREMISES (NL):")
        for pi, p in enumerate(rec.premises_nl):
            lines.append(f"  [{pi}] {p}")
        lines.append(f"QUESTION (NL):\n  {rec.question_nl}")
        if rec.options:
            lines.append("OPTIONS:")
            for oi, opt in enumerate(rec.options):
                lines.append(f"  {chr(ord('A') + oi)}. {opt}")

        # One prompt for YNU, one per option for MCQ.
        batches = build_messages_for_record(rec, n_fewshot=args.n_fewshot)
        prompt_inputs = [render_prompt_input(backend, messages) for messages in batches]
        raw = backend.chat_generate(
            batch_messages=batches,
            n=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            lora_path=translator.cfg.lora_path,
        )

        groups: list[list[Translation]] = []
        for bi, prompt_outputs in enumerate(raw):
            if rec.answer_type == AnswerType.MCQ and rec.options and bi < len(rec.options):
                tag = f"PROMPT {bi + 1}/{len(raw)}  (option {chr(ord('A') + bi)}: {rec.options[bi]})"
            else:
                tag = f"PROMPT {bi + 1}/{len(raw)}"
            lines.append("")
            lines.append("-" * 80)
            lines.append(tag)
            lines.append("-" * 80)
            lines.append("  --- PROMPT INPUT TO MODEL ---")
            lines.append(indent(prompt_inputs[bi]))

            per_prompt: list[Translation] = []
            for si, txt in enumerate(prompt_outputs):
                n_samples += 1
                thinking, answer = split_think(txt)
                raw_code = parse_translator_output(txt)
                ok = raw_code is not None
                n_parsed += int(ok)

                # Repair (auto-declare) then exec-check in the real Z3 sandbox:
                # "executable" is the metric that actually predicts solvability,
                # unlike "a <z3py> tag is present".
                code = repair_program(raw_code) if ok else None
                executable = code is not None and premises_of(code) is not None
                n_executable += int(executable)
                if code is not None:
                    per_prompt.append(Translation(
                        code=code, goal_expr=extract_goal_expr(code),
                        raw_text=txt, sample_index=si,
                    ))

                repaired_note = ""
                if ok and code != raw_code:
                    repaired_note = "  (repaired: added missing declarations)"
                lines.append(f"  SAMPLE {si + 1}/{len(prompt_outputs)}   "
                             f"parse: {'OK' if ok else 'FAIL'}   "
                             f"exec: {'OK' if executable else 'FAIL'}{repaired_note}   "
                             f"(think {len(thinking)} chars, answer {len(answer)} chars)")
                if not args.hide_raw:
                    lines.append("  --- MODEL OUTPUT (RAW COMPLETION) ---")
                    lines.append(indent(txt))
                lines.append("  --- PARSED / REPAIRED Z3 PROGRAM ---")
                lines.append(indent(code) if code else "    <<< PARSE FAILED: no <z3py> block found >>>")
                lines.append("")
            groups.append(per_prompt)

        # Solve the record end-to-end (the same path the real pipeline uses), so
        # the report shows whether these translations actually answer the question.
        verdict_answer = None
        if rec.answer_type == AnswerType.MCQ and rec.options and all(groups):
            verdicts, _ = _solve_mcq(groups, rec.options, pcfg)
            verdict_answer, _conf, _core = aggregate(verdicts, k=len(verdicts))
        elif rec.answer_type == AnswerType.YES_NO_UNCERTAIN and groups and groups[0]:
            verdicts, _ = _solve_ynu(groups[0], pcfg)
            verdict_answer, _conf, _core = aggregate(verdicts, k=len(verdicts))

        n_solved += int(verdict_answer is not None)
        correct = None
        if gold is not None:
            n_scored += 1
            correct = _verdict_matches_gold(verdict_answer, gold, rec)
            n_correct += int(correct)
        mark = "" if gold is None else f"   gold={gold!r}  {'OK' if correct else 'XX'}"
        lines.append(f"  >>> RECORD VERDICT: {verdict_answer!r}{mark}")
        lines.append("")

    rate = (n_parsed / n_samples) if n_samples else 0.0
    exec_rate = (n_executable / n_samples) if n_samples else 0.0
    summary_lines = [
        f"PARSE  (<z3py> tag present) : {n_parsed}/{n_samples} = {rate:.1%}",
        f"EXEC   (runs in Z3 sandbox) : {n_executable}/{n_samples} = {exec_rate:.1%}",
        f"SOLVED (definite verdict)   : {n_solved}/{len(records)} records",
    ]
    if n_scored:
        acc = n_correct / n_scored
        summary_lines.append(f"ACCURACY (verdict == gold)  : {n_correct}/{n_scored} = {acc:.1%}")
    lines.append("=" * 80)
    lines.extend(summary_lines)
    lines.append("=" * 80)

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("[done]")
    for s in summary_lines:
        print("       " + s)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
