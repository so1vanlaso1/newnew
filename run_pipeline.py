#!/usr/bin/env python
"""End-to-end neuro-symbolic runner for EXACT 2026 Track 1.

Sequentially walks every row of ``Logic_Based_Educational_Queries.json`` and,
using ONLY the natural-language premises and the question (never the gold
answer, never the gold FOL), produces a verdict:

    row ─► Qwen/Qwen3.5-4B-Base + translator-LoRA  (NL premises + question → Z3 program)
        ─► Z3 entailment solver               (K candidates, one solve each)
        ─► majority vote over the K verdicts  (high / medium confidence)
        ─► CoT fallback on the base model     (only if the vote is inconclusive)
        ─► Yes / No / Uncertain   or   the winning MCQ option

The answer type is decided structurally from the QUESTION alone: if the question
text carries lettered options it is treated as MCQ, otherwise as
Yes/No/Uncertain. The gold answer is loaded separately and used *only* for the
optional accuracy read-out — it never reaches the model or the solver.

Examples
--------
    # Real run on the RTX 5070 box (vLLM, bf16 base model + LoRA):
    python run_pipeline.py --limit 20

    # transformers + PEFT path with a precision toggle (4bit | bf16):
    python run_pipeline.py --backend hf --precision 4bit --limit 5
    python run_pipeline.py --backend hf --precision bf16 --limit 5

    # Wiring smoke test with no model at all (uses a stub translator):
    python run_pipeline.py --backend stub --limit 5 --show-gold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data.load import load_records                       # noqa: E402
from data.types import AnswerType, Record                # noqa: E402
from fallback.cot import CotConfig                        # noqa: E402
from pipeline import PipelineConfig, process_record       # noqa: E402
from translator.infer import Translator, TranslatorConfig  # noqa: E402

# ── Default local paths (override on the CLI) ──────────────────────────────
DEFAULT_DATA = ROOT / "Logic_Based_Educational_Queries.json"
# The single model for the whole pipeline. A HF repo id (resolved from the HF
# cache that setup.sh pre-populates), NOT a local dir — the old local
# models/Qwen3.5-4B was a multimodal checkpoint and caused LoRA/base mismatches.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B-Base"
DEFAULT_LORA = ROOT / "Finetune" / "Artifact" / "artifacts" / "artifacts" / "translator-lora"


# ── Input gating: the pipeline may see premises-NL + question only ─────────
def gate_inputs(r: Record) -> tuple[Record, str | None]:
    """Return (pipeline_record, gold_answer).

    The returned record carries ONLY the natural-language premises, the
    question, and any options parsed out of the question. The gold answer, gold
    FOL, and stored explanation are stripped so they can't influence the
    verdict; the gold answer is handed back separately for scoring/display.

    Answer type is re-derived from question structure (options present → MCQ,
    else Yes/No/Uncertain) so we never rely on the gold answer's value to know
    the task format.
    """
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


# ── Backends ───────────────────────────────────────────────────────────────
class StubBackend:
    """Zero-dependency backend for wiring tests.

    It does not understand language; for Yes/No/Uncertain it emits a trivially
    satisfiable Z3 program (the solver will return "Uncertain"), and for MCQ it
    emits a program with no entailed option ("Unknown"). Use it only to confirm
    the data → translate → solve → vote → explain plumbing end to end.
    """

    _YNU = (
        "<z3py>\n"
        "U = DeclareSort('U')\n"
        "P = Function('P', U, BoolSort())\n"
        "a = Const('a', U)\n"
        "premises = [Or(P(a), Not(P(a)))]\n"
        "goal = P(a)\n"
        "</z3py>"
    )

    def chat_generate(self, batch_messages, n, temperature, top_p, max_tokens, lora_path):  # noqa: ANN001
        return [[self._YNU for _ in range(n)] for _ in batch_messages]


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
    )

    if args.backend == "stub":
        print("[info] backend=stub (no model loaded -- wiring test only)")
        return Translator(StubBackend(), tcfg)

    if args.backend == "hf":
        import torch

        from translator.infer import TransformersBackend

        has_cuda = torch.cuda.is_available()
        load_4bit = args.precision == "4bit"
        if load_4bit and not has_cuda:
            print("[warn] --precision 4bit needs a CUDA GPU (bitsandbytes); CPU torch "
                  "detected, falling back to bf16.")
            load_4bit = False
        device_map = "auto" if has_cuda else "cpu"
        if not has_cuda:
            print("[warn] running on CPU (torch reports no CUDA). A 4B model is slow "
                  "on CPU -- use a small --limit and --k 1 for a smoke test.")
        print(
            f"[info] backend=transformers  model={tcfg.model}  device={device_map}\n"
            f"       lora={tcfg.lora_path or '(none)'}  precision={'4bit' if load_4bit else 'bf16'}"
        )
        backend = TransformersBackend(tcfg, load_4bit=load_4bit, device_map=device_map)
        return Translator(backend, tcfg)

    # default: vLLM
    from translator.infer import VLLMBackend

    tcfg.quantization = args.quantization
    if args.precision == "4bit":
        # vLLM in-flight 4-bit (bitsandbytes). Needs a recent vLLM build; if it
        # errors on the qwen3_5 arch, use `--backend hf --precision 4bit`.
        tcfg.quantization = "bitsandbytes"
        tcfg.load_format = "bitsandbytes"
    tcfg.gpu_memory_utilization = args.gpu_mem
    print(
        f"[info] backend=vllm  model={tcfg.model}  q={tcfg.quantization or 'none (bf16)'}\n"
        f"       lora={tcfg.lora_path or '(none)'}"
    )
    backend = VLLMBackend(tcfg)
    return Translator(backend, tcfg)


# ── Main loop ────────────────────────────────────────────────────────────
def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so unicode option text /
    # FOL symbols in printed verdicts don't crash on redirect.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="release JSON to read rows from")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="HF repo id or local dir of the base model")
    ap.add_argument("--lora", type=Path, default=DEFAULT_LORA, help="local LoRA adapter dir")
    ap.add_argument("--backend", choices=["vllm", "hf", "stub"], default="vllm")
    ap.add_argument("--no-lora", action="store_true", help="ignore the adapter, use base model only")
    ap.add_argument("--precision", choices=["bf16", "4bit"], default="bf16",
                    help="inference precision: bf16 (full bf16 weights) | 4bit (NF4 via bitsandbytes)")
    ap.add_argument("--quantization", default="none",
                    help="(vllm backend) none | fp8 | bitsandbytes | awq_marlin")
    ap.add_argument("--gpu-mem", type=float, default=0.80, help="(vllm backend) gpu_memory_utilization")
    ap.add_argument("--k", type=int, default=5, help="self-consistency samples per translate")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows (0 = all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N rows")
    ap.add_argument("--only", choices=["ynu", "mcq", "all"], default="all", help="filter by task type")
    ap.add_argument("--show-gold", action="store_true", help="print gold answer + running accuracy")
    ap.add_argument("--show-fol", action="store_true", help="print the winning Z3 program")
    ap.add_argument("--out", type=Path, default=None, help="optional path to write predictions JSON")
    args = ap.parse_args()

    if "none" == str(args.quantization).lower():
        args.quantization = None

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

    # ── Build pipeline ──────────────────────────────────────────────────
    translator = build_translator(args)
    pcfg = PipelineConfig(cot=CotConfig(k_samples=args.k))

    predictions: dict[str, dict] = {}
    n_correct = 0
    n_scored = 0
    t_start = time.perf_counter()

    for i, (rec, gold) in enumerate(gated, 1):
        final, timings = process_record(rec, translator, pcfg)
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


def _answers_match(pred: str | None, gold: str, rec: Record) -> bool:
    """Loose equality for the optional accuracy read-out."""
    if pred is None:
        return False
    p = pred.strip().rstrip(".").lower()
    g = gold.strip().rstrip(".").lower()
    if p == g:
        return True
    # MCQ: gold may be a letter while we emit the option text (or vice versa).
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
