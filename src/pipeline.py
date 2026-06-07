"""End-to-end orchestration for one record.

  Stage 1: NL → SMT-LIB (K samples) via translator (vLLM + optional LoRA)
  Stage 2: Z3 entailment check per sample
  Stage 3a: majority vote across surviving verdicts (high/medium confidence)
  Stage 3b: CoT fallback (open-ended, low confidence, or all-failed)
  Stage 4: assemble FinalAnswer (answer, explanation, fol, cot, premises, confidence)

Each stage logs wall time; `process_record` enforces an overall budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from data.types import AnswerType, FinalAnswer, Record, SolverVerdict, Translation
from explain import from_cot, from_failure, from_symbolic
from fallback.cot import CotConfig, run_cot
from solver.z3_runner import premises_of, run_mcq, run_yes_no_uncertain
from translator.parse import extract_goal_expr
from translator.repair import repair_program
from vote import aggregate

log = logging.getLogger(__name__)


class Translator(Protocol):
    """Structural type for anything `process_record` can drive: it must translate
    a record into candidate Z3 programs and optionally expose a chat `backend`
    (a truthy backend enables the Stage-3b CoT fallback; None disables it)."""

    backend: object | None

    def translate(self, record: Record) -> list[list[Translation]]: ...


@dataclass
class PipelineConfig:
    wall_clock_budget_s: float = 55.0
    solver_timeout_ms: int = 5000
    emit_unsat_core: bool = True
    vote_high_threshold: int = 4
    vote_medium_threshold: int = 3
    cot: CotConfig = field(default_factory=CotConfig)
    # Stage 3b (CoT) requires a chat-capable backend on `translator.backend`.
    # The secondary T5 translator is a translation-only seq2seq model with no
    # chat backend, so it disables this and runs symbolic-only. Default True
    # keeps the primary Qwen pipeline's behavior unchanged.
    enable_cot_fallback: bool = True
    # MCQ tie-break: when more than one option is entailed, pick the one whose
    # unsat core (proof) is smallest — the most directly supported claim — instead
    # of abstaining with 'Unknown'. Default False keeps the primary pipeline's
    # (sound) abstention; the T5 path enables it because its forward-chaining
    # records routinely entail several options along one chain.
    mcq_tiebreak_smallest_core: bool = False


@dataclass
class StageTimings:
    translate_s: float = 0.0
    solve_s: float = 0.0
    vote_s: float = 0.0
    cot_s: float = 0.0
    total_s: float = 0.0


def _best_premise_env(programs: list[str]) -> str | None:
    """Pick the shared premise environment for an MCQ record.

    Premises are option-independent, so every option's program is a candidate.
    The historical bug was blindly using option A's program: if A dropped facts
    or used an undeclared symbol, the whole record was lost even when another
    option produced a complete, runnable program. Instead we repair each
    candidate and keep the executable one with the MOST premises (the best proxy
    for fact-completeness)."""
    best: tuple[int, str] | None = None
    for prog in programs:
        repaired = repair_program(prog)
        premises = premises_of(repaired)
        if premises is None:
            continue
        if best is None or len(premises) > best[0]:
            best = (len(premises), repaired)
    return best[1] if best else None


def _augment_env_with_goal_symbols(env_code: str, goals: list[str]) -> str:
    """Declare in the premise environment any symbol an option goal references
    but the environment omits, by running the repair pass over env + goal probes.
    Auto-declaring a missing goal symbol is sound: it becomes a free (open-world)
    predicate, so the option simply isn't entailed rather than crashing the solve."""
    probes = "\n".join(f"_goal_probe_{i} = {g}" for i, g in enumerate(goals))
    repaired = repair_program(f"{env_code}\n{probes}")
    return "\n".join(
        ln for ln in repaired.splitlines() if not ln.lstrip().startswith("_goal_probe_")
    )


def _solve_ynu(
    translations: list[Translation],
    cfg: PipelineConfig,
) -> tuple[list[SolverVerdict], Translation | None]:
    """Run Z3 on each of the K Yes/No/Uncertain translations."""
    verdicts: list[SolverVerdict] = []
    for t in translations:
        v = run_yes_no_uncertain(
            t.code,
            timeout_ms=cfg.solver_timeout_ms,
            emit_unsat_core=cfg.emit_unsat_core,
        )
        verdicts.append(v)
    return verdicts, (translations[0] if translations else None)


def _solve_mcq(
    per_option_translations: list[list[Translation]],
    options: list[str],
    cfg: PipelineConfig,
) -> tuple[list[SolverVerdict], Translation | None]:
    """Pair up sample-k across options into one MCQ pass.

    Each Z3 Python program is self-contained (declares its own sort, predicates,
    constants). For MCQ we share the FIRST option's program as the premise
    environment and treat each option's `goal` line as the per-option goal —
    the runner re-evaluates each goal expression in the shared namespace.
    """
    if not per_option_translations or not all(per_option_translations):
        return [], None

    k = min(len(group) for group in per_option_translations)
    verdicts: list[SolverVerdict] = []
    winning: Translation | None = None
    for sample_i in range(k):
        sample_translations = [group[sample_i] for group in per_option_translations]
        # Shared premise environment = the most complete executable program among
        # the options, not blindly option A's.
        premise_code = _best_premise_env([t.code for t in sample_translations])
        if premise_code is None:
            verdicts.append(SolverVerdict(
                answer=None, status="parse_error",
                error="no executable premise program among option translations",
            ))
            continue
        option_goals: list[str] = [
            (t.goal_expr or extract_goal_expr(t.code) or "False")
            for t in sample_translations
        ]
        # Ensure every option goal's symbols are declared in the shared env.
        premise_code = _augment_env_with_goal_symbols(premise_code, option_goals)
        v = run_mcq(
            premise_code, option_goals,
            timeout_ms=cfg.solver_timeout_ms,
            emit_unsat_core=cfg.emit_unsat_core,
            tiebreak_smallest_core=cfg.mcq_tiebreak_smallest_core,
        )
        if v.answer is not None and v.answer.isdigit():
            opt_idx = int(v.answer)
            if 0 <= opt_idx < len(options):
                v = SolverVerdict(
                    answer=options[opt_idx],
                    status=v.status,
                    unsat_core=v.unsat_core,
                    elapsed_ms=v.elapsed_ms,
                )
            if winning is None:
                winning = sample_translations[0]
        verdicts.append(v)
    return verdicts, winning


def process_record(
    record: Record,
    translator: Translator,
    cfg: PipelineConfig | None = None,
) -> tuple[FinalAnswer, StageTimings]:
    cfg = cfg or PipelineConfig()
    t = StageTimings()
    t0 = time.perf_counter()

    # ── Stage 1: translate ────────────────────────────────────────────────
    t1 = time.perf_counter()
    translations_grouped: list[list[Translation]] = []
    if record.answer_type != AnswerType.OPEN_ENDED:
        translations_grouped = translator.translate(record)
    t.translate_s = time.perf_counter() - t1

    # ── Stage 2: Z3 ────────────────────────────────────────────────────────
    t2 = time.perf_counter()
    verdicts: list[SolverVerdict] = []
    winning_translation: Translation | None = None
    if translations_grouped:
        if record.answer_type == AnswerType.YES_NO_UNCERTAIN:
            verdicts, winning_translation = _solve_ynu(translations_grouped[0], cfg)
        elif record.answer_type == AnswerType.MCQ:
            verdicts, winning_translation = _solve_mcq(
                translations_grouped, record.options or [], cfg
            )
    t.solve_s = time.perf_counter() - t2

    # ── Stage 3a: vote ─────────────────────────────────────────────────────
    t3 = time.perf_counter()
    answer, confidence, unsat_core = aggregate(
        verdicts, k=len(verdicts) if verdicts else 0,
        high_threshold=cfg.vote_high_threshold,
        medium_threshold=cfg.vote_medium_threshold,
    )
    # For an Uncertain answer there is no unsat core; carry a counter-model
    # witness from a representative winning verdict so the explanation can show
    # why the goal is undetermined.
    witness = next(
        (v.witness for v in verdicts if v.answer == answer and v.witness),
        None,
    )
    t.vote_s = time.perf_counter() - t3

    # ── Stage 3b: CoT fallback if needed ───────────────────────────────────
    elapsed_so_far = time.perf_counter() - t0
    budget_left = cfg.wall_clock_budget_s - elapsed_so_far
    need_fallback = (
        answer is None
        or record.answer_type == AnswerType.OPEN_ENDED
        or confidence < 0.7
    )
    final: FinalAnswer
    cot_available = cfg.enable_cot_fallback and getattr(translator, "backend", None) is not None
    if need_fallback and budget_left > 5.0 and cot_available:
        t4 = time.perf_counter()
        cot_answer, cot_trace, cot_conf = run_cot(
            translator.backend, record, cfg.cot
        )
        t.cot_s = time.perf_counter() - t4
        if cot_answer is not None:
            final = from_cot(record, cot_answer, cot_conf, cot_trace)
        elif answer is not None:
            # CoT failed but we have a low-confidence symbolic answer — keep it.
            final = from_symbolic(record, answer, confidence, unsat_core, winning_translation, witness)
        else:
            final = from_failure(record)
    elif answer is not None:
        final = from_symbolic(record, answer, confidence, unsat_core, winning_translation)
    else:
        final = from_failure(record)

    t.total_s = time.perf_counter() - t0
    if cfg and cfg.wall_clock_budget_s < t.total_s:
        log.warning("record %s exceeded wall clock: %.2fs > %.2fs", record.id, t.total_s,
                    cfg.wall_clock_budget_s)
    return final, t
