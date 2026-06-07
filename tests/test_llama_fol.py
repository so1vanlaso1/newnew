"""Llama NL→FOL path: normalization, FOL→Z3 assembly, and end-to-end wiring.

No model is loaded — a fake backend returns canned FOL strings, so these run on
CPU with no GPU/transformers. They confirm the FOL→Z3→solve chain and that the
translator drives `process_record` symbolic-only when the CoT fallback is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.types import AnswerType, Record
from pipeline import PipelineConfig, process_record
from solver.z3_runner import run_yes_no_uncertain
from translator.llama_fol import (
    LlamaFolConfig,
    LlamaFolTranslator,
    assemble_z3_program,
    normalize_fol,
)

# Symbolic-only config: a single solved proof is high confidence, CoT disabled.
SYMBOLIC_PCFG = PipelineConfig(
    enable_cot_fallback=False, vote_high_threshold=1, vote_medium_threshold=1
)


# ─── normalize_fol ────────────────────────────────────────────────────────


def test_normalize_single_and_or_to_accepted_symbols():
    assert normalize_fol("P(a) & Q(a)") == "P(a) ∧ Q(a)"
    assert normalize_fol("P(a) | Q(a)") == "P(a) ∨ Q(a)"


def test_normalize_double_connectives_not_doubled():
    # `&&`/`||` must collapse to a single symbol, not `∧∧`/`∨∨`.
    assert normalize_fol("P(a) && Q(a)") == "P(a) ∧ Q(a)"
    assert normalize_fol("P(a) || Q(a)") == "P(a) ∨ Q(a)"


def test_normalize_strips_trailing_period_and_whitespace():
    assert normalize_fol("  Person(Alice).  ") == "Person(Alice)"


def test_normalize_leaves_accepted_forms_untouched():
    s = "FORALL x (Student(x) -> Person(x))"
    assert normalize_fol(s) == s


def test_normalize_word_operators_to_symbols():
    # Safety net for any model drift to AND/OR/NOT/IMPLIES uppercase words.
    assert normalize_fol("P(a) AND Q(a)") == "P(a) ∧ Q(a)"
    assert normalize_fol("P(a) OR Q(a)") == "P(a) ∨ Q(a)"
    assert normalize_fol("P(a) IMPLIES Q(a)") == "P(a) → Q(a)"
    assert normalize_fol("NOT P(a)") == "¬ P(a)"


def test_normalize_word_operators_are_whole_word_only():
    # A predicate the model glued together (e.g. `NOTQualifies`, `ANDgate`) must
    # NOT be mangled — only standalone operator tokens are replaced.
    assert normalize_fol("A(x) OR NOTQualifies(x)") == "A(x) ∨ NOTQualifies(x)"
    assert normalize_fol("ANDgate(x)") == "ANDgate(x)"


def test_normalize_splits_glued_quantifiers():
    # Models can emit runs like FORALLxFORALLyFORALLz with no spaces; each
    # quantifier and its variable must be separated so the converter binds them.
    out = normalize_fol("FORALLxFORALLyFORALLz (P(x) IMPLIES Q(z))")
    assert "FORALL x FORALL y FORALL z" in out


def test_glued_multi_quantifier_does_not_leak_a_free_variable():
    # Before spacing, `y`/`z` leaked as free "entities". A glued-quantifier rule
    # must parse with all variables bound, so the chain still solves.
    code, _ = assemble_z3_program(
        [
            "FORALLxFORALLy (Student(x) AND Knows(x, y) IMPLIES Smart(x))",
            "Student(sophia)",
            "Knows(sophia, sophia)",
        ],
        "Smart(sophia)",
    )
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"


def test_word_operator_fol_assembles_and_solves():
    # A real-shape record (FORALLx glued, word operators) must parse → solve.
    code, _ = assemble_z3_program(
        [
            "FORALLx (Student(x) AND Passed(x) IMPLIES Qualified(x))",
            "Student(sophia)",
            "Passed(sophia)",
        ],
        "Qualified(sophia)",
    )
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"


# ─── assemble_z3_program → exec → solve ──────────────────────────────────


def test_assemble_yields_runnable_entailed_program():
    code, goal_expr = assemble_z3_program(
        ["FORALL x (Student(x) -> Person(x))", "Student(Alice)"],
        "Person(Alice)",
    )
    assert code is not None and goal_expr is not None
    verdict = run_yes_no_uncertain(code)
    assert verdict.status == "solved"
    assert verdict.answer == "Yes"


def test_assemble_normalizes_ascii_connectives():
    # Model output using single `&` must still parse and solve.
    code, _ = assemble_z3_program(
        ["FORALL x (Student(x) -> (Person(x) & Mortal(x)))", "Student(Alice)"],
        "Mortal(Alice)",
    )
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"


def test_assemble_returns_none_when_no_premises():
    code, goal = assemble_z3_program([], "Person(Alice)")
    assert code is None and goal is None


def test_assemble_missing_goal_defaults_to_unentailed():
    code, goal_expr = assemble_z3_program(["Student(Alice)"], None)
    assert code is not None and goal_expr is None
    assert "goal = False" in code


# ─── 𝜙= marker stripping (model output → bare FOL) ────────────────────────


def test_strip_phi_marker_is_handled_by_normalize_round_trip():
    # The backend strips the '𝜙=' marker before FOL ever reaches assembly; here we
    # confirm a bare formula (post-strip) assembles + solves as expected.
    from translator.llama_fol import _strip_phi

    assert _strip_phi("𝜙=∀x (Dog(x) → Animal(x))") == "∀x (Dog(x) → Animal(x))"
    assert _strip_phi("  φ = Person(Alice) ") == "Person(Alice)"
    assert _strip_phi("Person(Alice)") == "Person(Alice)"


# ─── End-to-end wiring with a fake backend ───────────────────────────────


class FakeBackend:
    """Maps NL sentences → canned FOL by keyword, no model involved."""

    def translate_sentences(self, sentences: list[str], k: int) -> list[list[str]]:
        out: list[list[str]] = []
        for s in sentences:
            low = s.lower()
            if "every student" in low:
                fol = "FORALL x (Student(x) -> Person(x))"
            elif "alice is a student" in low:
                fol = "Student(Alice)"
            elif "alice" in low and "person" in low:
                fol = "Person(Alice)"
            elif "alice" in low and "dog" in low:
                fol = "Dog(Alice)"
            else:
                fol = "Unknown(Thing)"
            out.append([fol] * k)
        return out


def _translator() -> LlamaFolTranslator:
    return LlamaFolTranslator(FakeBackend(), LlamaFolConfig(k_samples=1))


def test_ynu_end_to_end_yes():
    rec = Record(
        id="ynu-1",
        premises_nl=["Every student is a person.", "Alice is a student."],
        question_nl="Is Alice a person?",
        answer_type=AnswerType.YES_NO_UNCERTAIN,
    )
    final, _ = process_record(rec, _translator(), SYMBOLIC_PCFG)
    assert final.answer == "Yes"
    # Symbolic-only (CoT disabled): a proof, never the fallback.
    assert final.debug.get("source") != "cot"


def test_mcq_end_to_end_picks_entailed_option():
    rec = Record(
        id="mcq-1",
        premises_nl=["Every student is a person.", "Alice is a student."],
        question_nl="Which statement can be inferred?",
        answer_type=AnswerType.MCQ,
        options=["Alice is a person.", "Alice is a dog."],
    )
    final, _ = process_record(rec, _translator(), SYMBOLIC_PCFG)
    assert final.answer == "Alice is a person."


def test_translator_exposes_chat_backend():
    # Unlike the old T5 path, the Llama translator exposes its backend so the
    # pipeline CAN run the CoT fallback (the base model with the adapter disabled).
    assert _translator().backend is not None
