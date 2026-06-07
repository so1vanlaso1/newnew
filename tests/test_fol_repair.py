"""Deterministic FOL repair bolt-ons (translator.fol_repair).

  A. ground_goal     — re-ground a goal T5 mangled into a universal / binary
                       formula back to `Pred(entity)` (arity from the premises).
  B. add_type_facts  — assert free sort guards (e.g. Student(sophia)) so gated
                       rules can fire.

No model is loaded; everything runs on CPU. The end-to-end tests exec the
assembled Z3 program through the real solver to prove the bolt-ons turn an
otherwise-dead record into a proof — and that disabling them (ablation) does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.types import AnswerType, Record
from pipeline import PipelineConfig, process_record
from solver.z3_runner import run_yes_no_uncertain
from translator.fol_repair import add_type_facts, ground_goal
from translator.llama_fol import (
    LlamaFolConfig,
    LlamaFolTranslator,
    assemble_z3_program,
)

SYMBOLIC_PCFG = PipelineConfig(
    enable_cot_fallback=False, vote_high_threshold=1, vote_medium_threshold=1
)


# ─── bolt-on A: ground_goal ───────────────────────────────────────────────


def test_ground_goal_universal_to_ground():
    # T5's signature failure: "Sophia is eligible" → "∀x (Student(x) → Eligible(x))".
    premises = ["FORALL x (Student(x) ∧ A(x) → B(x))", "A(sophia)"]
    goal = "FORALL x (Student(x) → B(x))"
    assert ground_goal(goal, premises) == "B(sophia)"


def test_ground_goal_binary_predicate_collapses_to_premise_arity():
    # The goal invents `y` and makes Eligible binary; the premises use it unary.
    premises = ["FORALL x (A(x) → Eligible(x))", "A(sophia)"]
    goal = "FORALL x (A(x) ∧ A(y) ∧ Foo(x, y) → Eligible(x, y))"
    assert ground_goal(goal, premises) == "Eligible(sophia)"


def test_ground_goal_already_ground_is_unchanged():
    premises = ["FORALL x (A(x) → B(x))", "A(sophia)"]
    assert ground_goal("B(sophia)", premises) == "B(sophia)"


def test_ground_goal_preserves_negation():
    premises = ["FORALL x (A(x) → B(x))", "A(sophia)"]
    goal = "FORALL x (A(x) → ¬ B(x))"
    assert ground_goal(goal, premises) == "¬ B(sophia)"


def test_ground_goal_no_entity_is_unchanged():
    # No ground facts → no entity to ground to → leave the goal alone.
    premises = ["FORALL x (A(x) → B(x))"]
    goal = "FORALL x (A(x) → B(x))"
    assert ground_goal(goal, premises) == goal


def test_ground_goal_prefers_positive_concept_over_glued_negation():
    # A tautology-shaped goal: pick the positive concept, not the `NOT…` glue.
    premises = ["FORALL x (A(x) → Eligible(x))", "A(sophia)"]
    goal = "FORALL x (Premises(x) → (QualifiesForScholarship(sophia) ∨ NOTQualifiesForScholarship(sophia)))"
    assert ground_goal(goal, premises) == "QualifiesForScholarship(sophia)"


def test_ground_goal_rejects_spurious_free_variable_as_entity():
    # The goal's only "constant" is T5's dangling free `y`; the real entity is in
    # the premises. We must ground to sophia, never to y.
    premises = ["FORALL x (A(x) → Eligible(x))", "A(sophia)"]
    goal = "FORALL x (A(x) → Eligible(x, y))"
    assert ground_goal(goal, premises) == "Eligible(sophia)"


def test_ground_goal_never_picks_a_sort_guard():
    # A distractor like "needs a recommendation to qualify" has the sort guard
    # `Student` in its antecedent. Grounding must NOT pick `Student` (which B makes
    # true for everyone) — that would make the distractor trivially "entailed".
    premises = [
        "FORALL x (Student(x) ∧ A(x) → B(x))",
        "FORALL x (Student(x) ∧ C(x) → D(x))",
        "A(sophia)",
    ]
    goal = "FORALL x (Student(x) ∧ NeedsRec(x) → Qualified(x))"
    out = ground_goal(goal, premises)
    assert "Student" not in out
    assert out == "Qualified(sophia)"  # the real consequent, grounded


def test_ground_goal_falls_back_to_antecedent_when_consequent_is_junk():
    # T5 parks the real claim (ReceivesDistinction) in the antecedent and fills the
    # consequent with the filler `AccordingToPremises`; recover the real claim.
    premises = [
        "FORALL x (Person(x) ∧ ReceivesDistinction(x) → AccordingToPremises(x))",
        "FORALL x (Person(x) ∧ Other(x) → Something(x))",
        "ReceivesDistinction(john)",
    ]
    goal = "FORALL x (Person(x) ∧ ReceivesDistinction(x) → AccordingToPremises(x))"
    assert ground_goal(goal, premises) == "ReceivesDistinction(john)"


# ─── bolt-on B: add_type_facts ────────────────────────────────────────────


def test_add_type_facts_asserts_sort_guard_gating_every_rule():
    premises = [
        "FORALL x (Student(x) ∧ CompletedCore(x) → Qualified(x))",
        "FORALL x (Student(x) ∧ Qualified(x) → Eligible(x))",
        "CompletedCore(sophia)",
    ]
    out = add_type_facts(premises)
    assert out == premises + ["Student(sophia)"]


def test_add_type_facts_skips_precondition_gating_one_rule():
    premises = [
        "FORALL x (Student(x) ∧ Extra(x) → B(x))",
        "FORALL x (Student(x) → C(x))",
        "A(sophia)",
    ]
    out = add_type_facts(premises)
    assert "Student(sophia)" in out
    # `Extra` gates only one rule → it is a precondition, not a sort. Never assert.
    assert "Extra(sophia)" not in out


def test_add_type_facts_skips_guard_already_asserted_as_fact():
    premises = [
        "FORALL x (Student(x) → B(x))",
        "FORALL x (Student(x) → C(x))",
        "Student(sophia)",
    ]
    assert add_type_facts(premises) == premises  # nothing added


def test_add_type_facts_skips_head_predicate():
    # `Qualified` gates rule 2 but is the HEAD of rule 1 — it is derivable, not a
    # sort guard, so it must not be asserted as a base fact.
    premises = [
        "FORALL x (Student(x) ∧ Core(x) → Qualified(x))",
        "FORALL x (Student(x) ∧ Qualified(x) → Eligible(x))",
        "Core(sophia)",
    ]
    out = add_type_facts(premises)
    assert "Qualified(sophia)" not in out
    assert "Student(sophia)" in out


def test_add_type_facts_respects_min_rules():
    premises = ["FORALL x (Student(x) → B(x))", "A(sophia)"]
    assert add_type_facts(premises, min_rules=2) == premises  # 1 rule < 2 → no-op
    # Lowering the threshold lets the single-rule guard through.
    assert "Student(sophia)" in add_type_facts(premises, min_rules=1)


def test_add_type_facts_excludes_goal_predicate():
    premises = [
        "FORALL x (Student(x) → B(x))",
        "FORALL x (Student(x) → C(x))",
        "A(sophia)",
    ]
    # If the goal asks about Student itself, don't beg the question by asserting it.
    assert add_type_facts(premises, goal_fol="Student(sophia)") == premises


def test_add_type_facts_asserts_for_every_person_entity():
    premises = [
        "FORALL x (Person(x) → Mortal(x))",
        "FORALL x (Person(x) → HasName(x))",
        "Greek(socrates)",
        "Greek(plato)",
    ]
    out = add_type_facts(premises)
    assert "Person(socrates)" in out and "Person(plato)" in out


def test_add_type_facts_never_types_non_person_objects():
    # The mis-parse from the run review: "Sophia has completed the core curriculum"
    # became a binary `Completed(Curriculum, sophia)`, pulling the OBJECT `Curriculum`
    # into argument position. The sort guard must be asserted for the person sophia
    # only — never `Student(Curriculum)` / `Student(capstoneProject)`.
    premises = [
        "FORALL x (Student(x) ∧ CompletedCoreCurriculum(x) → Qualified(x))",
        "FORALL x (Student(x) ∧ Qualified(x) → Eligible(x))",
        "CompletedCoreCurriculum(sophia)",        # unary fact → sophia is a person
        "Completed(Curriculum, sophia)",          # binary → Curriculum is an object
        "Passed(sophia, scienceAssessment)",      # binary → scienceAssessment is an object
    ]
    out = add_type_facts(premises)
    assert "Student(sophia)" in out
    assert "Student(Curriculum)" not in out
    assert "Student(scienceAssessment)" not in out


def test_add_type_facts_no_unary_subject_asserts_nothing():
    # When NO entity is ever a unary-predicate subject we cannot tell a person from
    # an object, so assert nothing rather than risk typing an object.
    premises = [
        "FORALL x (Student(x) ∧ A(x) → B(x))",
        "FORALL x (Student(x) ∧ B(x) → C(x))",
        "Completed(Curriculum, sophia)",
    ]
    assert add_type_facts(premises) == premises


# ─── the payoff: bolt-ons turn a dead record into a proof ─────────────────


_CHAIN_PREMISES = [
    "FORALL x (Student(x) ∧ CompletedCoreCurriculum(x) → QualifiedForAdvancedCourses(x))",
    "FORALL x (Student(x) ∧ QualifiedForAdvancedCourses(x) → EligibleForInternationalProgram(x))",
    "CompletedCoreCurriculum(sophia)",
]
# Exactly the shape from the real run: entity dropped, wrapped in a universal.
_MANGLED_GOAL = "FORALL x (Student(x) → EligibleForInternationalProgram(x))"


def test_bolt_ons_make_the_chain_solve():
    code, _ = assemble_z3_program(_CHAIN_PREMISES, _MANGLED_GOAL)  # bolt-ons ON
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"


def test_without_bolt_ons_the_chain_is_dead():
    code, _ = assemble_z3_program(
        _CHAIN_PREMISES, _MANGLED_GOAL, ground_goals=False, assert_type_facts=False
    )
    assert code is not None
    # Universal goal + no Student(sophia) → not entailed (the actual run's failure).
    assert run_yes_no_uncertain(code).answer != "Yes"


# ─── end-to-end through process_record with a backend that mangles like fvossel ─


class ManglingBackend:
    """Reproduces the fvossel NL→FOL failure modes: ground claims become universals
    and the `Student` sort fact is never emitted."""

    def translate_sentences(self, sentences: list[str], k: int) -> list[list[str]]:
        out: list[list[str]] = []
        for s in sentences:
            low = s.lower()
            if "core curriculum" in low and "students" in low:
                fol = ("FORALL x (Student(x) ∧ CompletedCoreCurriculum(x) "
                       "→ QualifiedForAdvancedCourses(x))")
            elif "qualified for advanced courses" in low and "students" in low:
                fol = ("FORALL x (Student(x) ∧ QualifiedForAdvancedCourses(x) "
                       "→ EligibleForInternationalProgram(x))")
            elif "sophia has completed the core curriculum" in low:
                fol = "CompletedCoreCurriculum(sophia)"
            elif "international program" in low:
                fol = "FORALL x (Student(x) → EligibleForInternationalProgram(x))"
            else:
                fol = "Unknown(thing)"
            out.append([fol] * k)
        return out


def test_process_record_solves_yes_with_bolt_ons():
    rec = Record(
        id="ynu-mangled",
        premises_nl=[
            "Students who have completed the core curriculum are qualified for advanced courses.",
            "Students who are qualified for advanced courses are eligible for the international program.",
            "Sophia has completed the core curriculum.",
        ],
        question_nl="Is Sophia eligible for the international program?",
        answer_type=AnswerType.YES_NO_UNCERTAIN,
    )
    translator = LlamaFolTranslator(ManglingBackend(), LlamaFolConfig(k_samples=1))
    final, _ = process_record(rec, translator, SYMBOLIC_PCFG)
    assert final.answer == "Yes"
    assert final.debug.get("source") != "cot"  # symbolic-only, never the fallback
