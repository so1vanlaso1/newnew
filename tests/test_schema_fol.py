"""Bolt-on E — schema-conditioned reshape (translator.schema_fol).

These reproduce the exact wrong-shape failures from the run review and prove the
deterministic reshape snaps facts/goals onto the rules' predicate registry so the
chain connects. The payoff tests exec the assembled Z3 program through the real
solver: a record that scored WRONG in the run now solves.
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
)
from translator.schema_fol import (
    harvest_registry,
    is_rule_fol,
    partition_rules_facts,
    record_person_entities,
    registry_targets,
    reshape_fact,
    reshape_goal,
    schema_condition,
    snap_text_to_registry,
)

SYMBOLIC_PCFG = PipelineConfig(
    enable_cot_fallback=False, vote_high_threshold=1, vote_medium_threshold=1
)

# The Sophia record's rules (records 1–4 in the run review).
_SOPHIA_RULES = [
    "∀x (Student(x) ∧ CompletedCoreCurriculum(x) ∧ PassedScienceAssessment(x) → QualifiedForAdvancedCourses(x))",
    "∀x (Student(x) ∧ QualifiedForAdvancedCourses(x) ∧ CompletedResearchMethodology(x) → EligibleForInternationalProgram(x))",
    "∀x (Student(x) ∧ EligibleForInternationalProgram(x) ∧ CompletedCapstoneProject(x) → HonorsDiploma(x))",
]

# The John academic-distinction record's rules (records 5–6).
_JOHN_RULES = [
    "∀x (Student(x) ∧ CompletesRequiredCourses(x) → EligibleForGraduation(x))",
    "∀x (Student(x) ∧ EligibleForGraduation(x) ∧ GPAAbove3_5(x) → GraduatesWithHonors(x))",
    "∀x (Student(x) ∧ GraduatesWithHonors(x) ∧ CompletesThesis(x) → ReceivesAcademicDistinction(x))",
    "∀x (Student(x) ∧ ReceivesAcademicDistinction(x) → QualifyForGraduateFellowship(x))",
]


# ─── partition / registry ─────────────────────────────────────────────────


def test_is_rule_fol_distinguishes_rules_from_facts():
    assert is_rule_fol("∀x (Student(x) ∧ A(x) → B(x))")
    assert not is_rule_fol("Completed(Curriculum, sophia)")
    assert not is_rule_fol("CompletedCoreCurriculum(sophia)")


def test_partition_rules_facts():
    prem = _SOPHIA_RULES + ["Completed(Curriculum, sophia)", "CompletedResearchMethodologyCourse(sophia)"]
    rules, facts = partition_rules_facts(prem)
    assert rules == [0, 1, 2]
    assert facts == [3, 4]


def test_registry_targets_excludes_sort_guard():
    targets = registry_targets(_SOPHIA_RULES)
    assert "CompletedCoreCurriculum" in targets
    assert "PassedScienceAssessment" in targets
    # `Student` gates every rule → a sort guard, never a snap target.
    assert "Student" not in targets


def test_harvest_registry_includes_all_rule_predicates():
    reg = harvest_registry(_SOPHIA_RULES)
    assert "Student" in reg and reg["Student"] == 1
    assert "QualifiedForAdvancedCourses" in reg


# ─── snapping an English claim onto the registry ──────────────────────────


def test_snap_clean_verb_object_facts():
    t = registry_targets(_SOPHIA_RULES)
    assert snap_text_to_registry("Sophia has completed the core curriculum.", t) == "CompletedCoreCurriculum"
    assert snap_text_to_registry("Sophia has passed the science assessment.", t) == "PassedScienceAssessment"
    assert snap_text_to_registry("Sophia has completed her capstone project.", t) == "CompletedCapstoneProject"


def test_snap_handles_numeric_paraphrase():
    # "maintains a GPA of 3.8" must reach GPAAbove3_5 (the rule's threshold predicate).
    t = registry_targets(_JOHN_RULES)
    assert snap_text_to_registry("John maintains a GPA of 3.8.", t) == "GPAAbove3_5"


def test_snap_recovers_goal_predicate_from_question():
    t = registry_targets(_JOHN_RULES)
    assert snap_text_to_registry("Does John receive academic distinction?", t) == "ReceivesAcademicDistinction"


def test_snap_rejects_incidental_single_token_overlap():
    # A claim sharing only one weak token with any target is not confidently mapped.
    t = registry_targets(_SOPHIA_RULES)
    assert snap_text_to_registry("Sophia visited the campus library yesterday.", t) is None


# ─── reshape a single fact / goal ─────────────────────────────────────────


def test_reshape_fact_binary_relation_to_unary_registry():
    targets = registry_targets(_SOPHIA_RULES)
    reg = harvest_registry(_SOPHIA_RULES)
    persons = {"sophia"}
    out = reshape_fact("Completed(Curriculum, sophia)", "Sophia has completed the core curriculum.",
                       targets, persons, reg)
    assert out == "CompletedCoreCurriculum(sophia)"


def test_reshape_fact_leaves_on_registry_fact_untouched():
    targets = registry_targets(_SOPHIA_RULES)
    reg = harvest_registry(_SOPHIA_RULES)
    # Already a registry predicate (correctly translated) → unchanged.
    fol = "QualifiedForAdvancedCourses(sophia)"
    assert reshape_fact(fol, "Sophia is qualified for advanced courses.", targets, {"sophia"}, reg) == fol


def test_reshape_fact_preserves_negation():
    rules = ["∀x (Driver(x) ∧ ReceivedSafetyEndorsement(x) → CanHaul(x))",
             "∀x (Driver(x) ∧ CanHaul(x) → CanCross(x))"]
    targets = registry_targets(rules)
    reg = harvest_registry(rules)
    out = reshape_fact("Lacks(john, endorsement)", "John has not received a safety endorsement.",
                       targets, {"john"}, reg)
    assert out == "¬ ReceivedSafetyEndorsement(john)"


def test_reshape_goal_uses_protagonist_when_fol_lost_entity():
    targets = registry_targets(_JOHN_RULES)
    reg = harvest_registry(_JOHN_RULES)
    # The mangled goal named no usable entity; fall back to the protagonist.
    out = reshape_goal("Receive(x)", "Does John receive academic distinction?",
                       targets, {"john"}, reg, protagonist="john")
    assert out == "ReceivesAcademicDistinction(john)"


def test_record_person_entities_finds_protagonist_in_wrong_shaped_facts():
    # John is buried in relation-argument position in every fact, never a unary
    # subject — but he recurs, so he is still recognised as the protagonist.
    fols = ["Completed(Courses, John)", "Maintain(John, 3.8)", "Completed(thesis, john)"]
    assert record_person_entities(fols) == {"john"}


# ─── schema_condition over a whole record ─────────────────────────────────


def test_schema_condition_noop_without_rules():
    # No rules → no registry → inputs returned unchanged.
    prem = ["Completed(Courses, John)", "Maintain(John, 3.8)"]
    goals = [("__goal__", ["Receive(John)"])]
    out_p, out_g = schema_condition(AnswerType.YES_NO_UNCERTAIN, prem, ["a", "b"], goals, ["q"])
    assert out_p == prem and out_g == goals


def test_schema_condition_reshapes_john_record():
    prem = ["Completed(Courses, John)", "Maintain(John, 3.8)", "Completed(thesis, john)"] + _JOHN_RULES
    prem_nl = [
        "John has completed all required courses.",
        "John maintains a GPA of 3.8.",
        "John has completed a thesis.",
        "rule", "rule", "rule", "rule",
    ]
    goals = [("__goal__", ["∃x (Person(x) ∧ John(x) → ∃y (Premise(y) ∧ Receive(x, y)))"])]
    out_p, out_g = schema_condition(
        AnswerType.YES_NO_UNCERTAIN, prem, prem_nl, goals,
        ["Does John receive academic distinction, according to the premises?"],
    )
    assert out_p[0] == "CompletesRequiredCourses(John)"
    assert out_p[1] == "GPAAbove3_5(John)"
    assert out_p[2] == "CompletesThesis(john)"
    assert out_g[0][1][0] == "ReceivesAcademicDistinction(John)"


# ─── the payoff: a record that scored WRONG now solves ────────────────────


def test_previously_wrong_john_record_now_solves_yes():
    """Record 6 in the run review: predicted 'Uncertain', gold 'Yes'. With the
    reshape it derives the full chain to ReceivesAcademicDistinction(john)."""
    prem = ["Completed(Courses, John)", "Maintain(John, 3.8)", "Completed(thesis, john)"] + _JOHN_RULES
    prem_nl = [
        "John has completed all required courses.",
        "John maintains a GPA of 3.8.",
        "John has completed a thesis.",
        "rule", "rule", "rule", "rule",
    ]
    goals = [("__goal__", ["∃x (Person(x) ∧ John(x) → ∃y (Premise(y) ∧ Receive(x, y)))"])]
    out_p, out_g = schema_condition(
        AnswerType.YES_NO_UNCERTAIN, prem, prem_nl, goals,
        ["Does John receive academic distinction, according to the premises?"],
    )
    code, _ = assemble_z3_program(out_p, out_g[0][1][0])  # bolt-ons + const-canon run here
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"


# ─── end-to-end through process_record with a wrong-shaping backend ────────


class WrongShapeBackend:
    """Reproduces the fvossel failure mode: rules translate cleanly, but ground
    FACTS come out as generic binary relations and the GOAL loses its predicate."""

    def translate_sentences(self, sentences: list[str], k: int) -> list[list[str]]:
        out: list[list[str]] = []
        for s in sentences:
            low = s.lower()
            if "completed the core curriculum" in low and "students" in low:
                fol = ("∀x (Student(x) ∧ CompletedCoreCurriculum(x) "
                       "∧ PassedScienceAssessment(x) → QualifiedForAdvancedCourses(x))")
            elif "qualified for advanced courses" in low and "students" in low:
                fol = ("∀x (Student(x) ∧ QualifiedForAdvancedCourses(x) "
                       "→ EligibleForInternationalProgram(x))")
            elif "sophia has completed the core curriculum" in low:
                fol = "Completed(Curriculum, sophia)"          # wrong shape: binary relation
            elif "sophia has passed the science assessment" in low:
                fol = "Passed(sophia, scienceAssessment)"      # wrong shape: binary relation
            elif "international program" in low:
                fol = "Eligible(sophia, internationalProgram)"  # mangled goal predicate
            else:
                fol = "Unknown(thing)"
            out.append([fol] * k)
        return out


def test_process_record_solves_with_schema_conditioning():
    rec = Record(
        id="ynu-wrongshape",
        premises_nl=[
            "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
            "Students who are qualified for advanced courses are eligible for the international program.",
            "Sophia has completed the core curriculum.",
            "Sophia has passed the science assessment.",
        ],
        question_nl="Is Sophia eligible for the international program?",
        answer_type=AnswerType.YES_NO_UNCERTAIN,
    )
    translator = LlamaFolTranslator(WrongShapeBackend(), LlamaFolConfig(k_samples=1))
    final, _ = process_record(rec, translator, SYMBOLIC_PCFG)
    assert final.answer == "Yes"
    assert final.debug.get("source") != "cot"  # solved symbolically, not via fallback


def test_disabling_schema_conditioning_leaves_the_record_dead():
    rec = Record(
        id="ynu-wrongshape-ablate",
        premises_nl=[
            "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
            "Students who are qualified for advanced courses are eligible for the international program.",
            "Sophia has completed the core curriculum.",
            "Sophia has passed the science assessment.",
        ],
        question_nl="Is Sophia eligible for the international program?",
        answer_type=AnswerType.YES_NO_UNCERTAIN,
    )
    cfg = LlamaFolConfig(k_samples=1, schema_conditioned=False)
    translator = LlamaFolTranslator(WrongShapeBackend(), cfg)
    final, _ = process_record(rec, translator, SYMBOLIC_PCFG)
    # Without the reshape the binary facts never connect to the unary rule guards.
    assert final.answer != "Yes"
