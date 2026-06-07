"""Predicate-name grouping: extraction, LLM-cluster parsing, deterministic
rename, and the payoff — aligning synonyms makes an otherwise-unsolved record
solve. No model is loaded; the LLM is a fake returning canned clusters.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solver.z3_runner import run_yes_no_uncertain
from translator.predicate_group import (
    apply_canonical,
    build_canonical_map,
    group_relations,
    parse_grouping_response,
    relation_name_counts,
    safe_canonical_map,
)
from translator.llama_fol import assemble_z3_program, normalize_fol


# ─── name extraction ──────────────────────────────────────────────────────


def test_relation_name_counts_collects_predicates():
    counts = relation_name_counts(
        ["FORALL x (Student(x) -> Person(x))", "Student(Alice)"]
    )
    assert counts["Student"] == 2
    assert counts["Person"] == 1
    # `Alice` is a constant/entity, not a relation → excluded.
    assert "Alice" not in counts


# ─── parsing the LLM response ─────────────────────────────────────────────


def test_parse_grouping_response_keeps_known_pairs():
    known = {"EligibleForScholarship", "QualifyForUniversityScholarship", "Student"}
    text = 'noise [["EligibleForScholarship", "QualifyForUniversityScholarship"]] trailing'
    clusters = parse_grouping_response(text, known)
    assert clusters == [["EligibleForScholarship", "QualifyForUniversityScholarship"]]


def test_parse_grouping_response_drops_unknown_and_singletons():
    known = {"A", "B"}
    text = '[["A", "ZZZ"], ["A", "B"]]'  # first group → only A known (singleton)
    assert parse_grouping_response(text, known) == [["A", "B"]]


def test_parse_grouping_response_handles_garbage():
    assert parse_grouping_response("no json here", {"A"}) == []
    assert parse_grouping_response("[not, valid, json]", {"A"}) == []


# ─── canonical map + rename ───────────────────────────────────────────────


def test_canonical_prefers_more_frequent_name():
    from collections import Counter

    clusters = [["Qualify", "Eligible"]]
    counts = Counter({"Qualify": 1, "Eligible": 3})
    mapping = build_canonical_map(clusters, counts)
    assert mapping == {"Qualify": "Eligible"}  # the dominant name wins


def test_apply_canonical_renames_whole_words_only():
    mapping = {"Qualify": "Eligible"}
    out = apply_canonical("Qualify(Bob) ∧ Disqualify(Bob)", mapping)
    # `Qualify` renamed; the substring inside `Disqualify` left intact.
    assert out == "Eligible(Bob) ∧ Disqualify(Bob)"


def test_apply_canonical_no_mapping_is_identity():
    assert apply_canonical("P(a)", {}) == "P(a)"


# ─── group_relations end-to-end with a fake LLM ───────────────────────────


def _fake_chat(_messages):
    return '[["QualifyForUniversityScholarship", "EligibleForScholarship"]]'


def test_group_relations_builds_mapping():
    fol = [
        "FORALL x (EligibleForScholarship(x) -> HasFunding(x))",
        "QualifyForUniversityScholarship(Sophia)",
    ]
    mapping = group_relations(fol, _fake_chat)
    # Both names appear once → tie broken by shorter, then alpha.
    assert set(mapping) <= {"QualifyForUniversityScholarship", "EligibleForScholarship"}
    assert len(mapping) == 1
    # After applying, both formulas use one symbol.
    canon = next(iter(mapping.values()))
    renamed = [apply_canonical(s, mapping) for s in fol]
    assert all(canon in r for r in renamed)


# ─── the payoff: grouping turns an unsolved record into a proof ───────────


def test_grouping_makes_record_solve():
    # The RULE uses one predicate name and the FACT uses a synonym, so the synonym
    # link is the ONLY way to fire the rule. Without grouping they're unrelated
    # symbols and the goal is not entailed.
    premises = [
        "FORALL x (EligibleForScholarship(x) -> ReceivesFunding(x))",
        "QualifyForUniversityScholarship(Sophia)",
    ]
    goal = "ReceivesFunding(Sophia)"

    # Baseline: distinct predicates → rule never fires → NOT entailed (Uncertain).
    code_before, _ = assemble_z3_program(premises, goal)
    assert run_yes_no_uncertain(code_before).answer != "Yes"

    # Group the synonyms, then re-assemble.
    all_fol = [normalize_fol(s) for s in premises + [goal]]
    mapping = group_relations(
        all_fol,
        lambda _m: '[["EligibleForScholarship", "QualifyForUniversityScholarship"]]',
    )
    prem2 = [apply_canonical(normalize_fol(s), mapping) for s in premises]
    goal2 = apply_canonical(normalize_fol(goal), mapping)
    code_after, _ = assemble_z3_program(prem2, goal2)
    assert run_yes_no_uncertain(code_after).answer == "Yes"


# ─── bolt-on D: safe_canonical_map (guarded LLM + deterministic matching) ──


def test_safe_merges_morphological_typo_twin_without_llm():
    # The rule says `…Assement`, the fact says `…Assessory`; no LLM cluster.
    fol = [
        "FORALL x (Student(x) ∧ PassedScienceAssement(x) → Qualified(x))",
        "PassedScienceAssessory(sophia)",
        "Student(sophia)",
    ]
    m = safe_canonical_map(fol, llm_clusters=[])
    # Both count once → canonical is the shorter name.
    assert m == {"PassedScienceAssessory": "PassedScienceAssement"}


def test_safe_merges_prefix_twin():
    fol = [
        "FORALL x (A(x) → CompletedResearchMethodology(x))",
        "CompletedResearchMethodologyCourse(sophia)",
    ]
    m = safe_canonical_map(fol, llm_clusters=[])
    assert m == {"CompletedResearchMethodologyCourse": "CompletedResearchMethodology"}


def test_safe_never_merges_an_entity_name():
    # T5 emitted the entity as a predicate `Sophia(x)`; the LLM wants Sophia≈Student.
    fol = ["FORALL x (Sophia(x) → Eligible(x))", "Eligible(sophia)"]
    m = safe_canonical_map(fol, llm_clusters=[["Sophia", "Student"], ["Sophia", "Eligible"]])
    assert "Sophia" not in m and "Sophia" not in m.values()


def test_safe_rejects_cross_arity_even_when_names_are_similar():
    fol = [
        "FORALL x (CompletedProject(x) → Done(x))",      # unary
        "CompletedProjects(sophia, math)",               # binary, near-identical name
    ]
    m = safe_canonical_map(fol, llm_clusters=[["CompletedProject", "CompletedProjects"]])
    # Same string family, but different arity → must NOT merge.
    assert "CompletedProjects" not in m and "CompletedProject" not in m


def test_safe_rejects_opposite_polarity():
    fol = [
        "FORALL x (Member(x) → QualifyScholarship(x))",
        "NOTQualifyScholarship(sophia)",
    ]
    # Tiny edit distance, but one is the other's negation → never merge.
    m = safe_canonical_map(fol, llm_clusters=[["QualifyScholarship", "NOTQualifyScholarship"]])
    assert m == {}


def test_safe_keeps_semantic_llm_merge_strings_cannot_see():
    fol = [
        "FORALL x (Member(x) → EligibleForScholarship(x))",
        "QualifyForUniversityScholarship(sophia)",
    ]
    m = safe_canonical_map(
        fol, llm_clusters=[["EligibleForScholarship", "QualifyForUniversityScholarship"]]
    )
    assert m == {"QualifyForUniversityScholarship": "EligibleForScholarship"}


# The capstone: the real q0 record (raw T5 FOL + the real Qwen clusters) solves.
_Q0_PREMISES = [
    "FORALLx (Student(x) AND CompletedCoreCurriculum(x) AND PassedScienceAssement(x) IMPLIES QualifiedForAdvancedCourses(x))",
    "FORALLx (Student(x) AND QualifiedForAdvancedCourses(x) AND CompletedResearchMethodology(x) IMPLIES EligibleForInternationalProgram(x))",
    "FORALLx (Student(x) AND PassedLanguageProficiencyExam(x) IMPLIES EligibleForInternationalProgram(x))",
    "FORALLx (Student(x) AND EligibleForInternationalProgram(x) AND CompletedCapstoneProject(x) IMPLIES AwardedHonorsDiploma(x))",
    "FORALLx (Student(x) AND AwardedHonorsDiploma(x) AND CompletedCommunityService(x) IMPLIES EligibleForScholarship(x))",
    "FORALLx (Student(x) AND AwardedHonorsDiploma(x) AND ReceivedFacultyRecommendation(x) IMPLIES QualifyForUniversityScholarship(x))",
    "CompletedCoreCurriculum(sophia)",
    "PassedScienceAssessory(sophia)",
    "CompletedResearchMethodologyCourse(sophia)",
    "CompletedCapstoneProject(sophia)",
    "CompletedCommunityService(sophia)",
]
_Q0_GOAL = "FORALLx (Sophia(x) IMPLIES EligibleForScholarship(x))"  # gold option, mangled
_Q0_LLM_CLUSTERS = [
    ["AwardedHonorsDiploma", "GetsHonorsDiploma", "ReceivedFacultyRecommendation"],
    ["EligibleForScholarship", "QualifyForUniversityScholarship"],
    ["CompletedResearchMethodology", "CompletedResearchMethodologyCourse"],
    ["PassedLanguageProficiencyExam", "PassesLanguageProficiencyExam"],
    ["Sophia", "Student"],
]


def test_safe_align_solves_real_q0_gold_option():
    all_fol = [normalize_fol(s) for s in _Q0_PREMISES + [_Q0_GOAL]]
    m = safe_canonical_map(all_fol, _Q0_LLM_CLUSTERS)
    # The catastrophic merges are blocked …
    assert "Sophia" not in m and "Sophia" not in m.values()
    # … the typo twin the LLM missed is fixed deterministically …
    assert m.get("PassedScienceAssessory") == "PassedScienceAssement"
    # … and the two concepts that must stay distinct are not fused.
    assert "EligibleForInternationalProgram" not in m

    prem = [apply_canonical(normalize_fol(s), m) for s in _Q0_PREMISES]
    goal = apply_canonical(normalize_fol(_Q0_GOAL), m)
    code, _ = assemble_z3_program(prem, goal)  # A+B+(C) inside
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"
