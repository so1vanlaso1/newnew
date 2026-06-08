"""Bolt-on C — case-variant entity unification (translator.fol_repair.canonicalize_constants).

The per-sentence translator splits a protagonist across `John`/`john`, `David`/`david`,
so Z3 sees two individuals and no chain about that person closes. These tests prove
the constants are unified (and that predicates are left alone), then exec the real
solver to show a record split across `John`/`john` now derives its goal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solver.z3_runner import run_yes_no_uncertain
from translator.fol_repair import apply_renames, canonicalize_constants
from translator.llama_fol import assemble_z3_program


def test_unifies_case_variant_constants():
    fols = ["Completed(Courses, John)", "Maintain(John, 3.8)", "Completed(thesis, john)"]
    m = canonicalize_constants(fols)
    # `John` (freq 2) wins over `john` (freq 1); the lone variant is remapped.
    assert m == {"john": "John"}


def test_prefers_lowercase_on_a_tie():
    fols = ["P(John)", "Q(john)"]  # 1 each → tie → lowercase canonical
    assert canonicalize_constants(fols) == {"John": "john"}


def test_no_collision_leaves_constants_alone():
    fols = ["P(sophia)", "Q(sophia)", "R(curriculum)"]
    assert canonicalize_constants(fols) == {}


def test_does_not_touch_predicate_names():
    # `John` here is a PREDICATE, not an argument — never canonicalised as a constant.
    fols = ["∀x (John(x) → Happy(x))", "Happy(john)"]
    m = canonicalize_constants(fols)
    assert "John" not in m  # predicate-position John is untouched


def test_apply_renames_is_whole_word_only():
    out = apply_renames("Completed(thesis, john)", {"john": "John"})
    assert out == "Completed(thesis, John)"
    # A substring must not be rewritten.
    assert apply_renames("Johnson(x)", {"John": "john"}) == "Johnson(x)"


def test_split_entity_chain_solves_after_unification():
    """A record whose protagonist is split across `John`/`john`: the goal predicate
    is proved of `john` but the supporting fact is about `John`. Only after
    canonicalisation (run inside assemble_z3_program) does the chain close."""
    premises = [
        "∀x (HasCompletedCourses(x) → Graduates(x))",
        "HasCompletedCourses(John)",   # fact about `John`
    ]
    goal = "Graduates(john)"           # asked about `john`
    code, _ = assemble_z3_program(premises, goal)
    assert code is not None
    assert run_yes_no_uncertain(code).answer == "Yes"

    # Ablation parity: prove the split genuinely blocks it without unification.
    from translator.fol_converter import convert_premises_to_z3py

    setup, prem, gexpr, _ = convert_premises_to_z3py(premises, goal_fol=goal)
    split_code = "\n".join(setup) + "\npremises = [\n" + ",\n".join(f"    {p}" for p in prem) + "\n]\n"
    split_code += f"goal = {gexpr}\n"
    assert run_yes_no_uncertain(split_code).answer != "Yes"
