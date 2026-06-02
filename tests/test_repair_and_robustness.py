"""Tests for the robustness fixes: program repair, MCQ premise-env selection,
and the LoRA/base compatibility preflight guard. No GPU needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.types import AnswerType, Record, Translation
from pipeline import PipelineConfig, _best_premise_env, _solve_mcq
from solver.z3_runner import premises_of
from translator.infer import _adapter_target_paths, assert_adapter_matches_base
from translator.parse import extract_goal_expr
from translator.repair import repair_program


# ─── repair_program ──────────────────────────────────────────────────────

# Mirrors the real failure in fol_nothink.txt option A: a predicate used in the
# premises and goal but never declared → NameError at exec time.
_UNDECLARED = """
Universe = DeclareSort('Universe')
honors = Function('honors', Universe, BoolSort())
sophia = Const('sophia', Universe)
x = Const('x', Universe)
premises = [
    ForAll([x], Implies(honors(x), university_scholarship(x))),
    honors(sophia),
]
goal = university_scholarship(sophia)
"""


def test_repair_makes_undeclared_program_executable() -> None:
    assert premises_of(_UNDECLARED) is None  # crashes before repair
    repaired = repair_program(_UNDECLARED)
    assert premises_of(repaired) is not None  # runs after repair
    assert "university_scholarship = Function(" in repaired


def test_repair_is_idempotent_on_complete_program() -> None:
    complete = (
        "Universe = DeclareSort('Universe')\n"
        "p = Function('p', Universe, BoolSort())\n"
        "a = Const('a', Universe)\n"
        "premises = [p(a)]\n"
        "goal = p(a)"
    )
    assert repair_program(complete) == complete


def test_repair_adds_missing_universe_sort() -> None:
    no_sort = "premises = [p(a)]\ngoal = p(a)"
    repaired = repair_program(no_sort)
    assert "DeclareSort(" in repaired
    assert premises_of(repaired) is not None


# ─── MCQ premise-env selection ───────────────────────────────────────────

# Option A: undeclared symbol AND no individual facts (mirrors the real bug).
_OPT_A = """
Universe = DeclareSort('Universe')
advanced = Function('advanced', Universe, BoolSort())
eligible = Function('eligible', Universe, BoolSort())
core = Function('core', Universe, BoolSort())
research = Function('research', Universe, BoolSort())
sophia = Const('sophia', Universe)
x = Const('x', Universe)
premises = [
    ForAll([x], Implies(core(x), advanced(x))),
    ForAll([x], Implies(And(advanced(x), research(x)), eligible(x))),
]
goal = scholarship(sophia)
"""

# Option C: complete — same rules PLUS the grounded facts about Sophia.
_OPT_C = """
Universe = DeclareSort('Universe')
advanced = Function('advanced', Universe, BoolSort())
eligible = Function('eligible', Universe, BoolSort())
core = Function('core', Universe, BoolSort())
research = Function('research', Universe, BoolSort())
sophia = Const('sophia', Universe)
x = Const('x', Universe)
premises = [
    ForAll([x], Implies(core(x), advanced(x))),
    ForAll([x], Implies(And(advanced(x), research(x)), eligible(x))),
    core(sophia),
    research(sophia),
]
goal = eligible(sophia)
"""


def test_best_premise_env_prefers_fact_complete_program() -> None:
    # Even though option A comes first, the fact-complete program must win.
    env = _best_premise_env([_OPT_A, _OPT_C])
    assert env is not None
    assert "core(sophia)" in env  # picked the one carrying the facts


def _mcq_groups() -> list[list[Translation]]:
    options_code = [_OPT_A, _OPT_A, _OPT_C, _OPT_A]  # only option C is complete
    return [
        [Translation(code=repair_program(c), goal_expr=extract_goal_expr(c),
                     raw_text="", sample_index=0)]
        for c in options_code
    ]


def test_solve_mcq_recovers_answer_despite_broken_option_a() -> None:
    options = ["scholarship", "faculty", "eligible", "language"]
    # Each option's goal comes from its own program; give every option a distinct,
    # correctly-declared goal so the entailed one (C: eligible) is identifiable.
    groups = _mcq_groups()
    groups[1][0].goal_expr = "faculty(sophia)"
    groups[3][0].goal_expr = "language(sophia)"
    verdicts, _ = _solve_mcq(groups, options, PipelineConfig())
    # Option C's goal (eligible(sophia)) is the only one entailed by the premises.
    assert verdicts[0].answer == "eligible", verdicts


def test_solve_mcq_parse_error_when_no_option_executes() -> None:
    broken = "this is not python ((("
    groups = [[Translation(code=broken, goal_expr=None, raw_text="", sample_index=0)]
              for _ in range(3)]
    verdicts, _ = _solve_mcq(groups, ["a", "b", "c"], PipelineConfig())
    assert verdicts[0].status == "parse_error"


# ─── LoRA / base preflight guard ─────────────────────────────────────────


class _FakeBase:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_modules(self):
        return [(n, object()) for n in self._names]


def test_adapter_target_paths_strips_peft_prefix_and_lora_suffix() -> None:
    keys = [
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
    ]
    paths = _adapter_target_paths(keys)
    assert paths == {
        "model.layers.0.mlp.down_proj",
        "model.layers.3.self_attn.q_proj",
    }


def _write_fake_adapter(tmp_path: Path) -> str:
    """A LoRA dir whose adapter targets `model.layers.*` (dense text model)."""
    from safetensors.torch import save_file
    import torch

    tensors = {
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros(2, 2),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros(2, 2),
    }
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    return str(tmp_path)


def test_preflight_raises_on_architecture_mismatch(tmp_path: Path) -> None:
    lora = _write_fake_adapter(tmp_path)
    # Base whose text layers live under `model.language_model.layers.*` (the
    # multimodal model that caused the real bug) — no overlap with the adapter.
    vl_base = _FakeBase(["model.language_model.layers.0.mlp.down_proj"])
    with pytest.raises(RuntimeError, match="LoRA/base mismatch"):
        assert_adapter_matches_base(vl_base, lora)


def test_preflight_passes_on_matching_base(tmp_path: Path) -> None:
    lora = _write_fake_adapter(tmp_path)
    text_base = _FakeBase(["model.layers.0.mlp.down_proj"])
    assert_adapter_matches_base(text_base, lora)  # no raise
