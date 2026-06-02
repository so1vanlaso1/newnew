"""Z3 runner using a Z3-Python DSL (safer subset of Python).

The translator emits a Z3 Python program shape:

    U = DeclareSort('U')
    WT = Function('WT', U, BoolSort())
    O  = Function('O',  U, BoolSort())
    x  = Const('x', U)
    alice = Const('alice', U)
    premises = [
        ForAll([x], Implies(WT(x), O(x))),
        WT(alice),
    ]
    goal = O(alice)

We `exec` this in a sandboxed namespace whose only globals are an allow-list
of Z3 constructors. AST validation runs first and rejects anything outside
that subset (no imports, no attribute access, no comprehensions, no lambdas,
no augmented assignments, etc.).

Entailment is then tested exactly like before: `premises ∧ ¬goal` unsat ⇒
goal is entailed; the symmetric check distinguishes Yes/No/Uncertain. MCQ
runs one entailment test per option and falls back to "Unknown" when none
of the options is entailed (a real answer in the EXACT dataset).
"""

from __future__ import annotations

import ast
import io
import keyword
import time
import tokenize
from dataclasses import dataclass
from typing import Iterable

import z3

from data.types import SolverVerdict


class UnsafeProgram(ValueError):
    pass


# Python reserved words can appear as predicate/constant names in the model's
# output (e.g. a predicate literally named `pass`). As a bare identifier that's
# a SyntaxError, so we rename such keyword NAME tokens to a safe form before
# exec. We only rename a keyword when it is used in *identifier* position —
# defined via `<kw> =` or called via `<kw>(` — then rename every reference to
# that name. Statement keywords (`import`, `for`, `lambda`, …) are left alone so
# the AST validator still rejects genuinely unsafe programs, and the value
# keywords `True`/`False`/`None` (valid allow-listed constants) are never
# touched. String literals (the Z3 symbol names) are also left untouched.
_RESERVED = set(keyword.kwlist) | set(getattr(keyword, "softkwlist", []))
_VALUE_KEYWORDS = {"True", "False", "None"}
_SKIP_TOK = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
}


def _sanitize_source(code: str) -> str:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code

    # Pass 1: collect keyword names used in identifier position.
    rename: set[str] = set()
    for i, tok in enumerate(toks):
        if tok.type != tokenize.NAME or tok.string not in _RESERVED:
            continue
        if tok.string in _VALUE_KEYWORDS:
            continue
        nxt = None
        for j in range(i + 1, len(toks)):
            if toks[j].type not in _SKIP_TOK:
                nxt = toks[j]
                break
        if nxt is not None and nxt.type == tokenize.OP and nxt.string in ("=", "("):
            rename.add(tok.string)

    if not rename:
        return code

    out = [
        tok._replace(string=tok.string + "_")
        if tok.type == tokenize.NAME and tok.string in rename
        else tok
        for tok in toks
    ]
    try:
        return tokenize.untokenize(out)
    except (ValueError, IndentationError):
        return code


# ─── Sandbox ─────────────────────────────────────────────────────────────
#
# Each program is built in its OWN fresh Z3 context. Z3 interns sorts and
# function symbols by name in a shared context, and quantifier reasoning (MBQI)
# is sensitive to that accumulated symbol table — so reusing the global context
# across the ~1.9k records (and across test files) makes some heavy nested-
# quantifier verdicts depend on evaluation order. A per-program context plus a
# fixed random seed makes every solve deterministic and fully isolated.


def _allowed_names(ctx: z3.Context) -> dict[str, object]:
    """Allow-list of Z3 constructors bound to a specific context. Sort/const/
    value builders take the context explicitly; connectives and quantifiers
    infer it from their (context-bound) arguments."""
    return {
        # Sorts / declarations
        "DeclareSort": lambda name: z3.DeclareSort(name, ctx),
        "BoolSort": lambda: z3.BoolSort(ctx),
        "IntSort": lambda: z3.IntSort(ctx),
        "RealSort": lambda: z3.RealSort(ctx),
        "Function": z3.Function,  # ctx inferred from the (ctx-bound) arg sorts
        "Const": z3.Const,        # ctx inferred from the sort
        "Consts": z3.Consts,
        "Bool": lambda name: z3.Bool(name, ctx),
        "Bools": lambda names: z3.Bools(names, ctx),
        "Int": lambda name: z3.Int(name, ctx),
        "Ints": lambda names: z3.Ints(names, ctx),
        "Real": lambda name: z3.Real(name, ctx),
        "Reals": lambda names: z3.Reals(names, ctx),
        "BoolVal": lambda v: z3.BoolVal(v, ctx),
        "IntVal": lambda v: z3.IntVal(v, ctx),
        "RealVal": lambda v: z3.RealVal(v, ctx),
        # Boolean connectives
        "Not": z3.Not,
        "And": z3.And,
        "Or": z3.Or,
        "Implies": z3.Implies,
        "Iff": lambda a, b: z3.And(z3.Implies(a, b), z3.Implies(b, a)),
        "Xor": z3.Xor,
        # Quantifiers
        "ForAll": z3.ForAll,
        "Exists": z3.Exists,
        # Equality
        "Distinct": z3.Distinct,
        # Constants
        "True": True,
        "False": False,
        "None": None,
    }


# Names the AST validator / sanitizer may need to recognise as allow-listed.
_ALLOWED_NAMES: frozenset[str] = frozenset(_allowed_names(z3.main_ctx()).keys())

# Fixed seed so MBQI / SAT tie-breaking is reproducible.
_SOLVER_SEED = 0


def _make_solver(ctx: z3.Context, timeout_ms: int, track_core: bool = False) -> z3.Solver:
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    s.set("random_seed", _SOLVER_SEED)
    if track_core:
        s.set(unsat_core=True)
    return s

# AST nodes allowed in the translated program.
_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Call,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Not,
    ast.IfExp,
)


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeProgram(f"disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            raise UnsafeProgram("attribute access is not allowed")
        if isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Name):
                raise UnsafeProgram("only direct calls to allow-listed names are permitted")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Loaded names must either be in the allow-list or have been assigned
            # earlier in the same program. Cheap check: defer assignment-set
            # tracking and rely on NameError at exec time for unknown names.
            pass


def _exec_program(code: str, ctx: z3.Context | None = None) -> dict[str, object]:
    code = _sanitize_source(code)
    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)
    ctx = ctx or z3.Context()
    ns: dict[str, object] = {"__builtins__": {}}
    ns.update(_allowed_names(ctx))
    exec(compile(tree, "<z3py>", "exec"), ns, ns)
    # Stash the context so callers can build the goal-override / extra solvers
    # in the same isolated context.
    ns["__z3_ctx__"] = ctx
    return ns


def premises_of(code: str) -> list[z3.BoolRef] | None:
    """Exec `code` and return its `premises` list if it is a non-empty list of
    Z3 BoolRefs, else None. Used to rank candidate premise environments for MCQ
    (the most premises ≈ the most fact-complete translation)."""
    try:
        ns = _exec_program(code)
    except Exception:
        return None
    premises = ns.get("premises")
    if (
        isinstance(premises, list)
        and premises
        and all(isinstance(p, z3.BoolRef) for p in premises)
    ):
        return premises
    return None


def _premises_and_goal(code: str, goal_override: str | None = None) -> tuple[list[z3.BoolRef], z3.BoolRef]:
    """Extract `premises` (list) and `goal` (Bool) from the executed namespace."""
    ns = _exec_program(code)
    premises = ns.get("premises")
    if not isinstance(premises, list) or not all(isinstance(p, z3.BoolRef) for p in premises):
        raise UnsafeProgram("`premises` must be a list of Z3 BoolRef")
    if goal_override is not None:
        # Re-exec a small extra snippet in the same namespace so the goal can
        # reference declared sorts/predicates/constants from `code`.
        goal_src = _sanitize_source(f"_goal_override = {goal_override}")
        goal_tree = ast.parse(goal_src, mode="exec")
        _validate_ast(goal_tree)
        exec(compile(goal_tree, "<goal>", "exec"), ns, ns)
        goal = ns.get("_goal_override")
    else:
        goal = ns.get("goal")
    if not isinstance(goal, z3.BoolRef):
        raise UnsafeProgram("`goal` must be a Z3 BoolRef")
    return premises, goal


# ─── Entailment checks ───────────────────────────────────────────────────


def _check_entailment(
    premises: list[z3.BoolRef],
    goal: z3.BoolRef,
    timeout_ms: int,
    track_core: bool,
) -> tuple[z3.CheckSatResult, list[str], z3.ModelRef | None]:
    """Check `premises ∧ ¬goal`. Returns (result, unsat-core labels, model).

    The model is populated only when the result is `sat` (i.e. ¬goal is
    consistent with the premises) — it is the counter-model that demonstrates
    the goal need not hold.
    """
    s = _make_solver(goal.ctx, timeout_ms, track_core=track_core)
    if track_core:
        for i, p in enumerate(premises):
            s.assert_and_track(p, f"p{i}")
        s.add(z3.Not(goal))
    else:
        for p in premises:
            s.add(p)
        s.add(z3.Not(goal))
    result = s.check()
    core_labels: list[str] = []
    model: z3.ModelRef | None = None
    if result == z3.unsat and track_core:
        core_labels = [str(c) for c in s.unsat_core()]
    elif result == z3.sat:
        model = s.model()
    return result, core_labels, model


# ─── Counter-model witnesses (for "Uncertain") ────────────────────────────

_BOOL_CONNECTIVES = frozenset({
    z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_NOT,
    z3.Z3_OP_IMPLIES, z3.Z3_OP_IFF, z3.Z3_OP_XOR,
})
_MAX_WITNESS_ATOMS = 16


def _has_bound_var(expr: z3.ExprRef) -> bool:
    if z3.is_var(expr):
        return True
    return any(_has_bound_var(c) for c in expr.children())


def _collect_ground_atoms(
    expr: z3.ExprRef, acc: list[z3.BoolRef], seen: set[str]
) -> None:
    """Gather the ground (quantifier-free) Boolean atoms a formula mentions —
    predicate applications and comparisons whose truth a model fixes concretely.
    Descends through quantifiers and Boolean connectives; skips anything that
    still references a bound variable."""
    if z3.is_quantifier(expr):
        _collect_ground_atoms(expr.body(), acc, seen)
        return
    if not z3.is_app(expr):
        return
    kind = expr.decl().kind()
    if kind in _BOOL_CONNECTIVES:
        for child in expr.children():
            _collect_ground_atoms(child, acc, seen)
        return
    if kind in (z3.Z3_OP_TRUE, z3.Z3_OP_FALSE):
        return
    if expr.sort_kind() == z3.Z3_BOOL_SORT and not _has_bound_var(expr):
        key = str(expr)
        if key not in seen:
            seen.add(key)
            acc.append(expr)


def _summarize_model(
    model: z3.ModelRef, atoms: list[z3.BoolRef]
) -> dict[str, bool]:
    """Evaluate each ground atom in the model -> {atom_text: truth value}."""
    summary: dict[str, bool] = {}
    for atom in atoms[:_MAX_WITNESS_ATOMS]:
        try:
            summary[str(atom)] = z3.is_true(model.eval(atom, model_completion=True))
        except z3.Z3Exception:
            continue
    return summary


def _skolemized_goal(goal: z3.BoolRef) -> tuple[z3.BoolRef, bool] | None:
    """If `goal` is a top-level universal `∀x… φ`, instantiate its bound
    variables with fresh constants and return (φ[c…], True). For a top-level
    existential return (φ[c…], False). Otherwise return None.

    A fresh constant that occurs nowhere in the premises can always play the
    witness/counter-witness element, so this turns a quantified Uncertain goal
    into ground atoms we can actually evaluate."""
    if not z3.is_quantifier(goal):
        return None
    consts = [
        z3.Const(f"_w_{goal.var_name(i)}", goal.var_sort(i))
        for i in range(goal.num_vars())
    ]
    # substitute_vars maps de Bruijn Var(0) to the LAST bound variable, so the
    # constant list must be reversed relative to var_name ordering.
    body = z3.substitute_vars(goal.body(), *reversed(consts))
    return body, bool(goal.is_forall())


def _uncertain_witness(
    premises: list[z3.BoolRef],
    goal: z3.BoolRef,
    goal_true_model: z3.ModelRef | None,
    goal_false_model: z3.ModelRef | None,
    timeout_ms: int,
) -> dict[str, dict[str, bool]] | None:
    """Build a two-scenario counter-model witness for an Uncertain verdict.

    Prefers the ground atoms the problem already mentions (evaluated in the two
    models from the entailment checks). If the goal is purely quantified and has
    no ground atoms, it skolemizes the goal and re-solves for a concrete witness
    element so the explanation still has something specific to point at."""
    atoms: list[z3.BoolRef] = []
    seen: set[str] = set()
    for p in premises:
        _collect_ground_atoms(p, atoms, seen)
    _collect_ground_atoms(goal, atoms, seen)
    if atoms and goal_true_model is not None and goal_false_model is not None:
        return {
            "goal_true": _summarize_model(goal_true_model, atoms),
            "goal_false": _summarize_model(goal_false_model, atoms),
        }

    skolem = _skolemized_goal(goal)
    if skolem is None:
        return None
    body, _is_forall = skolem
    body_atoms: list[z3.BoolRef] = []
    body_seen: set[str] = set()
    _collect_ground_atoms(body, body_atoms, body_seen)
    if not body_atoms:
        return None

    def _model_of(extra: z3.BoolRef) -> z3.ModelRef | None:
        s = _make_solver(goal.ctx, timeout_ms)
        for p in premises:
            s.add(p)
        s.add(extra)
        return s.model() if s.check() == z3.sat else None

    # Witness for the bound variable: one element where the body fails (goal can
    # be false) and one where it holds (goal can be true).
    m_false = _model_of(z3.Not(body))
    m_true = _model_of(body)
    if m_false is None or m_true is None:
        return None
    return {
        "goal_true": _summarize_model(m_true, body_atoms),
        "goal_false": _summarize_model(m_false, body_atoms),
    }


def _premise_core(premises: list[z3.BoolRef], timeout_ms: int) -> list[str]:
    """Unsat core of the premises on their own (used when they are mutually
    contradictory). Returns label strings like ['p1', 'p4'] or [] if not unsat."""
    if not premises:
        return []
    s = _make_solver(premises[0].ctx, timeout_ms, track_core=True)
    for i, p in enumerate(premises):
        s.assert_and_track(p, f"p{i}")
    if s.check() == z3.unsat:
        return [str(c) for c in s.unsat_core()]
    return []


def _minimize_core(
    premises: list[z3.BoolRef],
    extra: z3.BoolRef,
    core_labels: list[str],
    timeout_ms: int,
) -> list[str]:
    """Shrink an unsat core to an irreducible subset by greedy drop-one.

    `extra` is the assertion added alongside the premises (¬goal for a Yes
    verdict, goal for a No verdict). Z3's reported core is not guaranteed
    minimal; we drop premises one at a time, keeping the conflict unsat."""
    idx = sorted({int(l[1:]) for l in core_labels if l[1:].isdigit()})
    if len(idx) <= 1:
        return [f"p{i}" for i in idx]

    def _still_unsat(subset: list[int]) -> bool:
        s = _make_solver(extra.ctx, timeout_ms)
        for i in subset:
            s.add(premises[i])
        s.add(extra)
        return s.check() == z3.unsat

    current = list(idx)
    for candidate in idx:
        trial = [i for i in current if i != candidate]
        if trial and _still_unsat(trial):
            current = trial
    return [f"p{i}" for i in current]


def run_yes_no_uncertain(
    z3py_code: str,
    timeout_ms: int = 5000,
    emit_unsat_core: bool = True,
) -> SolverVerdict:
    t0 = time.perf_counter()
    try:
        premises, goal = _premises_and_goal(z3py_code)
    except (UnsafeProgram, SyntaxError, Exception) as e:
        return SolverVerdict(
            answer=None, status="parse_error", error=str(e),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # pos check asserts premises ∧ ¬goal: a model here is a scenario where the
    # goal is FALSE. neg check asserts premises ∧ goal: its model is a scenario
    # where the goal is TRUE.
    pos_result, pos_core, goal_false_model = _check_entailment(
        premises, goal, timeout_ms, emit_unsat_core
    )
    neg_result, neg_core, goal_true_model = _check_entailment(
        premises, z3.Not(goal), timeout_ms, emit_unsat_core
    )
    elapsed = (time.perf_counter() - t0) * 1000
    pos_unsat = pos_result == z3.unsat
    neg_unsat = neg_result == z3.unsat

    # Decide on any definitive contradiction first, so a timeout on one side
    # never discards a sound verdict already proven on the other.
    if pos_unsat and neg_unsat:
        # premises ∧ ¬goal AND premises ∧ goal are both unsat ⇒ the premises are
        # self-contradictory. Surface which premises clash (their own core).
        clash = _minimize_core(
            premises, z3.BoolVal(True, goal.ctx),
            _premise_core(premises, timeout_ms), timeout_ms,
        )
        return SolverVerdict(
            answer=None, status="inconsistent", unsat_core=clash,
            error=f"inconsistent premises: {clash}" if clash else "inconsistent premises",
            elapsed_ms=elapsed,
        )
    if pos_unsat:  # premises ⊨ goal
        core = _minimize_core(premises, z3.Not(goal), pos_core, timeout_ms) if pos_core else pos_core
        return SolverVerdict(answer="Yes", status="solved", unsat_core=core, elapsed_ms=elapsed)
    if neg_unsat:  # premises ⊨ ¬goal
        core = _minimize_core(premises, goal, neg_core, timeout_ms) if neg_core else neg_core
        return SolverVerdict(answer="No", status="solved", unsat_core=core, elapsed_ms=elapsed)
    if pos_result == z3.unknown or neg_result == z3.unknown:
        return SolverVerdict(answer=None, status="timeout", elapsed_ms=elapsed)

    # Uncertain: both the goal and its negation are consistent with the premises.
    witness = _uncertain_witness(
        premises, goal, goal_true_model, goal_false_model, timeout_ms
    )
    return SolverVerdict(
        answer="Uncertain", status="solved", witness=witness, elapsed_ms=elapsed
    )


def run_mcq(
    z3py_code: str,
    option_goals: list[str],
    timeout_ms: int = 5000,
    emit_unsat_core: bool = True,
) -> SolverVerdict:
    """Pick the option whose goal is entailed by the premises; otherwise emit
    'Unknown', which is a real answer in the EXACT dataset when no listed
    option follows from the premises."""
    t0 = time.perf_counter()
    candidates: list[tuple[int, list[str]]] = []
    for i, goal_src in enumerate(option_goals):
        try:
            premises, goal = _premises_and_goal(z3py_code, goal_override=goal_src)
        except Exception:
            continue
        result, core, _model = _check_entailment(premises, goal, timeout_ms, emit_unsat_core)
        if result == z3.unsat:
            candidates.append((i, core))

    elapsed = (time.perf_counter() - t0) * 1000
    if len(candidates) == 1:
        chosen_idx, chosen_core = candidates[0]
        core = chosen_core
        if chosen_core:
            c_premises, c_goal = _premises_and_goal(
                z3py_code, goal_override=option_goals[chosen_idx]
            )
            core = _minimize_core(c_premises, z3.Not(c_goal), chosen_core, timeout_ms)
        return SolverVerdict(
            answer=str(chosen_idx), status="solved",
            unsat_core=core, elapsed_ms=elapsed,
        )
    # Zero or >1 options entailed → no unique answer. Abstaining with 'Unknown'
    # is sound: a tie means the premises don't single out one option, and a
    # largest-core tiebreak would be arbitrary.
    return SolverVerdict(answer="Unknown", status="solved", elapsed_ms=elapsed)
