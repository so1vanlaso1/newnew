"""Mechanical repair of translator-emitted Z3-Python programs.

A translator (or, when the LoRA is weak or — as we found — applied to the wrong
base model, a near-base model) sometimes emits a program that *looks* like a
valid `<z3py>` block but will not `exec` in the solver sandbox: a predicate is
applied but never declared (`NameError`), the universe sort is referenced before
its `DeclareSort` line, etc.

These faults are detectable from the program's own AST and fixable WITHOUT
touching the logic. Declaring a previously-undeclared predicate adds **no**
constraint — it just introduces a free (unconstrained) symbol, which is the
correct open-world reading and exactly what Z3 needs to run. So repair only ever
*adds* declarations; it never edits or drops existing statements. If no safe
repair applies, the input is returned unchanged.

This is a safety net, not a substitute for a correct model: it salvages an
otherwise-executable program, but it cannot invent premises the model omitted.
"""

from __future__ import annotations

import ast
import re

# Mirror of the solver's allow-list (solver.z3_runner._allowed_names). Kept as a
# local literal so repair stays a pure-AST pass with no z3 import.
_Z3_BUILTINS: frozenset[str] = frozenset({
    "DeclareSort", "BoolSort", "IntSort", "RealSort", "Function", "Const",
    "Consts", "Bool", "Bools", "Int", "Ints", "Real", "Reals", "BoolVal",
    "IntVal", "RealVal", "Not", "And", "Or", "Implies", "Iff", "Xor",
    "ForAll", "Exists", "Distinct", "True", "False", "None",
})

# Names that are bound by the program structure itself, not declarations.
_STRUCTURAL = frozenset({"premises", "goal"})

_PREMISES_LINE = re.compile(r"^\s*premises\s*=")


def _mark_numeric(node: ast.AST, numeric: set[str]) -> None:
    """Record a name/function used in a numeric (RealSort) position."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        numeric.add(node.func.id)
    elif isinstance(node, ast.Name):
        numeric.add(node.id)


def _find_sort_name(tree: ast.AST) -> str | None:
    """Return the LHS name of `X = DeclareSort(...)`, or None if absent."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "DeclareSort"
            and node.targets
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id
    return None


def repair_program(code: str) -> str:
    """Return `code` with declarations added for any symbol it uses but never
    declares (and a universe sort if missing). Returns `code` unchanged when it
    is unparseable or already self-contained."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    declared.add(tgt.id)

    sort_name = _find_sort_name(tree)
    add_sort = sort_name is None
    if add_sort:
        sort_name = "Universe"

    used: set[str] = set()
    call_arity: dict[str, int] = {}   # name -> max arity seen (>=1 ⇒ function)
    arg_names: set[str] = set()        # names passed as call arguments (individuals)
    numeric: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if node.args:
                call_arity[name] = max(call_arity.get(name, 0), len(node.args))
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    arg_names.add(arg.id)
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for op in node.ops
        ):
            for operand in (node.left, *node.comparators):
                _mark_numeric(operand, numeric)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
        ):
            _mark_numeric(node.left, numeric)
            _mark_numeric(node.right, numeric)

    undeclared = (used | set(call_arity)) - declared - _Z3_BUILTINS - _STRUCTURAL - {sort_name}
    if not undeclared and not add_sort:
        return code

    decls: list[str] = []
    if add_sort:
        decls.append(f"{sort_name} = DeclareSort('{sort_name}')")
    for name in sorted(undeclared):
        if name in call_arity:                       # applied → Function
            argsorts = ", ".join([sort_name] * call_arity[name])
            ret = "RealSort()" if name in numeric else "BoolSort()"
            decls.append(f"{name} = Function('{name}', {argsorts}, {ret})")
        elif name in arg_names:                      # passed as an argument → individual
            decls.append(f"{name} = Const('{name}', {sort_name})")
        else:                                        # bare boolean atom → nullary predicate
            sort = "RealSort()" if name in numeric else "BoolSort()"
            decls.append(f"{name} = Const('{name}', {sort})")

    lines = code.splitlines()
    inject_at = next(
        (i for i, ln in enumerate(lines) if _PREMISES_LINE.match(ln)),
        None,
    )
    if inject_at is None:
        inject_at = 0
        if not add_sort:
            for i, ln in enumerate(lines):
                if "DeclareSort" in ln:
                    inject_at = i + 1
                    break

    repaired = "\n".join(lines[:inject_at] + decls + lines[inject_at:])
    try:
        ast.parse(repaired)
    except SyntaxError:
        return code
    return repaired
