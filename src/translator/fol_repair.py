"""Deterministic FOL repair bolt-ons for the Llama NL→FOL path.

Two pure, model-free transforms applied at Z3-assembly time (so BOTH the
single-phase `LlamaFolTranslator.translate` and the runner's grouping path in
`run_pipeline.translate_and_group` benefit, with no extra model loads):

  A. ``ground_goal`` — fvossel/t5-3b systematically rewrites a ground claim
     "Entity does X" into a universal rule "∀x (Type(x) → X(x))", and sometimes
     invents spare variables so the goal predicate comes out *binary*
     (``∀x (… → Eligible(x, y))``). Neither shape can be entailed by ground facts
     about a single entity, so the record is lost before Z3 even runs. We invert
     that transform: peel the quantifier/implication down to the conclusion, take
     its principal predicate, and re-ground it as ``Pred(entity)`` using the arity
     the PREMISES use for that predicate. ``∀x (Student(x) → Eligible(x))`` becomes
     ``Eligible(sophia)``; ``∀x (… → Eligible(x, y))`` becomes ``Eligible(sophia)``.

  B. ``add_type_facts`` — every rule in these records is gated by a sort guard
     like ``Student(x)``, but the facts only ever say
     ``CompletedCoreCurriculum(sophia)`` and never ``Student(sophia)``. With the
     guard unsatisfiable, NO rule can fire, so the whole chain is dead. We detect a
     *free sort guard* — a unary predicate that gates EVERY rule yet is never a
     rule head, never a ground fact, and not the goal — and assert it for each
     entity in the record. That derives ``Student(sophia)`` deterministically
     instead of hardcoding it. A precondition that gates only SOME rules (e.g. a
     misspelled ``PassedScienceAssement``) is left alone: aligning it with the
     fact's spelling is a name-canonicalization job (predicate_group / bolt-on D),
     not a typing one, and asserting it would be unsound.

Soundness notes
---------------
A is sound for this benchmark's single-protagonist Yes/No/Uncertain and MCQ
questions (the claim really is about one named entity). B applies a *closed-world
assumption on sort guards*: it assumes the protagonist satisfies the sort that
gates every rule. That can in principle flip a record whose answer hinges on the
entity NOT being of that sort; it is therefore gated by ``min_rules`` (default 2)
and the strict "gates every rule" coverage test, and can be disabled with
``--no-type-facts``. Measure on the dev split before trusting it blindly.

Both functions take and return normalized FOL strings (see
``translator.llama_fol.normalize_fol``) and reuse ``translator.fol_converter``'s
parser/AST, so the symbolic reading is exact — no string heuristics on the FOL.
"""

from __future__ import annotations

from collections import Counter

from translator.fol_converter import (
    App,
    Arith,
    BinOp,
    Cmp,
    Const,
    FolParseError,
    Not,
    Num,
    Quant,
    Var,
    parse,
)

# Predicate names T5 builds by gluing a negation onto a concept (e.g.
# `NOTQualifiesForScholarship`). When a mangled goal is a bare disjunction of a
# concept and its glued negation, we want the POSITIVE concept as the claim.
_NEG_PREFIXES = ("NOT", "Not", "Disqualif", "Cannot")

# Filler predicates T5 hallucinates to glue a "rule" together. They never carry
# the actual claim, so they must not be chosen as a goal's principal predicate
# nor asserted as a sort guard.
_JUNK_PREDS = frozenset({
    "AccordingToPremises", "AccordingToThePremises", "Premise", "Premises",
    "LogicalChain", "Combination", "TruthValue", "Statement", "Conclusion",
    "Holds", "True_", "Fact",
})


def _looks_negated(name: str) -> bool:
    return len(name) > 4 and name.startswith(_NEG_PREFIXES)


def _parse_or_none(s: str):
    try:
        return parse(s)
    except (FolParseError, Exception):
        return None


# ─────────────────────────────────────────────────────────────────────────
# Shared AST walkers
# ─────────────────────────────────────────────────────────────────────────


def _strip_quants(node):
    """Peel leading quantifiers only (not implications)."""
    while isinstance(node, Quant):
        node = node.body
    return node


def _peel_to_conclusion(node):
    """Strip leading quantifiers/negations and follow implications/iff to the
    right-hand (consequent) side. Returns (conclusion_node, negated)."""
    neg = False
    while True:
        if isinstance(node, Quant):
            node = node.body
        elif isinstance(node, Not):
            neg = not neg
            node = node.body
        elif isinstance(node, BinOp) and node.op in ("implies", "iff"):
            node = node.right
        else:
            return node, neg


def _antecedent_atoms(node):
    """Predicate atoms of the rule's antecedent (the left of a top-level, possibly
    quantified, implication). Empty when the goal is not an implication. Used as a
    fallback claim source when T5 dumps the real predicate into the antecedent and
    fills the consequent with a junk filler like `AccordingToPremises`."""
    n = node
    while isinstance(n, Quant):
        n = n.body
    if isinstance(n, BinOp) and n.op in ("implies", "iff"):
        return _pred_atoms(n.left)
    return []


def _pred_atoms(node, neg: bool = False):
    """Boolean-position predicate applications under a formula, left to right,
    as (name, arity, negated) — descending only through ¬ and ∧/∨/→/↔."""
    out: list[tuple[str, int, bool]] = []
    if isinstance(node, App):
        out.append((node.name, len(node.args), neg))
    elif isinstance(node, Not):
        out.extend(_pred_atoms(node.body, not neg))
    elif isinstance(node, BinOp):
        out.extend(_pred_atoms(node.left, neg))
        out.extend(_pred_atoms(node.right, neg))
    # Cmp / Arith / Quant inside a conclusion are not simple predicate claims.
    return out


def _all_pred_names(node) -> list[str]:
    """Every predicate-position application name anywhere in the formula."""
    out: list[str] = []

    def walk(n, in_term: bool) -> None:
        if n is None or isinstance(n, (Var, Const, Num)):
            return
        if isinstance(n, App):
            if not in_term:
                out.append(n.name)
            for a in n.args:
                walk(a, True)
        elif isinstance(n, Not):
            walk(n.body, in_term)
        elif isinstance(n, BinOp):
            walk(n.left, in_term)
            walk(n.right, in_term)
        elif isinstance(n, (Cmp, Arith)):
            walk(n.left, True)
            walk(n.right, True)
        elif isinstance(n, Quant):
            walk(n.body, in_term)

    walk(node, False)
    return out


def _pred_arities(nodes) -> dict[str, int]:
    """Most-common arity per predicate name across predicate positions."""
    seen: dict[str, Counter] = {}

    def walk(n, in_term: bool) -> None:
        if n is None or isinstance(n, (Var, Const, Num)):
            return
        if isinstance(n, App):
            if not in_term:
                seen.setdefault(n.name, Counter())[len(n.args)] += 1
            for a in n.args:
                walk(a, True)
        elif isinstance(n, Not):
            walk(n.body, in_term)
        elif isinstance(n, BinOp):
            walk(n.left, in_term)
            walk(n.right, in_term)
        elif isinstance(n, (Cmp, Arith)):
            walk(n.left, True)
            walk(n.right, True)
        elif isinstance(n, Quant):
            walk(n.body, in_term)

    for n in nodes:
        walk(n, False)
    return {name: c.most_common(1)[0][0] for name, c in seen.items()}


def _arg_names(node, bound: set[str], out: set[str]) -> None:
    """Names in argument (term) position that are not bound by a quantifier —
    i.e. the record's entities/constants (`sophia`, `Alice`)."""
    if node is None:
        return
    if isinstance(node, Var):
        if node.name not in bound:
            out.add(node.name)
    elif isinstance(node, Const):
        out.add(node.name)
    elif isinstance(node, App):
        for a in node.args:
            _arg_names(a, bound, out)
    elif isinstance(node, Not):
        _arg_names(node.body, bound, out)
    elif isinstance(node, Quant):
        _arg_names(node.body, bound | set(node.vars), out)
    elif isinstance(node, (BinOp, Cmp, Arith)):
        _arg_names(node.left, bound, out)
        _arg_names(node.right, bound, out)


def _entities(nodes) -> list[str]:
    """Entity/constant names across the formulas, most frequent first. Bound
    quantifier variables (rule variables like `x`) are excluded."""
    counts: Counter = Counter()
    for n in nodes:
        if n is None:
            continue
        names: set[str] = set()
        _arg_names(n, set(), names)
        for nm in names:
            counts[nm] += 1
    return [nm for nm, _ in counts.most_common()]


def _unary_subjects(node, bound: set[str], out: set[str]) -> None:
    """Collect entities that appear as the SOLE argument of a predicate, e.g.
    `sophia` in ``CompletedCoreCurriculum(sophia)``. Bound quantifier variables
    are skipped, so a rule's ``Student(x)`` contributes nothing."""
    if node is None:
        return
    if isinstance(node, App):
        if len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, Const):
                out.add(arg.name)
            elif isinstance(arg, Var) and arg.name not in bound:
                out.add(arg.name)
        for a in node.args:
            _unary_subjects(a, bound, out)
    elif isinstance(node, Not):
        _unary_subjects(node.body, bound, out)
    elif isinstance(node, Quant):
        _unary_subjects(node.body, bound | set(node.vars), out)
    elif isinstance(node, (BinOp, Cmp, Arith)):
        _unary_subjects(node.left, bound, out)
        _unary_subjects(node.right, bound, out)


def _person_entities(nodes) -> set[str]:
    """Entities that are the subject of at least one unary predicate — the
    record's *people* (`sophia`, `socrates`). An object that the model only ever
    parked in a relation's argument position (`Curriculum` in the mis-parsed
    ``Completed(Curriculum, sophia)``) is never a unary subject, so it is excluded.

    Sort guards like ``Student`` only make sense on people; asserting them for
    every constant produces the unsound ``Student(Curriculum)`` facts."""
    out: set[str] = set()
    for n in nodes:
        if n is not None:
            _unary_subjects(n, set(), out)
    return out


# Public string-level helpers (used by translator.predicate_group's bolt-on D).


def entity_names(fol_strings) -> set[str]:
    """The set of entity/constant names (argument-position symbols) across a
    batch of FOL strings — e.g. {"sophia"}. Used to forbid predicate-name
    alignment from ever merging an entity (the `Sophia → QualifiedForAdvancedCourses`
    catastrophe), comparing case-insensitively at the call site."""
    nodes = [n for n in (_parse_or_none(s) for s in fol_strings) if n is not None]
    return set(_entities(nodes))


def predicate_arities(fol_strings) -> dict[str, int]:
    """Most-common arity per predicate name across a batch of FOL strings — so
    alignment only ever merges same-arity predicates."""
    nodes = [n for n in (_parse_or_none(s) for s in fol_strings) if n is not None]
    return _pred_arities(nodes)


# ─────────────────────────────────────────────────────────────────────────
# Rule analysis (shared by bolt-on A's claim selection and bolt-on B)
# ─────────────────────────────────────────────────────────────────────────


def _analyze_rules(valid_nodes):
    """Classify a record's parsed premises into rule structure. Returns
    (guard_count, guard_arity, heads, facts, n_rules):
      * guard_count[p]  = #rules whose antecedent mentions predicate p
      * guard_arity[p]  = p's arity where it appears as a guard
      * heads           = predicates that are a rule consequent (derivable)
      * facts           = predicates asserted in a quantifier-free premise
      * n_rules         = number of (quantified) implication/iff premises"""
    guard_count: Counter = Counter()
    guard_arity: dict[str, int] = {}
    heads: set[str] = set()
    facts: set[str] = set()
    n_rules = 0
    for n in valid_nodes:
        body = _strip_quants(n)
        if isinstance(body, BinOp) and body.op == "implies":
            n_rules += 1
            ante: dict[str, int] = {}
            for name, ar, _neg in _pred_atoms(body.left):
                ante.setdefault(name, ar)
            for name, ar in ante.items():
                guard_count[name] += 1  # once per rule, even if repeated
                guard_arity.setdefault(name, ar)
            for name in _all_pred_names(body.right):
                heads.add(name)
        elif isinstance(body, BinOp) and body.op == "iff":
            n_rules += 1
            for name in _all_pred_names(body):
                heads.add(name)
        else:
            for name in _all_pred_names(body):
                facts.add(name)
    return guard_count, guard_arity, heads, facts, n_rules


def _free_sort_guards(valid_nodes, goal_preds=frozenset(), min_rules: int = 2,
                      coverage: float = 1.0) -> set[str]:
    """The unary predicates that gate ≥ coverage×(#rules) rules yet are never a
    head, a ground fact, the goal, or a junk filler — i.e. the record's pure sort
    guards (`Student`, `Person`). Used by B (to assert them) and by A (to avoid
    ever choosing one as the goal's claim)."""
    gc, ga, heads, facts, n_rules = _analyze_rules(valid_nodes)
    if n_rules < min_rules:
        return set()
    threshold = coverage * n_rules
    return {
        g for g, c in gc.items()
        if c >= threshold
        and ga.get(g) == 1
        and g not in heads
        and g not in facts
        and g not in goal_preds
        and g not in _JUNK_PREDS
    }


# ─────────────────────────────────────────────────────────────────────────
# Bolt-on A: re-ground a mangled goal
# ─────────────────────────────────────────────────────────────────────────


def ground_goal(goal_fol: str, premises_fol: list[str]) -> str:
    """Re-ground a goal T5 turned into a universal / binary-arity formula.

    Returns a ground atom ``Pred(entity)`` (optionally negated) when a principal
    predicate and an entity can be identified; otherwise returns the goal
    unchanged. The predicate's arity is taken from the PREMISES, so a spuriously
    binary goal predicate (``Eligible(x, y)``) collapses to the unary form the
    knowledge base actually uses (``Eligible(sophia)``)."""
    if not goal_fol or not goal_fol.strip():
        return goal_fol
    gnode = _parse_or_none(goal_fol)
    if gnode is None:
        return goal_fol

    prem_nodes = [n for n in (_parse_or_none(p) for p in premises_fol) if n is not None]
    prem_preds = _pred_arities(prem_nodes)
    prem_ents = _entities(prem_nodes)
    # Never choose a junk filler or a sort guard (`Student`) as the claim: the
    # guard is made true for every entity by bolt-on B, so picking it would make
    # any option trivially "entailed".
    exclude = set(_JUNK_PREDS) | _free_sort_guards(prem_nodes)

    # Principal predicate: try the CONSEQUENT's atoms first (the claim normally
    # lives there); only if the consequent yields nothing usable fall back to the
    # antecedent (T5 sometimes parks the real predicate there under a junk
    # consequent). Within a tier, prefer a premise predicate (only those can be
    # entailed), then a non-negation-looking name, then the first remaining.
    concl, neg = _peel_to_conclusion(gnode)
    concl_atoms = [(nm, neg ^ an) for nm, _ar, an in _pred_atoms(concl)]
    ante_atoms = [(nm, an) for nm, _ar, an in _antecedent_atoms(gnode)]
    pick = None
    for tier in (concl_atoms, ante_atoms):
        cands = [(nm, an) for nm, an in tier if nm not in exclude]
        if not cands:
            continue
        for keep in (
            lambda nm, an: nm in prem_preds and not _looks_negated(nm),
            lambda nm, an: nm in prem_preds,
            lambda nm, an: not _looks_negated(nm),
            lambda nm, an: True,
        ):
            pick = next(((nm, an) for nm, an in cands if keep(nm, an)), None)
            if pick is not None:
                break
        if pick is not None:
            break
    if pick is None:
        return goal_fol
    pred, negated = pick

    # Entity: prefer a name shared by the goal AND the premises (the real
    # protagonist), then the premises' principal entity, then any goal name.
    # This rejects T5's spare variables (a free `y` left dangling in the goal).
    goal_ents = _entities([gnode])
    entity = next((e for e in goal_ents if e in prem_ents), None)
    if entity is None:
        entity = prem_ents[0] if prem_ents else (goal_ents[0] if goal_ents else None)
    if entity is None:
        return goal_fol

    # Only re-ground to a unary atom. If the premises genuinely use the predicate
    # with arity ≥ 2 we cannot invent the extra arguments — leave the goal as-is.
    if prem_preds.get(pred, 1) != 1:
        return goal_fol

    atom = f"{pred}({entity})"
    return f"¬ {atom}" if negated else atom


# ─────────────────────────────────────────────────────────────────────────
# Bolt-on B: assert free sort guards
# ─────────────────────────────────────────────────────────────────────────


def add_type_facts(
    premises_fol: list[str],
    goal_fol: str | None = None,
    min_rules: int = 2,
    coverage: float = 1.0,
) -> list[str]:
    """Append ground sort-guard facts (e.g. ``Student(sophia)``) so gated rules
    can fire. Returns a new list (premises + additions); never mutates the input.

    A unary predicate ``G`` is asserted for every entity iff it
      * appears in the antecedent of at least ``coverage`` × (#rules) rules
        (default ``coverage=1.0`` → EVERY rule), and the record has ≥ ``min_rules``
        rules, and
      * is never a rule head/consequent, never an asserted ground fact, and not a
        predicate of the (already re-grounded) goal.
    These conditions single out a sort guard like ``Student`` while leaving
    rule-specific preconditions untouched."""
    nodes = [_parse_or_none(p) for p in premises_fol]
    valid = [n for n in nodes if n is not None]
    if not valid:
        return list(premises_fol)

    goal_preds: set[str] = set()
    if goal_fol and goal_fol.strip():
        gp = _parse_or_none(goal_fol)
        if gp is not None:
            goal_preds = set(_all_pred_names(gp))

    free_guards = sorted(_free_sort_guards(valid, goal_preds, min_rules, coverage))
    if not free_guards:
        return list(premises_fol)

    # Assert sort guards only for PEOPLE (unary-predicate subjects), never for
    # object constants the model dumped into a relation's argument position — that
    # is what produced the unsound `Student(Curriculum)` / `Student(capstoneProject)`.
    persons = _person_entities(valid)
    ents = [e for e in _entities(valid) if e in persons]
    if not ents:
        return list(premises_fol)

    existing = {s.strip() for s in premises_fol}
    additions: list[str] = []
    for g in free_guards:
        for e in ents:
            atom = f"{g}({e})"
            if atom not in existing:
                additions.append(atom)
                existing.add(atom)
    return list(premises_fol) + additions
