"""Schema-conditioned reshape (bolt-on E): make FACTS and GOALS reuse the RULES'
predicate vocabulary so the symbolic chain can actually connect.

The fvossel adapter translates one sentence at a time, so the RULES come out clean
and uniform (``∀x (Student(x) ∧ CompletedCoreCurriculum(x) → …)``) but a ground
FACT comes out as a generic relation whose symbol matches nothing in the rules:

    "Sophia has completed the core curriculum."  →  Completed(Curriculum, sophia)
    "John maintains a GPA of 3.8."               →  Maintain(John, 3.8)
    "Does John receive academic distinction?"    →  Receive(John)

Z3 then never connects ``Completed(Curriculum, sophia)`` to the rule's unary
``CompletedCoreCurriculum(x)``, and the whole derivation dies before it starts —
the single biggest accuracy killer in the run review.

This module fixes that at the source, deterministically (no extra model call):

  1. ``partition_rules_facts`` splits a record's premise FOLs into RULES (quantified
     implications, which the model gets right) and FACTS (everything else).
  2. ``harvest_registry`` collects the canonical *unary* predicate vocabulary the
     rules establish (``CompletedCoreCurriculum``, ``PassedScienceAssessment``,
     ``ReceivesAcademicDistinction``, …), excluding pure sort guards (``Student``).
  3. ``reshape_fact`` / ``reshape_goal`` snap an off-vocabulary fact/goal onto that
     registry by matching the SOURCE ENGLISH SENTENCE's tokens against each
     registry name (``"completed the core curriculum"`` ↔ ``CompletedCoreCurriculum``),
     and re-ground it as ``Pred(entity)`` reusing the entity the model already named.

Matching is on the source sentence (not the mangled FOL) because the FOL has often
*lost* the discriminating words — ``Receive(John)`` no longer says "academic
distinction", but the question still does. Everything here is pure and unit-tested;
``schema_condition`` ties it together over one ``RecordFol``.
"""

from __future__ import annotations

import re

from translator.fol_converter import App, BinOp, FolParseError, Not, Quant, parse
from translator.fol_repair import (
    _all_pred_names,
    _entities,
    _free_sort_guards,
    _JUNK_PREDS,
    _pred_arities,
    _pred_atoms,
    _strip_quants,
)

# CamelCase / snake_case → word tokens (same split predicate_group uses).
_TOKEN_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|\d+")

# Words that carry no discriminating meaning in either a predicate name or an
# English claim. Verbs like "completed"/"passed"/"received" are KEPT — they are
# exactly what distinguishes one requirement from another.
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with", "and",
    "or", "as", "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "his", "her", "their", "its", "they", "them", "he", "she", "it", "this",
    "that", "these", "those", "all", "any", "some", "does", "do", "did", "who",
    "whom", "which", "what", "according", "premise", "premises", "above",
    "based", "conclusion", "follows", "logically", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "if", "then", "not",
    "no", "but", "than", "from", "into", "about", "course",
})


def _stem(tok: str) -> str:
    """Crude Porter-lite stem so verb/number forms align across a predicate name
    and the English sentence (``receives``↔``receive``, ``qualifies``↔``qualify``,
    ``completed``↔``completes``↔``complete``)."""
    t = tok.lower()
    if len(t) <= 3 or t.isdigit():
        return t
    if t.endswith("ies"):
        return t[:-3] + "i"
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("es") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    if t.endswith("y"):
        t = t[:-1] + "i"
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def _name_tokens(name: str) -> set[str]:
    """Stemmed content tokens of a CamelCase/snake predicate name."""
    toks = (t for t in _TOKEN_SPLIT.findall(name))
    return {_stem(t) for t in toks if t.lower() not in _STOPWORDS} - {""}


def _nl_tokens(text: str) -> set[str]:
    """Stemmed content tokens of an English sentence."""
    toks = re.split(r"[^A-Za-z0-9]+", text)
    return {_stem(t) for t in toks if t and t.lower() not in _STOPWORDS} - {""}


def _parse_or_none(s: str):
    try:
        return parse(s)
    except (FolParseError, Exception):
        return None


# ─────────────────────────────────────────────────────────────────────────
# Rule / fact partition + registry harvest
# ─────────────────────────────────────────────────────────────────────────


def is_rule_fol(fol: str) -> bool:
    """True when `fol` reads as a rule — a (possibly quantified) implication or
    biconditional. Those are the formulas the adapter translates reliably, and the
    ones that define the canonical predicate vocabulary."""
    node = _parse_or_none(fol)
    if node is None:
        return False
    body = _strip_quants(node)
    return isinstance(body, BinOp) and body.op in ("implies", "iff")


def partition_rules_facts(premise_fols: list[str]) -> tuple[list[int], list[int]]:
    """Indices of (rules, facts) in `premise_fols`, classified by FOL shape."""
    rules, facts = [], []
    for i, fol in enumerate(premise_fols):
        (rules if is_rule_fol(fol) else facts).append(i)
    return rules, facts


def harvest_registry(rule_fols: list[str]) -> dict[str, int]:
    """Canonical predicate vocabulary the rules establish: name → most-common arity
    across rule predicate positions. This is the set facts/goals should reuse."""
    nodes = [n for n in (_parse_or_none(s) for s in rule_fols) if n is not None]
    return _pred_arities(nodes)


def registry_targets(rule_fols: list[str]) -> dict[str, int]:
    """Unary registry predicates that are valid SNAP TARGETS — i.e. the requirement
    / status / capability predicates, with pure sort guards (``Student``, ``Driver``)
    and junk fillers removed (a fact must never be snapped onto the sort it is gated
    by, nor onto a hallucinated filler)."""
    nodes = [n for n in (_parse_or_none(s) for s in rule_fols) if n is not None]
    arities = _pred_arities(nodes)
    guards = _free_sort_guards(nodes) if nodes else set()
    return {
        name: ar
        for name, ar in arities.items()
        if ar == 1 and name not in guards and name not in _JUNK_PREDS
    }


# ─────────────────────────────────────────────────────────────────────────
# Snap an English claim onto a registry predicate
# ─────────────────────────────────────────────────────────────────────────


def snap_text_to_registry(text: str, targets: dict[str, int]) -> str | None:
    """Best registry predicate for an English claim, or None when no match is
    confident enough.

    Score = #shared stemmed tokens between the claim and the predicate NAME;
    coverage = that share over the predicate name's own token count. We accept the
    single best target when it overlaps meaningfully (≥2 shared tokens, or a full
    cover of a short name) AND it is strictly better than the runner-up — so an
    incidental single shared token (``research`` between two different capabilities)
    is rejected, while ``"completed the core curriculum"`` ↔ ``CompletedCoreCurriculum``
    (a full cover) is accepted."""
    tt = _nl_tokens(text)
    if not tt:
        return None
    scored: list[tuple[int, float, str]] = []
    for name in targets:
        rt = _name_tokens(name)
        if not rt:
            continue
        shared = len(rt & tt)
        if shared == 0:
            continue
        coverage = shared / len(rt)
        scored.append((shared, coverage, name))
    if not scored:
        return None
    scored.sort(key=lambda s: (s[0], s[1], -len(s[2])), reverse=True)
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    shared, coverage, name = best
    # Confidence gate.
    strong = shared >= 2 or coverage >= 0.99
    if not strong or coverage < 0.5:
        return None
    # Ambiguity gate: a clear winner over the runner-up.
    if runner is not None and (runner[0], runner[1]) >= (shared, coverage):
        return None
    return name


# ─────────────────────────────────────────────────────────────────────────
# Reshape one fact / one goal
# ─────────────────────────────────────────────────────────────────────────

_NEG_NL = re.compile(r"\b(not|never|cannot|can't|won't|doesn't|don't|didn't|no longer|isn't|hasn't|haven't)\b",
                     re.IGNORECASE)


def _principal_entity(node, persons: set[str]) -> str | None:
    """The protagonist symbol among a formula's argument-position names: prefer one
    whose casefold is a known person of the record, else the first argument name."""
    names = _entities([node]) if node is not None else []
    for nm in names:
        if nm.casefold() in persons:
            return nm
    return names[0] if names else None


def _already_on_registry(node, registry: dict[str, int]) -> bool:
    """True when the formula's principal predicate is already a registry name — a
    correctly-shaped fact/goal we must leave alone."""
    if node is None:
        return False
    for name, _ar, _neg in _pred_atoms(_strip_quants(node)):
        if name in registry:
            return True
    # Ground atoms aren't implications, so also check raw predicate positions.
    return any(n in registry for n in _all_pred_names(node))


def reshape_fact(fact_fol: str, nl: str, targets: dict[str, int],
                 persons: set[str], registry: dict[str, int]) -> str:
    """Rewrite an off-registry ground fact to ``RegistryPred(entity)`` using the
    source sentence; otherwise return the fact unchanged.

    Leaves alone any fact whose predicate is already in the registry (a correct
    translation — including correctly-negated ones)."""
    node = _parse_or_none(fact_fol)
    if node is None or _already_on_registry(node, registry):
        return fact_fol
    target = snap_text_to_registry(nl, targets)
    if target is None:
        return fact_fol
    entity = _principal_entity(node, persons)
    if entity is None:
        return fact_fol
    negated = fact_fol.lstrip().startswith(("¬", "~")) or bool(_NEG_NL.search(nl))
    atom = f"{target}({entity})"
    return f"¬ {atom}" if negated else atom


def reshape_goal(goal_fol: str, nl: str, targets: dict[str, int],
                 persons: set[str], registry: dict[str, int],
                 protagonist: str | None) -> str:
    """Rewrite an off-registry goal/option to ``RegistryPred(entity)`` using the
    question/option text; otherwise return it unchanged for bolt-on A to handle."""
    node = _parse_or_none(goal_fol)
    if node is not None and _already_on_registry(node, registry):
        return goal_fol
    target = snap_text_to_registry(nl, targets)
    if target is None:
        return goal_fol
    entity = _principal_entity(node, persons) if node is not None else None
    if entity is None or entity.casefold() not in persons:
        entity = protagonist
    if entity is None:
        return goal_fol
    negated = goal_fol.lstrip().startswith(("¬", "~")) or bool(_NEG_NL.search(nl))
    atom = f"{target}({entity})"
    return f"¬ {atom}" if negated else atom


# ─────────────────────────────────────────────────────────────────────────
# Record-level driver
# ─────────────────────────────────────────────────────────────────────────


def record_person_entities(premise_fols: list[str]) -> set[str]:
    """Casefolded protagonist symbols of a record: any entity that recurs across
    ≥2 premises (the person every fact is about). Robust to the wrong-shaped facts
    where the protagonist is buried in a relation's argument list and so is never a
    unary subject."""
    nodes = [n for n in (_parse_or_none(s) for s in premise_fols) if n is not None]
    from collections import Counter

    counts: Counter = Counter()
    for n in nodes:
        names = set(_entities([n]))
        for nm in names:
            counts[nm.casefold()] += 1
    return {nm for nm, c in counts.items() if c >= 2}


def _protagonist(premise_fols: list[str], persons: set[str]) -> str | None:
    """A concrete spelling of the record's main person, for grounding a goal whose
    own FOL named no usable entity."""
    nodes = [n for n in (_parse_or_none(s) for s in premise_fols) if n is not None]
    from collections import Counter

    counts: Counter = Counter()
    for n in nodes:
        for nm in set(_entities([n])):
            if nm.casefold() in persons:
                counts[nm] += 1
    return counts.most_common(1)[0][0] if counts else None


def schema_condition(
    answer_type,
    premises_fol: list[str],
    premises_nl: list[str],
    goals: list[tuple[str, list[str]]],
    goals_nl: list[str],
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Reshape a record's facts and goals onto the rules' predicate registry.

    `goals_nl[i]` is the source English for `goals[i]` (the question for a YNU
    record, the option text for each MCQ option). Returns (new_premises_fol,
    new_goals). A no-op (returns the inputs) when the record has no rules to
    condition on."""
    from translator.llama_fol import normalize_fol  # local: avoid import cycle

    norm = [normalize_fol(p) if p else "" for p in premises_fol]
    rule_idx, fact_idx = partition_rules_facts(norm)
    rule_fols = [norm[i] for i in rule_idx]
    targets = registry_targets(rule_fols)
    if not targets:
        return premises_fol, goals
    registry = harvest_registry(rule_fols)
    persons = record_person_entities(norm)

    new_premises = list(premises_fol)
    for i in fact_idx:
        nl = premises_nl[i] if i < len(premises_nl) else ""
        new_premises[i] = reshape_fact(norm[i], nl, targets, persons, registry)

    protagonist = _protagonist(norm, persons)
    new_goals: list[tuple[str, list[str]]] = []
    for gi, (label, cands) in enumerate(goals):
        nl = goals_nl[gi] if gi < len(goals_nl) else ""
        reshaped = [
            reshape_goal(normalize_fol(c) if c else c, nl, targets, persons,
                         registry, protagonist)
            for c in cands
        ]
        new_goals.append((label, reshaped))
    return new_premises, new_goals
