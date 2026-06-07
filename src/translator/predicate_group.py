"""Align synonymous predicate/function names across a record's FOL formulas.

The T5 translator names predicates per-phrasing, so the SAME concept gets
different symbols across premises and the goal (e.g. `EligibleForScholarship(x)`
in a premise vs `QualifyForUniversityScholarship(x)` in the goal). Z3 then sees
two unrelated relations and the goal is never entailed.

Fix (used by the two-phase T5 runner):
  1. extract every relation name from a record's FOL  (deterministic, here)
  2. ask an LLM to GROUP names with the same meaning   (model — caller supplies)
  3. pick one canonical name per group + rewrite all   (deterministic, here)

The LLM ONLY clusters names; every rename is done by the deterministic code in
this module, so the symbolic guarantees are preserved. Scope: predicate and
function names only — constants/entities are left untouched.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Callable, Iterable

from translator.fol_converter import FolParseError, collect_signature, parse
from translator.fol_repair import entity_names, predicate_arities

# A chat function: takes chat `messages`, returns the model's raw completion text.
ChatFn = Callable[[list[dict]], str]

# Negation prefixes T5 glues onto a concept (e.g. `NOTQualifyScholarship`). Used
# to refuse merging a predicate with its own negation.
_NEG_PREFIXES = ("NOT", "Not", "Non", "Dis", "Cannot", "Un")


def relation_name_counts(fol_strings: Iterable[str]) -> Counter:
    """Count, per relation name, how many formulas use it. Predicate + function
    names only (Bool predicates, numeric functions, universe-valued functions).

    Inputs should already be normalized (FORALL/->/∧ etc.); unparseable formulas
    are skipped — we still collect names from the parseable majority."""
    counts: Counter = Counter()
    for s in fol_strings:
        if not s or not s.strip():
            continue
        try:
            node = parse(s)
        except (FolParseError, Exception):
            continue
        sig = collect_signature(node)
        names = set(sig.predicates) | set(sig.int_functions) | set(sig.universe_functions)
        for name in names:
            counts[name] += 1
    return counts


_GROUP_SYSTEM = """You are given a list of predicate names taken from first-order-logic formulas. Some names denote the SAME concept but are written differently. Your job is to group names that mean the same thing.

Rules:
- Output ONLY a JSON array of arrays. Each inner array is one group of names that are synonymous.
- Only group names you are confident mean the same thing. When in doubt, do NOT group them.
- Include only groups with 2 or more names. Omit any name that has no synonym.
- Use the given names exactly; never invent or alter a name.
- Output nothing except the JSON array."""

_GROUP_EXAMPLE_USER = (
    "Predicate names:\n"
    "- EligibleForScholarship\n"
    "- QualifyForUniversityScholarship\n"
    "- Student\n"
    "- IsStudent\n"
    "- GPA"
)
_GROUP_EXAMPLE_ASSISTANT = (
    '[["EligibleForScholarship", "QualifyForUniversityScholarship"], '
    '["Student", "IsStudent"]]'
)


def build_grouping_messages(names: list[str]) -> list[dict]:
    user = "Predicate names:\n" + "\n".join(f"- {n}" for n in names)
    return [
        {"role": "system", "content": _GROUP_SYSTEM},
        {"role": "user", "content": _GROUP_EXAMPLE_USER},
        {"role": "assistant", "content": _GROUP_EXAMPLE_ASSISTANT},
        {"role": "user", "content": user},
    ]


def parse_grouping_response(text: str, known: set[str]) -> list[list[str]]:
    """Pull the first balanced JSON array out of the model output and keep only
    groups of >=2 KNOWN names. Robust to leading/trailing chatter."""
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    clusters: list[list[str]] = []
    if isinstance(data, list):
        for grp in data:
            if not isinstance(grp, list):
                continue
            members = [x for x in grp if isinstance(x, str) and x in known]
            if len(set(members)) >= 2:
                clusters.append(list(dict.fromkeys(members)))
    return clusters


def build_canonical_map(
    clusters: list[list[str]], counts: Counter | None = None
) -> dict[str, str]:
    """Map every non-canonical name in a group to the group's canonical name.

    Canonical = most frequently used name (keeps the dominant symbol), breaking
    ties by shorter, then alphabetical — fully deterministic."""
    counts = counts or Counter()
    mapping: dict[str, str] = {}
    for group in clusters:
        uniq = list(dict.fromkeys(group))
        if len(uniq) < 2:
            continue
        canonical = sorted(uniq, key=lambda n: (-counts.get(n, 0), len(n), n))[0]
        for name in uniq:
            if name != canonical:
                mapping[name] = canonical
    return mapping


def apply_canonical(fol: str, mapping: dict[str, str]) -> str:
    """Rewrite whole-word relation names in `fol` per `mapping`, in a single pass
    (so A→B and B→C never chain). Identifier names only, so `\\b` boundaries are
    safe."""
    if not mapping:
        return fol
    # Longer names first so a name that is a prefix of another can't shadow it.
    alternation = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    pattern = re.compile(r"\b(" + alternation + r")\b")
    return pattern.sub(lambda m: mapping[m.group(0)], fol)


def group_relations_debug(fol_strings: list[str], chat: ChatFn) -> dict:
    """Like `group_relations` but also returns the intermediate I/O for logging:
    {"names", "raw_response", "clusters", "mapping"}."""
    counts = relation_name_counts(fol_strings)
    names = sorted(counts)
    if len(names) < 2:
        return {"names": names, "raw_response": "", "clusters": [], "mapping": {}}
    raw = chat(build_grouping_messages(names))
    clusters = parse_grouping_response(raw, set(names))
    mapping = build_canonical_map(clusters, counts)
    return {"names": names, "raw_response": raw, "clusters": clusters, "mapping": mapping}


def group_relations(fol_strings: list[str], chat: ChatFn) -> dict[str, str]:
    """End-to-end for one record: extract relation names → LLM clusters → canonical
    rename map. Returns {} when there are <2 names or the LLM finds no synonyms."""
    return group_relations_debug(fol_strings, chat)["mapping"]


# ─────────────────────────────────────────────────────────────────────────
# Bolt-on D: SAFE alignment — guarded LLM clusters + deterministic matching
# ─────────────────────────────────────────────────────────────────────────
#
# The raw LLM grouper over-merges (it grouped the ENTITY `Sophia` with `Student`,
# and a precondition `ReceivedFacultyRecommendation` with the status
# `AwardedHonorsDiploma`) and MISSES morphological twins (it left the rule's
# `PassedScienceAssement` and the fact's `PassedScienceAssessory` as separate
# singletons, so the chain could never fire). `safe_canonical_map` fixes both:
#   * it GUARDS every LLM cluster — never merge an entity name, never cross arity,
#     never merge a predicate with its own negation; and
#   * it ADDS deterministic merges for near-identical names (shared long prefix or
#     tiny edit distance), which need no model and catch the typo twins the LLM
#     misses.
# The LLM still contributes the *semantic* synonyms deterministic matching can't
# see (`EligibleForScholarship` ≈ `QualifyForUniversityScholarship`).


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lcp_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _strip_neg(name: str) -> str:
    for p in _NEG_PREFIXES:
        if name.startswith(p) and len(name) > len(p) + 2:
            return name[len(p):]
    return name


def _opposite_polarity(a: str, b: str) -> bool:
    """True if one name is the other with a negation prefix (Qualify vs NotQualify)."""
    return a != b and (_strip_neg(a) == b or _strip_neg(b) == a)


def _similar(a: str, b: str, min_len: int = 8, lcp_ratio: float = 0.7,
             edit_ratio: float = 0.15) -> bool:
    """Deterministic near-identity: a long shared prefix (covers `Foo`/`FooCourse`)
    OR a tiny normalized edit distance (covers `…Assement`/`…Assessory`,
    `Passed…`/`Passes…`). Conservative `min_len` avoids merging short tokens."""
    la, lb = len(a), len(b)
    if min(la, lb) < min_len:
        return False
    if _lcp_len(a, b) / min(la, lb) >= lcp_ratio:
        return True
    return _levenshtein(a, b) / max(la, lb) <= edit_ratio


def _majority_arity_members(members: list[str], arities: dict[str, int]) -> list[str]:
    """Keep only the members sharing the cluster's most common arity, so a cluster
    never fuses a unary and a binary predicate into one symbol."""
    if len(members) < 2:
        return members
    ar_counts = Counter(arities[m] for m in members if m in arities)
    if not ar_counts:
        return members
    top = ar_counts.most_common(1)[0][0]
    return [m for m in members if arities.get(m) == top]


def _drop_polarity_conflicts(members: list[str]) -> list[str]:
    """Within a cluster, drop the negated member of any positive/negative pair."""
    drop: set[str] = set()
    for m in members:
        for o in members:
            if _opposite_polarity(m, o):
                drop.add(m if _strip_neg(m) != m else o)
    return [m for m in members if m not in drop]


def safe_canonical_map(
    fol_strings: list[str],
    llm_clusters: list[list[str]] | None = None,
    deterministic: bool = True,
) -> dict[str, str]:
    """Build a SAFE predicate-rename map for one record (bolt-on D).

    Combines guarded LLM clusters with deterministic near-identity matching, then
    reuses `build_canonical_map` to pick one canonical name per merged group (most
    frequent → shortest → alphabetical). Entity names are never merged; merges are
    restricted to a single arity and never cross a positive/negative pair."""
    counts = relation_name_counts(fol_strings)
    if len(counts) < 2:
        return {}
    ents = {e.lower() for e in entity_names(fol_strings)}
    arities = predicate_arities(fol_strings)
    cand = [n for n in sorted(counts) if n.lower() not in ents]
    if len(cand) < 2:
        return {}

    parent = {n: n for n in cand}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # (1) guarded LLM clusters
    for cluster in (llm_clusters or []):
        members = [m for m in cluster if m in parent]          # known, non-entity
        members = _majority_arity_members(members, arities)    # single arity
        members = _drop_polarity_conflicts(members)            # no P vs ¬P
        for m in members[1:]:
            union(members[0], m)

    # (2) deterministic near-identity (same arity, same polarity)
    if deterministic:
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a, b = cand[i], cand[j]
                if (
                    arities.get(a) == arities.get(b)
                    and not _opposite_polarity(a, b)
                    and _similar(a, b)
                ):
                    union(a, b)

    groups: dict[str, list[str]] = {}
    for n in cand:
        groups.setdefault(find(n), []).append(n)
    clusters = [g for g in groups.values() if len(g) >= 2]
    return build_canonical_map(clusters, counts)
