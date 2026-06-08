"""Bolt-on D hardening — the predicate-grouping guard no longer over-merges.

The run review surfaced two catastrophic LLM over-merges that corrupt the logic:
  * a possession folded into the sort that gates every rule (`HasLicense → Driver`);
  * two distinct chain stages fused on one incidental shared token
    (`CanApplyForCollaborativeResearchProjects → CanSubmitResearchProposals`, both
     sharing only `research`).
Both must be refused, while genuine synonyms (`Eligible…Scholarship` ≈
`Qualify…Scholarship`) and morphological twins still merge.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from translator.predicate_group import _shared_object, safe_canonical_map

# Driver-record rules (record 11): `Driver` gates every rule → a sort/type guard.
_DRIVER_FOLS = [
    "∀x (Driver(x) ∧ PassedVehicleInspection(x) ∧ HasAppropriateLicense(x) → CanTransportStandardGoods(x))",
    "∀x (Driver(x) ∧ CanTransportStandardGoods(x) ∧ HasInterstatePermit(x) → CanCrossStateLines(x))",
    "PassedVehicleInspection(john)",
    "HasLicense(john)",
]


def test_type_predicate_never_merges_with_possession():
    types = frozenset({"Driver"})
    assert not _shared_object("HasLicense", "Driver", types)
    assert not _shared_object("HasInterstatePermit", "Driver", types)
    # Two non-types still merge on object overlap.
    assert _shared_object("HasLicense", "HasAppropriateLicense", types)


def test_safe_map_refuses_driver_overmerge():
    bad_cluster = [["HasLicense", "HasAppropriateLicense", "HasInterstatePermit", "Driver"]]
    m = safe_canonical_map(_DRIVER_FOLS, bad_cluster)
    # Driver must remain its own symbol — never a merge source or target.
    assert "Driver" not in m and "Driver" not in m.values()
    # The genuine possession synonym (shared object "license") still merges.
    assert m.get("HasAppropriateLicense") == "HasLicense"


def test_incidental_single_token_overlap_is_rejected():
    # Only "research" is shared; it is < half of {submit, research, proposals}.
    assert not _shared_object(
        "CanApplyForCollaborativeResearchProjects", "CanSubmitResearchProposals"
    )


def test_safe_map_refuses_distinct_capability_overmerge():
    fols = [
        "∀x (Professor(x) ∧ CanSubmitResearchProposals(x) → CanApplyForCollaborativeResearchProjects(x))",
        "CanSubmitResearchProposals(john)",
    ]
    bad = [["CanApplyForCollaborativeResearchProjects", "CanSubmitResearchProposals"]]
    assert safe_canonical_map(fols, bad) == {}


def test_genuine_synonyms_still_merge():
    # Shared object token "scholarship" → a real synonym pair.
    assert _shared_object("EligibleForScholarship", "QualifyForUniversityScholarship")
    fols = [
        "∀x (A(x) → EligibleForScholarship(x))",
        "∀x (B(x) → QualifyForUniversityScholarship(x))",
        "A(s)",
    ]
    m = safe_canonical_map(fols, [["EligibleForScholarship", "QualifyForUniversityScholarship"]])
    assert len(m) == 1  # the two are unified into one canonical name
