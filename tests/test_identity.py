from pathlib import Path

import pandas as pd
import pytest

from rookie_ranker.identity import (
    IdentityContractError,
    UnreviewedIdentityAmbiguity,
    UnreviewedIdentityDecision,
    match_college_identities,
    normalize_player_name,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rr02" / "college"


def college_stats() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "player_season_stats.csv")


def cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draft_season": 2024,
                "overall_pick": 1,
                "canonical_id": "2024-001",
                "player_name": "José Smith, Jr.",
                "position": "qb",
            },
            {
                "draft_season": 2024,
                "overall_pick": 2,
                "canonical_id": "2024-002",
                "player_name": "Chris Jones",
                "position": "WR",
            },
            {
                "draft_season": 2024,
                "overall_pick": 3,
                "canonical_id": "2024-003",
                "player_name": "John Doe",
                "position": "RB",
            },
            {
                "draft_season": 2024,
                "overall_pick": 4,
                "canonical_id": "2024-004",
                "player_name": "Terry Tight",
                "position": "WR",
            },
        ]
    )


def test_normalize_player_name_is_exact_and_deterministic():
    assert normalize_player_name("José Smith, Jr.") == "jose smith"
    assert normalize_player_name("Jose Smith Jr") == "jose smith"
    assert normalize_player_name("John Doe") != normalize_player_name("Johnathan Doe")


def test_unreviewed_compatible_name_ambiguity_fails_closed():
    with pytest.raises(UnreviewedIdentityAmbiguity, match="2024/2"):
        match_college_identities(cohort().iloc[:2], college_stats())


def test_exact_override_and_reviewed_quarantine_are_auditable():
    overrides = pd.read_csv(FIXTURE_DIR / "identity_overrides.csv")

    matched, quarantine = match_college_identities(cohort(), college_stats(), overrides)

    assert matched["canonical_id"].tolist() == [
        "2024-001",
        "2024-002",
        "2024-003",
        "2024-004",
    ]
    assert matched["identity_match_status"].tolist() == [
        "exact",
        "quarantined",
        "override",
        "quarantined",
    ]
    assert matched["identity_match_method"].tolist() == [
        "normalized_name_position",
        "none",
        "override",
        "none",
    ]
    assert matched.loc[0, "cfbd_player_id"] == "101"
    assert pd.isna(matched.loc[1, "cfbd_player_id"])
    assert matched.loc[2, "cfbd_player_id"] == "301"
    assert pd.isna(matched.loc[3, "cfbd_player_id"])
    assert quarantine["canonical_id"].tolist() == ["2024-002", "2024-004"]
    assert quarantine.loc[0, "candidate_cfbd_ids"] == "201|202"
    assert quarantine.loc[0, "override_status"] == "quarantine"
    assert quarantine.loc[1, "reason"] == "Only an incompatible-position player has this exact name"
    assert matched.loc[3, "quarantine_reason"] == "Only an incompatible-position player has this exact name"


def test_only_pre_draft_seasons_can_supply_identity():
    future_only = college_stats().query("playerId == 101 and season == 2024")

    with pytest.raises(UnreviewedIdentityDecision) as error:
        match_college_identities(cohort().iloc[:1], future_only)

    assert error.value.quarantine.loc[0, "candidate_count"] == 0
    assert error.value.quarantine.loc[0, "override_status"] == "unreviewed"
    assert error.value.quarantine.loc[0, "reason"] == "no_exact_name_and_position_candidate_in_prior_season"


def test_automatic_match_requires_prior_season_activity_but_override_allows_exception():
    older_stats = college_stats().query("playerId == 101 and season <= 2023")
    older_cohort = cohort().iloc[[0]].assign(
        draft_season=2025, canonical_id="2025-001"
    )

    with pytest.raises(UnreviewedIdentityDecision):
        match_college_identities(older_cohort, older_stats)

    override = pd.DataFrame(
        [
            {
                "draft_season": 2025,
                "overall_pick": 1,
                "resolution": "match",
                "cfbd_player_id": 101,
                "reason": "Reviewed sit-out exception",
                "evidence": "CFBD history confirms the older player ID",
            }
        ]
    )
    matched, quarantine = match_college_identities(older_cohort, older_stats, override)

    assert matched.loc[0, "identity_match_status"] == "override"
    assert matched.loc[0, "cfbd_player_id"] == "101"
    assert quarantine.empty


def test_identity_rejects_non_numeric_college_seasons():
    malformed = college_stats().iloc[[0]].assign(season="not-a-year")

    with pytest.raises(IdentityContractError, match="season must be numeric"):
        match_college_identities(cohort().iloc[[0]], malformed)


def test_a_cfbd_id_cannot_be_reused_within_a_draft_class():
    duplicate = pd.concat(
        [
            cohort().iloc[[0]],
            cohort().iloc[[0]].assign(
                overall_pick=5, canonical_id="2024-005", player_name="Alias Smith"
            ),
        ],
        ignore_index=True,
    )
    overrides = pd.DataFrame(
        [
            {
                "draft_season": 2024,
                "overall_pick": 5,
                "resolution": "match",
                "cfbd_player_id": 101,
                "reason": "Intentional collision test",
                "evidence": "Same fixture CFBD ID",
            }
        ]
    )

    with pytest.raises(IdentityContractError, match="reused within a draft class"):
        match_college_identities(duplicate, college_stats(), overrides)


def test_override_requires_review_evidence():
    overrides = pd.read_csv(FIXTURE_DIR / "identity_overrides.csv").iloc[[1]].copy()
    overrides["evidence"] = ""

    with pytest.raises(IdentityContractError, match="requires evidence"):
        match_college_identities(cohort().iloc[[2]], college_stats(), overrides)


def test_every_override_key_must_match_exactly_one_cohort_pick():
    typoed = pd.DataFrame(
        [
            {
                "draft_season": 2024,
                "overall_pick": 999,
                "resolution": "quarantine",
                "cfbd_player_id": pd.NA,
                "reason": "Intentional typo regression",
                "evidence": "No cohort pick 999 exists",
            }
        ]
    )

    with pytest.raises(IdentityContractError, match="do not match cohort rows"):
        match_college_identities(cohort().iloc[[0]], college_stats(), typoed)


def test_quarantine_override_must_not_assign_a_cfbd_player_id():
    invalid = pd.DataFrame(
        [
            {
                "draft_season": 2024,
                "overall_pick": 2,
                "resolution": "quarantine",
                "cfbd_player_id": 201,
                "reason": "Invalid populated quarantine",
                "evidence": "Regression fixture",
            }
        ]
    )

    with pytest.raises(IdentityContractError, match="must leave cfbd_player_id blank"):
        match_college_identities(cohort().iloc[[1]], college_stats(), invalid)
