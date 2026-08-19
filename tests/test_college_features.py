from pathlib import Path

import pandas as pd
import pytest

from rookie_ranker.college_features import (
    CollegeFeatureContractError,
    build_college_features,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rr02" / "college"


def college_stats() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "player_season_stats.csv")


def identities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"canonical_id": "2024-001", "draft_season": 2024, "cfbd_player_id": "101"},
            {"canonical_id": "2024-003", "draft_season": 2024, "cfbd_player_id": "301"},
            {"canonical_id": "2024-004", "draft_season": 2024, "cfbd_player_id": pd.NA},
            {"canonical_id": "2024-999", "draft_season": 2024, "cfbd_player_id": "999"},
        ]
    )


def test_final_active_and_career_features_aggregate_transfers_without_leakage():
    features = build_college_features(college_stats(), identities()).set_index("canonical_id")
    player = features.loc["2024-001"]

    assert player["college_stats_status"] == "observed"
    assert player["college_final_active_season"] == 2023
    assert player["college_seasons_observed"] == 2
    assert player["college_team"] == "Alpha State|Beta Tech"
    assert player["college_conference"] == "Big Ten|SEC"
    assert player["final_passing_attempts"] == 150
    assert player["final_passing_yards"] == 1300
    assert player["final_passing_yards_observed"] == True
    assert player["final_passing_yards_per_attempt"] == pytest.approx(1300 / 150)
    assert player["final_passing_yards_per_attempt_observed"] == True
    assert player["final_rushing_carries"] == 40
    assert player["final_rushing_yards"] == 250
    assert player["final_rushing_yards_per_carry"] == pytest.approx(6.25)
    assert player["final_yards_from_scrimmage"] == 250
    assert player["final_yards_from_scrimmage_observed"] == True
    assert player["final_total_touchdowns"] == 16
    assert player["final_total_touchdowns_observed"] == True
    assert player["career_passing_attempts"] == 250
    assert player["career_passing_yards"] == 2000
    assert player["career_passing_yards_per_attempt"] == pytest.approx(8.0)
    assert player["career_total_touchdowns"] == 25
    assert player["career_total_touchdowns_observed"] == True
    assert player["career_passing_attempts"] != 1249  # excludes the post-draft row


def test_observation_flags_and_missing_stats_do_not_invent_zeroes():
    features = build_college_features(college_stats(), identities()).set_index("canonical_id")
    running_back = features.loc["2024-003"]

    assert running_back["college_passing_observed"] == False
    assert running_back["college_rushing_observed"] == True
    assert running_back["college_receiving_observed"] == False
    assert pd.isna(running_back["final_passing_yards"])
    assert running_back["final_passing_yards_observed"] == False
    assert pd.isna(running_back["final_receiving_yards"])
    assert running_back["final_receiving_yards_observed"] == False
    assert pd.isna(running_back["final_yards_from_scrimmage"])
    assert running_back["final_yards_from_scrimmage_observed"] == False
    assert pd.isna(running_back["final_total_touchdowns"])
    assert running_back["final_total_touchdowns_observed"] == False


def test_unresolved_and_missing_college_rows_have_distinct_statuses():
    features = build_college_features(college_stats(), identities()).set_index("canonical_id")
    unresolved = features.loc["2024-004"]
    missing = features.loc["2024-999"]

    assert unresolved["college_stats_status"] == "unresolved_identity"
    assert pd.isna(unresolved["college_seasons_observed"])
    assert pd.isna(unresolved["final_rushing_yards"])
    assert unresolved["final_rushing_yards_observed"] == False
    assert missing["college_stats_status"] == "missing"
    assert missing["college_seasons_observed"] == 0
    assert pd.isna(missing["career_receiving_yards"])
    assert missing["career_receiving_yards_observed"] == False


def test_feature_output_remains_one_row_per_canonical_identity():
    features = build_college_features(college_stats(), identities())

    assert features["canonical_id"].tolist() == identities()["canonical_id"].tolist()
    assert features["canonical_id"].is_unique


def test_supported_stats_must_be_numeric_and_populated():
    malformed = college_stats().iloc[[0]].assign(stat="not-a-number")

    with pytest.raises(CollegeFeatureContractError, match="must be numeric"):
        build_college_features(malformed, identities().iloc[[0]])


def test_duplicate_source_grain_is_rejected_but_transfers_are_allowed():
    stats = college_stats()
    duplicate = pd.concat([stats, stats.iloc[[0]]], ignore_index=True)

    with pytest.raises(CollegeFeatureContractError, match="duplicate college source-grain"):
        build_college_features(duplicate, identities().iloc[[0]])

    transfer_features = build_college_features(stats, identities().iloc[[0]])
    assert transfer_features.loc[0, "final_passing_attempts"] == 150


def test_college_features_reject_non_numeric_seasons():
    malformed = college_stats().iloc[[0]].assign(season="not-a-year")

    with pytest.raises(CollegeFeatureContractError, match="season must be numeric"):
        build_college_features(malformed, identities().iloc[[0]])
