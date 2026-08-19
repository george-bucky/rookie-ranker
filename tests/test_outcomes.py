from pathlib import Path

import pandas as pd
import pytest

from rookie_ranker.cohort import build_draft_cohort
from rookie_ranker.outcomes import build_targets, load_regular_season_stats


FIXTURES = Path(__file__).parent / "fixtures" / "rr02" / "nfl"


def cohort() -> pd.DataFrame:
    draft = pd.read_csv(FIXTURES / "draft_picks.csv")
    return build_draft_cohort(draft, [2023, 2025, 2026])


def player_stats() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "player_stats_reg.csv")


def row_for(targets: pd.DataFrame, player_name: str) -> pd.Series:
    return targets.loc[targets["player_name"] == player_name].iloc[0]


def test_rookie_target_is_only_the_first_regular_season():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)
    alex = row_for(targets, "Alex Accurate")

    assert alex["rookie_year_ppr_points"] == 100.5
    assert alex["rookie_target_status"] == "complete"


def test_three_year_target_is_exact_first_three_season_sum():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)
    alex = row_for(targets, "Alex Accurate")

    assert alex["three_year_ppr_points"] == 330.5
    assert alex["three_year_target_status"] == "complete"


def test_mature_gsis_player_with_missing_rows_receives_zero():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)
    terry = row_for(targets, "Terry Tightend")

    assert terry["rookie_year_ppr_points"] == 0.0
    assert terry["rookie_target_status"] == "complete"
    assert terry["three_year_ppr_points"] == 50.0
    assert terry["three_year_target_status"] == "complete"


def test_missing_gsis_never_receives_fabricated_zero():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)

    for name in ["Riley Runner", "Casey Catcher"]:
        player = row_for(targets, name)
        assert pd.isna(player["rookie_year_ppr_points"])
        assert player["rookie_target_status"] == "missing_gsis"
        assert pd.isna(player["three_year_ppr_points"])
        assert player["three_year_target_status"] == "missing_gsis"


def test_immature_three_year_window_stays_null_despite_partial_stats():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)
    pat = row_for(targets, "Pat Present")

    assert pat["rookie_year_ppr_points"] == 75.25
    assert pat["rookie_target_status"] == "complete"
    assert pd.isna(pat["three_year_ppr_points"])
    assert pat["three_year_target_status"] == "immature"


def test_immature_rookie_and_three_year_targets_stay_null():
    targets = build_targets(cohort(), player_stats(), outcomes_through_year=2025)
    future = row_for(targets, "Future Runner")

    assert pd.isna(future["rookie_year_ppr_points"])
    assert future["rookie_target_status"] == "immature"
    assert pd.isna(future["three_year_ppr_points"])
    assert future["three_year_target_status"] == "immature"


def test_future_nfl_season_is_rejected():
    stats = pd.concat(
        [
            player_stats(),
            pd.DataFrame(
                {
                    "player_id": ["G001"],
                    "season": [2026],
                    "fantasy_points_ppr": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="after outcomes_through_year: 2026"):
        build_targets(cohort(), stats, outcomes_through_year=2025)


def test_duplicate_player_season_is_rejected():
    stats = player_stats()
    stats = pd.concat([stats, stats.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate player-season stats"):
        build_targets(cohort(), stats, outcomes_through_year=2025)


def test_invalid_stat_value_is_rejected_instead_of_becoming_zero():
    stats = player_stats()
    stats.loc[0, "fantasy_points_ppr"] = None

    with pytest.raises(ValueError, match="invalid fantasy_points_ppr"):
        build_targets(cohort(), stats, outcomes_through_year=2025)


def test_zero_point_missing_id_sentinel_is_ignored():
    stats = pd.concat(
        [
            player_stats(),
            pd.DataFrame(
                {
                    "player_id": [None],
                    "season": [2024],
                    "fantasy_points_ppr": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    targets = build_targets(cohort(), stats, outcomes_through_year=2025)

    assert row_for(targets, "Alex Accurate")["three_year_ppr_points"] == 330.5


@pytest.mark.parametrize("bad_points", [1.0, "not-a-number"])
def test_missing_id_with_nonzero_or_invalid_points_is_rejected(bad_points):
    stats = pd.concat(
        [
            player_stats(),
            pd.DataFrame(
                {
                    "player_id": [None],
                    "season": [2024],
                    "fantasy_points_ppr": [bad_points],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="player_id|fantasy_points_ppr"):
        build_targets(cohort(), stats, outcomes_through_year=2025)


def test_missing_global_source_season_fails_before_zero_imputation():
    stats = player_stats().loc[lambda frame: frame["season"] != 2024]

    with pytest.raises(ValueError, match="missing required source seasons: 2024"):
        build_targets(cohort(), stats, outcomes_through_year=2025)


def test_regular_season_loader_explicitly_excludes_postseason(monkeypatch):
    calls = []

    class Loaded:
        @staticmethod
        def to_dicts():
            return [{"loaded": True}]

    def fake_load(years, *, summary_level):
        calls.append((years, summary_level))
        return Loaded()

    monkeypatch.setattr("rookie_ranker.outcomes.nfl.load_player_stats", fake_load)

    loaded = load_regular_season_stats([2023, 2024])

    assert calls == [([2023, 2024], "reg")]
    assert loaded.to_dict("records") == [{"loaded": True}]
