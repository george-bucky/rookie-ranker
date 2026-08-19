from pathlib import Path

import pandas as pd
import pytest

from rookie_ranker.cohort import (
    COHORT_COLUMNS,
    build_draft_cohort,
    load_draft_picks,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rr02" / "nfl" / "draft_picks.csv"


def draft_picks() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_cohort_keeps_every_configured_drafted_skill_position():
    cohort = build_draft_cohort(draft_picks(), [2023])

    assert cohort["position"].tolist() == ["QB", "RB", "WR", "TE"]
    assert cohort["player_name"].tolist() == [
        "Alex Accurate",
        "Riley Runner",
        "Casey Catcher",
        "Terry Tightend",
    ]
    assert "Frank Fullback" not in cohort["player_name"].tolist()
    assert "Les Linebacker" not in cohort["player_name"].tolist()


def test_canonical_ids_prefer_gsis_then_pfr_then_draft_pick():
    cohort = build_draft_cohort(draft_picks(), [2023])

    assert cohort[["canonical_id", "canonical_id_source"]].to_dict("records") == [
        {"canonical_id": "gsis:G001", "canonical_id_source": "gsis"},
        {"canonical_id": "pfr:RRun00", "canonical_id_source": "pfr"},
        {"canonical_id": "draft:2023:3", "canonical_id_source": "draft"},
        {"canonical_id": "gsis:G004", "canonical_id_source": "gsis"},
    ]


def test_only_requested_draft_years_enter_cohort():
    cohort = build_draft_cohort(draft_picks(), [2025, 2026])

    assert cohort[["draft_season", "overall_pick"]].to_dict("records") == [
        {"draft_season": 2025, "overall_pick": 5},
        {"draft_season": 2026, "overall_pick": 6},
    ]


def test_duplicate_draft_pick_fails():
    source = draft_picks()
    duplicate = source.iloc[[0]].copy()
    duplicate["gsis_id"] = "DIFFERENT"
    duplicate["pfr_player_id"] = "Different00"

    with pytest.raises(ValueError, match="duplicate draft picks"):
        build_draft_cohort(pd.concat([source, duplicate], ignore_index=True), [2023])


def test_duplicate_canonical_id_fails():
    source = draft_picks()
    duplicate = source.iloc[[0]].copy()
    duplicate["pick"] = 99

    with pytest.raises(ValueError, match="duplicate canonical IDs"):
        build_draft_cohort(pd.concat([source, duplicate], ignore_index=True), [2023])


def test_required_draft_schema_is_enforced():
    with pytest.raises(ValueError, match="pfr_player_id"):
        build_draft_cohort(draft_picks().drop(columns="pfr_player_id"), [2023])


def test_empty_configured_cohort_has_stable_schema():
    cohort = build_draft_cohort(draft_picks(), [2000])

    assert cohort.empty
    assert cohort.columns.tolist() == list(COHORT_COLUMNS)


def test_draft_loader_uses_only_explicit_years(monkeypatch):
    calls = []

    class Loaded:
        @staticmethod
        def to_dicts():
            return [{"loaded": True}]

    def fake_load(years):
        calls.append(years)
        return Loaded()

    monkeypatch.setattr("rookie_ranker.cohort.nfl.load_draft_picks", fake_load)

    loaded = load_draft_picks([2023, 2024])

    assert calls == [[2023, 2024]]
    assert loaded.to_dict("records") == [{"loaded": True}]
