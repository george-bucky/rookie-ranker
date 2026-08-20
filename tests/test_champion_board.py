from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rookie_ranker.artifact_contract import SourceNotice, load_handoff
from rookie_ranker.champion_board import (
    CurrentClassEvidence,
    GLOBAL_RESIDUAL_MINIMUM,
    POSITION_RESIDUAL_MINIMUM,
    PublicationMetadata,
    _macro_report,
    _residual_bounds,
    add_prior_fold_intervals,
    build_random_forest,
    build_rookie_board,
    evaluate_challenger,
    interval_metrics,
    prepare_model_features,
    publish_rookie_board,
    random_forest_predictions,
    select_champion,
)
from rookie_ranker.college_features import FEATURE_NAMES


def feature_rows(years=range(2013, 2024), players_per_year=8):
    rows = []
    positions = ("QB", "RB", "WR", "TE")
    for year in years:
        for index in range(players_per_year):
            position = positions[index % len(positions)]
            talent = float((index * 7 + year) % 17)
            row = {
                "canonical_id": f"draft:{year}:{index + 1}",
                "draft_season": year,
                "round": 1,
                "overall_pick": index + 1,
                "nfl_team": f"T{index:02d}",
                "player_name": f"Player {year} {index}",
                "position": position,
                "identity_match_status": "quarantined",
                "identity_match_method": "none",
                "college_stats_status": "observed",
                "college_seasons_observed": 3,
                "college_passing_observed": position == "QB",
                "college_rushing_observed": position in {"QB", "RB"},
                "college_receiving_observed": position in {"RB", "WR", "TE"},
                "three_year_ppr_points": 40 + talent * talent + (12 if position == "QB" else 0),
                "three_year_target_status": "complete",
                "rookie_year_ppr_points": 10 + talent * 5 + (4 if position == "RB" else 0),
                "rookie_target_status": "complete",
                "data_quality_warnings": ("synthetic fixture",),
            }
            for scope_index, scope in enumerate(("final", "career"), start=1):
                for feature_index, feature in enumerate(FEATURE_NAMES, start=1):
                    name = f"{scope}_{feature}"
                    row[name] = talent * feature_index * scope_index
                    row[f"{name}_observed"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def current_class():
    current = feature_rows(years=[2026], players_per_year=4)
    return current.drop(
        columns=[
            "three_year_ppr_points",
            "three_year_target_status",
            "rookie_year_ppr_points",
            "rookie_target_status",
        ]
    )


def publication_metadata():
    return PublicationMetadata(
        producer_commit="f111442868c81b102729563f01149f214faeb119",
        draft_event_date=date(2026, 4, 25),
        generated_at=datetime(2026, 5, 1, 12, tzinfo=UTC),
        data_cutoff=date(2026, 4, 30),
        outcomes_cutoff_season=2025,
        source_notices=(
            SourceNotice(
                name="fixture",
                url="https://example.test/fixture",
                license="fixture-only",
                notice="Synthetic derived test values only.",
                version="1",
            ),
        ),
    )


def current_evidence(current):
    return CurrentClassEvidence(
        draft_class=2026,
        draft_keys=frozenset(
            zip(
                current["canonical_id"].astype(str),
                current["overall_pick"].astype(int),
                current["position"].astype(str),
                strict=True,
            )
        ),
    )


def test_features_are_approved_explicit_and_observed_flags_fail_closed():
    table = feature_rows(years=[2020], players_per_year=2)
    table["conference"] = "do-not-use"
    features = prepare_model_features(table)

    assert "conference" not in features
    assert "draft_season" not in features
    assert "canonical_id" not in features
    assert "college_seasons_observed_missing" in features

    table.loc[0, "final_passing_attempts"] = np.nan
    with pytest.raises(ValueError, match="missing despite its observed flag"):
        prepare_model_features(table)


def test_random_forest_parameters_and_predictions_are_frozen_and_deterministic():
    training = feature_rows(years=range(2013, 2019), players_per_year=4)
    held_out = feature_rows(years=[2019], players_per_year=4)
    first = random_forest_predictions(training, held_out, "rookie_year_ppr_points")
    second = random_forest_predictions(training, held_out, "rookie_year_ppr_points")

    assert np.array_equal(first, second)
    model = build_random_forest().named_steps["random_forest"]
    assert model.random_state == 20260819
    assert model.n_jobs == 1
    assert model.min_samples_leaf == 5


def test_temporal_evaluation_uses_target_specific_availability_gaps():
    table = feature_rows()
    three_year = evaluate_challenger(table, "three_year_ppr_points")
    rookie_year = evaluate_challenger(table, "rookie_year_ppr_points")

    assert three_year.per_year["test_year"].min() == 2020
    assert rookie_year.per_year["test_year"].min() == 2018
    assert all(max(years) <= test_year - 3 for test_year, years in zip(three_year.per_year["test_year"], three_year.per_year["train_years"], strict=True))
    assert all(max(years) <= test_year - 1 for test_year, years in zip(rookie_year.per_year["test_year"], rookie_year.per_year["train_years"], strict=True))
    for report in (three_year.macro_class_ranking, three_year.position_slices):
        assert {
            "ndcg_24",
            "ndcg_12",
            "spearman",
            "top_12_hit_recall",
            "top_24_hit_recall",
        }.issubset(report.columns)


def test_macro_pooling_gives_each_class_equal_weight():
    per_year = pd.DataFrame(
        [
            {"test_year": 2020, "model": "draft_capital", "ndcg_24": 0.0},
            {"test_year": 2021, "model": "draft_capital", "ndcg_24": 1.0},
            {"test_year": 2020, "model": "random_forest", "ndcg_24": 0.4},
            {"test_year": 2021, "model": "random_forest", "ndcg_24": 0.8},
        ]
    )
    for metric in ("ndcg_12", "spearman", "top_12_hit_recall", "top_24_hit_recall"):
        per_year[metric] = per_year["ndcg_24"]
    report = _macro_report(per_year).set_index("model")
    assert report.at["draft_capital", "ndcg_24"] == 0.5
    assert report.at["random_forest", "ndcg_24"] == pytest.approx(0.6)


def champion_reports(*, challenger_mae=105.0, integrity=True):
    macro = pd.DataFrame(
        [
            {"model": "draft_capital", "ndcg_24": 0.7, "ndcg_24_eligible_class_count": 5},
            {"model": "random_forest", "ndcg_24": 0.8, "ndcg_24_eligible_class_count": 5},
        ]
    )
    pooled = pd.DataFrame(
        [
            {"model": "draft_capital", "mae": 100.0},
            {"model": "random_forest", "mae": challenger_mae},
        ]
    )
    challenger_scores = [0.9, 0.9, 0.9, 0.5, 0.4]
    baseline_scores = [0.8, 0.8, 0.8, 0.5, 0.6]
    per_year = pd.DataFrame(
        [
            {"test_year": 2020 + index, "model": model, "ndcg_24": score}
            for index, pair in enumerate(zip(challenger_scores, baseline_scores, strict=True))
            for model, score in zip(("random_forest", "draft_capital"), pair, strict=True)
        ]
    )
    return macro, pooled, per_year, {
        "data_integrity": integrity,
        "temporal_leakage": True,
        "identity": True,
        "interval_validation": True,
    }


def test_exact_champion_gate_accepts_60_percent_and_105_percent_boundaries():
    decision = select_champion(
        "three_year_ppr_points", *champion_reports()[:3], integrity_gates=champion_reports()[3]
    )
    assert decision.champion == "random_forest"
    assert decision.strict_win_count == 3
    assert decision.eligible_fold_count == 5
    assert decision.strict_win_rate == 0.6

    macro, pooled, per_year, gates = champion_reports(challenger_mae=105.0001)
    losing = select_champion(
        "three_year_ppr_points", macro, pooled, per_year, integrity_gates=gates
    )
    assert losing.champion == "draft_capital"


def test_tied_macro_or_failed_integrity_falls_back_to_draft_capital():
    macro, pooled, per_year, gates = champion_reports()
    macro.loc[macro["model"].eq("random_forest"), "ndcg_24"] = 0.7
    tied = select_champion(
        "rookie_year_ppr_points", macro, pooled, per_year, integrity_gates=gates
    )
    assert tied.champion == "draft_capital"

    macro, pooled, per_year, gates = champion_reports(integrity=False)
    failed = select_champion(
        "rookie_year_ppr_points", macro, pooled, per_year, integrity_gates=gates
    )
    assert failed.champion == "draft_capital"


@pytest.mark.parametrize(
    "gates",
    [
        {"data_integrity": True},
        {
            "data_integrity": True,
            "temporal_leakage": True,
            "identity": True,
            "interval_validation": True,
            "unknown": True,
        },
    ],
)
def test_missing_or_unknown_integrity_gate_is_rejected(gates):
    macro, pooled, per_year, _ = champion_reports()
    with pytest.raises(ValueError, match="exactly the frozen set"):
        select_champion(
            "three_year_ppr_points", macro, pooled, per_year, integrity_gates=gates
        )


def test_first_fold_is_unavailable_then_position_and_global_thresholds_apply():
    rows = []
    positions = ["QB"] * POSITION_RESIDUAL_MINIMUM + ["RB"] * 20 + ["WR"] * 10
    assert len(positions) == GLOBAL_RESIDUAL_MINIMUM
    for index, position in enumerate(positions):
        rows.append(
            {
                "test_year": 2020,
                "model": "random_forest",
                "position": position,
                "actual": float(index + 1),
                "prediction": float(index),
            }
        )
    rows.extend(
        [
            {"test_year": 2021, "model": "random_forest", "position": "QB", "actual": 11.0, "prediction": 10.0},
            {"test_year": 2021, "model": "random_forest", "position": "WR", "actual": 11.0, "prediction": 10.0},
        ]
    )
    result = add_prior_fold_intervals(pd.DataFrame(rows))

    assert result[result["test_year"].eq(2020)]["interval_status"].eq("unavailable").all()
    later = result[result["test_year"].eq(2021)].set_index("position")
    assert later.at["QB", "interval_source"] == "position"
    assert later.at["WR", "interval_source"] == "global"

    short = pd.DataFrame({"position": ["QB"] * 29 + ["RB"] * 30, "residual": [1.0] * 59})
    assert _residual_bounds(short, "QB") is None


def test_interval_metrics_report_only_measured_held_out_rows():
    predictions = pd.DataFrame(
        [
            {"model": "draft_capital", "position": "QB", "actual": 5.0, "interval_status": "unavailable", "interval_lower": np.nan, "interval_upper": np.nan},
            {"model": "draft_capital", "position": "QB", "actual": 5.0, "interval_status": "available", "interval_lower": 4.0, "interval_upper": 6.0},
            {"model": "draft_capital", "position": "RB", "actual": 8.0, "interval_status": "available", "interval_lower": 0.0, "interval_upper": 7.0},
        ]
    )
    overall = interval_metrics(predictions).query("slice == 'overall'").iloc[0]
    assert overall["held_out_row_count"] == 3
    assert overall["interval_row_count"] == 2
    assert overall["interval_coverage"] == 0.5
    assert overall["mean_interval_width"] == 4.5


def test_board_publication_is_valid_deterministic_and_ordered_by_rookie_year(tmp_path):
    training = feature_rows()
    position_bonus = training["position"].map({"QB": 4.0, "RB": 3.0, "WR": 2.0, "TE": 1.0})
    training["rookie_year_ppr_points"] = (
        50.0 - 5.0 * np.log1p(training["overall_pick"]) + position_bonus
    )
    current = current_class()
    board, evaluations = build_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
    )

    assert set(evaluations) == {"three_year_ppr_points", "rookie_year_ppr_points"}
    assert board.metadata.champion_by_target.three_year_ppr_points == "random_forest"
    assert board.metadata.champion_by_target.rookie_year_ppr_points == "draft_capital"
    ordered = sorted(board.players, key=lambda player: player.base_rank)
    assert [player.base_rank for player in ordered] == list(range(1, len(ordered) + 1))
    assert [player.rookie_year_ppr.p50 for player in ordered] == sorted(
        [player.rookie_year_ppr.p50 for player in ordered], reverse=True
    )
    assert all(player.tier == 1 for player in board.players)
    assert all(player.champion_by_target == board.metadata.champion_by_target for player in board.players)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = publish_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
        first,
    )
    second_paths = publish_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
        second,
    )
    assert first_paths["artifact"].read_bytes() == second_paths["artifact"].read_bytes()
    assert first_paths["manifest"].read_bytes() == second_paths["manifest"].read_bytes()
    validated = load_handoff(
        first_paths["artifact"], first_paths["manifest"], schema_path=first_paths["schema"]
    )
    assert validated.metadata.champion_by_target == board.metadata.champion_by_target
    assert [player.canonical_id for player in validated.players] == [
        player.canonical_id for player in ordered
    ]


def test_rookie_year_forecast_drives_rank_tiers_and_confidence(monkeypatch):
    training = feature_rows()
    current = feature_rows(years=[2026], players_per_year=13).drop(
        columns=[
            "three_year_ppr_points",
            "three_year_target_status",
            "rookie_year_ppr_points",
            "rookie_target_status",
            "data_quality_warnings",
        ]
    )
    current["identity_match_status"] = "exact"
    current["identity_match_method"] = "normalized_name_position"
    rookie_points = np.arange(13, dtype=float)
    three_year_points = rookie_points[::-1]

    def forecasts(_training, supplied_current, evaluation):
        if evaluation.target == "rookie_year_ppr_points":
            points = rookie_points
            widths = np.array([2.0] * 7 + [20.0] * 6)
        else:
            points = three_year_points
            widths = np.array([100.0] * 7 + [1.0] * 6)
        interval_status = ["available"] * len(points)
        if evaluation.target == "three_year_ppr_points":
            interval_status[0] = "unavailable"
        return pd.DataFrame(
            {
                "p10": points - widths / 2,
                "p50": points,
                "p90": points + widths / 2,
                "interval_status": interval_status,
            },
            index=supplied_current.index,
        )

    monkeypatch.setattr("rookie_ranker.champion_board.fit_selected_target", forecasts)
    board, _ = build_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
    )
    ordered = sorted(board.players, key=lambda player: player.base_rank)

    assert [player.canonical_id for player in ordered] == list(
        reversed(current["canonical_id"].astype(str).tolist())
    )
    assert [player.rookie_year_ppr.p50 for player in ordered] == sorted(
        rookie_points, reverse=True
    )
    assert [player.three_year_ppr.p50 for player in ordered] == sorted(three_year_points)
    assert ordered[0].tier == 1
    assert ordered[11].tier == 1
    assert ordered[12].tier == 2
    quarterbacks = [player for player in ordered if player.position == "QB"]
    assert [player.position_rank for player in quarterbacks] == [1, 2, 3, 4]
    assert [player.canonical_id for player in quarterbacks] == [
        str(current.loc[index, "canonical_id"]) for index in (12, 8, 4, 0)
    ]
    by_id = {player.canonical_id: player for player in board.players}
    assert by_id[str(current.loc[0, "canonical_id"])].confidence == "high"
    assert "three_year_ppr_points interval unavailable" in by_id[
        str(current.loc[0, "canonical_id"])
    ].data_quality_warnings
    assert by_id[str(current.loc[12, "canonical_id"])].confidence == "medium"


def test_confidence_uses_published_quantiles_at_median_boundary(tmp_path, monkeypatch):
    training = feature_rows()
    current = current_class().iloc[:3].copy().drop(columns=["data_quality_warnings"])
    current["identity_match_status"] = "exact"
    current["identity_match_method"] = "normalized_name_position"
    raw_widths = np.array([1.00001, 1.00002, 1.00003])

    def forecasts(_training, supplied_current, evaluation):
        widths = (
            raw_widths
            if evaluation.target == "rookie_year_ppr_points"
            else np.ones(len(supplied_current))
        )
        return pd.DataFrame(
            {
                "p10": np.zeros(len(supplied_current)),
                "p50": np.full(len(supplied_current), 0.5),
                "p90": widths,
                "interval_status": "available",
            },
            index=supplied_current.index,
        )

    monkeypatch.setattr("rookie_ranker.champion_board.fit_selected_target", forecasts)
    paths = publish_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
        tmp_path,
    )
    published = load_handoff(
        paths["artifact"], paths["manifest"], schema_path=paths["schema"]
    )
    published_widths = {
        player.canonical_id: player.rookie_year_ppr.p90 - player.rookie_year_ppr.p10
        for player in published.players
    }
    published_median = float(np.median(list(published_widths.values())))
    boundary_id = str(current.loc[2, "canonical_id"])
    boundary = next(player for player in published.players if player.canonical_id == boundary_id)

    assert raw_widths[2] > float(np.median(raw_widths))
    assert published_widths[boundary_id] == published_median
    assert boundary.confidence == "high"
    assert all(
        player.confidence
        == ("high" if published_widths[player.canonical_id] <= published_median else "medium")
        for player in published.players
    )


def test_board_derives_warnings_from_identity_and_college_status():
    training = feature_rows()
    current = current_class().drop(columns=["data_quality_warnings"])
    current.loc[0, "quarantine_reason"] = "reviewed live identity gap"
    current.loc[0, "college_stats_status"] = "missing"
    current.loc[1, "identity_match_status"] = "exact"
    current.loc[1, "identity_match_method"] = "normalized_name_position"
    current.loc[1, "college_stats_status"] = "missing"

    board, _ = build_rookie_board(
        {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
        current,
        current_evidence(current),
        publication_metadata(),
    )
    players = {player.canonical_id: player for player in board.players}

    quarantined = players[str(current.loc[0, "canonical_id"])]
    assert quarantined.confidence not in {"high", "medium"}
    assert {
        "college identity quarantined: reviewed live identity gap",
        "college stats status: missing",
    }.issubset(quarantined.data_quality_warnings)
    missing = players[str(current.loc[1, "canonical_id"])]
    assert missing.confidence not in {"high", "medium"}
    assert "college stats status: missing" in missing.data_quality_warnings


def test_incomplete_or_mismatched_current_cohort_is_rejected():
    training = feature_rows()
    complete = current_class()
    evidence = current_evidence(complete)

    with pytest.raises(ValueError, match="missing publication columns"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            complete.drop(columns=["college_stats_status"]),
            evidence,
            publication_metadata(),
        )

    null_status = complete.copy()
    null_status.loc[0, "college_stats_status"] = None
    with pytest.raises(ValueError, match="college_stats_status must be populated"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            null_status,
            evidence,
            publication_metadata(),
        )

    with pytest.raises(ValueError, match="does not match expected complete cohort"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            complete.iloc[:1].copy(),
            evidence,
            publication_metadata(),
        )

    wrong_year = complete.copy()
    wrong_year["draft_season"] = 2025
    with pytest.raises(ValueError, match="draft season does not match"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            wrong_year,
            evidence,
            publication_metadata(),
        )

    duplicate_pick = complete.copy()
    duplicate_pick.loc[1, "overall_pick"] = duplicate_pick.loc[0, "overall_pick"]
    with pytest.raises(ValueError, match="overall picks must be populated, whole, and unique"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            duplicate_pick,
            evidence,
            publication_metadata(),
        )

    swapped_picks = complete.copy()
    swapped_picks.loc[[0, 1], "overall_pick"] = swapped_picks.loc[[1, 0], "overall_pick"].to_numpy()
    with pytest.raises(ValueError, match="does not match expected complete cohort"):
        build_rookie_board(
            {"three_year_ppr_points": training, "rookie_year_ppr_points": training},
            swapped_picks,
            evidence,
            publication_metadata(),
        )
