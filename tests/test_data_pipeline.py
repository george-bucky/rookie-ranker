import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rookie_ranker.data_pipeline import (
    PipelineConfig,
    _fetch_live_sources,
    _producer_commit,
    build_parser,
    compose_training_table,
    parse_years,
    run_build,
    validate_config,
    validate_year_configuration,
)


EXPECTED = Path(__file__).parent / "fixtures" / "rr02" / "expected"


def test_year_parser_supports_ranges_and_rejects_duplicates():
    assert parse_years("2019,2021:2023") == (2019, 2021, 2022, 2023)
    with pytest.raises(Exception, match="duplicates"):
        parse_years("2023,2022:2023")


def test_cli_requires_explicit_years_timeout_overrides_and_output():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--draft-years", "2023"])


def test_year_validation_rejects_future_and_post_draft_college_years():
    with pytest.raises(ValueError, match="between 1900 and 2026"):
        validate_year_configuration([2027], [2026], 2026, current_year=2026)
    with pytest.raises(ValueError, match="must end before"):
        validate_year_configuration([2023], [2022, 2023], 2023, current_year=2026)


def test_outcomes_cutoff_must_precede_current_year():
    with pytest.raises(ValueError, match="strictly before the current year"):
        validate_year_configuration([2026], [2025], 2026, current_year=2026)


def test_live_nflreadpy_frames_convert_without_pyarrow(monkeypatch, tmp_path):
    class PolarsLike:
        def __init__(self, rows):
            self.rows = rows

        def to_dicts(self):
            return self.rows

        def to_pandas(self):
            raise AssertionError("to_pandas must not be used")

    monkeypatch.setattr(
        "nflreadpy.load_draft_picks",
        lambda _: PolarsLike([{"season": 2024, "pick": 1}]),
    )
    monkeypatch.setattr(
        "nflreadpy.load_player_stats",
        lambda _, summary_level: PolarsLike(
            [{"player_id": "G1", "season": 2024, "fantasy_points_ppr": 1.0}]
        ),
    )
    monkeypatch.setattr("nflreadpy.config.update_config", lambda **_: None)
    monkeypatch.setattr(
        "rookie_ranker.data_pipeline._fetch_college_year",
        lambda year, timeout: pd.DataFrame({"season": [year], "playerId": [1]}),
    )
    config = PipelineConfig(
        draft_years=(2024,),
        college_years=(2023,),
        outcomes_through_year=2024,
        http_timeout_seconds=30,
        identity_overrides=tmp_path / "overrides.csv",
        output_dir=tmp_path / "output",
    )

    draft, stats, college = _fetch_live_sources(config)

    assert draft.to_dict("records") == [{"season": 2024, "pick": 1}]
    assert stats.loc[0, "fantasy_points_ppr"] == 1.0
    assert college.loc[0, "season"] == 2023


def test_auto_producer_commit_rejects_dirty_git_tree(monkeypatch):
    results = iter(
        [SimpleNamespace(stdout="abc123\n"), SimpleNamespace(stdout="?? generated.csv\n")]
    )
    monkeypatch.setattr("rookie_ranker.data_pipeline.subprocess.run", lambda *_, **__: next(results))

    with pytest.raises(RuntimeError, match="dirty Git tree"):
        _producer_commit()


def test_producer_git_commands_are_pinned_to_rookie_ranker_repo(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "abc123\n" if command[:3] == ["git", "rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rookie_ranker.data_pipeline.subprocess.run", fake_run)

    assert _producer_commit() == "abc123"
    expected_root = Path(__file__).resolve().parents[1]
    assert len(calls) == 2
    assert all(call_kwargs["cwd"] == expected_root for _, call_kwargs in calls)


def test_local_source_mode_requires_all_three_inputs(tmp_path):
    config = PipelineConfig(
        draft_years=(2023,),
        college_years=(2022,),
        outcomes_through_year=2025,
        http_timeout_seconds=30,
        identity_overrides=tmp_path / "overrides.csv",
        output_dir=tmp_path / "output",
        draft_input=tmp_path / "draft.csv",
    )
    with pytest.raises(ValueError, match="requires draft, player-stats, and college"):
        validate_config(config)


def test_compose_training_table_is_a_left_join_and_rejects_duplicate_additions():
    cohort = pd.DataFrame(
        {"canonical_id": ["one", "two"], "draft_season": [2023, 2023], "overall_pick": [1, 2]}
    )
    targets = pd.DataFrame({"canonical_id": ["one"], "rookie_year_ppr_points": [0.0]})
    identities = pd.DataFrame({"canonical_id": ["one", "two"], "identity_match_status": ["exact", "quarantined"]})
    features = pd.DataFrame({"canonical_id": ["one"], "college_stats_status": ["observed"]})

    table = compose_training_table(cohort, targets, identities, features)
    assert table["canonical_id"].tolist() == ["one", "two"]
    assert pd.isna(table.loc[1, "rookie_year_ppr_points"])

    with pytest.raises(ValueError, match="duplicate canonical IDs"):
        compose_training_table(cohort, pd.concat([targets, targets]), identities, features)


def test_offline_build_writes_expected_truth_coverage_quarantine_and_manifest(
    tmp_path, monkeypatch
):
    draft = pd.DataFrame(
        {
            "season": [2023, 2023],
            "player": ["Drafted One", "Drafted Two"],
        }
    )
    stats = pd.DataFrame(
        {"player_id": ["G1"], "season": [2023], "fantasy_points_ppr": [0.0]}
    )
    college = pd.DataFrame(
        {"season": [2022], "playerId": [101], "player": ["Drafted One"]}
    )
    cohort = pd.DataFrame(
        {
            "draft_season": [2023, 2023],
            "round": [1, 1],
            "overall_pick": [1, 2],
            "nfl_team": ["AAA", "BBB"],
            "player_name": ["Drafted One", "Drafted Two"],
            "position": ["QB", "RB"],
            "canonical_id": ["gsis:G1", "pfr:P2"],
            "canonical_id_source": ["gsis", "pfr"],
            "gsis_id": ["G1", pd.NA],
            "pfr_player_id": ["P1", "P2"],
            "cfb_player_id": [pd.NA, pd.NA],
        }
    )
    targets = pd.DataFrame(
        {
            "canonical_id": ["gsis:G1", "pfr:P2"],
            "rookie_year_ppr_points": [0.0, pd.NA],
            "rookie_target_status": ["complete", "missing_gsis"],
            "three_year_ppr_points": [0.0, pd.NA],
            "three_year_target_status": ["complete", "missing_gsis"],
        }
    )
    identities = pd.DataFrame(
        {
            "canonical_id": ["gsis:G1", "pfr:P2"],
            "cfbd_player_id": [101, pd.NA],
            "identity_match_status": ["exact", "quarantined"],
            "identity_match_method": ["normalized_name_position", "none"],
            "quarantine_reason": [pd.NA, "missing_match"],
        }
    )
    features = pd.DataFrame(
        {
            "canonical_id": ["gsis:G1", "pfr:P2"],
            "college_stats_status": ["observed", "unresolved_identity"],
        }
    )
    quarantine = pd.DataFrame(
        {
            "draft_season": [2023],
            "overall_pick": [2],
            "canonical_id": ["pfr:P2"],
            "player_name": ["Drafted Two"],
            "position": ["RB"],
            "normalized_name": ["drafted two"],
            "candidate_count": [0],
            "candidate_cfbd_ids": [pd.NA],
            "reason": ["missing_match"],
            "override_status": ["quarantine"],
        }
    )

    monkeypatch.setattr("rookie_ranker.cohort.build_draft_cohort", lambda *_: cohort)
    monkeypatch.setattr("rookie_ranker.outcomes.build_targets", lambda *_: targets)
    monkeypatch.setattr(
        "rookie_ranker.identity.match_college_identities", lambda *_, **__: (identities, quarantine)
    )
    monkeypatch.setattr("rookie_ranker.college_features.build_college_features", lambda *_: features)

    overrides = tmp_path / "identity_overrides.csv"
    overrides.write_text(
        "draft_season,overall_pick,resolution,cfbd_player_id,reason,evidence\n",
        encoding="utf-8",
    )
    config = PipelineConfig(
        draft_years=(2023,),
        college_years=(2022,),
        outcomes_through_year=2025,
        http_timeout_seconds=30,
        identity_overrides=overrides,
        output_dir=tmp_path / "output",
    )

    paths = run_build(
        config,
        frames=(draft, stats, college),
        generated_at_utc="2026-08-19T12:00:00Z",
        producer_commit="abc123",
    )

    for key, expected_name in [
        ("training_table", "training-table.csv"),
        ("quarantine", "identity-quarantine.csv"),
        ("coverage", "coverage.csv"),
    ]:
        assert paths[key].read_text(encoding="utf-8") == (EXPECTED / expected_name).read_text(
            encoding="utf-8"
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["arguments"]["draft_years"] == [2023]
    assert manifest["producer_commit"] == "abc123"
    assert [output["filename"] for output in manifest["outputs"]] == [
        "training-table.csv",
        "identity-quarantine.csv",
        "coverage.csv",
    ]


def test_real_worker_contracts_integrate_offline(tmp_path):
    draft = pd.DataFrame(
        {
            "season": [2024] * 4,
            "round": [1] * 4,
            "pick": [1, 2, 3, 4],
            "team": ["AAA", "BBB", "CCC", "DDD"],
            "position": ["QB", "WR", "RB", "WR"],
            "pfr_player_name": ["José Smith, Jr.", "Chris Jones", "John Doe", "Terry Tight"],
            "gsis_id": ["G1", "G2", "G3", "G4"],
            "pfr_player_id": ["P1", "P2", "P3", "P4"],
            "cfb_player_id": [pd.NA] * 4,
        }
    )
    stats = pd.DataFrame(
        {"player_id": ["G1"], "season": [2024], "fantasy_points_ppr": [0.0]}
    )
    college_fixture = Path(__file__).parent / "fixtures" / "rr02" / "college"
    college = pd.read_csv(college_fixture / "player_season_stats.csv")
    config = PipelineConfig(
        draft_years=(2024,),
        college_years=(2022, 2023),
        outcomes_through_year=2024,
        http_timeout_seconds=30,
        identity_overrides=college_fixture / "identity_overrides.csv",
        output_dir=tmp_path / "integrated",
    )

    paths = run_build(
        config,
        frames=(draft, stats, college),
        generated_at_utc="2026-08-19T12:00:00Z",
        producer_commit="abc123",
    )
    truth = pd.read_csv(paths["training_table"])
    coverage = pd.read_csv(paths["coverage"])

    assert len(truth) == 4
    assert truth["identity_match_status"].tolist() == [
        "exact",
        "quarantined",
        "override",
        "quarantined",
    ]
    assert truth["rookie_year_ppr_points"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert truth["three_year_target_status"].tolist() == ["immature"] * 4
    assert coverage.loc[0, "cohort_count"] == 4
    assert coverage.loc[0, "quarantined_count"] == 2
