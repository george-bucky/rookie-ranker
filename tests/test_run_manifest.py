import json

import pandas as pd
import pytest

from rookie_ranker.run_manifest import (
    build_coverage_report,
    build_run_manifest,
    dataframe_sha256,
    output_record,
    source_record,
    validate_target_status_values,
    validate_coverage_partitions,
    write_manifest,
)


def truth_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draft_season": [2023, 2023, 2024],
            "canonical_id": ["gsis:one", "pfr:two", "gsis:three"],
            "gsis_id": ["one", pd.NA, "three"],
            "identity_match_status": ["exact", "quarantined", "override"],
            "college_stats_status": ["observed", "unresolved_identity", "missing"],
            "rookie_year_ppr_points": [0.0, pd.NA, 10.0],
            "rookie_target_status": ["complete", "missing_gsis", "complete"],
            "three_year_ppr_points": [12.0, pd.NA, pd.NA],
            "three_year_target_status": ["complete", "missing_gsis", "immature"],
        }
    )


def test_coverage_counts_reconcile_for_each_draft_class():
    coverage = build_coverage_report(truth_table())

    year = coverage.set_index("draft_season").loc[2023]
    assert year["cohort_count"] == 2
    assert year["gsis_count"] == 1
    assert year["exact_match_count"] == 1
    assert year["quarantined_count"] == 1
    assert year["rookie_target_complete_count"] == 1
    assert year["rookie_zero_count"] == 1
    assert year["rookie_missing_gsis_count"] == 1


def test_unknown_status_fails_coverage_partition():
    table = truth_table()
    table.loc[0, "identity_match_status"] = "unknown"

    with pytest.raises(ValueError, match="Identity coverage does not reconcile"):
        build_coverage_report(table)


def test_explicit_partition_validation_rejects_bad_counts():
    coverage = build_coverage_report(truth_table())
    coverage.loc[0, "college_observed_count"] = 0

    with pytest.raises(ValueError, match="College coverage does not reconcile"):
        validate_coverage_partitions(coverage)


@pytest.mark.parametrize(
    ("status_column", "value_column", "status", "value", "message"),
    [
        (
            "rookie_target_status",
            "rookie_year_ppr_points",
            "complete",
            pd.NA,
            "complete targets require a finite numeric value",
        ),
        (
            "rookie_target_status",
            "rookie_year_ppr_points",
            "immature",
            1.0,
            "immature or missing_gsis targets must be null",
        ),
        (
            "three_year_target_status",
            "three_year_ppr_points",
            "missing_gsis",
            0.0,
            "immature or missing_gsis targets must be null",
        ),
        (
            "three_year_target_status",
            "three_year_ppr_points",
            "complete",
            float("inf"),
            "complete targets require a finite numeric value",
        ),
    ],
)
def test_target_status_and_value_must_agree(
    status_column, value_column, status, value, message
):
    table = truth_table()
    table.loc[0, status_column] = status
    table.loc[0, value_column] = value

    with pytest.raises(ValueError, match=message):
        validate_target_status_values(table)


def test_dataframe_hash_changes_when_input_changes():
    frame = pd.DataFrame({"value": [1]})
    changed = pd.DataFrame({"value": [2]})

    assert dataframe_sha256(frame) != dataframe_sha256(changed)


def test_manifest_records_source_schema_output_hash_and_attribution(tmp_path):
    source = pd.DataFrame({"season": [2023], "player_id": ["one"]})
    output_path = tmp_path / "training-table.csv"
    output_path.write_text("canonical_id\ngsis:one\n", encoding="utf-8")
    coverage = build_coverage_report(truth_table())
    manifest = build_run_manifest(
        arguments={"draft_years": [2023]},
        generated_at_utc="2026-08-19T12:00:00Z",
        producer_commit="abc123",
        sources=[
            source_record(
                name="nflverse_draft_picks",
                frame=source,
                query_years=[2023],
                fetched_at_utc="2026-08-19T12:00:00Z",
                source_url="https://example.test/source",
                loader_version="0.1.5",
                source_version="fixture-v1",
            )
        ],
        outputs=[output_record(output_path, row_count=1)],
        coverage=coverage,
    )
    manifest_path = tmp_path / "run-manifest.json"
    write_manifest(manifest, manifest_path)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert loaded["producer_commit"] == "abc123"
    assert loaded["sources"][0]["query_years"] == [2023]
    assert loaded["sources"][0]["schema"]["columns"][0]["name"] == "season"
    assert len(loaded["outputs"][0]["sha256"]) == 64
    assert {notice["name"] for notice in loaded["source_notices"]} == {
        "nflverse",
        "CollegeFootballData",
    }
