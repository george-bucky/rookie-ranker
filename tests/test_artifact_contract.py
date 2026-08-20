from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import rookie_ranker.artifact_contract as artifact_contract
from rookie_ranker.artifact_contract import (
    ArtifactContractError,
    RookieBoard,
    load_handoff,
    write_handoff,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "rr04"
GOLDEN_ARTIFACT = FIXTURES / "rookie-board-2026.json"
GOLDEN_MANIFEST = FIXTURES / "rookie-board-2026.manifest.json"
SCHEMA = REPO_ROOT / "schemas" / "rookie-board.schema.json"


def golden_board() -> RookieBoard:
    return RookieBoard.model_validate_json(GOLDEN_ARTIFACT.read_bytes())


def write_mutated_handoff(
    tmp_path: Path,
    artifact: dict,
    *,
    manifest_updates: dict | None = None,
    year: int = 2026,
    allow_nan: bool = False,
) -> tuple[Path, Path, Path]:
    artifact_bytes = (
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=allow_nan,
        )
        + "\n"
    ).encode()
    manifest = json.loads(GOLDEN_MANIFEST.read_text())
    manifest.update(
        {
            "artifact_filename": f"rookie-board-{year}.json",
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "draft_class": year,
        }
    )
    if manifest_updates:
        manifest.update(manifest_updates)
    artifact_path = tmp_path / f"rookie-board-{year}.json"
    manifest_path = tmp_path / f"rookie-board-{year}.manifest.json"
    schema_path = tmp_path / "rookie-board.schema.json"
    artifact_path.write_bytes(artifact_bytes)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    schema_path.write_bytes(SCHEMA.read_bytes())
    return artifact_path, manifest_path, schema_path


def test_golden_handoff_loads_and_contains_only_public_contract_data():
    board = load_handoff(GOLDEN_ARTIFACT, GOLDEN_MANIFEST, schema_path=SCHEMA)
    text = GOLDEN_ARTIFACT.read_text()

    assert board.metadata.draft_class == 2026
    assert [player.canonical_id for player in board.players] == [
        "gsis:00-0030001",
        "pfr:FixtRu00",
    ]
    assert "passing_ATT" not in text
    assert "raw_response" not in text
    assert ".pkl" not in text


def test_golden_round_trip_is_byte_deterministic(tmp_path):
    board = golden_board()
    first = write_handoff(board, tmp_path / "first")

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"].reverse()
    payload["metadata"]["source_notices"].reverse()
    payload["metadata"]["training_cohorts"]["three_year_ppr_points"].reverse()
    payload["players"][0]["data_quality_warnings"].reverse()
    reordered = RookieBoard.model_validate_json(json.dumps(payload))
    second = write_handoff(reordered, tmp_path / "second")

    assert first["artifact"].read_bytes() == GOLDEN_ARTIFACT.read_bytes()
    assert first["manifest"].read_bytes() == GOLDEN_MANIFEST.read_bytes()
    assert first["schema"].read_bytes() == SCHEMA.read_bytes()
    assert second["artifact"].read_bytes() == first["artifact"].read_bytes()
    assert second["manifest"].read_bytes() == first["manifest"].read_bytes()


def test_numeric_output_uses_frozen_precision(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["rookie_year_ppr"]["p50"] = 180.567891234
    board = RookieBoard.model_validate_json(json.dumps(payload))

    paths = write_handoff(board, tmp_path)
    written = json.loads(paths["artifact"].read_text())

    assert written["players"][0]["rookie_year_ppr"]["p50"] == 180.5679


def test_writer_revalidates_constructed_models_before_any_output(tmp_path):
    board = golden_board()
    invalid = board.model_copy(update={"players": (board.players[0], board.players[0])})
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ArtifactContractError, match="canonical IDs must be unique"):
        write_handoff(invalid, output_dir)

    assert not output_dir.exists()


def test_unknown_schema_version_fails_closed(tmp_path):
    manifest = json.loads(GOLDEN_MANIFEST.read_text())
    manifest["schema_version"] = "2.0.0"
    manifest_path = tmp_path / GOLDEN_MANIFEST.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="unsupported schema version"):
        load_handoff(GOLDEN_ARTIFACT, manifest_path, schema_path=SCHEMA)


def test_breaking_artifact_version_fails_closed(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["artifact_version"] = "2.0.0"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(
        tmp_path, payload, manifest_updates={"artifact_version": "2.0.0"}
    )

    with pytest.raises(ArtifactContractError, match="unsupported artifact version"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_duplicate_canonical_and_verified_source_ids_fail(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][1]["canonical_id"] = payload["players"][0]["canonical_id"]
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="canonical IDs must be unique"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][1]["source_ids"]["gsis_id"] = "00-0030001"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="verified gsis_id values must be unique"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_non_finite_and_unordered_quantiles_fail(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["three_year_ppr"]["p50"] = float("nan")
    artifact_path, manifest_path, schema_path = write_mutated_handoff(
        tmp_path, payload, allow_nan=True
    )
    with pytest.raises(ArtifactContractError, match="non-finite JSON number"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["three_year_ppr"] = {"p10": 500.0, "p50": 400.0, "p90": 600.0}
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="p10 <= p50 <= p90"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_missing_or_mismatched_provenance_fails(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    del payload["metadata"]["producer_commit"]
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="producer_commit"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    artifact_path, manifest_path, schema_path = write_mutated_handoff(
        tmp_path,
        payload,
        manifest_updates={"producer_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    )
    with pytest.raises(ArtifactContractError, match="producer commit"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_wrong_class_year_fails(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload, year=2025)

    with pytest.raises(ArtifactContractError, match="draft class"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_checksum_detects_any_artifact_byte_change(tmp_path):
    artifact_path = tmp_path / GOLDEN_ARTIFACT.name
    manifest_path = tmp_path / GOLDEN_MANIFEST.name
    schema_path = tmp_path / SCHEMA.name
    artifact_path.write_bytes(GOLDEN_ARTIFACT.read_bytes() + b" ")
    manifest_path.write_bytes(GOLDEN_MANIFEST.read_bytes())
    schema_path.write_bytes(SCHEMA.read_bytes())

    with pytest.raises(ArtifactContractError, match="checksum"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_extra_fields_and_target_champion_mismatch_fail(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["raw_model"] = "forbidden"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="JSON Schema validation failed.*raw_model"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["champion_by_target"]["three_year_ppr_points"] = "random_forest"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="player champion_by_target"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_canonical_identity_and_match_evidence_must_agree(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["source_ids"]["gsis_id"] = "different"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="GSIS canonical ID"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["players"][0]["identity_match_method"] = "override"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="method and status are inconsistent"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_schema_is_strict_and_schema_version_must_match(tmp_path):
    schema = json.loads(SCHEMA.read_text())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    for definition in schema["$defs"].values():
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False

    schema["x-schema-version"] = "2.0.0"
    schema_path = tmp_path / SCHEMA.name
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ArtifactContractError, match="JSON Schema bytes"):
        load_handoff(GOLDEN_ARTIFACT, GOLDEN_MANIFEST, schema_path=schema_path)


def test_altered_player_schema_is_rejected_and_schema_is_actively_enforced(
    tmp_path, monkeypatch
):
    schema = json.loads(SCHEMA.read_text())
    schema["$defs"]["rookiePlayer"]["properties"]["name"]["const"] = "Impossible Name"
    schema_bytes = (json.dumps(schema, sort_keys=True, separators=(",", ":")) + "\n").encode()
    schema_path = tmp_path / SCHEMA.name
    schema_path.write_bytes(schema_bytes)

    with pytest.raises(ArtifactContractError, match="JSON Schema bytes"):
        load_handoff(GOLDEN_ARTIFACT, GOLDEN_MANIFEST, schema_path=schema_path)

    monkeypatch.setattr(
        artifact_contract,
        "SCHEMA_SHA256",
        hashlib.sha256(schema_bytes).hexdigest(),
    )
    with pytest.raises(ArtifactContractError, match="JSON Schema validation failed.*name"):
        load_handoff(GOLDEN_ARTIFACT, GOLDEN_MANIFEST, schema_path=schema_path)


def test_duplicate_json_keys_fail_for_artifact_manifest_and_schema(tmp_path):
    artifact_path = tmp_path / GOLDEN_ARTIFACT.name
    manifest_path = tmp_path / GOLDEN_MANIFEST.name
    schema_path = tmp_path / SCHEMA.name
    manifest_path.write_bytes(GOLDEN_MANIFEST.read_bytes())
    schema_path.write_bytes(SCHEMA.read_bytes())

    artifact_path.write_bytes(
        GOLDEN_ARTIFACT.read_bytes().replace(b'"base_rank":1', b'"base_rank":1,"base_rank":1', 1)
    )
    with pytest.raises(ArtifactContractError, match="duplicate JSON object key.*base_rank"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    artifact_path.write_bytes(GOLDEN_ARTIFACT.read_bytes())
    manifest_path.write_bytes(
        GOLDEN_MANIFEST.read_bytes().replace(
            b'"artifact_filename":', b'"artifact_filename":"duplicate","artifact_filename":', 1
        )
    )
    with pytest.raises(ArtifactContractError, match="duplicate JSON object key.*artifact_filename"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    manifest_path.write_bytes(GOLDEN_MANIFEST.read_bytes())
    schema_path.write_bytes(
        SCHEMA.read_bytes().replace(b'"title": "Rookie Board",', b'"title": "duplicate",\n  "title": "Rookie Board",', 1)
    )
    with pytest.raises(ArtifactContractError, match="duplicate JSON object key.*title"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


def test_outcome_maturity_and_post_draft_provenance_fail_closed(tmp_path):
    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["training_cohorts"]["three_year_ppr_points"].append(2024)
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="training cohorts exceed target maturity"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["target_maturity"]["three_year_ppr_points"] = 2024
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="target maturity must agree"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["outcomes_cutoff_season"] = 2026
    payload["metadata"]["target_maturity"] = {
        "rookie_year_ppr_points": 2026,
        "three_year_ppr_points": 2024,
    }
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="outcomes cutoff season is not complete"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["draft_event_date"] = "2026-05-02"
    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="cannot predate draft_event_date"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)


@pytest.mark.parametrize(
    ("metric", "value", "schema_bound", "error"),
    [
        ("ndcg_at_24", 1.1, "maximum", "ndcg_at_24 must be between"),
        ("mae", -1.0, "minimum", "magnitude metrics must not be negative"),
        ("mean_interval_width", -1.0, "minimum", "magnitude metrics must not be negative"),
    ],
)
def test_metric_bounds_agree_between_schema_and_pydantic(
    tmp_path, metric, value, schema_bound, error
):
    schema = json.loads(SCHEMA.read_text())
    metric_schema = schema["$defs"]["targetEvaluationSummary"]["properties"][metric]
    assert metric_schema[schema_bound] == (1 if schema_bound == "maximum" else 0)

    payload = json.loads(GOLDEN_ARTIFACT.read_text())
    payload["metadata"]["evaluation_summary"]["rookie_year_ppr_points"][metric] = value
    with pytest.raises(ValueError, match=error):
        RookieBoard.model_validate_json(json.dumps(payload))

    artifact_path, manifest_path, schema_path = write_mutated_handoff(tmp_path, payload)
    with pytest.raises(ArtifactContractError, match="JSON Schema validation failed"):
        load_handoff(artifact_path, manifest_path, schema_path=schema_path)
