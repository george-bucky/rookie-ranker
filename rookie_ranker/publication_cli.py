"""Offline, provenance-checked RR-05 rookie-board publication."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifact_contract import SourceNotice, write_handoff
from .champion_board import (
    CurrentClassEvidence,
    PublicationMetadata,
    TARGETS,
    TargetEvaluation,
    build_rookie_board,
)
from .data_pipeline import _producer_commit
from .run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    SOURCE_NOTICES,
    build_coverage_report,
    sha256_bytes,
    sha256_file,
)


AUDIT_SCHEMA_VERSION = "1.0.0"
REQUIRED_SOURCE_NAMES = frozenset(
    {
        "nflverse_draft_picks",
        "nflverse_player_stats_reg",
        "college_football_data_player_season",
        "identity_overrides",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


class PublicationInputError(ValueError):
    """Raised when offline publication inputs do not prove their provenance."""


@dataclass(frozen=True)
class OfflinePublicationConfig:
    training_table: Path
    training_table_sha256: str
    run_manifest: Path
    run_manifest_sha256: str
    draft_class: int
    draft_event_date: date
    data_cutoff: date
    outcomes_cutoff_season: int
    output_dir: Path


def _valid_sha256(value: str, label: str) -> str:
    value = value.strip().lower()
    if not _SHA256.fullmatch(value):
        raise PublicationInputError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PublicationInputError(f"run manifest contains invalid number: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationInputError(f"run manifest contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationInputError(f"invalid run manifest: {error}") from error
    if not isinstance(payload, dict):
        raise PublicationInputError("run manifest root must be an object")
    return payload


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PublicationInputError(f"{label} must be an ISO-8601 timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PublicationInputError(f"{label} must be an ISO-8601 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise PublicationInputError(f"{label} must include a timezone")
    return result


def _validate_source_metadata(
    manifest: Mapping[str, Any], arguments: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
        raise PublicationInputError("run manifest sources must be objects")
    by_name = {source.get("name"): source for source in sources}
    if len(by_name) != len(sources) or set(by_name) != REQUIRED_SOURCE_NAMES:
        raise PublicationInputError("run manifest must contain the exact RR-02 source set")
    for name, source in by_name.items():
        content_hash = source.get("content_sha256")
        if not isinstance(content_hash, str) or not _SHA256.fullmatch(content_hash):
            raise PublicationInputError(f"source {name} has an invalid content SHA-256")
        if source.get("source_version") != f"sha256:{content_hash}":
            raise PublicationInputError(f"source {name} version must match its content SHA-256")
        if not isinstance(source.get("source_url"), str) or not source["source_url"].strip():
            raise PublicationInputError(f"source {name} URL must be populated")
        _timestamp(source.get("fetched_at_utc"), f"source {name} fetched_at_utc")
        years = source.get("query_years")
        if (
            not isinstance(years, list)
            or not years
            or any(not isinstance(year, int) or isinstance(year, bool) for year in years)
            or years != sorted(set(years))
        ):
            raise PublicationInputError(f"source {name} query years must be sorted and unique")
        if not isinstance(source.get("row_count"), int) or source["row_count"] < 0:
            raise PublicationInputError(f"source {name} row count must be nonnegative")
        schema = source.get("schema")
        if (
            not isinstance(schema, dict)
            or not isinstance(schema.get("sha256"), str)
            or not _SHA256.fullmatch(schema["sha256"])
            or not isinstance(schema.get("columns"), list)
            or not schema["columns"]
        ):
            raise PublicationInputError(f"source {name} schema metadata is invalid")
        encoded_columns = json.dumps(
            schema["columns"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if schema["sha256"] != sha256_bytes(encoded_columns):
            raise PublicationInputError(f"source {name} schema hash does not match its columns")

    draft_years = arguments.get("draft_years")
    college_years = arguments.get("college_years")
    outcomes_year = arguments.get("outcomes_through_year")
    if (
        not isinstance(draft_years, list)
        or not draft_years
        or not isinstance(college_years, list)
        or not college_years
        or not isinstance(outcomes_year, int)
    ):
        raise PublicationInputError("run manifest year arguments are invalid")
    expected_years = {
        "nflverse_draft_picks": draft_years,
        "identity_overrides": draft_years,
        "nflverse_player_stats_reg": list(range(min(draft_years), outcomes_year + 1)),
        "college_football_data_player_season": college_years,
    }
    for name, years in expected_years.items():
        if by_name[name]["query_years"] != years:
            raise PublicationInputError(f"source {name} query years do not match RR-02 arguments")

    notices = manifest.get("source_notices")
    if not isinstance(notices, list) or any(not isinstance(notice, dict) for notice in notices):
        raise PublicationInputError("run manifest source notices must be objects")
    actual_notices = {notice.get("name"): notice for notice in notices}
    expected_notices = {notice["name"]: notice for notice in SOURCE_NOTICES.values()}
    if actual_notices != expected_notices:
        raise PublicationInputError("run manifest source notices do not match RR-02 metadata")
    return by_name


def _validate_inputs(
    config: OfflinePublicationConfig,
) -> tuple[
    pd.DataFrame,
    Mapping[str, Any],
    dict[str, Mapping[str, Any]],
    str,
    str,
    datetime,
]:
    expected_training_hash = _valid_sha256(
        config.training_table_sha256, "training table hash"
    )
    expected_manifest_hash = _valid_sha256(config.run_manifest_sha256, "run manifest hash")
    actual_training_hash = sha256_file(config.training_table)
    actual_manifest_hash = sha256_file(config.run_manifest)
    if actual_training_hash != expected_training_hash:
        raise PublicationInputError("training table SHA-256 does not match the expected hash")
    if actual_manifest_hash != expected_manifest_hash:
        raise PublicationInputError("run manifest SHA-256 does not match the expected hash")

    manifest = _load_json(config.run_manifest)
    expected_keys = {
        "schema_version",
        "generated_at_utc",
        "producer_commit",
        "arguments",
        "sources",
        "outputs",
        "coverage",
        "source_notices",
    }
    if set(manifest) != expected_keys or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PublicationInputError("run manifest does not match the RR-02 v1 topology")
    rr02_commit = manifest.get("producer_commit")
    if not isinstance(rr02_commit, str) or not _COMMIT.fullmatch(rr02_commit):
        raise PublicationInputError("run manifest producer commit is invalid")
    manifest_generated_at = _timestamp(manifest.get("generated_at_utc"), "run generated_at_utc")

    arguments = manifest.get("arguments")
    if not isinstance(arguments, dict):
        raise PublicationInputError("run manifest arguments must be an object")
    if arguments.get("outcomes_through_year") != config.outcomes_cutoff_season:
        raise PublicationInputError("outcomes cutoff does not match the RR-02 run manifest")
    draft_years = arguments.get("draft_years")
    if not isinstance(draft_years, list) or config.draft_class not in draft_years:
        raise PublicationInputError("draft class is absent from the RR-02 run manifest")

    sources = _validate_source_metadata(manifest, arguments)
    try:
        training = pd.read_csv(config.training_table)
    except (OSError, pd.errors.ParserError) as error:
        raise PublicationInputError(f"invalid training table: {error}") from error
    training_years = sorted(
        pd.to_numeric(training.get("draft_season"), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if training_years != draft_years:
        raise PublicationInputError("training table draft years do not match RR-02 arguments")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or any(not isinstance(output, dict) for output in outputs):
        raise PublicationInputError("run manifest outputs must be objects")
    matching_outputs = [
        output for output in outputs if output.get("filename") == config.training_table.name
    ]
    if len(matching_outputs) != 1:
        raise PublicationInputError("run manifest must identify the training table exactly once")
    training_output = matching_outputs[0]
    if (
        training_output.get("sha256") != actual_training_hash
        or training_output.get("byte_count") != config.training_table.stat().st_size
        or training_output.get("row_count") != len(training)
    ):
        raise PublicationInputError("run manifest training-table output metadata does not match")

    try:
        actual_coverage = build_coverage_report(training).to_dict("records")
    except ValueError as error:
        raise PublicationInputError(f"training table coverage is invalid: {error}") from error
    if _json_ready(actual_coverage) != manifest.get("coverage"):
        raise PublicationInputError("training table coverage does not match the run manifest")
    return (
        training,
        manifest,
        sources,
        actual_training_hash,
        actual_manifest_hash,
        manifest_generated_at,
    )


def _source_notices(sources: Mapping[str, Mapping[str, Any]]) -> tuple[SourceNotice, ...]:
    return (
        SourceNotice(
            name="nflverse",
            url="https://github.com/nflverse/nflverse-data",
            license="CC-BY-4.0",
            notice="Draft cohort and regular-season PPR outcomes are derived from nflverse data.",
            version=(
                f"draft={sources['nflverse_draft_picks']['source_version']}; "
                f"stats={sources['nflverse_player_stats_reg']['source_version']}"
            ),
        ),
        SourceNotice(
            name="CollegeFootballData",
            url="https://collegefootballdata.com/",
            license="Provider terms apply",
            notice=(
                "Only derived college features and reviewed identity decisions are published; "
                "raw API responses and credentials are excluded."
            ),
            version=str(sources["college_football_data_player_season"]["source_version"]),
        ),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return 0.0 if value == 0 else value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _model_metrics(evaluation: TargetEvaluation, model: str) -> dict[str, Any]:
    def records(table: pd.DataFrame) -> list[dict[str, Any]]:
        selected = table[table["model"].eq(model)] if "model" in table else table
        return _json_ready(selected.to_dict("records"))

    return {
        "macro_class_ranking": records(evaluation.macro_class_ranking)[0],
        "pooled_row_magnitude": records(evaluation.pooled_row_magnitude)[0],
        "per_year": records(evaluation.per_year),
        "position_slices": records(evaluation.position_slices),
        "interval_summary": records(evaluation.interval_summary),
    }


def _target_audit(evaluation: TargetEvaluation) -> dict[str, Any]:
    decision = evaluation.decision
    return {
        "selected_champion": decision.champion,
        "decision": {
            "eligible_fold_count": decision.eligible_fold_count,
            "strict_win_count": decision.strict_win_count,
            "strict_win_rate": decision.strict_win_rate,
            "challenger_ndcg_24": decision.challenger_ndcg_24,
            "baseline_ndcg_24": decision.baseline_ndcg_24,
            "challenger_mae": decision.challenger_mae,
            "baseline_mae": decision.baseline_mae,
            "gates": dict(decision.gates),
            "integrity_gates": dict(decision.integrity_gates),
        },
        "models": {
            "draft_capital": _model_metrics(evaluation, "draft_capital"),
            "random_forest": _model_metrics(evaluation, "random_forest"),
        },
    }


def _audit_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def publish_offline(config: OfflinePublicationConfig) -> dict[str, Path]:
    """Publish from RR-02 files only after clean-repository and provenance checks."""
    producer_commit = _producer_commit()
    if not _COMMIT.fullmatch(producer_commit):
        raise PublicationInputError("current producer commit is invalid")
    (
        training,
        manifest,
        sources,
        training_hash,
        manifest_hash,
        manifest_generated_at,
    ) = _validate_inputs(config)
    current = training[training["draft_season"].eq(config.draft_class)].copy()
    if current.empty:
        raise PublicationInputError("training table has no rows for the requested draft class")
    evidence = CurrentClassEvidence(
        draft_class=config.draft_class,
        draft_keys=frozenset(
            zip(
                current["canonical_id"].astype(str),
                current["overall_pick"].astype(int),
                current["position"].astype(str),
                strict=True,
            )
        ),
    )
    training_by_target = {
        "three_year_ppr_points": training[
            training["three_year_target_status"].eq("complete")
        ].copy(),
        "rookie_year_ppr_points": training[
            training["rookie_target_status"].eq("complete")
        ].copy(),
    }
    metadata = PublicationMetadata(
        producer_commit=producer_commit,
        draft_event_date=config.draft_event_date,
        generated_at=manifest_generated_at,
        data_cutoff=config.data_cutoff,
        outcomes_cutoff_season=config.outcomes_cutoff_season,
        source_notices=_source_notices(sources),
    )
    board, evaluations = build_rookie_board(
        training_by_target, current, evidence, metadata
    )
    audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "draft_class": config.draft_class,
        "generated_at": manifest_generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "producer_commit": producer_commit,
        "inputs": {
            "training_table": {
                "filename": config.training_table.name,
                "sha256": training_hash,
                "byte_count": config.training_table.stat().st_size,
                "row_count": len(training),
            },
            "run_manifest": {
                "filename": config.run_manifest.name,
                "sha256": manifest_hash,
                "byte_count": config.run_manifest.stat().st_size,
                "rr02_producer_commit": manifest["producer_commit"],
            },
        },
        "source_versions": {
            name: source["source_version"] for name, source in sorted(sources.items())
        },
        "targets": {target: _target_audit(evaluations[target]) for target in TARGETS},
    }
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".rookie-board-{config.draft_class}-", dir=config.output_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_paths = write_handoff(board, staging_dir)
        audit["outputs"] = {
            name: {
                "filename": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in sorted(staged_paths.items())
        }
        staged_audit = staging_dir / f"rookie-board-{config.draft_class}.audit.json"
        staged_audit.write_bytes(_audit_bytes(audit))
        staged_paths = {**staged_paths, "audit": staged_audit}

        final_paths = {
            name: config.output_dir / path.name for name, path in staged_paths.items()
        }
        config.output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = staging_dir / "previous"
        backup_dir.mkdir()
        backups: dict[str, Path] = {}
        promoted: list[str] = []
        try:
            for name, final_path in final_paths.items():
                if final_path.exists():
                    backup_path = backup_dir / final_path.name
                    os.replace(final_path, backup_path)
                    backups[name] = backup_path
            for name in ("schema", "artifact", "manifest", "audit"):
                os.replace(staged_paths[name], final_paths[name])
                promoted.append(name)
        except Exception:
            for name in promoted:
                if final_paths[name].exists():
                    final_paths[name].unlink()
            for name, backup_path in backups.items():
                if backup_path.exists():
                    os.replace(backup_path, final_paths[name])
            raise
    return final_paths


def _date_argument(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish an offline RR-05 rookie board")
    parser.add_argument("--training-table", required=True, type=Path)
    parser.add_argument("--training-table-sha256", required=True)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--draft-class", required=True, type=int)
    parser.add_argument("--draft-event-date", required=True, type=_date_argument)
    parser.add_argument("--data-cutoff", required=True, type=_date_argument)
    parser.add_argument("--outcomes-cutoff-season", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    publish_offline(OfflinePublicationConfig(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
