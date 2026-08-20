"""Strict, deterministic public artifact contract for rookie boards."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Annotated, Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
SUPPORTED_ARTIFACT_MAJOR = 1
NUMERIC_PRECISION = 4
SCHEMA_FILENAME = "rookie-board.schema.json"
SCHEMA_SHA256 = "d51a8f4e267aaff7fb7d11f1cd555e882c6d1813db2f89fde55ffe0255022641"
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Champion = Literal["draft_capital", "random_forest"]


class ArtifactContractError(ValueError):
    """Raised when a handoff is incompatible, incomplete, or tampered with."""


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _nonempty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _semver(value: str, field_name: str) -> str:
    value = _nonempty(value, field_name)
    if not _SEMVER.fullmatch(value):
        raise ValueError(f"{field_name} must be semantic version X.Y.Z")
    return value


class PredictionQuantiles(StrictContractModel):
    p10: FiniteFloat
    p50: FiniteFloat
    p90: FiniteFloat

    @model_validator(mode="after")
    def ordered(self) -> "PredictionQuantiles":
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("prediction quantiles must satisfy p10 <= p50 <= p90")
        return self


class ChampionByTarget(StrictContractModel):
    three_year_ppr_points: Champion
    rookie_year_ppr_points: Champion


class VerifiedSourceIds(StrictContractModel):
    gsis_id: str | None = None
    pfr_player_id: str | None = None
    cfb_player_id: str | None = None
    cfbd_player_id: str | None = None
    yahoo_id: str | None = None
    sleeper_id: str | None = None

    @field_validator("gsis_id", "pfr_player_id", "cfb_player_id", "cfbd_player_id", "yahoo_id", "sleeper_id")
    @classmethod
    def source_id_is_nonempty(cls, value: str | None) -> str | None:
        return None if value is None else _nonempty(value, "source ID")


class TargetDefinitions(StrictContractModel):
    three_year_ppr_points: str
    rookie_year_ppr_points: str

    @field_validator("three_year_ppr_points", "rookie_year_ppr_points")
    @classmethod
    def definition_is_nonempty(cls, value: str) -> str:
        return _nonempty(value, "target definition")


class TrainingCohorts(StrictContractModel):
    three_year_ppr_points: tuple[int, ...]
    rookie_year_ppr_points: tuple[int, ...]

    @field_validator("three_year_ppr_points", "rookie_year_ppr_points")
    @classmethod
    def cohorts_are_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("training cohorts must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("training cohorts must be unique")
        return value


class TargetMaturity(StrictContractModel):
    three_year_ppr_points: int = Field(ge=1936, le=2200)
    rookie_year_ppr_points: int = Field(ge=1936, le=2200)


class Capabilities(StrictContractModel):
    three_year_predictions: bool
    rookie_year_predictions: bool
    prediction_intervals: bool
    interval_level: FiniteFloat

    @field_validator("interval_level")
    @classmethod
    def interval_level_is_probability(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("interval_level must be between zero and one")
        return value


class TargetEvaluationSummary(StrictContractModel):
    selected_champion: Champion
    eligible_held_out_classes: int = Field(ge=0)
    ndcg_at_24: FiniteFloat | None
    mae: FiniteFloat | None
    interval_coverage: FiniteFloat | None
    mean_interval_width: FiniteFloat | None

    @field_validator("interval_coverage")
    @classmethod
    def coverage_is_probability(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("interval_coverage must be between zero and one")
        return value

    @field_validator("ndcg_at_24")
    @classmethod
    def ndcg_is_probability(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("ndcg_at_24 must be between zero and one")
        return value

    @field_validator("mae", "mean_interval_width")
    @classmethod
    def magnitude_is_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("magnitude metrics must not be negative")
        return value


class EvaluationSummary(StrictContractModel):
    three_year_ppr_points: TargetEvaluationSummary
    rookie_year_ppr_points: TargetEvaluationSummary


class SourceNotice(StrictContractModel):
    name: str
    url: str
    license: str
    notice: str
    version: str | None = None

    @field_validator("name", "url", "license", "notice")
    @classmethod
    def notice_field_is_nonempty(cls, value: str) -> str:
        return _nonempty(value, "source notice field")


class ArtifactMetadata(StrictContractModel):
    schema_version: str
    artifact_version: str
    model_version: str
    producer_commit: str
    draft_class: int = Field(ge=1936, le=2200)
    mode: Literal["post_draft"]
    draft_event_date: date
    generated_at: datetime
    data_cutoff: date
    outcomes_cutoff_season: int = Field(ge=1936, le=2200)
    scoring_basis: Literal["ppr"]
    target_definitions: TargetDefinitions
    target_maturity: TargetMaturity
    training_cohorts: TrainingCohorts
    champion_by_target: ChampionByTarget
    capabilities: Capabilities
    evaluation_summary: EvaluationSummary
    source_notices: tuple[SourceNotice, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_semver(cls, value: str) -> str:
        return _semver(value, "schema_version")

    @field_validator("artifact_version")
    @classmethod
    def artifact_version_is_semver(cls, value: str) -> str:
        return _semver(value, "artifact_version")

    @field_validator("model_version")
    @classmethod
    def model_version_is_nonempty(cls, value: str) -> str:
        return _nonempty(value, "model_version")

    @field_validator("producer_commit")
    @classmethod
    def commit_is_valid(cls, value: str) -> str:
        value = value.strip().lower()
        if not _COMMIT.fullmatch(value):
            raise ValueError("producer_commit must be a 7-40 character lowercase Git SHA")
        return value

    @field_validator("generated_at")
    @classmethod
    def generated_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @field_validator("source_notices")
    @classmethod
    def source_notices_are_complete(cls, value: tuple[SourceNotice, ...]) -> tuple[SourceNotice, ...]:
        if not value:
            raise ValueError("source_notices must not be empty")
        names = [notice.name for notice in value]
        if len(names) != len(set(names)):
            raise ValueError("source notice names must be unique")
        return value

    @model_validator(mode="after")
    def metadata_is_consistent(self) -> "ArtifactMetadata":
        if self.data_cutoff > self.generated_at.date():
            raise ValueError("data_cutoff cannot be after generated_at")
        if self.generated_at.year < self.draft_class:
            raise ValueError("generated_at cannot predate the artifact draft class")
        if self.draft_event_date.year != self.draft_class:
            raise ValueError("draft_event_date must fall within the artifact draft class")
        if self.data_cutoff < self.draft_event_date or self.generated_at.date() < self.draft_event_date:
            raise ValueError("post_draft artifact dates cannot predate draft_event_date")
        conservative_outcome_ready_date = date(self.outcomes_cutoff_season + 1, 2, 15)
        if self.data_cutoff < conservative_outcome_ready_date:
            raise ValueError("outcomes cutoff season is not complete by data_cutoff")
        expected_maturity = {
            "rookie_year_ppr_points": self.outcomes_cutoff_season,
            "three_year_ppr_points": self.outcomes_cutoff_season - 2,
        }
        if (
            self.target_maturity.rookie_year_ppr_points
            != expected_maturity["rookie_year_ppr_points"]
            or self.target_maturity.three_year_ppr_points
            != expected_maturity["three_year_ppr_points"]
        ):
            raise ValueError("target maturity must agree with outcomes cutoff season")
        cohorts_by_target = {
            "three_year_ppr_points": self.training_cohorts.three_year_ppr_points,
            "rookie_year_ppr_points": self.training_cohorts.rookie_year_ppr_points,
        }
        for target, cohorts in cohorts_by_target.items():
            if any(year >= self.draft_class for year in cohorts):
                raise ValueError("training cohorts must predate the artifact draft class")
            if any(year > expected_maturity[target] for year in cohorts):
                raise ValueError(f"{target} training cohorts exceed target maturity")
        if (
            self.evaluation_summary.three_year_ppr_points.selected_champion
            != self.champion_by_target.three_year_ppr_points
            or self.evaluation_summary.rookie_year_ppr_points.selected_champion
            != self.champion_by_target.rookie_year_ppr_points
        ):
            raise ValueError("evaluation champions must match champion_by_target")
        return self


class RookiePlayer(StrictContractModel):
    canonical_id: str
    source_ids: VerifiedSourceIds
    name: str
    position: Literal["QB", "RB", "WR", "TE"]
    nfl_team: str
    round: int = Field(ge=1, le=7)
    overall_pick: int = Field(ge=1, le=400)
    base_rank: int = Field(ge=1)
    position_rank: int = Field(ge=1)
    tier: int = Field(ge=1)
    three_year_ppr: PredictionQuantiles
    rookie_year_ppr: PredictionQuantiles
    confidence: Literal["high", "medium", "low", "unavailable"]
    data_quality_warnings: tuple[str, ...]
    identity_match_method: Literal["normalized_name_position", "override", "none"]
    identity_match_status: Literal["exact", "override", "quarantined"]
    champion_by_target: ChampionByTarget

    @field_validator("canonical_id", "name", "nfl_team")
    @classmethod
    def player_text_is_nonempty(cls, value: str) -> str:
        return _nonempty(value, "player field")

    @field_validator("data_quality_warnings")
    @classmethod
    def warnings_are_unique_and_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(_nonempty(item, "data quality warning") for item in value)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("data_quality_warnings must be unique")
        return cleaned

    @model_validator(mode="after")
    def identity_evidence_is_consistent(self) -> "RookiePlayer":
        expected_methods = {
            "exact": "normalized_name_position",
            "override": "override",
            "quarantined": "none",
        }
        if self.identity_match_method != expected_methods[self.identity_match_status]:
            raise ValueError("identity match method and status are inconsistent")
        return self


class RookieBoard(StrictContractModel):
    metadata: ArtifactMetadata
    players: tuple[RookiePlayer, ...]

    @model_validator(mode="after")
    def board_is_consistent(self) -> "RookieBoard":
        if not self.players:
            raise ValueError("players must not be empty")
        canonical_ids = [player.canonical_id for player in self.players]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("canonical IDs must be unique")
        picks = [player.overall_pick for player in self.players]
        if len(picks) != len(set(picks)):
            raise ValueError("overall picks must be unique")
        ranks = sorted(player.base_rank for player in self.players)
        if ranks != list(range(1, len(self.players) + 1)):
            raise ValueError("base ranks must be unique and contiguous from one")
        for position in ("QB", "RB", "WR", "TE"):
            position_ranks = sorted(
                player.position_rank for player in self.players if player.position == position
            )
            if position_ranks and position_ranks != list(range(1, len(position_ranks) + 1)):
                raise ValueError(f"{position} position ranks must be unique and contiguous from one")
        for player in self.players:
            if player.champion_by_target != self.metadata.champion_by_target:
                raise ValueError("player champion_by_target must match artifact metadata")
            if player.canonical_id.startswith("gsis:"):
                if player.source_ids.gsis_id != player.canonical_id.removeprefix("gsis:"):
                    raise ValueError("GSIS canonical ID must match verified gsis_id")
            elif player.canonical_id.startswith("pfr:"):
                if player.source_ids.pfr_player_id != player.canonical_id.removeprefix("pfr:"):
                    raise ValueError("PFR canonical ID must match verified pfr_player_id")
            elif player.canonical_id.startswith("draft:"):
                expected = f"draft:{self.metadata.draft_class}:{player.overall_pick}"
                if player.canonical_id != expected:
                    raise ValueError("draft fallback canonical ID must match class and overall pick")
            else:
                raise ValueError("canonical ID must use gsis, pfr, or draft namespace")
        for field_name in VerifiedSourceIds.model_fields:
            values = [
                getattr(player.source_ids, field_name)
                for player in self.players
                if getattr(player.source_ids, field_name) is not None
            ]
            if len(values) != len(set(values)):
                raise ValueError(f"verified {field_name} values must be unique")
        return self


class ArtifactManifest(StrictContractModel):
    schema_version: str
    artifact_version: str
    draft_class: int
    artifact_filename: str
    artifact_sha256: str
    producer_commit: str
    generated_at: datetime

    @field_validator("schema_version", "artifact_version")
    @classmethod
    def manifest_version_is_semver(cls, value: str, info: Any) -> str:
        return _semver(value, info.field_name)

    @field_validator("artifact_filename")
    @classmethod
    def filename_is_safe(cls, value: str) -> str:
        value = _nonempty(value, "artifact_filename")
        if Path(value).name != value:
            raise ValueError("artifact_filename must not contain a path")
        return value

    @field_validator("artifact_sha256")
    @classmethod
    def checksum_is_sha256(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("producer_commit")
    @classmethod
    def manifest_commit_is_valid(cls, value: str) -> str:
        value = value.strip().lower()
        if not _COMMIT.fullmatch(value):
            raise ValueError("producer_commit must be a 7-40 character lowercase Git SHA")
        return value

    @field_validator("generated_at")
    @classmethod
    def manifest_generated_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value


def _reject_json_constant(value: str) -> None:
    raise ArtifactContractError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactContractError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes) -> Any:
    return json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _read_json_bytes(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
        _parse_json(payload)
        return payload
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError(f"invalid JSON file {path.name}: {error}") from error


def _validate_supported_versions(schema_version: str, artifact_version: str) -> None:
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ArtifactContractError(f"unsupported schema version: {schema_version}")
    match = _SEMVER.fullmatch(artifact_version)
    if match is None or int(match.group(1)) != SUPPORTED_ARTIFACT_MAJOR:
        raise ArtifactContractError(f"unsupported artifact version: {artifact_version}")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactContractError("non-finite numbers cannot be serialized")
        rounded = round(value, NUMERIC_PRECISION)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _board_payload(board: RookieBoard) -> dict[str, Any]:
    payload = board.model_dump(mode="json", exclude_none=True)
    # Evaluation fields are required-but-nullable in the public schema. Keep
    # unavailable held-out evidence explicit without adding null optional IDs.
    for target in ("three_year_ppr_points", "rookie_year_ppr_points"):
        summary = getattr(board.metadata.evaluation_summary, target)
        for field_name in ("ndcg_at_24", "mae", "interval_coverage", "mean_interval_width"):
            if getattr(summary, field_name) is None:
                payload["metadata"]["evaluation_summary"][target][field_name] = None
    payload["players"] = sorted(payload["players"], key=lambda row: (row["base_rank"], row["canonical_id"]))
    for player in payload["players"]:
        player["data_quality_warnings"] = sorted(player["data_quality_warnings"])
    metadata = payload["metadata"]
    metadata["source_notices"] = sorted(metadata["source_notices"], key=lambda row: row["name"])
    for target in ("three_year_ppr_points", "rookie_year_ppr_points"):
        metadata["training_cohorts"][target] = sorted(metadata["training_cohorts"][target])
    return _canonicalize(payload)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _default_schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / SCHEMA_FILENAME


def _validate_schema_file(schema_path: Path, expected_version: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read_json_bytes(schema_path)
    schema = _parse_json(payload)
    if not isinstance(schema, dict):
        raise ArtifactContractError("JSON Schema root must be an object")
    if _sha256(payload) != SCHEMA_SHA256:
        raise ArtifactContractError("JSON Schema bytes do not match the supported public schema")
    if schema.get("x-schema-version") != expected_version:
        raise ArtifactContractError("JSON Schema version does not match the artifact")
    if schema.get("additionalProperties") is not False:
        raise ArtifactContractError("JSON Schema root must reject additional properties")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ArtifactContractError(f"invalid Draft 2020-12 JSON Schema: {error.message}") from error
    return payload, schema


def _validate_against_schema(instance: Any, schema: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ArtifactContractError(f"JSON Schema validation failed at {location}: {error.message}")


def write_handoff(
    board: RookieBoard,
    output_dir: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write schema, artifact, and external checksum manifest deterministically."""
    try:
        board = RookieBoard.model_validate(board.model_dump(mode="python"))
    except ValidationError as error:
        raise ArtifactContractError(f"invalid rookie board: {error}") from error
    _validate_supported_versions(board.metadata.schema_version, board.metadata.artifact_version)
    schema_source = Path(schema_path) if schema_path is not None else _default_schema_path()
    schema_bytes, schema = _validate_schema_file(schema_source, board.metadata.schema_version)
    output_dir = Path(output_dir)

    artifact_filename = f"rookie-board-{board.metadata.draft_class}.json"
    artifact_path = output_dir / artifact_filename
    manifest_path = output_dir / f"rookie-board-{board.metadata.draft_class}.manifest.json"
    schema_output = output_dir / SCHEMA_FILENAME
    artifact_payload = _board_payload(board)
    _validate_against_schema(artifact_payload, schema)
    artifact_bytes = _json_bytes(artifact_payload)
    manifest = ArtifactManifest(
        schema_version=board.metadata.schema_version,
        artifact_version=board.metadata.artifact_version,
        draft_class=board.metadata.draft_class,
        artifact_filename=artifact_filename,
        artifact_sha256=_sha256(artifact_bytes),
        producer_commit=board.metadata.producer_commit,
        generated_at=board.metadata.generated_at,
    )
    manifest_bytes = _json_bytes(manifest.model_dump(mode="json"))

    output_dir.mkdir(parents=True, exist_ok=True)
    schema_output.write_bytes(schema_bytes)
    artifact_path.write_bytes(artifact_bytes)
    manifest_path.write_bytes(manifest_bytes)
    return {"schema": schema_output, "artifact": artifact_path, "manifest": manifest_path}


def load_handoff(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    schema_path: str | Path | None = None,
) -> RookieBoard:
    """Load a board only after schema, checksum, provenance, and class checks pass."""
    artifact_path = Path(artifact_path)
    manifest_path = Path(manifest_path)
    schema_path = Path(schema_path) if schema_path is not None else artifact_path.with_name(SCHEMA_FILENAME)
    artifact_bytes = _read_json_bytes(artifact_path)
    manifest_bytes = _read_json_bytes(manifest_path)
    artifact_payload = _parse_json(artifact_bytes)
    try:
        manifest = ArtifactManifest.model_validate_json(manifest_bytes)
    except ValidationError as error:
        raise ArtifactContractError(f"invalid artifact manifest: {error}") from error
    _validate_supported_versions(manifest.schema_version, manifest.artifact_version)
    _, schema = _validate_schema_file(schema_path, manifest.schema_version)

    if manifest.artifact_filename != artifact_path.name:
        raise ArtifactContractError("manifest artifact filename does not match the loaded file")
    expected_filename = f"rookie-board-{manifest.draft_class}.json"
    expected_manifest_filename = f"rookie-board-{manifest.draft_class}.manifest.json"
    if artifact_path.name != expected_filename:
        raise ArtifactContractError("artifact filename does not match manifest draft class")
    if manifest_path.name != expected_manifest_filename:
        raise ArtifactContractError("manifest filename does not match manifest draft class")
    if _sha256(artifact_bytes) != manifest.artifact_sha256:
        raise ArtifactContractError("artifact checksum does not match manifest")
    _validate_against_schema(artifact_payload, schema)
    try:
        board = RookieBoard.model_validate_json(artifact_bytes)
    except ValidationError as error:
        raise ArtifactContractError(f"invalid rookie board: {error}") from error
    _validate_supported_versions(board.metadata.schema_version, board.metadata.artifact_version)

    comparisons = {
        "schema version": board.metadata.schema_version == manifest.schema_version,
        "artifact version": board.metadata.artifact_version == manifest.artifact_version,
        "draft class": board.metadata.draft_class == manifest.draft_class,
        "producer commit": board.metadata.producer_commit == manifest.producer_commit,
        "generation time": board.metadata.generated_at == manifest.generated_at,
    }
    mismatches = [name for name, matches in comparisons.items() if not matches]
    if mismatches:
        raise ArtifactContractError("manifest provenance mismatch: " + ", ".join(mismatches))
    return board
