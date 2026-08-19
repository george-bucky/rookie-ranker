"""Auditable RR-02 cohort, identity, college-feature, and target build."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import importlib.metadata
import os
from pathlib import Path
import subprocess
from typing import Sequence

import pandas as pd
import requests

from .run_manifest import (
    build_coverage_report,
    build_run_manifest,
    output_record,
    source_record,
    write_manifest,
)


TRAINING_FILENAME = "training-table.csv"
QUARANTINE_FILENAME = "identity-quarantine.csv"
COVERAGE_FILENAME = "coverage.csv"
MANIFEST_FILENAME = "run-manifest.json"


@dataclass(frozen=True)
class PipelineConfig:
    draft_years: tuple[int, ...]
    college_years: tuple[int, ...]
    outcomes_through_year: int
    http_timeout_seconds: float
    identity_overrides: Path
    output_dir: Path
    draft_input: Path | None = None
    player_stats_input: Path | None = None
    college_input: Path | None = None


def parse_years(value: str) -> tuple[int, ...]:
    """Parse comma-separated years and inclusive ranges such as 2015:2018."""
    years: list[int] = []
    try:
        for token in value.split(","):
            token = token.strip()
            if not token:
                raise ValueError
            if ":" in token:
                start_text, end_text = token.split(":", maxsplit=1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError
                years.extend(range(start, end + 1))
            else:
                years.append(int(token))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid year specification: {value!r}") from error

    if len(years) != len(set(years)):
        raise argparse.ArgumentTypeError(f"Year specification contains duplicates: {value!r}")
    if not years:
        raise argparse.ArgumentTypeError("At least one year is required")
    return tuple(sorted(years))


def validate_year_configuration(
    draft_years: Sequence[int],
    college_years: Sequence[int],
    outcomes_through_year: int,
    *,
    current_year: int | None = None,
) -> None:
    current_year = current_year or datetime.now(UTC).year
    if not draft_years or not college_years:
        raise ValueError("Draft years and college years are required")
    if len(draft_years) != len(set(draft_years)) or len(college_years) != len(set(college_years)):
        raise ValueError("Configured years must be unique")
    all_years = [*draft_years, *college_years, outcomes_through_year]
    if any(year < 1900 or year > current_year for year in all_years):
        raise ValueError(f"Years must be between 1900 and {current_year}")
    if outcomes_through_year >= current_year:
        raise ValueError(
            "outcomes_through_year must be strictly before the current year "
            "so only complete regular seasons can become targets"
        )
    if max(college_years) >= max(draft_years):
        raise ValueError("College query years must end before the newest configured draft class")
    if not all(any(college_year < draft_year for college_year in college_years) for draft_year in draft_years):
        raise ValueError("Every draft class requires at least one earlier college query year")


def validate_config(config: PipelineConfig) -> None:
    validate_year_configuration(
        config.draft_years,
        config.college_years,
        config.outcomes_through_year,
    )
    if config.http_timeout_seconds <= 0:
        raise ValueError("HTTP timeout must be greater than zero")
    inputs = [config.draft_input, config.player_stats_input, config.college_input]
    if any(path is not None for path in inputs) and not all(path is not None for path in inputs):
        raise ValueError("Local source mode requires draft, player-stats, and college input files")


def _read_local_sources(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert config.draft_input and config.player_stats_input and config.college_input
    return (
        pd.read_csv(config.draft_input),
        pd.read_csv(config.player_stats_input),
        pd.read_csv(config.college_input),
    )


def _fetch_college_year(year: int, timeout: float) -> pd.DataFrame:
    api_key = os.getenv("CFD_API_KEY")
    if not api_key:
        raise RuntimeError("CFD_API_KEY is required for live CollegeFootballData fetches")
    response = requests.get(
        "https://api.collegefootballdata.com/stats/player/season",
        params={"year": year},
        headers={"Authorization": f"Bearer {api_key}", "accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def _fetch_live_sources(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import nflreadpy as nfl
    from nflreadpy.config import update_config

    update_config(timeout=config.http_timeout_seconds)
    draft = pd.DataFrame(nfl.load_draft_picks(list(config.draft_years)).to_dicts())
    outcome_years = list(range(min(config.draft_years), config.outcomes_through_year + 1))
    if outcome_years:
        stats = pd.DataFrame(
            nfl.load_player_stats(outcome_years, summary_level="reg").to_dicts()
        )
    else:
        stats = pd.DataFrame(columns=["player_id", "season", "fantasy_points_ppr"])
    college = pd.concat(
        [_fetch_college_year(year, config.http_timeout_seconds) for year in config.college_years],
        ignore_index=True,
    )
    return draft, stats, college


def load_sources(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.draft_input is not None:
        return _read_local_sources(config)
    return _fetch_live_sources(config)


def _validated_season_column(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "season" not in frame.columns:
        raise ValueError(f"{label} source is missing required column: season")
    result = frame.copy()
    seasons = pd.to_numeric(result["season"], errors="coerce")
    if seasons.isna().any() or (seasons % 1 != 0).any():
        raise ValueError(f"{label} source contains invalid season values")
    result["season"] = seasons.astype("int64")
    return result


def _prepare_sources(
    draft: pd.DataFrame,
    stats: pd.DataFrame,
    college: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    draft = _validated_season_column(draft, "Draft")
    stats = _validated_season_column(stats, "Player stats")
    college = _validated_season_column(college, "College")

    draft = draft[draft["season"].isin(config.draft_years)].copy()
    college = college[college["season"].isin(config.college_years)].copy()
    returned_years = set(draft["season"].dropna().astype(int))
    missing_draft_years = sorted(set(config.draft_years).difference(returned_years))
    if missing_draft_years:
        raise ValueError(f"Draft source returned no rows for configured years: {missing_draft_years}")
    returned_college_years = set(college["season"].dropna().astype(int))
    missing_college_years = sorted(set(config.college_years).difference(returned_college_years))
    if missing_college_years:
        raise ValueError(
            f"College source returned no rows for configured years: {missing_college_years}"
        )
    return draft, stats, college


def _merge_additions(base: pd.DataFrame, addition: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if "canonical_id" not in addition.columns:
        raise ValueError(f"{label} is missing canonical_id")
    if addition["canonical_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate canonical IDs")
    new_columns = [column for column in addition.columns if column not in base.columns]
    return base.merge(
        addition[["canonical_id", *new_columns]],
        on="canonical_id",
        how="left",
        validate="one_to_one",
    )


def compose_training_table(
    cohort: pd.DataFrame,
    targets: pd.DataFrame,
    identities: pd.DataFrame,
    college_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine worker outputs while preserving exactly one row per draft pick."""
    if cohort["canonical_id"].duplicated().any():
        raise ValueError("Cohort contains duplicate canonical IDs")
    table = cohort.copy()
    table = _merge_additions(table, targets, label="Targets")
    table = _merge_additions(table, identities, label="Identities")
    table = _merge_additions(table, college_features, label="College features")
    if len(table) != len(cohort):
        raise ValueError("A left join changed the draft cohort row count")
    sort_columns = [column for column in ("draft_season", "overall_pick") if column in table]
    return table.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def _producer_commit() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    if status.stdout.strip():
        raise RuntimeError(
            "Refusing to publish provenance from a dirty Git tree; commit or remove "
            "all tracked and untracked changes first"
        )
    return completed.stdout.strip()


def run_build(
    config: PipelineConfig,
    *,
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    generated_at_utc: str | None = None,
    producer_commit: str | None = None,
) -> dict[str, Path]:
    """Run RR-02 from live/local sources or injected offline fixture frames."""
    validate_config(config)
    resolved_producer_commit = producer_commit or _producer_commit()
    draft, stats, college = frames if frames is not None else load_sources(config)
    draft, stats, college = _prepare_sources(draft, stats, college, config)

    from .cohort import build_draft_cohort
    from .college_features import build_college_features
    from .identity import match_college_identities
    from .outcomes import build_targets

    overrides = pd.read_csv(config.identity_overrides)

    cohort = build_draft_cohort(draft, config.draft_years)
    targets = build_targets(cohort, stats, config.outcomes_through_year)
    identities, quarantine = match_college_identities(cohort, college, overrides=overrides)
    college_features = build_college_features(college, identities)
    training_table = compose_training_table(cohort, targets, identities, college_features)
    coverage = build_coverage_report(training_table)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_table": config.output_dir / TRAINING_FILENAME,
        "quarantine": config.output_dir / QUARANTINE_FILENAME,
        "coverage": config.output_dir / COVERAGE_FILENAME,
        "manifest": config.output_dir / MANIFEST_FILENAME,
    }
    quarantine = quarantine.sort_values(
        [column for column in ("draft_season", "overall_pick") if column in quarantine],
        kind="stable",
    ).reset_index(drop=True)
    _write_csv(training_table, paths["training_table"])
    _write_csv(quarantine, paths["quarantine"])
    _write_csv(coverage, paths["coverage"])

    generated_at_utc = generated_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        nflreadpy_version = importlib.metadata.version("nflreadpy")
    except importlib.metadata.PackageNotFoundError:
        nflreadpy_version = None
    sources = [
        source_record(
            name="nflverse_draft_picks",
            frame=draft,
            query_years=config.draft_years,
            fetched_at_utc=generated_at_utc,
            source_url="https://github.com/nflverse/nflverse-data/releases/tag/draft_picks",
            loader_version=nflreadpy_version,
        ),
        source_record(
            name="nflverse_player_stats_reg",
            frame=stats,
            query_years=range(min(config.draft_years), config.outcomes_through_year + 1),
            fetched_at_utc=generated_at_utc,
            source_url="https://github.com/nflverse/nflverse-data/releases/tag/player_stats",
            loader_version=nflreadpy_version,
        ),
        source_record(
            name="college_football_data_player_season",
            frame=college,
            query_years=config.college_years,
            fetched_at_utc=generated_at_utc,
            source_url="https://api.collegefootballdata.com/stats/player/season",
        ),
        source_record(
            name="identity_overrides",
            frame=overrides,
            query_years=config.draft_years,
            fetched_at_utc=generated_at_utc,
            source_url=str(config.identity_overrides),
        ),
    ]
    manifest_arguments = asdict(config)
    manifest_arguments = {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in manifest_arguments.items()
    }
    manifest = build_run_manifest(
        arguments=manifest_arguments,
        generated_at_utc=generated_at_utc,
        producer_commit=resolved_producer_commit,
        sources=sources,
        outputs=[
            output_record(paths["training_table"], row_count=len(training_table)),
            output_record(paths["quarantine"], row_count=len(quarantine)),
            output_record(paths["coverage"], row_count=len(coverage)),
        ],
        coverage=coverage,
    )
    write_manifest(manifest, paths["manifest"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the auditable RR-02 data truth table")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build cohort, identities, targets, and provenance")
    build.add_argument("--draft-years", required=True, type=parse_years)
    build.add_argument("--college-years", required=True, type=parse_years)
    build.add_argument("--outcomes-through-year", required=True, type=int)
    build.add_argument("--http-timeout-seconds", required=True, type=float)
    build.add_argument("--identity-overrides", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--draft-input", type=Path, help="Optional offline draft CSV")
    build.add_argument("--player-stats-input", type=Path, help="Optional offline regular-season stats CSV")
    build.add_argument("--college-input", type=Path, help="Optional offline college stats CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        draft_years=args.draft_years,
        college_years=args.college_years,
        outcomes_through_year=args.outcomes_through_year,
        http_timeout_seconds=args.http_timeout_seconds,
        identity_overrides=args.identity_overrides,
        output_dir=args.output_dir,
        draft_input=args.draft_input,
        player_stats_input=args.player_stats_input,
        college_input=args.college_input,
    )
    run_build(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
