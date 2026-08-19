"""Deterministic coverage and provenance helpers for the RR-02 data build."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


MANIFEST_SCHEMA_VERSION = "1.0"

COVERAGE_COLUMNS = (
    "draft_season",
    "cohort_count",
    "gsis_count",
    "exact_match_count",
    "override_match_count",
    "quarantined_count",
    "college_observed_count",
    "college_missing_count",
    "college_unresolved_count",
    "rookie_target_complete_count",
    "rookie_zero_count",
    "rookie_immature_count",
    "rookie_missing_gsis_count",
    "three_year_target_complete_count",
    "three_year_zero_count",
    "three_year_immature_count",
    "three_year_missing_gsis_count",
)

SOURCE_NOTICES = {
    "nflverse": {
        "name": "nflverse",
        "license": "CC-BY-4.0",
        "url": "https://github.com/nflverse/nflverse-data",
        "notice": "NFL draft and regular-season player statistics are derived from nflverse data.",
    },
    "college_football_data": {
        "name": "CollegeFootballData",
        "license": "Provider terms apply",
        "url": "https://collegefootballdata.com/",
        "notice": (
            "Only derived features, identity decisions, and coverage summaries are published; "
            "raw API responses and credentials are not distributed."
        ),
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a stable CSV representation without mutating the frame."""
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.10g")
    return sha256_bytes(payload.encode("utf-8"))


def dataframe_schema(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [
        {"name": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    encoded = json.dumps(columns, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"columns": columns, "sha256": sha256_bytes(encoded)}


def source_record(
    *,
    name: str,
    frame: pd.DataFrame,
    query_years: Iterable[int],
    fetched_at_utc: str,
    source_url: str,
    loader_version: str | None = None,
    source_version: str | None = None,
) -> dict[str, Any]:
    """Describe a normalized source frame without retaining raw source rows."""
    content_sha256 = dataframe_sha256(frame)
    return {
        "name": name,
        "source_url": source_url,
        "source_version": source_version or f"sha256:{content_sha256}",
        "loader_version": loader_version,
        "fetched_at_utc": fetched_at_utc,
        "query_years": sorted(int(year) for year in query_years),
        "row_count": int(len(frame)),
        "content_sha256": content_sha256,
        "schema": dataframe_schema(frame),
    }


def build_coverage_report(training_table: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable coverage row per draft class."""
    required = {
        "draft_season",
        "canonical_id",
        "gsis_id",
        "identity_match_status",
        "college_stats_status",
        "rookie_year_ppr_points",
        "rookie_target_status",
        "three_year_ppr_points",
        "three_year_target_status",
    }
    missing = sorted(required.difference(training_table.columns))
    if missing:
        raise ValueError(f"Training table is missing coverage columns: {missing}")
    validate_target_status_values(training_table)

    rows: list[dict[str, int]] = []
    for draft_season, group in training_table.groupby("draft_season", sort=True):
        identity = group["identity_match_status"]
        college = group["college_stats_status"]
        rookie_status = group["rookie_target_status"]
        three_status = group["three_year_target_status"]
        rows.append(
            {
                "draft_season": int(draft_season),
                "cohort_count": int(len(group)),
                "gsis_count": int(group["gsis_id"].notna().sum()),
                "exact_match_count": int((identity == "exact").sum()),
                "override_match_count": int((identity == "override").sum()),
                "quarantined_count": int((identity == "quarantined").sum()),
                "college_observed_count": int((college == "observed").sum()),
                "college_missing_count": int((college == "missing").sum()),
                "college_unresolved_count": int((college == "unresolved_identity").sum()),
                "rookie_target_complete_count": int((rookie_status == "complete").sum()),
                "rookie_zero_count": int(
                    ((rookie_status == "complete") & (group["rookie_year_ppr_points"] == 0)).sum()
                ),
                "rookie_immature_count": int((rookie_status == "immature").sum()),
                "rookie_missing_gsis_count": int((rookie_status == "missing_gsis").sum()),
                "three_year_target_complete_count": int((three_status == "complete").sum()),
                "three_year_zero_count": int(
                    ((three_status == "complete") & (group["three_year_ppr_points"] == 0)).sum()
                ),
                "three_year_immature_count": int((three_status == "immature").sum()),
                "three_year_missing_gsis_count": int((three_status == "missing_gsis").sum()),
            }
        )

    coverage = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    validate_coverage_partitions(coverage)
    return coverage


def validate_target_status_values(training_table: pd.DataFrame) -> None:
    """Reject target states that confuse zero, missing identity, and immaturity."""
    target_contracts = (
        ("rookie", "rookie_target_status", "rookie_year_ppr_points"),
        ("three-year", "three_year_target_status", "three_year_ppr_points"),
    )
    allowed_statuses = {"complete", "immature", "missing_gsis"}
    for label, status_column, value_column in target_contracts:
        missing = {status_column, value_column}.difference(training_table.columns)
        if missing:
            raise ValueError(f"Training table is missing {label} target columns: {sorted(missing)}")

        invalid_status = ~training_table[status_column].isin(allowed_statuses)
        if invalid_status.any():
            rows = training_table.index[invalid_status].tolist()
            raise ValueError(f"{label.title()} target has invalid status at rows: {rows}")

        complete = training_table[status_column].eq("complete")
        bad_complete = complete & ~training_table[value_column].map(
            lambda value: isinstance(value, Number)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        if bad_complete.any():
            rows = training_table.index[bad_complete].tolist()
            raise ValueError(
                f"{label.title()} complete targets require a finite numeric value at rows: {rows}"
            )

        bad_unavailable = ~complete & training_table[value_column].notna()
        if bad_unavailable.any():
            rows = training_table.index[bad_unavailable].tolist()
            raise ValueError(
                f"{label.title()} immature or missing_gsis targets must be null at rows: {rows}"
            )


def validate_coverage_partitions(coverage: pd.DataFrame) -> None:
    """Fail when identity, college, or target statuses do not cover the cohort."""
    partitions = {
        "identity": ["exact_match_count", "override_match_count", "quarantined_count"],
        "college": ["college_observed_count", "college_missing_count", "college_unresolved_count"],
        "rookie target": [
            "rookie_target_complete_count",
            "rookie_immature_count",
            "rookie_missing_gsis_count",
        ],
        "three-year target": [
            "three_year_target_complete_count",
            "three_year_immature_count",
            "three_year_missing_gsis_count",
        ],
    }
    for label, columns in partitions.items():
        if not set(columns).issubset(coverage.columns):
            raise ValueError(f"Coverage is missing the {label} partition")
        bad = coverage[coverage[columns].sum(axis=1) != coverage["cohort_count"]]
        if not bad.empty:
            years = bad["draft_season"].astype(str).tolist()
            raise ValueError(f"{label.title()} coverage does not reconcile for draft years: {years}")


def output_record(path: str | Path, *, row_count: int | None = None) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    return record


def build_run_manifest(
    *,
    arguments: Mapping[str, Any],
    generated_at_utc: str,
    producer_commit: str,
    sources: Iterable[Mapping[str, Any]],
    outputs: Iterable[Mapping[str, Any]],
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-serializable run manifest."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "producer_commit": producer_commit,
        "arguments": dict(arguments),
        "sources": list(sources),
        "outputs": list(outputs),
        "coverage": coverage.to_dict("records"),
        "source_notices": list(SOURCE_NOTICES.values()),
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
