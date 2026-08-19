"""Build regular-season rookie and three-year fantasy targets."""

from __future__ import annotations

from collections.abc import Iterable

import nflreadpy as nfl
import pandas as pd


OUTCOME_REQUIRED_COLUMNS = ("player_id", "season", "fantasy_points_ppr")
TARGET_COLUMNS = (
    "rookie_year_ppr_points",
    "rookie_target_status",
    "three_year_ppr_points",
    "three_year_target_status",
)


def load_regular_season_stats(years: Iterable[int]) -> pd.DataFrame:
    """Load explicit nflverse player seasons, excluding postseason stats."""
    frame = nfl.load_player_stats(list(years), summary_level="reg")
    return pd.DataFrame(frame.to_dicts())


def build_targets(
    cohort: pd.DataFrame,
    player_stats: pd.DataFrame,
    outcomes_through_year: int,
) -> pd.DataFrame:
    """Append exact rookie-year and first-three-year PPR targets to a cohort.

    Missing regular-season rows become zero only for GSIS-identified players
    and mature target windows. Immature windows and players without a GSIS ID
    retain null targets.
    """
    _require_columns(cohort, ("draft_season", "gsis_id"), "cohort")
    _require_columns(player_stats, OUTCOME_REQUIRED_COLUMNS, "player stats")
    try:
        through_year = int(outcomes_through_year)
    except (TypeError, ValueError) as error:
        raise ValueError("outcomes_through_year must be an integer") from error

    result = cohort.copy()
    draft_seasons = _integer_column(result["draft_season"], "draft_season")
    stats, source_seasons = _validated_stats(player_stats, through_year)
    if not result.empty:
        required_seasons = set(range(int(draft_seasons.min()), through_year + 1))
        missing_seasons = sorted(required_seasons.difference(source_seasons))
        if missing_seasons:
            raise ValueError(
                "player stats missing required source seasons: "
                + ", ".join(str(year) for year in missing_seasons)
            )

    point_lookup = stats.set_index(["player_id", "season"])[
        "fantasy_points_ppr"
    ].to_dict()

    gsis_ids = result["gsis_id"].map(_optional_text)

    rookie_points: list[object] = []
    rookie_statuses: list[str] = []
    three_year_points: list[object] = []
    three_year_statuses: list[str] = []

    for draft_year, gsis_id in zip(draft_seasons, gsis_ids, strict=True):
        if pd.isna(gsis_id):
            rookie_points.append(pd.NA)
            rookie_statuses.append("missing_gsis")
            three_year_points.append(pd.NA)
            three_year_statuses.append("missing_gsis")
            continue

        player_id = str(gsis_id)
        if draft_year <= through_year:
            rookie_points.append(float(point_lookup.get((player_id, draft_year), 0.0)))
            rookie_statuses.append("complete")
        else:
            rookie_points.append(pd.NA)
            rookie_statuses.append("immature")

        if draft_year + 2 <= through_year:
            points = sum(
                float(point_lookup.get((player_id, season), 0.0))
                for season in range(draft_year, draft_year + 3)
            )
            three_year_points.append(points)
            three_year_statuses.append("complete")
        else:
            three_year_points.append(pd.NA)
            three_year_statuses.append("immature")

    result["rookie_year_ppr_points"] = pd.array(rookie_points, dtype="Float64")
    result["rookie_target_status"] = pd.array(rookie_statuses, dtype="string")
    result["three_year_ppr_points"] = pd.array(three_year_points, dtype="Float64")
    result["three_year_target_status"] = pd.array(
        three_year_statuses, dtype="string"
    )
    return result


# Descriptive alias for callers that do not use the shorter pipeline contract.
build_outcome_targets = build_targets


def _validated_stats(
    player_stats: pd.DataFrame, through_year: int
) -> tuple[pd.DataFrame, frozenset[int]]:
    stats = player_stats.loc[:, OUTCOME_REQUIRED_COLUMNS].copy()
    stats["player_id"] = stats["player_id"].map(_optional_text)
    stats["season"] = _integer_column(stats["season"], "season")

    points = pd.to_numeric(stats["fantasy_points_ppr"], errors="coerce")
    if points.isna().any():
        raise ValueError("player stats contain invalid fantasy_points_ppr values")
    stats["fantasy_points_ppr"] = points.astype("float64")

    future_seasons = sorted(stats.loc[stats["season"] > through_year, "season"].unique())
    if future_seasons:
        raise ValueError(
            "player stats include seasons after outcomes_through_year: "
            + ", ".join(str(year) for year in future_seasons)
        )

    source_seasons = frozenset(int(year) for year in stats["season"].unique())
    missing_id = stats["player_id"].isna()
    invalid_missing_id = missing_id & stats["fantasy_points_ppr"].ne(0.0)
    if invalid_missing_id.any():
        raise ValueError("player stats contain missing player_id with nonzero PPR")
    stats = stats.loc[~missing_id].copy()
    stats["player_id"] = stats["player_id"].astype("string")

    duplicate_mask = stats.duplicated(["player_id", "season"], keep=False)
    if duplicate_mask.any():
        duplicates = (
            stats.loc[duplicate_mask, ["player_id", "season"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"duplicate player-season stats: {duplicates}")
    return stats, source_seasons


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], source_name: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {', '.join(missing)}")


def _integer_column(values: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise ValueError(f"invalid {column_name} values")
    return numeric.astype("int64")


def _optional_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA
