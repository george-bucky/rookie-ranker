"""Leakage-safe college feature aggregation for resolved rookie identities."""

from __future__ import annotations

import pandas as pd


IDENTITY_COLUMNS = {"canonical_id", "draft_season", "cfbd_player_id"}
COLLEGE_COLUMNS = {
    "season",
    "playerId",
    "team",
    "conference",
    "category",
    "statType",
    "stat",
}
SOURCE_GRAIN = ("season", "playerId", "team", "category", "statType")
SOURCE_STATS = {
    "passing_attempts": "passing_ATT",
    "passing_yards": "passing_YDS",
    "passing_touchdowns": "passing_TD",
    "rushing_carries": "rushing_CAR",
    "rushing_yards": "rushing_YDS",
    "rushing_touchdowns": "rushing_TD",
    "receiving_receptions": "receiving_REC",
    "receiving_yards": "receiving_YDS",
    "receiving_touchdowns": "receiving_TD",
}
RATE_INPUTS = {
    "passing_yards_per_attempt": ("passing_yards", "passing_attempts"),
    "rushing_yards_per_carry": ("rushing_yards", "rushing_carries"),
    "receiving_yards_per_reception": ("receiving_yards", "receiving_receptions"),
}
DERIVED_INPUTS = {
    "yards_from_scrimmage": ("rushing_yards", "receiving_yards"),
    "total_touchdowns": (
        "passing_touchdowns",
        "rushing_touchdowns",
        "receiving_touchdowns",
    ),
}
FEATURE_NAMES = tuple(SOURCE_STATS) + tuple(RATE_INPUTS) + tuple(DERIVED_INPUTS)


class CollegeFeatureContractError(ValueError):
    """Raised when college feature inputs violate their declared grain."""


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise CollegeFeatureContractError(f"{label} is missing required columns: {missing}")


def _id_string(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _numeric_seasons(values: pd.Series) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise CollegeFeatureContractError("college stats season must be numeric") from error
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise CollegeFeatureContractError(
            "college stats season must contain whole numeric years"
        )
    return numeric.astype("int64")


def _joined(values: pd.Series) -> str | pd.NA:
    observed = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
    return "|".join(observed) if observed else pd.NA


def _aggregate_scope(rows: pd.DataFrame, prefix: str) -> dict[str, object]:
    result: dict[str, object] = {}
    grouped = rows.groupby("stat_name", dropna=False)["stat"].sum(min_count=1)

    for feature_name, source_name in SOURCE_STATS.items():
        observed = source_name in grouped.index and pd.notna(grouped.loc[source_name])
        result[f"{prefix}_{feature_name}"] = grouped.loc[source_name] if observed else pd.NA
        result[f"{prefix}_{feature_name}_observed"] = bool(observed)

    for feature_name, (numerator_name, denominator_name) in RATE_INPUTS.items():
        numerator = result[f"{prefix}_{numerator_name}"]
        denominator = result[f"{prefix}_{denominator_name}"]
        observed = (
            pd.notna(numerator)
            and pd.notna(denominator)
            and float(denominator) > 0
        )
        result[f"{prefix}_{feature_name}"] = (
            float(numerator) / float(denominator) if observed else pd.NA
        )
        result[f"{prefix}_{feature_name}_observed"] = bool(observed)

    for feature_name, input_names in DERIVED_INPUTS.items():
        values = [result[f"{prefix}_{input_name}"] for input_name in input_names]
        observed = all(pd.notna(value) for value in values)
        result[f"{prefix}_{feature_name}"] = (
            sum(float(value) for value in values) if observed else pd.NA
        )
        result[f"{prefix}_{feature_name}_observed"] = bool(observed)
    return result


def _empty_stats(prefix: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for feature_name in FEATURE_NAMES:
        result[f"{prefix}_{feature_name}"] = pd.NA
        result[f"{prefix}_{feature_name}_observed"] = False
    return result


def build_college_features(
    college_stats: pd.DataFrame, identities: pd.DataFrame
) -> pd.DataFrame:
    """Build final-active-season and career features at canonical rookie grain.

    Only pre-draft rows are included. Transfers are aggregated across all teams
    in a season. Every output feature has an observed flag. An unobserved
    statistic stays null; it is never silently converted to zero.
    """
    _require_columns(college_stats, COLLEGE_COLUMNS, "college stats")
    _require_columns(identities, IDENTITY_COLUMNS, "identities")
    if identities["canonical_id"].isna().any() or identities["canonical_id"].duplicated().any():
        raise CollegeFeatureContractError("identities canonical_id must be populated and unique")

    stats = college_stats.copy()
    stats["season"] = _numeric_seasons(stats["season"])
    stats["_cfbd_player_id"] = stats["playerId"].apply(_id_string)
    stats["stat_name"] = stats["category"].astype(str) + "_" + stats["statType"].astype(str)
    stats = stats[stats["stat_name"].isin(SOURCE_STATS.values())].copy()
    duplicate_rows = stats.duplicated(list(SOURCE_GRAIN), keep=False)
    if duplicate_rows.any():
        duplicates = stats.loc[duplicate_rows, list(SOURCE_GRAIN)].to_dict("records")
        raise CollegeFeatureContractError(
            f"duplicate college source-grain rows: {duplicates}"
        )
    try:
        stats["stat"] = pd.to_numeric(stats["stat"], errors="raise")
    except (TypeError, ValueError) as error:
        raise CollegeFeatureContractError(
            "supported college stat values must be numeric"
        ) from error
    if stats["stat"].isna().any():
        raise CollegeFeatureContractError(
            "supported college stat values must be populated numeric values"
        )

    feature_rows: list[dict[str, object]] = []
    for _, identity in identities.iterrows():
        cfbd_id = _id_string(identity["cfbd_player_id"])
        base: dict[str, object] = {
            "canonical_id": identity["canonical_id"],
            "college_stats_status": "unresolved_identity",
            "college_passing_observed": False,
            "college_rushing_observed": False,
            "college_receiving_observed": False,
            "college_first_observed_season": pd.NA,
            "college_final_active_season": pd.NA,
            "college_seasons_observed": pd.NA,
            "college_team": pd.NA,
            "college_conference": pd.NA,
        }

        if cfbd_id is None:
            base.update(_empty_stats("final"))
            base.update(_empty_stats("career"))
            feature_rows.append(base)
            continue

        player_rows = stats[
            (stats["_cfbd_player_id"] == cfbd_id)
            & (stats["season"] < identity["draft_season"])
        ].copy()
        observed_rows = player_rows[player_rows["stat"].notna()]
        if observed_rows.empty:
            base["college_stats_status"] = "missing"
            base["college_seasons_observed"] = 0
            base.update(_empty_stats("final"))
            base.update(_empty_stats("career"))
            feature_rows.append(base)
            continue

        final_season = observed_rows["season"].max()
        final_rows = observed_rows[observed_rows["season"] == final_season]
        seasons = sorted(observed_rows["season"].unique().tolist())
        base.update(
            {
                "college_stats_status": "observed",
                "college_passing_observed": bool(
                    observed_rows["category"].eq("passing").any()
                ),
                "college_rushing_observed": bool(
                    observed_rows["category"].eq("rushing").any()
                ),
                "college_receiving_observed": bool(
                    observed_rows["category"].eq("receiving").any()
                ),
                "college_first_observed_season": seasons[0],
                "college_final_active_season": final_season,
                "college_seasons_observed": len(seasons),
                "college_team": _joined(final_rows["team"]),
                "college_conference": _joined(final_rows["conference"]),
            }
        )
        base.update(_aggregate_scope(final_rows, "final"))
        base.update(_aggregate_scope(observed_rows, "career"))
        feature_rows.append(base)

    result = pd.DataFrame(feature_rows)
    numeric_columns = [
        f"{prefix}_{stat_name}"
        for prefix in ("final", "career")
        for stat_name in FEATURE_NAMES
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")
    return result
