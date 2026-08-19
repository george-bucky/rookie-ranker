"""Build the drafted rookie cohort used by the RR-02 data pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import nflreadpy as nfl
import pandas as pd


SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
DRAFT_REQUIRED_COLUMNS = (
    "season",
    "round",
    "pick",
    "team",
    "position",
    "pfr_player_name",
    "gsis_id",
    "pfr_player_id",
    "cfb_player_id",
)
COHORT_COLUMNS = (
    "draft_season",
    "round",
    "overall_pick",
    "nfl_team",
    "player_name",
    "position",
    "canonical_id",
    "canonical_id_source",
    "gsis_id",
    "pfr_player_id",
    "cfb_player_id",
)


def load_draft_picks(years: Iterable[int]) -> pd.DataFrame:
    """Load nflverse draft picks for explicit years."""
    frame = nfl.load_draft_picks(list(years))
    return pd.DataFrame(frame.to_dicts())


def build_draft_cohort(
    draft_picks: pd.DataFrame, draft_years: Iterable[int]
) -> pd.DataFrame:
    """Return one canonical row for every configured drafted QB/RB/WR/TE.

    Canonical IDs prefer GSIS, then PFR, and finally the stable draft
    season/overall-pick pair. Invalid source identity is rejected instead of
    being silently deduplicated.
    """
    _require_columns(draft_picks, DRAFT_REQUIRED_COLUMNS, "draft picks")
    configured_years = _configured_years(draft_years)

    source = draft_picks.copy()
    source["season"] = _integer_column(source["season"], "season")
    source["round"] = _integer_column(source["round"], "round")
    source["pick"] = _integer_column(source["pick"], "pick")
    source["position"] = source["position"].astype("string").str.strip().str.upper()

    configured = source[source["season"].isin(configured_years)].copy()
    _fail_duplicates(configured, ["season", "pick"], "draft picks")

    cohort = configured[configured["position"].isin(SKILL_POSITIONS)].copy()
    cohort["gsis_id"] = cohort["gsis_id"].map(_optional_text).astype("string")
    cohort["pfr_player_id"] = (
        cohort["pfr_player_id"].map(_optional_text).astype("string")
    )
    cohort["cfb_player_id"] = (
        cohort["cfb_player_id"].map(_optional_text).astype("string")
    )

    canonical = [_canonical_id(row) for _, row in cohort.iterrows()]
    cohort["canonical_id"] = [value for value, _ in canonical]
    cohort["canonical_id_source"] = [source for _, source in canonical]
    _fail_duplicates(cohort, ["canonical_id"], "canonical IDs")

    cohort = cohort.rename(
        columns={
            "season": "draft_season",
            "pick": "overall_pick",
            "team": "nfl_team",
            "pfr_player_name": "player_name",
        }
    )
    return (
        cohort.loc[:, COHORT_COLUMNS]
        .sort_values(["draft_season", "overall_pick"], kind="stable")
        .reset_index(drop=True)
    )


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], source_name: str
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {', '.join(missing)}")


def _configured_years(years: Iterable[int]) -> frozenset[int]:
    try:
        configured = frozenset(int(year) for year in years)
    except (TypeError, ValueError) as error:
        raise ValueError("draft_years must contain integers") from error
    if not configured:
        raise ValueError("draft_years must not be empty")
    return configured


def _integer_column(values: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise ValueError(f"draft picks contain invalid {column_name} values")
    return numeric.astype("int64")


def _optional_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _canonical_id(row: pd.Series) -> tuple[str, str]:
    if pd.notna(row["gsis_id"]):
        return f"gsis:{row['gsis_id']}", "gsis"
    if pd.notna(row["pfr_player_id"]):
        return f"pfr:{row['pfr_player_id']}", "pfr"
    return f"draft:{row['season']}:{row['pick']}", "draft"


def _fail_duplicates(
    frame: pd.DataFrame, columns: list[str], description: str
) -> None:
    duplicate_mask = frame.duplicated(columns, keep=False)
    if not duplicate_mask.any():
        return
    duplicate_values = frame.loc[duplicate_mask, columns].drop_duplicates()
    rendered = duplicate_values.to_dict("records")
    raise ValueError(f"duplicate {description}: {rendered}")
