"""Deterministic NFL draft-to-CFBD player identity matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


COHORT_COLUMNS = {
    "draft_season",
    "overall_pick",
    "canonical_id",
    "player_name",
    "position",
}
COLLEGE_COLUMNS = {"season", "playerId", "player", "position"}
OVERRIDE_COLUMNS = {
    "draft_season",
    "overall_pick",
    "resolution",
    "cfbd_player_id",
    "reason",
    "evidence",
}
QUARANTINE_COLUMNS = [
    "draft_season",
    "overall_pick",
    "canonical_id",
    "player_name",
    "position",
    "normalized_name",
    "candidate_count",
    "candidate_cfbd_ids",
    "reason",
    "override_status",
]


class IdentityContractError(ValueError):
    """Raised when identity inputs violate a deterministic matching rule."""


class UnreviewedIdentityDecision(IdentityContractError):
    """Raised with diagnostics when a draft pick still needs explicit review."""

    def __init__(self, quarantine: pd.DataFrame):
        self.quarantine = quarantine
        keys = [
            f"{row.draft_season}/{row.overall_pick}"
            for row in quarantine.itertuples(index=False)
        ]
        super().__init__(f"identity decisions require reviewed overrides: {keys}")


class UnreviewedIdentityAmbiguity(UnreviewedIdentityDecision):
    """Raised when more than one candidate exists without a reviewed override."""


_SUPPORTED_POSITIONS = {"QB", "RB", "WR", "TE"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_player_name(value: object) -> str:
    """Return a conservative comparison key without using fuzzy similarity."""
    if pd.isna(value):
        return ""
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", ascii_name.lower())
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise IdentityContractError(f"{label} is missing required columns: {missing}")


def _id_string(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _numeric_seasons(values: pd.Series, label: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise IdentityContractError(f"{label} season must be numeric") from error
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise IdentityContractError(f"{label} season must contain whole numeric years")
    return numeric.astype("int64")


def _compatible_positions(position: object) -> set[str]:
    normalized = str(position).strip().upper()
    return {normalized} if normalized in _SUPPORTED_POSITIONS else set()


def _candidate_ids(
    pre_draft_stats: pd.DataFrame, normalized_name: str, position: object
) -> list[str]:
    compatible = _compatible_positions(position)
    candidates = pre_draft_stats[
        (pre_draft_stats["_normalized_name"] == normalized_name)
        & pre_draft_stats["position"].astype(str).str.strip().str.upper().isin(compatible)
    ]["_cfbd_player_id"]
    return sorted(candidates.dropna().unique().tolist())


def _validate_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
    if overrides.empty and not set(overrides.columns):
        return pd.DataFrame(columns=sorted(OVERRIDE_COLUMNS))
    _require_columns(overrides, OVERRIDE_COLUMNS, "identity overrides")
    result = overrides.copy()
    result["resolution"] = result["resolution"].astype(str).str.strip().str.lower()
    invalid = sorted(set(result["resolution"]).difference({"match", "quarantine"}))
    if invalid:
        raise IdentityContractError(f"unknown resolution values: {invalid}")
    duplicate_keys = result.duplicated(["draft_season", "overall_pick"], keep=False)
    if duplicate_keys.any():
        keys = result.loc[duplicate_keys, ["draft_season", "overall_pick"]].to_dict("records")
        raise IdentityContractError(f"duplicate identity override keys: {keys}")
    for review_column in ("reason", "evidence"):
        missing_review = result[review_column].isna() | result[review_column].astype(str).str.strip().eq("")
        if missing_review.any():
            raise IdentityContractError(
                f"every identity override requires {review_column}"
            )
    missing_match_id = result["resolution"].eq("match") & result["cfbd_player_id"].apply(
        lambda value: _id_string(value) is None
    )
    if missing_match_id.any():
        raise IdentityContractError("match overrides require cfbd_player_id")
    populated_quarantine_id = result["resolution"].eq("quarantine") & result[
        "cfbd_player_id"
    ].apply(lambda value: _id_string(value) is not None)
    if populated_quarantine_id.any():
        raise IdentityContractError("quarantine overrides must leave cfbd_player_id blank")
    return result


def _quarantine_row(
    cohort_row: pd.Series,
    normalized_name: str,
    candidates: Iterable[str],
    reason: str,
    override_status: str,
) -> dict[str, object]:
    candidate_ids = list(candidates)
    return {
        "draft_season": cohort_row["draft_season"],
        "overall_pick": cohort_row["overall_pick"],
        "canonical_id": cohort_row["canonical_id"],
        "player_name": cohort_row["player_name"],
        "position": cohort_row["position"],
        "normalized_name": normalized_name,
        "candidate_count": len(candidate_ids),
        "candidate_cfbd_ids": "|".join(candidate_ids),
        "reason": reason,
        "override_status": override_status,
    }


def match_college_identities(
    cohort: pd.DataFrame,
    college_stats: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match a draft cohort to CFBD IDs with exact, fail-closed rules.

    Names are normalized but never fuzzy matched. Automatic candidates require
    exact position and activity in the season immediately before the draft.
    Older pre-draft candidates and ambiguities require a reviewed override.
    """
    _require_columns(cohort, COHORT_COLUMNS, "cohort")
    _require_columns(college_stats, COLLEGE_COLUMNS, "college stats")
    reviewed = _validate_overrides(overrides if overrides is not None else pd.DataFrame())

    if cohort["canonical_id"].isna().any() or cohort["canonical_id"].duplicated().any():
        raise IdentityContractError("cohort canonical_id must be populated and unique")
    if cohort.duplicated(["draft_season", "overall_pick"]).any():
        raise IdentityContractError("cohort draft_season/overall_pick keys must be unique")
    cohort_keys = set(zip(cohort["draft_season"], cohort["overall_pick"]))
    unused_override_keys = [
        (row.draft_season, row.overall_pick)
        for row in reviewed.itertuples(index=False)
        if (row.draft_season, row.overall_pick) not in cohort_keys
    ]
    if unused_override_keys:
        raise IdentityContractError(
            "identity override keys do not match cohort rows: "
            f"{unused_override_keys}"
        )

    stats = college_stats.copy()
    stats["season"] = _numeric_seasons(stats["season"], "college stats")
    stats["_normalized_name"] = stats["player"].apply(normalize_player_name)
    stats["_cfbd_player_id"] = stats["playerId"].apply(_id_string)

    override_lookup = {
        (row.draft_season, row.overall_pick): row
        for row in reviewed.itertuples(index=False)
    }
    identity_values: list[dict[str, object]] = []
    quarantine_values: list[dict[str, object]] = []
    unreviewed_values: list[dict[str, object]] = []

    for _, cohort_row in cohort.iterrows():
        normalized_name = normalize_player_name(cohort_row["player_name"])
        pre_draft = stats[stats["season"] < cohort_row["draft_season"]]
        prior_season = pre_draft[
            pre_draft["season"] == cohort_row["draft_season"] - 1
        ]
        candidates = _candidate_ids(prior_season, normalized_name, cohort_row["position"])
        override = override_lookup.get((cohort_row["draft_season"], cohort_row["overall_pick"]))

        identity = {
            "cfbd_player_id": pd.NA,
            "identity_match_status": "quarantined",
            "identity_match_method": "none",
            "quarantine_reason": pd.NA,
        }

        if override is not None and override.resolution == "quarantine":
            reason = str(override.reason).strip()
            identity["identity_match_method"] = "none"
            identity["quarantine_reason"] = reason
            quarantine_values.append(
                _quarantine_row(cohort_row, normalized_name, candidates, reason, "quarantine")
            )
        elif override is not None:
            override_id = _id_string(override.cfbd_player_id)
            eligible_id_rows = pre_draft[pre_draft["_cfbd_player_id"] == override_id]
            compatible = eligible_id_rows["position"].astype(str).str.strip().str.upper().isin(
                _compatible_positions(cohort_row["position"])
            )
            if eligible_id_rows.empty or not compatible.any():
                raise IdentityContractError(
                    "match override for "
                    f"{cohort_row['draft_season']}/{cohort_row['overall_pick']} points to "
                    f"ineligible CFBD player {override_id}"
                )
            identity.update(
                {
                    "cfbd_player_id": override_id,
                    "identity_match_status": "override",
                    "identity_match_method": "override",
                }
            )
        elif len(candidates) == 1:
            identity.update(
                {
                    "cfbd_player_id": candidates[0],
                    "identity_match_status": "exact",
                    "identity_match_method": "normalized_name_position",
                }
            )
        elif len(candidates) > 1:
            reason = "multiple_exact_name_position_candidates"
            identity["quarantine_reason"] = reason
            unreviewed_values.append(
                _quarantine_row(cohort_row, normalized_name, candidates, reason, "unreviewed")
            )
        else:
            reason = "no_exact_name_and_position_candidate_in_prior_season"
            identity["quarantine_reason"] = reason
            unreviewed_values.append(
                _quarantine_row(cohort_row, normalized_name, candidates, reason, "unreviewed")
            )

        identity_values.append(identity)

    if unreviewed_values:
        diagnostics = pd.DataFrame(unreviewed_values, columns=QUARANTINE_COLUMNS)
        if diagnostics["candidate_count"].gt(1).any():
            raise UnreviewedIdentityAmbiguity(diagnostics)
        raise UnreviewedIdentityDecision(diagnostics)

    matched = pd.concat(
        [cohort.reset_index(drop=True), pd.DataFrame(identity_values)], axis=1
    )
    reused = matched.dropna(subset=["cfbd_player_id"]).duplicated(
        ["draft_season", "cfbd_player_id"], keep=False
    )
    if reused.any():
        collisions = matched.loc[
            reused, ["draft_season", "canonical_id", "cfbd_player_id"]
        ].to_dict("records")
        raise IdentityContractError(f"CFBD player ID reused within a draft class: {collisions}")

    quarantine = pd.DataFrame(quarantine_values, columns=QUARANTINE_COLUMNS)
    return matched, quarantine
