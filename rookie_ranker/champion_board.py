"""Leakage-safe RR-05 challenger selection and rookie-board publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .artifact_contract import (
    SCHEMA_VERSION,
    ArtifactMetadata,
    Capabilities,
    ChampionByTarget,
    EvaluationSummary,
    PredictionQuantiles,
    RookieBoard,
    RookiePlayer,
    SourceNotice,
    TargetDefinitions,
    TargetEvaluationSummary,
    TargetMaturity,
    TrainingCohorts,
    VerifiedSourceIds,
    write_handoff,
)
from .college_features import FEATURE_NAMES
from .evaluation import (
    RANKING_METRICS,
    SUPPORTED_POSITIONS,
    TARGET_AVAILABILITY_GAPS,
    draft_capital_predictions,
    expanding_year_folds,
    magnitude_metrics,
    ranking_metrics,
    strict_fold_wins,
    validate_evaluation_table,
)


RANDOM_SEED = 20260819
RANDOM_FOREST_PARAMETERS: Mapping[str, Any] = MappingProxyType({
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
})
INTERVAL_LEVEL = 0.8
POSITION_RESIDUAL_MINIMUM = 30
GLOBAL_RESIDUAL_MINIMUM = 60
CHAMPION_WIN_RATE = 0.60
CHAMPION_MAE_RATIO = 1.05
MODEL_VERSION = "rr05-rf-v1"
TARGETS = ("three_year_ppr_points", "rookie_year_ppr_points")
REQUIRED_INTEGRITY_GATES = frozenset(
    {"data_integrity", "temporal_leakage", "identity", "interval_validation"}
)

VALUE_FEATURES = tuple(
    f"{scope}_{feature}"
    for scope in ("final", "career")
    for feature in FEATURE_NAMES
)
OBSERVED_FEATURES = tuple(f"{feature}_observed" for feature in VALUE_FEATURES)
NUMERIC_FEATURES = (
    "overall_pick",
    "college_seasons_observed",
    "college_seasons_observed_missing",
    "college_passing_observed",
    "college_rushing_observed",
    "college_receiving_observed",
    *VALUE_FEATURES,
    *OBSERVED_FEATURES,
)
CURRENT_REQUIRED_COLUMNS = {
    "canonical_id",
    "draft_season",
    "round",
    "overall_pick",
    "nfl_team",
    "player_name",
    "position",
    "identity_match_status",
    "identity_match_method",
}
SOURCE_ID_COLUMNS = (
    "gsis_id",
    "pfr_player_id",
    "cfb_player_id",
    "cfbd_player_id",
    "yahoo_id",
    "sleeper_id",
)


class ChampionBoardError(ValueError):
    """Raised when RR-05 evidence or publication inputs fail closed."""


@dataclass(frozen=True)
class ChampionDecision:
    target: str
    champion: str
    gates: Mapping[str, bool]
    integrity_gates: Mapping[str, bool]
    challenger_ndcg_24: float | None
    baseline_ndcg_24: float | None
    challenger_mae: float | None
    baseline_mae: float | None
    eligible_fold_count: int
    strict_win_count: int
    strict_win_rate: float | None


@dataclass(frozen=True)
class TargetEvaluation:
    target: str
    predictions: pd.DataFrame
    per_year: pd.DataFrame
    macro_class_ranking: pd.DataFrame
    pooled_row_magnitude: pd.DataFrame
    position_slices: pd.DataFrame
    interval_summary: pd.DataFrame
    decision: ChampionDecision


@dataclass(frozen=True)
class PublicationMetadata:
    producer_commit: str
    draft_event_date: date
    generated_at: datetime
    data_cutoff: date
    outcomes_cutoff_season: int
    source_notices: tuple[SourceNotice, ...]
    artifact_version: str = "1.0.0"
    model_version: str = MODEL_VERSION


@dataclass(frozen=True)
class CurrentClassEvidence:
    """Expected complete RR-02 cohort identity for one current draft class."""

    draft_class: int
    draft_keys: frozenset[tuple[str, int, str]]

    def __post_init__(self) -> None:
        if not self.draft_keys:
            raise ChampionBoardError("expected current cohort draft keys must be populated")
        ids = [canonical_id for canonical_id, _, _ in self.draft_keys]
        picks = [overall_pick for _, overall_pick, _ in self.draft_keys]
        if (
            any(not canonical_id.strip() for canonical_id in ids)
            or len(ids) != len(set(ids))
            or len(picks) != len(set(picks))
            or any(not isinstance(pick, int) or isinstance(pick, bool) or pick <= 0 for pick in picks)
            or any(position not in SUPPORTED_POSITIONS for _, _, position in self.draft_keys)
        ):
            raise ChampionBoardError(
                "expected current cohort draft keys require unique IDs, unique positive picks, and supported positions"
            )


def _required_feature_columns() -> set[str]:
    return {
        "position",
        "overall_pick",
        "college_seasons_observed",
        "college_passing_observed",
        "college_rushing_observed",
        "college_receiving_observed",
        *VALUE_FEATURES,
        *OBSERVED_FEATURES,
    }


def _boolean_feature(values: pd.Series, name: str) -> pd.Series:
    def normalize(value: Any) -> float | None:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return float(value)
        if pd.notna(value):
            return {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}.get(
                str(value).strip().lower()
            )
        return None

    mapped = values.map(normalize)
    if mapped.isna().any():
        raise ChampionBoardError(f"{name} must contain explicit boolean values")
    return mapped.astype("float64")


def prepare_model_features(table: pd.DataFrame) -> pd.DataFrame:
    """Return only the approved RR-05 features with explicit missingness."""
    missing = sorted(_required_feature_columns().difference(table.columns))
    if missing:
        raise ChampionBoardError(f"model table missing approved feature columns: {missing}")
    result = pd.DataFrame(index=table.index)
    result["position"] = table["position"].astype("string").str.strip().str.upper()
    invalid_positions = sorted(set(result["position"]).difference(SUPPORTED_POSITIONS))
    if invalid_positions:
        raise ChampionBoardError(f"unsupported model positions: {invalid_positions}")

    result["overall_pick"] = pd.to_numeric(table["overall_pick"], errors="coerce")
    if result["overall_pick"].isna().any() or (result["overall_pick"] <= 0).any():
        raise ChampionBoardError("overall_pick must be populated and positive")
    seasons = pd.to_numeric(table["college_seasons_observed"], errors="coerce")
    if (seasons.dropna() < 0).any():
        raise ChampionBoardError("college_seasons_observed must not be negative")
    result["college_seasons_observed"] = seasons.astype("float64")
    result["college_seasons_observed_missing"] = seasons.isna().astype("float64")

    for name in (
        "college_passing_observed",
        "college_rushing_observed",
        "college_receiving_observed",
        *OBSERVED_FEATURES,
    ):
        result[name] = _boolean_feature(table[name], name)
    for name in VALUE_FEATURES:
        values = pd.to_numeric(table[name], errors="coerce")
        invalid_text = table[name].notna() & values.isna()
        if invalid_text.any():
            raise ChampionBoardError(f"{name} must be numeric when populated")
        if (result[f"{name}_observed"].eq(1.0) & values.isna()).any():
            raise ChampionBoardError(f"{name} is missing despite its observed flag")
        if (result[f"{name}_observed"].eq(0.0) & values.notna()).any():
            raise ChampionBoardError(f"{name} is populated despite its unobserved flag")
        result[name] = values.astype("float64")
    return result.loc[:, ["position", *NUMERIC_FEATURES]]


def build_random_forest() -> Pipeline:
    """Build the single frozen pooled Random Forest challenger."""
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                list(NUMERIC_FEATURES),
            ),
            (
                "position",
                OneHotEncoder(
                    categories=[list(SUPPORTED_POSITIONS)],
                    handle_unknown="error",
                    sparse_output=False,
                ),
                ["position"],
            ),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(**RANDOM_FOREST_PARAMETERS)),
        ]
    )


def random_forest_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> np.ndarray:
    model = build_random_forest()
    model.fit(prepare_model_features(train), train[target].to_numpy(dtype=float))
    predictions = np.asarray(model.predict(prepare_model_features(test)), dtype=float)
    if not np.isfinite(predictions).all():
        raise ChampionBoardError("Random Forest produced non-finite predictions")
    return predictions


def _prediction_rows(table: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in expanding_year_folds(table, target):
        for model, predictor in (
            ("draft_capital", draft_capital_predictions),
            ("random_forest", random_forest_predictions),
        ):
            prediction = predictor(fold.train, fold.test, target)
            output = fold.test[["canonical_id", "position", "overall_pick"]].copy()
            output.insert(0, "test_year", fold.test_year)
            output.insert(1, "train_years", "|".join(str(year) for year in fold.train_years))
            output.insert(2, "model", model)
            output["actual"] = fold.test[target].to_numpy(dtype=float)
            output["prediction"] = prediction
            rows.append(output)
    predictions = pd.concat(rows, ignore_index=True).sort_values(
        ["test_year", "model", "overall_pick", "canonical_id"], kind="stable"
    ).reset_index(drop=True)
    return add_prior_fold_intervals(predictions)


def _residual_bounds(history: pd.DataFrame, position: str) -> tuple[str, float, float] | None:
    position_residuals = history.loc[history["position"].eq(position), "residual"]
    if len(position_residuals) >= POSITION_RESIDUAL_MINIMUM:
        residuals = position_residuals.to_numpy(dtype=float)
        source = "position"
    elif len(history) >= GLOBAL_RESIDUAL_MINIMUM:
        residuals = history["residual"].to_numpy(dtype=float)
        source = "global"
    else:
        return None
    radius = float(np.quantile(np.abs(residuals), INTERVAL_LEVEL, method="linear"))
    lower, upper = -radius, radius
    if not np.isfinite([lower, upper]).all():
        raise ChampionBoardError("residual interval validation failed")
    return source, float(lower), float(upper)


def add_prior_fold_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add 80% intervals using residuals from strictly earlier folds only."""
    required = {"test_year", "model", "position", "actual", "prediction"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ChampionBoardError(f"prediction table missing interval columns: {missing}")
    result = predictions.copy()
    result["interval_status"] = "unavailable"
    result["interval_source"] = pd.NA
    result["interval_lower"] = np.nan
    result["interval_upper"] = np.nan
    histories: dict[str, list[dict[str, Any]]] = {}
    for test_year in sorted(result["test_year"].unique()):
        for model in sorted(result["model"].unique()):
            mask = result["test_year"].eq(test_year) & result["model"].eq(model)
            history = pd.DataFrame(
                histories.get(model, []), columns=["position", "residual"]
            )
            for index, row in result.loc[mask].iterrows():
                bounds = _residual_bounds(history, str(row["position"]))
                if bounds is None:
                    continue
                source, lower_residual, upper_residual = bounds
                lower = float(row["prediction"]) + lower_residual
                upper = float(row["prediction"]) + upper_residual
                if not np.isfinite([lower, upper]).all() or lower > upper:
                    raise ChampionBoardError("residual interval validation failed")
                result.at[index, "interval_status"] = "available"
                result.at[index, "interval_source"] = source
                result.at[index, "interval_lower"] = lower
                result.at[index, "interval_upper"] = upper
            current = result.loc[mask, ["position", "actual", "prediction"]]
            histories.setdefault(model, []).extend(
                {
                    "position": row.position,
                    "residual": float(row.actual - row.prediction),
                }
                for row in current.itertuples(index=False)
            )
    return result


def interval_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available = predictions[predictions["interval_status"].eq("available")].copy()
    for model in sorted(predictions["model"].unique()):
        model_all = predictions[predictions["model"].eq(model)]
        model_available = available[available["model"].eq(model)]
        slices: list[tuple[str, str | None, pd.DataFrame, pd.DataFrame]] = [
            ("overall", None, model_all, model_available)
        ]
        for position in sorted(model_all["position"].unique()):
            slices.append(
                (
                    "position",
                    str(position),
                    model_all[model_all["position"].eq(position)],
                    model_available[model_available["position"].eq(position)],
                )
            )
        for slice_name, position, all_rows, covered_rows in slices:
            if covered_rows.empty:
                coverage = None
                width = None
            else:
                inside = covered_rows["actual"].between(
                    covered_rows["interval_lower"], covered_rows["interval_upper"], inclusive="both"
                )
                coverage = float(inside.mean())
                width = float(
                    (covered_rows["interval_upper"] - covered_rows["interval_lower"]).mean()
                )
            rows.append(
                {
                    "model": model,
                    "slice": slice_name,
                    "position": position,
                    "held_out_row_count": int(len(all_rows)),
                    "interval_row_count": int(len(covered_rows)),
                    "interval_coverage": coverage,
                    "mean_interval_width": width,
                }
            )
    return pd.DataFrame(rows)


def _per_year_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (test_year, model), group in predictions.groupby(["test_year", "model"], sort=True):
        rows.append(
            {
                "test_year": int(test_year),
                "model": model,
                "train_years": tuple(int(year) for year in group["train_years"].iloc[0].split("|")),
                "test_row_count": int(len(group)),
                **ranking_metrics(group),
                **magnitude_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def _macro_report(per_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in per_year.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model, "class_count": int(len(group))}
        for metric in RANKING_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = None if values.empty else float(values.mean())
            row[f"{metric}_eligible_class_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _pooled_report(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": model, "row_count": int(len(group)), **magnitude_metrics(group)}
            for model, group in predictions.groupby("model", sort=True)
        ]
    )


def _position_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, position), group in predictions.groupby(["model", "position"], sort=True):
        yearly = [ranking_metrics(rows) for _, rows in group.groupby("test_year", sort=True)]
        row: dict[str, Any] = {
            "model": model,
            "position": position,
            "row_count": int(len(group)),
            "class_count": int(group["test_year"].nunique()),
            **magnitude_metrics(group),
        }
        for metric in RANKING_METRICS:
            defined = [item[metric] for item in yearly if item[metric] is not None]
            row[metric] = None if not defined else float(np.mean(defined))
            row[f"{metric}_eligible_class_count"] = len(defined)
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_gates(
    table: pd.DataFrame, target: str, predictions: pd.DataFrame
) -> Mapping[str, bool]:
    identity_columns = {"identity_match_status", "identity_match_method"}
    missing_identity = sorted(identity_columns.difference(table.columns))
    if missing_identity:
        raise ChampionBoardError(f"model table missing identity evidence: {missing_identity}")
    expected_methods = {
        "exact": "normalized_name_position",
        "override": "override",
        "quarantined": "none",
    }
    identity_valid = table.apply(
        lambda row: expected_methods.get(str(row["identity_match_status"]))
        == str(row["identity_match_method"]),
        axis=1,
    ).all()
    if not identity_valid:
        raise ChampionBoardError("model table identity status and method are inconsistent")

    gap = TARGET_AVAILABILITY_GAPS[target]
    temporal_valid = all(
        max(int(year) for year in str(row.train_years).split("|"))
        <= int(row.test_year) - gap
        for row in predictions[["test_year", "train_years"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    if not temporal_valid:
        raise ChampionBoardError("temporal evaluation contains target-availability leakage")

    available = predictions["interval_status"].eq("available")
    unavailable = predictions["interval_status"].eq("unavailable")
    interval_valid = (
        (available | unavailable).all()
        and np.isfinite(
            predictions.loc[available, ["interval_lower", "interval_upper"]].to_numpy(dtype=float)
        ).all()
        and (
            predictions.loc[available, "interval_lower"]
            <= predictions.loc[available, "prediction"]
        ).all()
        and (
            predictions.loc[available, "prediction"]
            <= predictions.loc[available, "interval_upper"]
        ).all()
        and predictions.loc[unavailable, ["interval_lower", "interval_upper"]].isna().all().all()
    )
    for model, group in predictions.groupby("model", sort=True):
        first_year = group["test_year"].min()
        interval_valid = bool(
            interval_valid
            and group.loc[group["test_year"].eq(first_year), "interval_status"]
            .eq("unavailable")
            .all()
        )
    if not interval_valid:
        raise ChampionBoardError("held-out interval evidence failed validation")
    return {
        "data_integrity": True,
        "temporal_leakage": bool(temporal_valid),
        "identity": bool(identity_valid),
        "interval_validation": bool(interval_valid),
    }


def select_champion(
    target: str,
    macro_class_ranking: pd.DataFrame,
    pooled_row_magnitude: pd.DataFrame,
    per_year: pd.DataFrame,
    *,
    integrity_gates: Mapping[str, bool],
) -> ChampionDecision:
    """Apply the exact frozen target-specific gate; every tie is a loss."""
    supplied_integrity_gates = set(integrity_gates)
    if supplied_integrity_gates != REQUIRED_INTEGRITY_GATES:
        missing = sorted(REQUIRED_INTEGRITY_GATES.difference(supplied_integrity_gates))
        unknown = sorted(supplied_integrity_gates.difference(REQUIRED_INTEGRITY_GATES))
        raise ChampionBoardError(
            f"integrity gates must be exactly the frozen set; missing={missing}, unknown={unknown}"
        )
    if any(not isinstance(value, (bool, np.bool_)) for value in integrity_gates.values()):
        raise ChampionBoardError("integrity gate results must be boolean")
    macro = macro_class_ranking.set_index("model")
    pooled = pooled_row_magnitude.set_index("model")
    try:
        challenger_ndcg = macro.at["random_forest", "ndcg_24"]
        baseline_ndcg = macro.at["draft_capital", "ndcg_24"]
        challenger_mae = pooled.at["random_forest", "mae"]
        baseline_mae = pooled.at["draft_capital", "mae"]
    except KeyError as error:
        raise ChampionBoardError("champion reports must contain both candidate models") from error
    wins = strict_fold_wins(
        per_year,
        challenger="random_forest",
        baseline="draft_capital",
        metric="ndcg_24",
    )
    metrics_defined = not any(
        pd.isna(value)
        for value in (challenger_ndcg, baseline_ndcg, challenger_mae, baseline_mae)
    )
    gates = {
        "macro_ndcg_24_strictly_greater": bool(
            metrics_defined and challenger_ndcg > baseline_ndcg
        ),
        "strict_fold_win_rate_at_least_60_percent": bool(
            wins["win_rate"] is not None and wins["win_rate"] >= CHAMPION_WIN_RATE
        ),
        "pooled_mae_no_more_than_105_percent": bool(
            metrics_defined and challenger_mae <= CHAMPION_MAE_RATIO * baseline_mae
        ),
        **{name: bool(integrity_gates[name]) for name in sorted(REQUIRED_INTEGRITY_GATES)},
    }
    return ChampionDecision(
        target=target,
        champion="random_forest" if all(gates.values()) else "draft_capital",
        gates=gates,
        integrity_gates={name: bool(integrity_gates[name]) for name in sorted(REQUIRED_INTEGRITY_GATES)},
        challenger_ndcg_24=None if pd.isna(challenger_ndcg) else float(challenger_ndcg),
        baseline_ndcg_24=None if pd.isna(baseline_ndcg) else float(baseline_ndcg),
        challenger_mae=None if pd.isna(challenger_mae) else float(challenger_mae),
        baseline_mae=None if pd.isna(baseline_mae) else float(baseline_mae),
        eligible_fold_count=int(wins["eligible_fold_count"]),
        strict_win_count=int(wins["win_count"]),
        strict_win_rate=wins["win_rate"],
    )


def evaluate_challenger(table: pd.DataFrame, target: str) -> TargetEvaluation:
    """Evaluate the two candidates on target-specific RR-03 temporal folds."""
    validated, _ = validate_evaluation_table(table, target)
    prepare_model_features(validated)
    predictions = _prediction_rows(validated, target)
    per_year = _per_year_report(predictions)
    macro = _macro_report(per_year)
    pooled = _pooled_report(predictions)
    positions = _position_report(predictions)
    intervals = interval_metrics(predictions)
    integrity_gates = _validation_gates(validated, target, predictions)
    decision = select_champion(
        target,
        macro,
        pooled,
        per_year,
        integrity_gates=integrity_gates,
    )
    return TargetEvaluation(
        target=target,
        predictions=predictions,
        per_year=per_year,
        macro_class_ranking=macro,
        pooled_row_magnitude=pooled,
        position_slices=positions,
        interval_summary=intervals,
        decision=decision,
    )


def _current_intervals(
    point_predictions: np.ndarray,
    current: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for prediction, (_, player) in zip(point_predictions, current.iterrows(), strict=True):
        bounds = _residual_bounds(residuals, str(player["position"]))
        if bounds is None:
            rows.append(
                {"p10": float(prediction), "p50": float(prediction), "p90": float(prediction), "interval_status": "unavailable"}
            )
            continue
        _, lower, upper = bounds
        rows.append(
            {
                "p10": float(prediction + lower),
                "p50": float(prediction),
                "p90": float(prediction + upper),
                "interval_status": "available",
            }
        )
    return pd.DataFrame(rows, index=current.index)


def fit_selected_target(
    training: pd.DataFrame,
    current: pd.DataFrame,
    evaluation: TargetEvaluation,
) -> pd.DataFrame:
    target = evaluation.target
    validated, _ = validate_evaluation_table(training, target)
    prepare_model_features(current)
    if current["draft_season"].nunique() != 1:
        raise ChampionBoardError("current class must contain exactly one draft season")
    current_year = int(current["draft_season"].iloc[0])
    maximum_training_year = current_year - TARGET_AVAILABILITY_GAPS[target]
    if validated["draft_season"].max() > maximum_training_year:
        raise ChampionBoardError(
            f"{target} training data exceeds the current-class availability cutoff"
        )
    champion = evaluation.decision.champion
    if champion == "random_forest":
        point = random_forest_predictions(validated, current, target)
    else:
        point = draft_capital_predictions(validated, current, target)
    residuals = evaluation.predictions[
        evaluation.predictions["model"].eq(champion)
    ][["position", "actual", "prediction"]].copy()
    residuals["residual"] = residuals["actual"] - residuals["prediction"]
    return _current_intervals(point, current, residuals[["position", "residual"]])


def _clean_optional(value: Any) -> str | None:
    return None if pd.isna(value) or not str(value).strip() else str(value).strip()


def _warnings(value: Any) -> list[str]:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _evaluation_summary(evaluation: TargetEvaluation) -> TargetEvaluationSummary:
    champion = evaluation.decision.champion
    macro = evaluation.macro_class_ranking.set_index("model").loc[champion]
    pooled = evaluation.pooled_row_magnitude.set_index("model").loc[champion]
    interval = evaluation.interval_summary[
        evaluation.interval_summary["model"].eq(champion)
        & evaluation.interval_summary["slice"].eq("overall")
    ].iloc[0]
    return TargetEvaluationSummary(
        selected_champion=champion,
        eligible_held_out_classes=int(macro["ndcg_24_eligible_class_count"]),
        ndcg_at_24=None if pd.isna(macro["ndcg_24"]) else float(macro["ndcg_24"]),
        mae=None if pd.isna(pooled["mae"]) else float(pooled["mae"]),
        interval_coverage=None if pd.isna(interval["interval_coverage"]) else float(interval["interval_coverage"]),
        mean_interval_width=None if pd.isna(interval["mean_interval_width"]) else float(interval["mean_interval_width"]),
    )


def build_rookie_board(
    training_by_target: Mapping[str, pd.DataFrame],
    current_class: pd.DataFrame,
    current_class_evidence: CurrentClassEvidence,
    metadata: PublicationMetadata,
) -> tuple[RookieBoard, Mapping[str, TargetEvaluation]]:
    """Select each target independently and populate the frozen RR-04 board."""
    missing_targets = sorted(set(TARGETS).difference(training_by_target))
    if missing_targets:
        raise ChampionBoardError(f"missing target-specific training tables: {missing_targets}")
    missing_columns = sorted(CURRENT_REQUIRED_COLUMNS.difference(current_class.columns))
    if missing_columns:
        raise ChampionBoardError(f"current class missing publication columns: {missing_columns}")
    if current_class.empty or current_class["canonical_id"].isna().any() or current_class["canonical_id"].duplicated().any():
        raise ChampionBoardError("current class canonical IDs must be populated and unique")
    numeric_picks = pd.to_numeric(current_class["overall_pick"], errors="coerce")
    if numeric_picks.isna().any() or (numeric_picks % 1 != 0).any() or numeric_picks.duplicated().any():
        raise ChampionBoardError("current class overall picks must be populated, whole, and unique")
    supplied_keys = frozenset(
        zip(
            current_class["canonical_id"].astype(str),
            numeric_picks.astype(int),
            current_class["position"].astype(str).str.strip().str.upper(),
            strict=True,
        )
    )
    if supplied_keys != current_class_evidence.draft_keys:
        missing_keys = sorted(current_class_evidence.draft_keys.difference(supplied_keys))
        extra_keys = sorted(supplied_keys.difference(current_class_evidence.draft_keys))
        raise ChampionBoardError(
            f"current class does not match expected complete cohort; missing={missing_keys}, extra={extra_keys}"
        )
    current = current_class.sort_values(["overall_pick", "canonical_id"], kind="stable").reset_index(drop=True).copy()
    class_years = pd.to_numeric(current["draft_season"], errors="coerce")
    if class_years.isna().any() or class_years.nunique() != 1:
        raise ChampionBoardError("current class must contain one numeric draft season")
    draft_class = int(class_years.iloc[0])
    if draft_class != current_class_evidence.draft_class:
        raise ChampionBoardError("current class draft season does not match expected cohort evidence")

    evaluations = {target: evaluate_challenger(training_by_target[target], target) for target in TARGETS}
    forecasts = {
        target: fit_selected_target(training_by_target[target], current, evaluations[target])
        for target in TARGETS
    }
    champions = ChampionByTarget(
        three_year_ppr_points=evaluations["three_year_ppr_points"].decision.champion,
        rookie_year_ppr_points=evaluations["rookie_year_ppr_points"].decision.champion,
    )
    ordering = pd.DataFrame(
        {
            "canonical_id": current["canonical_id"].astype(str),
            "prediction": forecasts["three_year_ppr_points"]["p50"].to_numpy(dtype=float),
        }
    ).sort_values(["prediction", "canonical_id"], ascending=[False, True], kind="stable")
    base_rank = {canonical_id: rank for rank, canonical_id in enumerate(ordering["canonical_id"], start=1)}
    current["base_rank"] = current["canonical_id"].astype(str).map(base_rank)
    current["position_rank"] = current.sort_values(
        ["position", "base_rank", "canonical_id"], kind="stable"
    ).groupby("position", sort=True).cumcount().add(1).sort_index()

    three_width = forecasts["three_year_ppr_points"]["p90"] - forecasts["three_year_ppr_points"]["p10"]
    available_widths = three_width[forecasts["three_year_ppr_points"]["interval_status"].eq("available")]
    median_width = None if available_widths.empty else float(available_widths.median())
    players = []
    for index, row in current.iterrows():
        warnings = _warnings(row.get("data_quality_warnings"))
        unavailable_targets = [
            target
            for target in TARGETS
            if forecasts[target].at[index, "interval_status"] == "unavailable"
        ]
        warnings.extend(f"{target} interval unavailable" for target in unavailable_targets)
        warnings = sorted(set(warnings))
        if unavailable_targets:
            confidence = "unavailable"
        elif warnings:
            confidence = "low"
        elif median_width is not None and float(three_width.at[index]) <= median_width:
            confidence = "high"
        else:
            confidence = "medium"
        source_ids = {
            name: _clean_optional(row.get(name))
            for name in SOURCE_ID_COLUMNS
            if _clean_optional(row.get(name)) is not None
        }
        players.append(
            RookiePlayer(
                canonical_id=str(row["canonical_id"]),
                source_ids=VerifiedSourceIds(**source_ids),
                name=str(row["player_name"]),
                position=str(row["position"]),
                nfl_team=str(row["nfl_team"]),
                round=int(row["round"]),
                overall_pick=int(row["overall_pick"]),
                base_rank=int(row["base_rank"]),
                position_rank=int(row["position_rank"]),
                tier=1 + (int(row["base_rank"]) - 1) // 12,
                three_year_ppr=PredictionQuantiles(**forecasts["three_year_ppr_points"].loc[index, ["p10", "p50", "p90"]].to_dict()),
                rookie_year_ppr=PredictionQuantiles(**forecasts["rookie_year_ppr_points"].loc[index, ["p10", "p50", "p90"]].to_dict()),
                confidence=confidence,
                data_quality_warnings=tuple(warnings),
                identity_match_method=str(row["identity_match_method"]),
                identity_match_status=str(row["identity_match_status"]),
                champion_by_target=champions,
            )
        )
    cohorts = {
        target: tuple(sorted(int(year) for year in training_by_target[target]["draft_season"].unique()))
        for target in TARGETS
    }
    all_intervals_available = all(
        forecasts[target]["interval_status"].eq("available").all() for target in TARGETS
    )
    artifact_metadata = ArtifactMetadata(
        schema_version=SCHEMA_VERSION,
        artifact_version=metadata.artifact_version,
        model_version=metadata.model_version,
        producer_commit=metadata.producer_commit,
        draft_class=draft_class,
        mode="post_draft",
        draft_event_date=metadata.draft_event_date,
        generated_at=metadata.generated_at,
        data_cutoff=metadata.data_cutoff,
        outcomes_cutoff_season=metadata.outcomes_cutoff_season,
        scoring_basis="ppr",
        target_definitions=TargetDefinitions(
            three_year_ppr_points="Sum of PPR points in the first three NFL regular seasons.",
            rookie_year_ppr_points="PPR points in the first NFL regular season.",
        ),
        target_maturity=TargetMaturity(
            three_year_ppr_points=metadata.outcomes_cutoff_season - 2,
            rookie_year_ppr_points=metadata.outcomes_cutoff_season,
        ),
        training_cohorts=TrainingCohorts(**cohorts),
        champion_by_target=champions,
        capabilities=Capabilities(
            three_year_predictions=True,
            rookie_year_predictions=True,
            prediction_intervals=all_intervals_available,
            interval_level=INTERVAL_LEVEL,
        ),
        evaluation_summary=EvaluationSummary(
            three_year_ppr_points=_evaluation_summary(evaluations["three_year_ppr_points"]),
            rookie_year_ppr_points=_evaluation_summary(evaluations["rookie_year_ppr_points"]),
        ),
        source_notices=metadata.source_notices,
    )
    return RookieBoard(metadata=artifact_metadata, players=tuple(players)), evaluations


def publish_rookie_board(
    training_by_target: Mapping[str, pd.DataFrame],
    current_class: pd.DataFrame,
    current_class_evidence: CurrentClassEvidence,
    metadata: PublicationMetadata,
    output_dir: str | Path,
) -> dict[str, Path]:
    board, _ = build_rookie_board(
        training_by_target, current_class, current_class_evidence, metadata
    )
    return write_handoff(board, output_dir)
