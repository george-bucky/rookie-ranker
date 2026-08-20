"""Frozen temporal evaluation contracts and transparent rookie baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


MIN_TRAINING_CLASSES = 5
SUPPORTED_POSITIONS = ("QB", "RB", "TE", "WR")
BASELINE_NAMES = ("position_mean", "draft_capital")
TARGET_AVAILABILITY_GAPS = {
    "three_year_ppr_points": 3,
    "rookie_year_ppr_points": 1,
}
TARGET_STATUS_COLUMNS = {
    "three_year_ppr_points": "three_year_target_status",
    "rookie_year_ppr_points": "rookie_target_status",
}
REQUIRED_COLUMNS = {
    "canonical_id",
    "draft_season",
    "position",
    "overall_pick",
}
RANKING_METRICS = (
    "ndcg_24",
    "ndcg_12",
    "spearman",
    "top_12_hit_recall",
    "top_24_hit_recall",
)


class EvaluationContractError(ValueError):
    """Raised when an evaluation input could invalidate temporal evidence."""


@dataclass(frozen=True)
class TemporalFold:
    """One expanding-year split with immutable year membership."""

    test_year: int
    train_years: tuple[int, ...]
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class EvaluationResult:
    """All frozen RR-03 reports for one target."""

    target: str
    status_column: str
    min_training_classes: int
    folds: pd.DataFrame
    predictions: pd.DataFrame
    per_year: pd.DataFrame
    macro_class_ranking: pd.DataFrame
    pooled_row_magnitude: pd.DataFrame
    position_slices: pd.DataFrame
    strict_wins: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "target": self.target,
            "status_column": self.status_column,
            "target_availability_gap": TARGET_AVAILABILITY_GAPS[self.target],
            "min_training_classes": self.min_training_classes,
            "baselines": list(BASELINE_NAMES),
            "tie_rule": (
                "Higher score first; equal scores use lexicographically smaller "
                "canonical_id. This neutral tie-break keeps overall_pick out of the "
                "position-mean baseline. Spearman uses standard average ranks for "
                "equal numeric values."
            ),
            "hit_rule": (
                "For K in {12,24}, actual hits are exactly the first min(K, class size) "
                "players under the deterministic actual-target ordering; recall is the "
                "share of those IDs in the first min(K, class size) predicted players."
            ),
            "reports": {
                "folds": _records(self.folds),
                "predictions": _records(self.predictions),
                "per_year": _records(self.per_year),
                "macro_class_ranking": _records(self.macro_class_ranking),
                "pooled_row_magnitude": _records(self.pooled_row_magnitude),
                "position_slices": _records(self.position_slices),
                "strict_wins": _records(self.strict_wins),
            },
        }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if math.isinf(numeric):
            raise EvaluationContractError("evaluation output contains an infinite value")
        return numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return None
    return value


def _status_column(target: str, status_column: str | None) -> str:
    expected = TARGET_STATUS_COLUMNS.get(target)
    if expected is None:
        raise EvaluationContractError(
            f"unsupported evaluation target {target!r}; expected one of "
            f"{sorted(TARGET_STATUS_COLUMNS)}"
        )
    if status_column is not None and status_column != expected:
        raise EvaluationContractError(
            f"target {target!r} requires status column {expected!r}"
        )
    return expected


def validate_evaluation_table(
    table: pd.DataFrame,
    target: str,
    *,
    status_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Validate and normalize a mature-only evaluation table.

    RR-03 intentionally fails when even one supplied row is immature, missing a
    target, or outside the drafted QB/RB/WR/TE contract. Callers must provide a
    complete, mature class extract rather than silently dropping rows.
    """
    resolved_status = _status_column(target, status_column)
    required = REQUIRED_COLUMNS | {target, resolved_status}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise EvaluationContractError(f"evaluation table missing columns: {missing}")
    if table.empty:
        raise EvaluationContractError("evaluation table must not be empty")

    result = table.copy()
    result["canonical_id"] = result["canonical_id"].astype("string").str.strip()
    if result["canonical_id"].isna().any() or result["canonical_id"].eq("").any():
        raise EvaluationContractError("canonical_id must be populated")
    if result["canonical_id"].duplicated().any():
        raise EvaluationContractError("canonical_id must be unique across evaluation rows")

    result["draft_season"] = _whole_numbers(result["draft_season"], "draft_season")
    result["overall_pick"] = _whole_numbers(result["overall_pick"], "overall_pick")
    if result["overall_pick"].le(0).any():
        raise EvaluationContractError("overall_pick must be greater than zero")
    duplicate_picks = result.duplicated(
        ["draft_season", "overall_pick"], keep=False
    )
    if duplicate_picks.any():
        keys = result.loc[
            duplicate_picks, ["draft_season", "overall_pick"]
        ].drop_duplicates().to_dict("records")
        raise EvaluationContractError(f"duplicate draft-season/overall-pick keys: {keys}")

    result["position"] = result["position"].astype("string").str.strip().str.upper()
    invalid_positions = sorted(
        str(value)
        for value in result.loc[~result["position"].isin(SUPPORTED_POSITIONS), "position"].unique()
    )
    if invalid_positions:
        raise EvaluationContractError(f"unsupported positions: {invalid_positions}")

    statuses = result[resolved_status].astype("string").str.strip().str.lower()
    incomplete = statuses.ne("complete").fillna(True)
    numeric_target = pd.to_numeric(result[target], errors="coerce")
    invalid_target = numeric_target.isna() | ~np.isfinite(numeric_target.astype(float))
    if incomplete.any() or invalid_target.any():
        bad_years = sorted(
            result.loc[incomplete | invalid_target, "draft_season"].unique().tolist()
        )
        raise EvaluationContractError(
            "evaluation requires complete, finite targets for every supplied row; "
            f"invalid draft classes: {bad_years}"
        )
    result[resolved_status] = statuses
    result[target] = numeric_target.astype("float64")
    return result.sort_values(
        ["draft_season", "overall_pick", "canonical_id"], kind="stable"
    ).reset_index(drop=True), resolved_status


def _whole_numbers(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or (numeric % 1 != 0).any():
        raise EvaluationContractError(f"{label} must contain whole numbers")
    return numeric.astype("int64")


def expanding_year_folds(
    table: pd.DataFrame,
    target: str,
    *,
    status_column: str | None = None,
    min_training_classes: int = MIN_TRAINING_CLASSES,
) -> tuple[TemporalFold, ...]:
    """Return expanding folds that train only on draft classes before test Y."""
    if min_training_classes < MIN_TRAINING_CLASSES:
        raise EvaluationContractError(
            f"min_training_classes must be at least {MIN_TRAINING_CLASSES}"
        )
    validated, _ = validate_evaluation_table(
        table, target, status_column=status_column
    )
    years = sorted(int(year) for year in validated["draft_season"].unique())
    availability_gap = TARGET_AVAILABILITY_GAPS[target]

    folds: list[TemporalFold] = []
    for test_year in years:
        train_years = tuple(
            year for year in years if year <= test_year - availability_gap
        )
        if len(train_years) < min_training_classes:
            continue
        train = validated[validated["draft_season"].isin(train_years)].copy()
        test = validated[validated["draft_season"].eq(test_year)].copy()
        if train["draft_season"].max() >= test_year:
            raise EvaluationContractError("temporal fold contains test-year leakage")
        folds.append(TemporalFold(test_year, train_years, train, test))
    if not folds:
        raise EvaluationContractError(
            "evaluation requires at least one test class with "
            f"{min_training_classes} target-available training classes"
        )
    return tuple(folds)


def _require_test_positions(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing = sorted(set(test["position"]).difference(train["position"]))
    if missing:
        raise EvaluationContractError(
            f"test positions have no historical training rows: {missing}"
        )


def position_mean_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> np.ndarray:
    """Predict each held-out player with the prior-class mean for that position."""
    _require_test_positions(train, test)
    means = train.groupby("position", sort=True)[target].mean()
    predictions = test["position"].map(means).to_numpy(dtype=float)
    _require_finite_computation(predictions, "position-mean predictions")
    return predictions


def _draft_capital_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [np.ones(len(frame)), np.log1p(frame["overall_pick"].to_numpy(dtype=float))]
    for position in SUPPORTED_POSITIONS[1:]:
        columns.append(frame["position"].eq(position).to_numpy(dtype=float))
    return np.column_stack(columns)


def draft_capital_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> np.ndarray:
    """Fit transparent OLS: target ~ position + log1p(overall_pick)."""
    _require_test_positions(train, test)
    try:
        coefficients, _, _, _ = np.linalg.lstsq(
            _draft_capital_matrix(train),
            train[target].to_numpy(dtype=float),
            rcond=None,
        )
    except np.linalg.LinAlgError as error:
        raise EvaluationContractError("draft-capital baseline fit failed") from error
    _require_finite_computation(coefficients, "draft-capital coefficients")
    predictions = _draft_capital_matrix(test) @ coefficients
    _require_finite_computation(predictions, "draft-capital predictions")
    return predictions


def _require_finite_computation(values: Any, label: str) -> None:
    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise EvaluationContractError(f"{label} produced non-finite values")


def _ordered_ids(frame: pd.DataFrame, score_column: str, k: int) -> list[str]:
    ordered = frame.sort_values(
        [score_column, "canonical_id"],
        ascending=[False, True],
        kind="stable",
    )
    return ordered["canonical_id"].head(min(k, len(ordered))).astype(str).tolist()


def ndcg_at_k(frame: pd.DataFrame, *, k: int) -> float | None:
    """Compute linear-gain NDCG after deterministic prediction ordering."""
    if frame.empty:
        return None
    ranked = frame.sort_values(
        ["prediction", "canonical_id"],
        ascending=[False, True],
        kind="stable",
    )
    ideal = frame.sort_values(
        ["actual", "canonical_id"],
        ascending=[False, True],
        kind="stable",
    )
    limit = min(k, len(frame))
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    ranked_relevance = np.maximum(ranked["actual"].to_numpy(dtype=float)[:limit], 0.0)
    ideal_relevance = np.maximum(ideal["actual"].to_numpy(dtype=float)[:limit], 0.0)
    dcg = float(np.sum(ranked_relevance / discounts))
    ideal_dcg = float(np.sum(ideal_relevance / discounts))
    _require_finite_computation([dcg, ideal_dcg], "NDCG")
    if ideal_dcg == 0:
        return None
    score = dcg / ideal_dcg
    _require_finite_computation([score], "NDCG")
    return score


def spearman_correlation(frame: pd.DataFrame) -> float | None:
    """Compute standard Spearman correlation using average ranks for ties."""
    if len(frame) < 2:
        return None
    actual_rank = frame["actual"].rank(method="average")
    prediction_rank = frame["prediction"].rank(method="average")
    if actual_rank.nunique() < 2 or prediction_rank.nunique() < 2:
        return None
    correlation = float(actual_rank.corr(prediction_rank))
    _require_finite_computation([correlation], "Spearman correlation")
    return correlation


def top_k_hit_recall(frame: pd.DataFrame, *, k: int) -> float | None:
    """Recall of deterministic actual top-K IDs in deterministic predicted top K."""
    if frame.empty:
        return None
    actual_ids = set(_ordered_ids(frame, "actual", k))
    predicted_ids = set(_ordered_ids(frame, "prediction", k))
    return len(actual_ids.intersection(predicted_ids)) / len(actual_ids)


def magnitude_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {"mae": None, "r2": None}
    actual = frame["actual"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        mae = float(np.mean(np.abs(actual - prediction)))
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        numerator = float(np.sum((actual - prediction) ** 2))
    _require_finite_computation([mae, denominator, numerator], "magnitude metrics")
    r2 = None if len(frame) < 2 or denominator == 0 else 1 - numerator / denominator
    if r2 is not None:
        _require_finite_computation([r2], "R-squared")
    return {"mae": mae, "r2": r2}


def ranking_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    return {
        "ndcg_24": ndcg_at_k(frame, k=24),
        "ndcg_12": ndcg_at_k(frame, k=12),
        "spearman": spearman_correlation(frame),
        "top_12_hit_recall": top_k_hit_recall(frame, k=12),
        "top_24_hit_recall": top_k_hit_recall(frame, k=24),
    }


def _prediction_rows(folds: Iterable[TemporalFold], target: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in folds:
        for model, predictor in (
            ("position_mean", position_mean_predictions),
            ("draft_capital", draft_capital_predictions),
        ):
            prediction = predictor(fold.train, fold.test, target)
            output = fold.test[
                ["canonical_id", "position", "overall_pick"]
            ].copy()
            output.insert(0, "test_year", fold.test_year)
            output.insert(1, "train_years", "|".join(str(year) for year in fold.train_years))
            output.insert(2, "model", model)
            output["actual"] = fold.test[target].to_numpy(dtype=float)
            output["prediction"] = prediction
            rows.append(output)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["test_year", "model", "overall_pick", "canonical_id"], kind="stable"
    ).reset_index(drop=True)


def _per_year_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (test_year, model), group in predictions.groupby(
        ["test_year", "model"], sort=True
    ):
        train_years = tuple(int(year) for year in group["train_years"].iloc[0].split("|"))
        rows.append(
            {
                "test_year": int(test_year),
                "model": model,
                "train_years": train_years,
                "train_class_count": len(train_years),
                "test_row_count": len(group),
                **ranking_metrics(group),
                **magnitude_metrics(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_year", "model"], kind="stable").reset_index(
        drop=True
    )


def _mean_defined(values: pd.Series) -> tuple[float | None, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None, 0
    _require_finite_computation(numeric.to_numpy(), "macro ranking metrics")
    mean = float(numeric.mean())
    _require_finite_computation([mean], "macro ranking mean")
    return mean, int(len(numeric))


def _macro_class_report(per_year: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in per_year.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model, "class_count": int(len(group))}
        for metric in RANKING_METRICS:
            value, count = _mean_defined(group[metric])
            row[metric] = value
            row[f"{metric}_eligible_class_count"] = count
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model", kind="stable").reset_index(drop=True)


def _pooled_magnitude_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        rows.append(
            {"model": model, "row_count": int(len(group)), **magnitude_metrics(group)}
        )
    return pd.DataFrame(rows).sort_values("model", kind="stable").reset_index(drop=True)


def _position_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, position), group in predictions.groupby(["model", "position"], sort=True):
        class_metrics = [
            ranking_metrics(class_rows)
            for _, class_rows in group.groupby("test_year", sort=True)
        ]
        row: dict[str, Any] = {
            "model": model,
            "position": position,
            "row_count": int(len(group)),
            "class_count": int(group["test_year"].nunique()),
            **magnitude_metrics(group),
        }
        metric_frame = pd.DataFrame(class_metrics)
        for metric in RANKING_METRICS:
            value, count = _mean_defined(metric_frame[metric])
            row[metric] = value
            row[f"{metric}_eligible_class_count"] = count
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "position"], kind="stable").reset_index(
        drop=True
    )


def strict_fold_wins(
    per_year: pd.DataFrame,
    *,
    challenger: str,
    baseline: str,
    metric: str = "ndcg_24",
) -> dict[str, Any]:
    """Count strict held-out class wins; equal or undefined values are not wins."""
    required = {"test_year", "model", metric}
    missing = sorted(required.difference(per_year.columns))
    if missing:
        raise EvaluationContractError(f"per-year report missing columns: {missing}")
    values = per_year.pivot(index="test_year", columns="model", values=metric)
    missing_models = sorted({challenger, baseline}.difference(values.columns))
    if missing_models:
        raise EvaluationContractError(f"per-year report missing models: {missing_models}")
    paired = values[[challenger, baseline]].dropna()
    wins = paired[challenger] > paired[baseline]
    ties = paired[challenger] == paired[baseline]
    eligible = int(len(paired))
    win_count = int(wins.sum())
    return {
        "challenger": challenger,
        "baseline": baseline,
        "metric": metric,
        "eligible_fold_count": eligible,
        "win_count": win_count,
        "tie_count": int(ties.sum()),
        "non_win_count": eligible - win_count,
        "win_rate": None if eligible == 0 else win_count / eligible,
    }


def evaluate_baselines(
    table: pd.DataFrame,
    target: str,
    *,
    status_column: str | None = None,
    min_training_classes: int = MIN_TRAINING_CLASSES,
) -> EvaluationResult:
    """Run the complete deterministic RR-03 baseline evaluation."""
    validated, resolved_status = validate_evaluation_table(
        table, target, status_column=status_column
    )
    folds = expanding_year_folds(
        validated,
        target,
        status_column=resolved_status,
        min_training_classes=min_training_classes,
    )
    fold_report = pd.DataFrame(
        [
            {
                "test_year": fold.test_year,
                "train_years": fold.train_years,
                "train_class_count": len(fold.train_years),
                "train_row_count": len(fold.train),
                "test_row_count": len(fold.test),
            }
            for fold in folds
        ]
    )
    predictions = _prediction_rows(folds, target)
    per_year = _per_year_report(predictions)
    strict_wins = pd.DataFrame(
        [
            strict_fold_wins(
                per_year,
                challenger="draft_capital",
                baseline="position_mean",
            )
        ]
    )
    return EvaluationResult(
        target=target,
        status_column=resolved_status,
        min_training_classes=min_training_classes,
        folds=fold_report,
        predictions=predictions,
        per_year=per_year,
        macro_class_ranking=_macro_class_report(per_year),
        pooled_row_magnitude=_pooled_magnitude_report(predictions),
        position_slices=_position_report(predictions),
        strict_wins=strict_wins,
    )


def evaluation_json_bytes(result: EvaluationResult) -> bytes:
    """Serialize an evaluation with stable ordering and no non-JSON NaN values."""
    return (
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def evaluation_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a stable report CSV with explicit tuple and float formats."""
    normalized = frame.copy()
    for column in normalized.select_dtypes(include="number").columns:
        numeric = normalized[column].to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise EvaluationContractError(
                f"evaluation CSV column {column!r} contains an infinite value"
            )
    for column in normalized.columns:
        if normalized[column].map(lambda value: isinstance(value, tuple)).any():
            normalized[column] = normalized[column].map(
                lambda value: "|".join(str(item) for item in value)
                if isinstance(value, tuple)
                else value
            )
    return normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def write_evaluation(result: EvaluationResult, path: str | Path) -> None:
    Path(path).write_bytes(evaluation_json_bytes(result))


def write_evaluation_bundle(
    result: EvaluationResult, output_dir: str | Path
) -> dict[str, Path]:
    """Stage and promote the three reports without exposing a partial bundle."""
    output_dir = Path(output_dir)
    paths = {
        "json": output_dir / "evaluation.json",
        "predictions": output_dir / "predictions.csv",
        "per_year": output_dir / "per-year.csv",
    }
    payloads = {
        "json": evaluation_json_bytes(result),
        "predictions": evaluation_csv_bytes(result.predictions),
        "per_year": evaluation_csv_bytes(result.per_year),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name or 'evaluation'}.rr03-",
            dir=output_dir.parent,
        )
    )
    staged_paths = {
        name: staging_dir / path.name for name, path in paths.items()
    }
    created_output_dir = not output_dir.exists()
    backups: dict[str, Path] = {}
    promoted: list[Path] = []
    try:
        for name, staged_path in staged_paths.items():
            staged_path.write_bytes(payloads[name])

        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)

        try:
            for name, final_path in paths.items():
                if final_path.exists() or final_path.is_symlink():
                    backup_path = staging_dir / f"{final_path.name}.previous"
                    os.replace(final_path, backup_path)
                    backups[name] = backup_path
            for name, final_path in paths.items():
                os.replace(staged_paths[name], final_path)
                promoted.append(final_path)
        except BaseException:
            for final_path in promoted:
                final_path.unlink(missing_ok=True)
            for name, backup_path in backups.items():
                os.replace(backup_path, paths[name])
            if created_output_dir:
                try:
                    output_dir.rmdir()
                except OSError:
                    pass
            raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return paths
