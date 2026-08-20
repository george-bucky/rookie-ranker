from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rookie_ranker.evaluation import (
    EvaluationContractError,
    evaluation_csv_bytes,
    evaluate_baselines,
    expanding_year_folds,
    magnitude_metrics,
    ndcg_at_k,
    draft_capital_predictions,
    strict_fold_wins,
    top_k_hit_recall,
    write_evaluation_bundle,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rr03" / "mature-training-table.csv"
TARGET = "three_year_ppr_points"


def mature_table() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def test_three_year_fold_uses_five_target_available_classes():
    folds = expanding_year_folds(mature_table(), TARGET)

    assert len(folds) == 2
    fold = folds[0]
    assert fold.test_year == 2020
    assert fold.train_years == (2013, 2014, 2015, 2016, 2017)
    assert set(fold.train["draft_season"]) == set(fold.train_years)
    assert set(fold.test["draft_season"]) == {2020}
    assert fold.train["draft_season"].max() < fold.test_year

    second = folds[1]
    assert second.test_year == 2021
    assert second.train_years == (2013, 2014, 2015, 2016, 2017, 2018)


def test_rookie_year_fold_can_use_the_immediately_prior_class():
    folds = expanding_year_folds(mature_table(), "rookie_year_ppr_points")

    assert folds[0].test_year == 2018
    assert folds[0].train_years == (2013, 2014, 2015, 2016, 2017)


def test_test_year_targets_cannot_change_its_baseline_predictions():
    original = evaluate_baselines(mature_table(), TARGET).predictions
    changed = mature_table()
    changed.loc[changed["draft_season"].eq(2020), TARGET] += 10_000
    rerun = evaluate_baselines(changed, TARGET).predictions

    original = original.query("test_year == 2020")
    rerun = rerun.query("test_year == 2020")
    assert original[["canonical_id", "model", "prediction"]].to_dict("records") == rerun[
        ["canonical_id", "model", "prediction"]
    ].to_dict("records")


def test_three_year_2020_labels_cannot_change_2021_predictions():
    original = evaluate_baselines(mature_table(), TARGET).predictions
    changed = mature_table()
    changed.loc[changed["draft_season"].eq(2020), TARGET] += 10_000
    rerun = evaluate_baselines(changed, TARGET).predictions

    columns = ["canonical_id", "model", "prediction"]
    assert original.query("test_year == 2021")[columns].to_dict(
        "records"
    ) == rerun.query("test_year == 2021")[columns].to_dict("records")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("three_year_target_status", "immature"),
        ("three_year_target_status", pd.NA),
        ("three_year_ppr_points", pd.NA),
    ],
)
def test_immature_or_null_target_row_rejects_the_entire_class(column, value):
    table = mature_table()
    table.loc[table["draft_season"].eq(2020), column] = value

    with pytest.raises(EvaluationContractError, match=r"invalid draft classes: \[2020\]"):
        evaluate_baselines(table, TARGET)


def test_fewer_than_five_training_classes_is_never_allowed():
    with pytest.raises(EvaluationContractError, match="must be at least 5"):
        expanding_year_folds(mature_table(), TARGET, min_training_classes=4)

    five_classes = mature_table().query("draft_season <= 2019")
    with pytest.raises(EvaluationContractError, match="at least one test class"):
        expanding_year_folds(five_classes, TARGET)


def test_known_target_cannot_use_a_different_maturity_status():
    with pytest.raises(EvaluationContractError, match="requires status column"):
        evaluate_baselines(
            mature_table(), TARGET, status_column="rookie_target_status"
        )


def test_unknown_target_is_rejected_even_with_a_status_column():
    table = mature_table().assign(made_up_target=1.0, made_up_status="complete")
    with pytest.raises(EvaluationContractError, match="unsupported evaluation target"):
        evaluate_baselines(
            table, "made_up_target", status_column="made_up_status"
        )


def test_hand_calculated_ranking_and_magnitude_metrics():
    frame = pd.DataFrame(
        {
            "canonical_id": ["a", "b", "c", "d"],
            "overall_pick": [1, 2, 3, 4],
            "actual": [4.0, 3.0, 2.0, 1.0],
            "prediction": [1.0, 2.0, 3.0, 4.0],
        }
    )
    expected_ndcg_2 = (1 + 2 / np.log2(3)) / (4 + 3 / np.log2(3))

    assert ndcg_at_k(frame, k=2) == pytest.approx(expected_ndcg_2)
    assert top_k_hit_recall(frame, k=2) == 0.0
    assert magnitude_metrics(frame) == pytest.approx({"mae": 2.0, "r2": -3.0})


def test_top_12_and_top_24_recall_use_their_exact_cutoffs():
    frame = pd.DataFrame(
        {
            "canonical_id": [f"p-{number:02d}" for number in range(30)],
            "overall_pick": range(1, 31),
            "actual": range(30, 0, -1),
            "prediction": range(1, 31),
        }
    )

    assert top_k_hit_recall(frame, k=12) == 0.0
    assert top_k_hit_recall(frame, k=24) == pytest.approx(18 / 24)


def test_integrated_report_wires_ndcg_12_and_24_to_distinct_cutoffs():
    rows = []
    for year in range(2013, 2021):
        for pick in range(1, 31):
            is_test = year == 2020
            rows.append(
                {
                    "canonical_id": f"{year}-{pick:02d}",
                    "draft_season": year,
                    "position": "QB",
                    "overall_pick": pick,
                    TARGET: float(pick if is_test else 31 - pick),
                    "three_year_target_status": "complete",
                }
            )
    result = evaluate_baselines(pd.DataFrame(rows), TARGET)
    predictions = result.predictions.query(
        "test_year == 2020 and model == 'position_mean'"
    )
    report = result.per_year.query(
        "test_year == 2020 and model == 'position_mean'"
    ).iloc[0]

    assert report["ndcg_12"] == pytest.approx(ndcg_at_k(predictions, k=12))
    assert report["ndcg_24"] == pytest.approx(ndcg_at_k(predictions, k=24))
    assert report["ndcg_12"] != pytest.approx(report["ndcg_24"])


def test_hit_ties_use_neutral_canonical_id_deterministically():
    frame = pd.DataFrame(
        {
            "canonical_id": ["z", "a", "b"],
            "overall_pick": [1, 99, 3],
            "actual": [10.0, 10.0, 0.0],
            "prediction": [1.0, 1.0, 2.0],
        }
    )

    # Actual top one is "a" by canonical ID despite z having better draft capital.
    # Predicted top one is b, so the hit recall is zero.
    assert top_k_hit_recall(frame, k=1) == 0.0


def test_draft_capital_baseline_is_position_plus_log_pick_ols():
    train = pd.DataFrame(
        {
            "position": ["QB", "QB", "QB"],
            "overall_pick": [1, 3, 7],
        }
    )
    train[TARGET] = 100 - 10 * np.log1p(train["overall_pick"])
    test = pd.DataFrame({"position": ["QB"], "overall_pick": [15]})

    prediction = draft_capital_predictions(train, test, TARGET)

    assert prediction[0] == pytest.approx(100 - 10 * np.log1p(15))


def test_reports_keep_class_macro_ranking_separate_from_row_pooled_magnitude():
    result = evaluate_baselines(mature_table(), TARGET)

    assert result.per_year[["test_year", "model"]].to_dict("records") == [
        {"test_year": 2020, "model": "draft_capital"},
        {"test_year": 2020, "model": "position_mean"},
        {"test_year": 2021, "model": "draft_capital"},
        {"test_year": 2021, "model": "position_mean"},
    ]
    assert result.macro_class_ranking["class_count"].tolist() == [2, 2]
    assert result.pooled_row_magnitude["row_count"].tolist() == [6, 6]
    assert set(result.position_slices["position"]) == {"QB", "RB"}
    assert result.position_slices["row_count"].sum() == 12

    position_years = result.per_year.query("model == 'position_mean'")
    macro = result.macro_class_ranking.set_index("model").loc[
        "position_mean", "ndcg_24"
    ]
    assert macro == pytest.approx(position_years["ndcg_24"].mean())
    row_weighted = np.average(
        position_years["ndcg_24"], weights=position_years["test_row_count"]
    )
    assert macro != pytest.approx(row_weighted)


def test_fixture_metrics_are_stable_and_hand_checkable():
    result = evaluate_baselines(mature_table(), TARGET)
    rows = result.per_year.set_index(["test_year", "model"])

    # Position means are QB=(300+120)/2=210 and RB=(240+60)/2=150.
    position_predictions = result.predictions.query(
        "model == 'position_mean' and test_year == 2020"
    ).set_index("canonical_id")["prediction"]
    assert position_predictions.to_dict() == {
        "2020-qb-1": 210.0,
        "2020-rb-5": 150.0,
        "2020-qb-20": 210.0,
        "2020-rb-40": 150.0,
    }
    # Absolute errors are 90, 70, 110, and 100, for a pooled mean of 92.5.
    position_2020 = rows.loc[(2020, "position_mean")]
    assert position_2020["mae"] == pytest.approx(92.5)
    assert position_2020["top_12_hit_recall"] == 1.0
    assert position_2020["top_24_hit_recall"] == 1.0


def test_strict_win_count_excludes_undefined_folds_and_ties_are_not_wins():
    per_year = pd.DataFrame(
        {
            "test_year": [2020, 2020, 2021, 2021, 2022, 2022],
            "model": ["challenger", "baseline"] * 3,
            "ndcg_24": [0.9, 0.8, 0.7, 0.7, pd.NA, 0.6],
        }
    )

    count = strict_fold_wins(
        per_year, challenger="challenger", baseline="baseline", metric="ndcg_24"
    )

    assert count == {
        "challenger": "challenger",
        "baseline": "baseline",
        "metric": "ndcg_24",
        "eligible_fold_count": 2,
        "win_count": 1,
        "tie_count": 1,
        "non_win_count": 1,
        "win_rate": 0.5,
    }


def test_extreme_finite_targets_fail_on_nonfinite_computation():
    table = mature_table()
    table[TARGET] = np.where(table.index % 2, 1e308, -1e308)

    with pytest.raises(EvaluationContractError, match="non-finite"):
        evaluate_baselines(table, TARGET)


def test_csv_float_representation_is_frozen_to_17_significant_digits():
    payload = evaluation_csv_bytes(pd.DataFrame({"metric": [1 / 3]}))

    assert payload == b"metric\n0.33333333333333331\n"

    with pytest.raises(EvaluationContractError, match="infinite value"):
        evaluation_csv_bytes(pd.DataFrame({"metric": [float("inf")]}))


def test_bundle_second_staging_write_failure_leaves_no_outputs_or_temp(
    tmp_path, monkeypatch
):
    result = evaluate_baselines(mature_table(), TARGET)
    output_dir = tmp_path / "evaluation"
    original_write_bytes = Path.write_bytes
    call_count = 0

    def fail_second_write(path, payload):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected second staging write failure")
        return original_write_bytes(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail_second_write)

    with pytest.raises(OSError, match="injected second staging write failure"):
        write_evaluation_bundle(result, output_dir)

    assert not output_dir.exists()
    assert list(tmp_path.glob(".evaluation.rr03-*")) == []


def test_bundle_preserves_unrelated_output_directory_contents(tmp_path):
    result = evaluate_baselines(mature_table(), TARGET)
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    marker = output_dir / "keep-me.txt"
    marker.write_text("unrelated\n", encoding="utf-8")

    paths = write_evaluation_bundle(result, output_dir)

    assert marker.read_text(encoding="utf-8") == "unrelated\n"
    assert all(path.exists() for path in paths.values())
    assert list(tmp_path.glob(".evaluation.rr03-*")) == []


def test_duplicate_identity_and_unseen_test_position_fail_closed():
    duplicate = mature_table()
    duplicate.loc[1, "canonical_id"] = duplicate.loc[0, "canonical_id"]
    with pytest.raises(EvaluationContractError, match="must be unique"):
        evaluate_baselines(duplicate, TARGET)

    duplicate_pick = mature_table()
    duplicate_pick.loc[1, "overall_pick"] = duplicate_pick.loc[0, "overall_pick"]
    with pytest.raises(EvaluationContractError, match="duplicate draft-season"):
        evaluate_baselines(duplicate_pick, TARGET)

    unseen = mature_table()
    unseen.loc[unseen["draft_season"].eq(2020), "position"] = "TE"
    with pytest.raises(EvaluationContractError, match="no historical training rows"):
        evaluate_baselines(unseen, TARGET)
