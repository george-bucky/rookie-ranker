import json
from pathlib import Path

import pytest

from rookie_ranker.evaluation import EvaluationContractError
from rookie_ranker.evaluation_cli import build_parser, main


FIXTURE = Path(__file__).parent / "fixtures" / "rr03" / "mature-training-table.csv"


def test_cli_requires_input_target_and_output():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_writes_identical_json_and_csv_bytes_across_runs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = [
        "--input",
        str(FIXTURE),
        "--target",
        "three_year_ppr_points",
    ]

    assert main([*common, "--output-dir", str(first)]) == 0
    assert main([*common, "--output-dir", str(second)]) == 0

    for filename in ("evaluation.json", "predictions.csv", "per-year.csv"):
        first_bytes = (first / filename).read_bytes()
        assert first_bytes == (second / filename).read_bytes()
        assert first_bytes.endswith(b"\n")
        assert b"\r\n" not in first_bytes
    payload = json.loads((first / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["target"] == "three_year_ppr_points"
    assert payload["target_availability_gap"] == 3
    assert payload["reports"]["folds"][0]["train_years"] == [
        2013,
        2014,
        2015,
        2016,
        2017,
    ]
    assert {row["model"] for row in payload["reports"]["per_year"]} == {
        "position_mean",
        "draft_capital",
    }
    assert "NaN" not in (first / "evaluation.json").read_text(encoding="utf-8")
    predictions = (first / "predictions.csv").read_text(encoding="utf-8")
    assert predictions.startswith(
        "test_year,train_years,model,canonical_id,position,overall_pick,actual,prediction\n"
    )
    assert "2020,2013|2014|2015|2016|2017,draft_capital" in predictions
    assert (first / "per-year.csv").read_text(encoding="utf-8").startswith(
        "test_year,model,train_years,train_class_count,test_row_count,"
    )


def test_cli_rejects_immature_input_without_writing_output(tmp_path):
    immature = tmp_path / "immature.csv"
    rows = FIXTURE.read_text(encoding="utf-8").replace(
        "2020-qb-1,2020,QB,1,300,complete",
        "2020-qb-1,2020,QB,1,,immature",
    )
    immature.write_text(rows, encoding="utf-8")
    output = tmp_path / "must-not-exist"

    with pytest.raises(EvaluationContractError, match="invalid draft classes"):
        main(
            [
                "--input",
                str(immature),
                "--target",
                "three_year_ppr_points",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "colliding_name", ["evaluation.json", "predictions.csv", "per-year.csv"]
)
def test_cli_rejects_input_collision_with_every_output(tmp_path, colliding_name):
    input_path = tmp_path / colliding_name
    original = FIXTURE.read_bytes()
    input_path.write_bytes(original)

    with pytest.raises(EvaluationContractError, match="must not equal"):
        main(
            [
                "--input",
                str(input_path),
                "--target",
                "three_year_ppr_points",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert input_path.read_bytes() == original
