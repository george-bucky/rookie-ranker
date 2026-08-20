"""Command-line entry point for deterministic RR-03 baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .evaluation import EvaluationContractError, evaluate_baselines, write_evaluation_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen temporal rookie baselines"
    )
    parser.add_argument("--input", required=True, type=Path, help="Mature RR-02 training CSV")
    parser.add_argument(
        "--target",
        required=True,
        choices=("three_year_ppr_points", "rookie_year_ppr_points"),
    )
    parser.add_argument("--min-training-classes", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for evaluation.json, predictions.csv, and per-year.csv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = (
        args.output_dir / "evaluation.json",
        args.output_dir / "predictions.csv",
        args.output_dir / "per-year.csv",
    )
    resolved_input = args.input.resolve()
    if resolved_input in {path.resolve() for path in output_paths}:
        raise EvaluationContractError("input path must not equal an evaluation output path")
    table = pd.read_csv(args.input)
    result = evaluate_baselines(
        table,
        args.target,
        min_training_classes=args.min_training_classes,
    )
    write_evaluation_bundle(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
