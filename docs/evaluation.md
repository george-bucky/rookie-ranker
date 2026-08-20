# RR-03 temporal evaluation contract

RR-03 freezes the evidence harness that later models must use. It evaluates
transparent baselines only; it does not train a Random Forest, tune features, or
publish a rookie-board artifact.

## Eligible data

Evaluation starts from an RR-02 training table restricted to complete outcome
classes for one target. Every supplied row must have a finite target and the
matching status `complete`. Any immature, missing-identity, or null-target row
rejects its entire input rather than being dropped. Canonical IDs must be unique,
positions must be QB/RB/WR/TE, and overall picks must be positive whole numbers.

The primary target is `three_year_ppr_points`; the secondary target is
`rookie_year_ppr_points`. Each target is evaluated separately.

## Temporal folds and baselines

Draft years are sorted ascending. Evaluation is explicitly as-of the NFL draft in
held-out year Y, when the Y-1 NFL regular season is complete. Therefore rookie-year
training labels may come only from draft classes `<= Y-1`, while three-year labels
may come only from classes `<= Y-3` because class D's window ends after season
`D+2`. A fold is emitted only after at least five target-available training
classes. Later classes with recorded outcomes remain excluded from that fold. The
held-out class never enters fitting or calibration.

Two frozen baselines are fitted independently in every fold:

1. `position_mean`: the mean training target for the player's position.
2. `draft_capital`: ordinary least squares with an intercept, position indicators,
   and `log1p(overall_pick)`.

A test position without any prior training row fails closed.

## Metrics and pooling

- Ranking: linear-gain NDCG@24, NDCG@12, and Spearman correlation.
- Magnitude: MAE and R-squared.
- Decisions: top-12 and top-24 hit recall.
- Per-year results retain each held-out class independently.
- `macro_class_ranking` averages each defined ranking metric across classes, so
  every eligible class has equal weight.
- `pooled_row_magnitude` computes MAE and R-squared once across all held-out player
  rows.
- `position_slices` report class-macro ranking and pooled magnitude by position.

NDCG uses nonnegative relevance (`max(actual PPR, 0)`) with linear gain. A class
with zero ideal gain has undefined NDCG and is excluded only from that metric's
eligible-class denominator. R-squared is undefined for fewer than two rows or a
constant actual target. Spearman uses standard average ranks for equal values.

### Frozen hit and tie rule

For K equal to 12 or 24, the actual hit set is exactly the first `min(K, class
size)` players ordered by actual target descending. Recall is the share of those
IDs found in the first `min(K, class size)` predicted players.

Whenever equal scores cross an ordering boundary, lexicographically smaller
`canonical_id` wins the tie. This neutral rule prevents hidden draft-capital signal
from entering the position-mean baseline. It is used for actual and predicted
top-K membership and NDCG ordering. Spearman retains its standard average-rank tie
semantics.

Strict fold-win counts include only held-out classes where both compared metrics
are defined. A challenger wins only when its value is strictly greater; ties are
reported and count as non-wins.

## Deterministic CLI

Run the harness against a mature-only RR-02 CSV:

```bash
python -m rookie_ranker.evaluation_cli \
  --input data/processed/rr02/mature-training-table.csv \
  --target three_year_ppr_points \
  --output-dir data/processed/rr03/three-year
```

Use `rookie_year_ppr_points` for the separate immediate-impact evaluation. The
output directory contains `evaluation.json`, `predictions.csv`, and
`per-year.csv`. JSON contains folds, held-out predictions, per-year metrics, macro
ranking, pooled magnitude, position slices, and strict baseline win counts. The
CSVs freeze the two most reusable row-level reports. Outputs contain no timestamp
or filesystem path; JSON uses sorted keys and rejects NaN, while CSV uses fixed
ordering, `%.17g` float formatting, Unix newlines, and pipe-delimited year tuples.
Identical inputs and code therefore produce identical bytes. The CLI rejects an
input path that resolves to any of its three output paths before reading or
writing the bundle.

The writer first renders every payload in memory, stages all three complete files
in a sibling temporary directory, and only then promotes the named outputs. A
staging or promotion error cleans the temporary directory and rolls back named
outputs; unrelated files already present in the output directory are preserved.
