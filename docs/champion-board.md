# RR-05 champion selection and board publication

RR-05 adds one deliberately small challenger to the frozen RR-03 evaluation:
a pooled `RandomForestRegressor`. It does not replace or retune the RR-01
prototype. The forest uses seed `20260819` and the parameters checked into
`rookie_ranker/champion_board.py`; model selection never changes them from
held-out results.

## Features and target separation

The challenger uses only position, `overall_pick`, college seasons observed,
final-active-season and career passing/rushing/receiving values, the approved
efficiency and derived values, and explicit observed/missing flags. Conference,
college team, draft year, player identity, combine data, ADP, expert rankings,
and landing spot are not model inputs.

Three-year PPR and rookie-year PPR are fitted and evaluated independently. Each
uses RR-03 expanding draft-class folds, its target-availability gap, and at
least five target-available training classes. No held-out class is used for
training, preprocessing, or its own interval.

Publication also requires `CurrentClassEvidence`: the expected draft class and
exact `(canonical_id, overall_pick, position)` keys from the complete RR-02
draft cohort. Missing, extra, or changed keys, a different class, or duplicate
overall picks fail before fitting or writing. A nonempty subset is never treated
as proof of cohort completeness.

## Frozen champion gate

For each target, Random Forest is selected only when all four conditions pass:

1. Its class-macro NDCG@24 is strictly greater than draft capital.
2. Its NDCG@24 strictly wins at least 60% of defined paired folds. Ties are not
   wins.
3. Its pooled held-out MAE is no more than 105% of draft capital MAE.
4. Every data-integrity, leakage, identity, and interval-validation gate passes.
   These four named results are required exactly; missing or unknown gate names
   are rejected.

A loss or tie publishes draft capital for that target. Mixed champions are
valid. For the personal-use board, overall rank and tier always use the selected
rookie-year PPR P50 only. The three-year forecast remains available as secondary
information and does not change the published order.

## Residual intervals and claims

The 80% interval is a point prediction plus or minus the 80th percentile of
absolute residuals. For a held-out class, only residuals from earlier eligible
folds may be used. Position residuals are used with at least 30 prior rows for
that position; otherwise global residuals are used with at least 60 prior rows.
The interval is unavailable below both thresholds, including the first eligible
fold.

Coverage and mean width include only held-out rows with available intervals.
Unavailable evidence is emitted as `null` in the artifact evaluation summary,
not described as calibrated. A current-class player without enough residual
history receives a collapsed P10/P50/P90 at the point prediction and a
target-specific warning. Missing rookie-year residual history makes first-year
confidence `unavailable`; missing three-year history does not.

## Frozen rank, tier, and confidence rules

- Base rank: selected rookie-year P50 descending, then `canonical_id` ascending.
- Position rank: base-rank order within position.
- Tier: `1 + floor((base_rank - 1) / 12)`; each consecutive block of 12 ranks is
  one tier.
- Confidence: `unavailable` if the rookie-year interval is unavailable;
  otherwise `low` when an identity or college-data warning exists; otherwise
  `high` when the rookie-year interval width is no wider than the current class
  median and `medium` when it is wider. A secondary three-year interval warning
  remains visible but does not lower first-year confidence. Widths and the
  median are calculated after P10 and P90 are rounded to the artifact's
  four-decimal numeric precision, so confidence is reproducible from the
  published values.

`publish_rookie_board` writes only the RR-04 schema, deterministic JSON board,
and external checksum manifest. It does not fetch live data, serialize a model,
or schedule/upload output.
