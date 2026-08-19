# RR-02 data sources and handling

RR-02 builds a derived, auditable training table. It does not commit raw API
responses, credentials, serialized models, or generated data under `data/`.

## nflverse

- Draft cohort: nflverse draft picks, loaded for explicit `--draft-years`.
- Outcomes: nflverse regular-season player statistics, loaded with
  `summary_level="reg"` through the explicit `--outcomes-through-year`.
- Maturity guard: `--outcomes-through-year` must be strictly earlier than the
  current calendar year. The in-progress current regular season can never be
  declared complete.
- Identity: `gsis_id` is preferred, followed by `pfr_player_id`, then the
  deterministic draft-season and overall-pick fallback.
- Target: `fantasy_points_ppr`; postseason rows are excluded.
- Attribution: data is provided by the
  [nflverse project](https://github.com/nflverse/nflverse-data) under CC BY 4.0.

The run manifest records the nflreadpy version, requested years, fetch time,
row count, normalized content hash, columns, and schema hash.

Production provenance is emitted only from a clean Git tree. Tracked or
untracked changes make the build fail before it records the producer commit.

## CollegeFootballData

- Source: the CollegeFootballData player-season statistics endpoint, queried
  only for explicit `--college-years`.
- Authentication: `CFD_API_KEY` is read from the environment for a live fetch.
  It is never written to an output or manifest.
- Publication boundary: only derived final-active-season and career features,
  identity decisions, and coverage summaries are published. Raw responses are
  not redistributed.
- Terms: use remains subject to the current
  [CollegeFootballData terms and documentation](https://collegefootballdata.com/).

The run manifest records the endpoint, requested years, fetch time, row count,
normalized content hash, columns, and schema hash. If the provider does not
publish a version identifier, the normalized content SHA-256 is also used as
the immutable run-level source version rather than inventing a release number.

## Identity review

Automatic college matching is limited to one unique normalized name with the
same QB, RB, WR, or TE position and only pre-draft college seasons. Fuzzy
matching is not used. Missing, duplicated, or incompatible matches require a
tracked row in `config/identity_overrides.csv`:

```text
draft_season,overall_pick,resolution,cfbd_player_id,reason,evidence
```

`resolution` is `match` or `quarantine`. A match requires a CFBD player ID,
reason, and evidence. A quarantine leaves the CFBD player ID blank and records
why the identity remains unresolved.

## Reproducing a build

```bash
python -m rookie_ranker.data_pipeline build \
  --draft-years 2015:2026 \
  --college-years 2012:2025 \
  --outcomes-through-year 2025 \
  --http-timeout-seconds 30 \
  --identity-overrides config/identity_overrides.csv \
  --output-dir data/processed/rr02
```

For an offline or previously fetched run, pass all three together:

```text
--draft-input PATH
--player-stats-input PATH
--college-input PATH
```

The output directory contains `training-table.csv`,
`identity-quarantine.csv`, `coverage.csv`, and `run-manifest.json`.
