# Rookie board artifact contract

RR-04 defines the static boundary between Rookie Ranker and any recommendation
consumer. Consumers read JSON files; they do not import Rookie Ranker, load a
serialized model, or require training dependencies.

## Handoff files

Each draft class is delivered as exactly three files:

```text
rookie-board.schema.json
rookie-board-<class-year>.json
rookie-board-<class-year>.manifest.json
```

`rookie-board.schema.json` is the public Draft 2020-12 JSON Schema. The board is
the public data artifact. The manifest contains the SHA-256 of the exact board
bytes; the board never contains its own checksum.

The supported schema version is `1.0.0`. Schema versions outside the explicit
allowlist and artifact versions with an unsupported major version fail closed.
Breaking field changes require a new schema version.

## Metadata

The board records:

- Schema, artifact, and model versions.
- Producer Git commit.
- Draft class and `post_draft` mode.
- Explicit NFL draft event date, timezone-aware generation time, and data cutoff.
- Latest complete NFL outcome season and target-specific maturity class.
- PPR scoring basis and both target definitions.
- Training cohorts independently for rookie-year and three-year targets.
- The independently selected champion for each target.
- Prediction and interval capabilities.
- Evaluation summaries for both targets.
- Source, URL, license, notice, and optional source version.

Training cohorts must predate the artifact class and may not exceed that
target's mature class. Rookie-year maturity equals the outcome cutoff season;
three-year maturity is two classes earlier. An outcome season is conservatively
considered complete on February 15 of the following calendar year.

For `post_draft` mode, both generation time and data cutoff must be on or after
the explicit `draft_event_date`, and that event date must fall in the stated
draft class. Evaluation champions must match `champion_by_target`. Missing or
inconsistent provenance is rejected.

## Player rows

Every drafted QB, RB, WR, or TE row contains:

- Canonical ID plus verified GSIS, PFR, CFB, CFBD, Yahoo, or Sleeper IDs when
  available.
- Name, position, NFL team, round, and overall pick.
- Base rank, position rank, and tier.
- Three-year and rookie-year P10, P50, and P90 PPR predictions.
- Confidence, data-quality warnings, and identity match evidence state.
- The champion variant used for each target.

Canonical IDs, overall picks, ranks, and every populated source-ID namespace
must be unique. Quantiles must be finite and ordered `P10 <= P50 <= P90`.
Player target champions must match artifact metadata.

## Deterministic bytes

`write_handoff` validates a `RookieBoard`, sorts semantic sets, orders players
by base rank, rounds floating-point values to at most four decimal places, and
writes compact UTF-8 JSON with one trailing newline. Repeated writes of the
same board produce identical bytes and therefore the same SHA-256.

`load_handoff` verifies, in order:

1. Strict JSON and manifest shape, including rejection of duplicate keys.
2. Supported schema and artifact versions.
3. Exact supported JSON Schema bytes, Draft 2020-12 validity, and artifact
   validation against the supplied schema.
4. Artifact and manifest filenames for the class year.
5. SHA-256 of the exact artifact bytes.
6. Strict Pydantic validation and cross-row invariants.
7. Schema, artifact, class, commit, and generation-time agreement between the
   artifact and manifest.

Unknown fields, non-finite values, wrong years, duplicate IDs, missing
provenance, and byte tampering are rejected rather than partially loaded.

## Python usage

```python
from rookie_ranker.artifact_contract import load_handoff, write_handoff

paths = write_handoff(board, "data/output/rookie_board/2026")
validated = load_handoff(
    paths["artifact"],
    paths["manifest"],
    schema_path=paths["schema"],
)
```

The checked-in golden fixture is synthetic. It contains no raw source response,
credential, serialized model, or claim of real player performance.
