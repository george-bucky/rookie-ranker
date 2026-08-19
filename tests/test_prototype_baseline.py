import numpy as np
import pandas as pd

from rookie_ranker.college_data import merge_college_and_nfl
from rookie_ranker.features import NUMERIC_FEATURES, prepare_features
from rookie_ranker.model import build_model, predict
from rookie_ranker.nfl_data import merge_draft_and_fantasy


def feature_rows() -> pd.DataFrame:
    rows = []
    positions = ["QB", "RB", "WR", "TE"]
    for index in range(12):
        position = positions[index % len(positions)]
        rows.append(
            {
                "player": f"Player {index}",
                "position_x": position,
                "conference": "SEC",
                "team": "TEST",
                "passing_ATT": 100 if position == "QB" else 0,
                "passing_YDS": 2000 + index,
                "passing_TD": 15 + index,
                "passing_YPA": 7.0 + index / 10,
                "rushing_YDS": 500 + index,
                "rushing_TD": 5 + index,
                "rushing_CAR": 100 + index,
                "receiving_YDS": 600 + index,
                "receiving_TD": 6 + index,
                "receiving_REC": 50 + index,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_features_preserves_prototype_cleanup_and_derivations():
    raw = feature_rows().iloc[:4].copy()
    raw.loc[0, "passing_ATT"] = 4
    raw.loc[1, "conference"] = "Ivy League"

    prepared = prepare_features(raw)

    assert prepared["position_x"].tolist() == ["QB", "WR", "TE"]
    assert prepared.loc[0, ["passing_YDS", "passing_TD", "passing_YPA"]].tolist() == [0, 0, 0]
    assert prepared.loc[1, ["passing_YDS", "passing_TD", "passing_YPA"]].tolist() == [0, 0, 0]
    assert prepared.loc[2, ["rushing_YDS", "rushing_TD", "rushing_CAR"]].tolist() == [0, 0, 0]
    assert prepared.loc[1, "total_touchdowns"] == 15
    assert prepared.loc[1, "yards_from_scrimmage"] == 1104
    assert "conference" not in prepared.columns


def test_current_nfl_join_uses_matching_player_ids_and_inner_join():
    draft = pd.DataFrame(
        {
            "gsis_id": ["keep", "draft-only"],
            "pfr_player_name": ["Matched Player", "Draft Only"],
        }
    )
    fantasy = pd.DataFrame(
        {"player_id": ["keep", "stats-only"], "fantasy_points": [123.0, 50.0]}
    )

    joined = merge_draft_and_fantasy(draft, fantasy)

    assert joined[["gsis_id", "player_id", "fantasy_points"]].to_dict("records") == [
        {"gsis_id": "keep", "player_id": "keep", "fantasy_points": 123.0}
    ]


def test_current_college_join_matches_names_and_keeps_joined_values(capsys):
    college = pd.DataFrame(
        {
            "player": ["Exact Player", "Other Prospect"],
            "position": ["WR", "RB"],
            "receiving_YDS": [900, 100],
        }
    )
    nfl = pd.DataFrame(
        {"pfr_player_name": ["Exact Player"], "fantasy_points": [88.5]}
    )

    joined = merge_college_and_nfl(college, nfl)

    assert joined.loc[0, "player"] == "Exact Player"
    assert joined.loc[0, "fantasy_points"] == 88.5
    assert "Total matched: 1" in capsys.readouterr().out


def test_prediction_output_shape_metadata_and_descending_order():
    raw = feature_rows()
    training = prepare_features(raw)
    model = build_model()
    model.fit(training, np.arange(len(training), dtype=float))

    rankings = predict(model, raw.iloc[:4].rename(columns={"position_x": "position"}))

    assert rankings.columns.tolist() == [
        "player",
        "position",
        "predicted_fantasy_points",
        "conference",
        "team",
    ]
    assert rankings.shape == (4, 5)
    assert rankings["player"].notna().all()
    assert rankings["predicted_fantasy_points"].is_monotonic_decreasing


def test_tiny_model_smoke_is_deterministic():
    training = prepare_features(feature_rows())
    target = np.arange(len(training), dtype=float)
    first = build_model(NUMERIC_FEATURES).fit(training, target)
    second = build_model(NUMERIC_FEATURES).fit(training, target)

    first_predictions = first.predict(training)
    second_predictions = second.predict(training)

    np.testing.assert_allclose(first_predictions, second_predictions)
    assert first_predictions.shape == (len(training),)
    assert np.isfinite(first_predictions).all()
