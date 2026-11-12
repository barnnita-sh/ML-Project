"""Fast replay unit tests; no pipeline_output or trained models needed."""

import numpy as np
import pandas as pd
import pytest

from taxi.models.data import split_of
from taxi.replay.splits import decision_snapshots, replay_days
from taxi.replay.candidates import reachable_candidates, candidate_features
from taxi.replay.scoring import drive_wait_score, rank_curves, model_b_curves


def test_split_boundaries_and_bad_day(tmp_path):
    assert split_of("20250915") == "train"
    assert split_of("20250916") == "val"
    assert split_of("20251109") == "val"
    assert split_of("20251110") == "test"
    assert split_of("20250829") is None
    for day in ("20250829", "20250916", "20251109", "20251110"):
        (tmp_path / f"{day}.pkl").touch()
    assert replay_days("val", tmp_path) == ["20250916", "20251109"]
    assert replay_days("test", tmp_path) == ["20251110"]
    with pytest.raises(ValueError):
        replay_days("train", tmp_path)


def test_one_snapshot_per_episode_per_checkpoint():
    df = pd.DataFrame({"episode_id": ["a", "a", "a", "a", "b"],
                       "snapshot_t": [100, 101, 400, 1000, 100],
                       "minutes_empty": [0.2, 0.8, 5.1, 9.8, 15.0],
                       "hex": [1, 1, 1, 1, 2]})
    got = decision_snapshots(df)
    assert set(zip(got.episode_id, got.decision_minute)) == {("a", 0), ("a", 5), ("b", 15)}
    assert got.loc[(got.episode_id == "a") & (got.decision_minute == 0), "snapshot_t"].item() == 100


def test_candidate_current_hex_and_drive_cap():
    # With explicit road-time lookup, current hex is included even if absent from universe.
    times = {(1, 2): 14.0, (1, 3): 16.0}
    got = reachable_candidates(1, [2, 3], speed_kmh=20,
                               travel_time_lookup=lambda a, b: times[(a, b)])
    assert got.hex.tolist() == [1, 2]
    assert got.drive_min.tolist() == [0.0, 14.0]


def test_candidate_feature_hour_and_projected_empty():
    class FakeWindow:
        def __init__(self):
            self.days = {}

        def features(self, hexes, hours):
            assert list(hours) == [100, 100]
            return pd.DataFrame({"hour_of_day": hours % 24})
    c = pd.DataFrame({"hex": ["h1", "h2"], "drive_min": [0.0, 7.0]})
    got = candidate_features(FakeWindow(), c, snapshot_t=100 * 3600 + 3599, minutes_empty=15)
    assert got.minutes_empty.tolist() == [15, 22]


def test_scoring_and_ranking():
    never_pickup = np.ones(182)
    immediate_pickup = np.zeros(182)
    assert drive_wait_score(never_pickup, 0) == 60
    assert drive_wait_score(immediate_pickup, 6) == 6
    assert drive_wait_score(never_pickup, 6) == 60
    candidates = pd.DataFrame({"hex": ["a", "b"], "drive_min": [0.0, 6.0]})
    ranked = rank_curves(candidates, np.stack([never_pickup, immediate_pickup]))
    assert ranked.hex.tolist() == ["b", "a"]


def test_model_b_gbm_adapter():
    class Curve:
        def predict_curve(self, X):
            assert X.columns.tolist() == ["minutes_empty"]
            return np.ones((len(X), 182))
    class Model:
        features = ["minutes_empty"]
        curve = Curve()
    got = model_b_curves(Model(), pd.DataFrame({"minutes_empty": [0, 5]}))
    assert got.shape == (2, 182)
