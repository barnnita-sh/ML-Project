"""Synthetic tests for Phase 2; no taxi outputs or trained joblib files needed."""
import numpy as np
import pandas as pd

from taxi.replay.strategies import ranked_strategies
from taxi.replay.outcomes import prepare_reference, comparison_taxis, estimate_candidate
from taxi.replay.engine import replay_decision


class FakeA:
    def predict(self, df):
        return df['a_score'].to_numpy()


class FakeHeatmap:
    def predict(self, df):
        return df['h_score'].to_numpy()


class FakeB:
    features = ['minutes_empty']
    class Curve:
        def predict_curve(self, df):
            return np.tile(np.ones(182), (len(df), 1))
    curve = Curve()


def test_strategies_rank_and_stay():
    c = pd.DataFrame({'hex': ['x', 'y', 'z'], 'drive_min': [0., 5., 9.]})
    f = c.assign(a_score=[2, 8, 4], h_score=[7, 1, 3], minutes_empty=[0, 5, 9])
    r = ranked_strategies(c, f, origin_hex='x', model_a=FakeA(), model_b=FakeB(),
                          heatmap=FakeHeatmap(), seed=1)
    assert r['model_a'][0] == 'y'
    assert r['heatmap'][0] == 'x'
    assert r['stay'] == ['x']
    assert len(set(r['random'])) == 3
    assert r['model_b'][0] == 'x'  # all waits 60, tie broken by travel time


def make_reference(hex_id):
    return pd.DataFrame({
        'episode_id': ['own', 'p1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'],
        'vid': ['ownv', 'v1', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        'snapshot_t': [3600, 3600, 3601, 3600, 3600, 3600, 3600, 7200],
        'minutes_empty': [5, 5, 5, 5, 5, 5, 5, 5],
        'hex': [hex_id] * 8,
        'minutes_until_end': [3, 3, 3, 3, 3, 3, 3, 3],
        'pickup_seen': [True] * 8,
    })


def test_comparison_excludes_self_and_deduplicates():
    from taxi.geo import to_hex
    h = to_hex([13.75], [100.5])[0]
    r = prepare_reference(make_reference(h))
    peers = comparison_taxis(r, destination_hex=h, arrival_t=3600,
                             arrival_minutes_empty=5, own_episode_id='own', own_vid='ownv')
    assert peers['episode_id'].nunique() == 5
    assert 'own' not in set(peers['episode_id'])
    est = estimate_candidate(r, destination_hex=h, drive_min=0,
                             decision_t=3600, minutes_empty=5,
                             own_episode_id='own', own_vid='ownv')
    assert est['covered'] and est['n_episodes'] == 5
    assert 0 <= est['score_min'] <= 60


def test_not_enough_comparisons():
    from taxi.geo import to_hex
    h = to_hex([13.75], [100.5])[0]
    r = prepare_reference(make_reference(h).iloc[:4])
    est = estimate_candidate(r, destination_hex=h, drive_min=0,
                             decision_t=3600, minutes_empty=5,
                             own_episode_id='own', own_vid='ownv')
    assert est['covered'] is False
    assert est['score_min'] is None


def test_engine_returns_five_policy_rows():
    from taxi.geo import to_hex
    h = to_hex([13.75], [100.5])[0]
    r = prepare_reference(make_reference(h))
    class Window:
        def features(self, hexes, hours):
            return pd.DataFrame({'a_score': [1] * len(hexes),
                                 'h_score': [1] * len(hexes),
                                 'minutes_empty': [0] * len(hexes)})
    d = {'hex': h, 'snapshot_t': 3600, 'minutes_empty': 5,
         'episode_id': 'own', 'vid': 'ownv'}
    out = replay_decision(d, available_hexes=[], feature_window=Window(),
                          model_a=FakeA(), model_b=FakeB(), heatmap=FakeHeatmap(),
                          comparison_reference=r, speed_kmh=20)
    assert set(out['strategy']) == {'heatmap', 'model_a', 'model_b', 'stay', 'random'}
    assert out['recommended_covered'].all()
    assert not out['training_coverage_checked'].any()
