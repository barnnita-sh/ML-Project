"""Phase 4 unit tests. Synthetic inputs only, no frozen test access."""
import numpy as np
import pandas as pd
import pytest

from taxi.replay.runner import build_training_coverage, load_frozen_models
from taxi.replay.engine import replay_decision
from scripts.run_replay import check_test_guard


def test_training_coverage_counts_correct_band():
    from taxi.geo import to_hex
    h = to_hex([13.75], [100.5])[0]
    t = int(pd.Timestamp('2025-02-01 10:00:00').timestamp())
    df = pd.DataFrame({'hex': [h] * 30,
                       'snapshot_t': [t] * 30,
                       'minutes_empty': [5.0] * 30})
    f = build_training_coverage(df)
    assert f(h, t // 3600, 5.0) == 30
    assert f(h, t // 3600, 35.0) == 0


def test_test_guard_requires_confirm_and_full_run(tmp_path):
    lock = tmp_path / "TEST_RUN_DONE"

    with pytest.raises(ValueError):
        check_test_guard("test", False, None, None, 100, lock)

    with pytest.raises(ValueError):
        check_test_guard("test", True, 1, None, 100, lock)

    with pytest.raises(ValueError):
        check_test_guard("test", True, None, None, None, lock)

    check_test_guard("test", True, None, None, 100, lock)

    lock.write_text("done")

    with pytest.raises(FileExistsError):
        check_test_guard("test", True, None, None, 100, lock)

    check_test_guard("val", False, 1, 1, None, lock)


def test_missing_frozen_manifest_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match='final.json'):
        load_frozen_models(tmp_path)


def test_engine_can_return_candidate_outcomes():
    from taxi.geo import to_hex
    h = to_hex([13.75], [100.5])[0]
    class FakeA:
        def predict(self, df): return np.ones(len(df))
    class FakeB:
        features = ['minutes_empty']
        class Curve:
            def predict_curve(self, df): return np.ones((len(df), 182))
        curve = Curve()
    class Window:
        def __init__(self):
                self.days = {}
                
        def features(self, hexes, hours):
            return pd.DataFrame({
                'minutes_empty': np.zeros(len(hexes)),
                'hour_of_day': np.asarray(hours),
            })
        
    ref = pd.DataFrame({'episode_id': [f'p{i}' for i in range(5)],
                        'vid': [f'v{i}' for i in range(5)],
                        'snapshot_t': [3600] * 5,
                        'minutes_empty': [5] * 5,
                        'hex': [h] * 5,
                        'minutes_until_end': [3] * 5,
                        'pickup_seen': [True] * 5})
    from taxi.replay.outcomes import prepare_reference
    p, c = replay_decision({'hex': h, 'snapshot_t': 3600, 'minutes_empty': 5,
                            'episode_id': 'own', 'vid': 'ownv'},
                           available_hexes=[], feature_window=Window(),
                           model_a=FakeA(), model_b=FakeB(), fallback_b=FakeB(), heatmap=FakeA(),
                           comparison_reference=prepare_reference(ref), speed_kmh=20,
                           training_coverage=lambda *args: 30,
                           return_candidate_outcomes=True)
    assert len(p) == 5 and len(c) == 1
    assert c.iloc[0]['covered']
    assert c.iloc[0]['training_n'] == 30
