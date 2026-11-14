"""Post-hoc replay strategies: they must reproduce the saved heat map and Model B picks, and the
move margin must fall back to staying when the predicted saving is too small."""
import numpy as np
import pandas as pd

from taxi.replay.posthoc import derive_strategies


class Heat:
    def predict(self, df):
        return np.where(df["hex"].to_numpy() == "far", 9.0, 1.0)


class Hist:
    tables = [{}, {}, {str(b): np.exp(-np.arange(121) / 20) for b in range(4)}]


def _run(b_scores):
    t = int(pd.Timestamp("2025-11-20 10:00").value // 10**9)
    cand = pd.DataFrame({"episode_id": "e1", "snapshot_t": t, "hex": ["home", "near", "far"],
                         "drive_min": [0.0, 3.0, 12.0], "score_min": [20.0, 15.0, 30.0],
                         "covered": True, "comparison_episodes": 5, "model_b_score": b_scores})
    pol = pd.DataFrame([{"episode_id": "e1", "snapshot_t": t, "origin_hex": "home", "decision_minute": 5.0,
                         "strategy": "heatmap", "recommended_hex": "far", "replay_day": "20251120"}])
    import taxi.replay.posthoc as ph
    ph.hex_center = lambda h: (np.full(len(h), 13.75), np.full(len(h), 100.53))
    return derive_strategies(pol, cand, heatmap=Heat(), fallback_b=Hist())


def test_heatmap_reproduced_and_near_version_excludes_far_areas():
    rows, check = _run([22.0, 18.0, 25.0])
    assert check["heatmap_first_pick_reproduced"] == 1
    near = rows[rows.strategy == "heatmap_near5"].iloc[0]
    assert near.recommended_hex in {"home", "near"}


def test_historical_curve_counts_the_drive():
    rows, _ = _run([22.0, 18.0, 25.0])
    hist = rows[rows.strategy == "hist_curve"].iloc[0]
    assert hist.recommended_hex == "home"          # same curve everywhere, so any drive only costs time


def test_margin_switches_to_stay_only_when_the_saving_is_small():
    rows, _ = _run([22.0, 18.0, 25.0])             # Model B predicts moving saves 4 minutes
    pick = rows.set_index("strategy")["recommended_hex"]
    assert pick["model_b_m0"] == "near"
    assert pick["model_b_m4"] == "near"            # saves exactly 4: at least the margin, so it moves
    assert pick["model_b_m5"] == "home"            # saves less than 5: stays
    assert pick["model_b_m6"] == "home"
