"""Timing (leakage) and correctness tests for the feature builder.
Run:  python tests/test_features.py   (or  python -m pytest tests)

Uses the stand-in grid's integer hexagon IDs, so it runs with or without h3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi.features import DATA_START_HOUR, FeatureWindow  # noqa: E402

H = 50005000                 # a hexagon (stand-in grid ID)
NB = H + 10000               # one of its 6 neighbours
FAR = H + 30000              # three hexagons away: not a neighbour
D0 = DATA_START_HOUR + 24 * 40   # a day well after the data starts (hour of 00:00)
H1 = D0 + 14                 # current hour: 14:00-15:00, so k = 13:00-14:00


def window(rows, waits=None):
    """rows: list of (hex, hour_id, vacant, pickups); waits: list of (hex, end hour, minutes)."""
    hh = pd.DataFrame(rows, columns=["hex", "hour_id", "vacant_taxis", "pickups"])
    hh["speed_med"], hh["n_speed"] = 30.0, 1
    w = pd.DataFrame(waits or [], columns=["hex", "hour_id", "minutes"])
    win = FeatureWindow()
    win.add_day("d", hh, w)
    return win


def feats(win, hexes=(H,), h1=(H1,)):
    return win.features(np.array(hexes, dtype=np.int64), np.array(h1)).iloc[0]


def test_current_hour_is_never_used():
    # lots of pickups and a wait ending DURING the current hour: must not show up
    f = feats(window([(H, H1, 9, 50)], [(H, H1, 3.0)]))
    assert f["pickups_hex_1h"] == 0 and f["pickups_nbr_1h"] == 0 and f["pickups_nbr_3h"] == 0
    assert f["n_waits_nbr_3h"] == 0 and np.isnan(f["median_wait_nbr_3h"])


def test_last_completed_hour_and_neighbourhood():
    f = feats(window([(H, H1 - 1, 4, 2), (NB, H1 - 1, 6, 3), (FAR, H1 - 1, 9, 7)]))
    assert f["pickups_hex_1h"] == 2 and f["vacant_hex_1h"] == 4
    assert f["pickups_nbr_1h"] == 5 and f["vacant_nbr_1h"] == 10      # FAR is not included
    assert f["vacant_per_pickup_nbr_1h"] == 2


def test_three_hour_window():
    rows = [(H, H1 - 1, 1, 1), (H, H1 - 2, 1, 2), (H, H1 - 3, 1, 4), (H, H1 - 4, 1, 8)]
    assert feats(window(rows))["pickups_nbr_3h"] == 7                  # hours k, k-1, k-2 only


def test_recent_waits_only_completed_before_now():
    waits = [(H, H1 - 1, 10.0), (NB, H1 - 3, 20.0), (H, H1 - 4, 99.0), (H, H1, 1.0)]
    f = feats(window([(H, H1 - 1, 1, 1)], waits))
    assert f["n_waits_nbr_3h"] == 2 and f["median_wait_nbr_3h"] == 15.0


def test_lags_and_four_week_history():
    rows = [(H, H1 - 24, 1, 3), (H, H1 - 168, 1, 5)] + [(H, H1 - 168 * w, 1, 4 * w) for w in range(2, 5)]
    f = feats(window(rows))
    assert f["pickups_nbr_lag24"] == 3 and f["pickups_nbr_lag168"] == 5
    assert f["hist_pickups_hex_4w"] == (5 + 8 + 12 + 16) / 4


def test_before_data_start_is_missing_not_zero():
    f = feats(window([(H, DATA_START_HOUR, 1, 1)]), h1=(DATA_START_HOUR + 1,))
    assert f["pickups_hex_1h"] == 1                                     # 1 Jan 00:00-01:00 exists
    assert np.isnan(f["pickups_nbr_lag24"]) and np.isnan(f["hist_pickups_hex_4w"])
    assert np.isnan(f["pickups_nbr_3h"])                                # needs hours before the start


def test_calendar_is_of_current_hour():
    f = feats(window([(H, H1 - 1, 1, 1)]))
    ts = pd.to_datetime(H1 * 3600, unit="s")
    assert f["hour_of_day"] == 14 and f["day_of_week"] == ts.dayofweek and f["month"] == ts.month


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
            n += 1
    print(f"all {n} tests passed")
