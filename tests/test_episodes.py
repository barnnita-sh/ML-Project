"""Small hand-made traces with known answers.  Run:  python tests/test_episodes.py  (or  python -m pytest tests)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi.clean import clean  # noqa: E402
from taxi.episodes import build_episodes  # noqa: E402
from taxi.snapshots import build_snapshots  # noqa: E402

DAY_END = 10**9
MOVE, PARK = "move", "park"          # driving at 20 km/h, engine on  /  parked, engine off


def trace(parts, gaps=None, step_deg=0.0005, zigzag=False, vid="taxi1"):
    """parts = [(occ, n_records, MOVE|PARK), ...], one record per minute.
    gaps = {record index: extra seconds of silence before it}.
    zigzag = drive back and forth (turn around every 20 records) instead of in a straight line."""
    rows, now, lon = [], 0, 100.50
    for occ, n, mode in parts:
        for _ in range(n):
            now += (gaps or {}).get(len(rows), 0)
            if mode == MOVE:
                lon += step_deg * (-1 if zigzag and (len(rows) // 20) % 2 else 1)
            rows.append((vid, now, 13.75, lon, 20 if mode == MOVE else 0, occ, 1 if mode == MOVE else 0))
            now += 60
    return pd.DataFrame(rows, columns=["vid", "t", "lat", "lon", "speed", "occ", "acc"])


def run(parts, **kw):
    d, stats = clean(trace(parts, **{k: v for k, v in kw.items() if k != "has_next_day"}))
    trips, empty = build_episodes(d, DAY_END, kw.get("has_next_day", True))
    return d, stats, trips, empty


# ---------------------------------------------------------------- basic behaviour
def test_simple_wait_and_snapshots():
    d, _, trips, empty = run([(0, 3, MOVE), (1, 5, MOVE), (0, 20, MOVE), (1, 5, MOVE)])
    assert len(empty) == 1
    ep = empty.iloc[0]
    assert ep["minutes"] == 20 and ep["pickup_seen"] and ep["censor_reason"] == "" and ep["start_type"] == "dropoff"
    snaps = build_snapshots(d, empty)
    assert list(snaps["minutes_empty"]) == [0, 5, 10, 15]
    assert list(snaps["minutes_until_end"]) == [20, 15, 10, 5]
    assert len(trips) == 1 and trips.iloc[0]["minutes"] == 5 and not trips.iloc[0]["is_fake"]


def test_gap_censors():
    _, _, _, empty = run([(1, 5, MOVE), (0, 20, MOVE), (1, 5, MOVE)], gaps={13: 900})
    ep = empty.iloc[0]
    assert not ep["pickup_seen"] and ep["censor_reason"] == "gap" and ep["minutes"] == 7


def test_flicker_is_fixed():
    _, stats, _, empty = run([(1, 5, MOVE), (0, 8, MOVE), (1, 1, MOVE), (0, 8, MOVE), (1, 5, MOVE)])
    assert stats["flicker_fixed"] == 1 and len(empty) == 1 and empty.iloc[0]["minutes"] == 17


def test_no_pickup_end_of_data():
    _, _, _, empty = run([(1, 5, MOVE), (0, 10, MOVE)], has_next_day=False)
    assert empty.iloc[0]["censor_reason"] == "end_of_data"


# ---------------------------------------------------------------- rules added after the audit
def test_fake_pickup_is_censored_off_duty():
    # wait 10 min, then light 'occupied' while parked with the engine off (driver resting)
    _, _, trips, empty = run([(1, 5, MOVE), (0, 10, MOVE), (1, 15, PARK), (0, 3, MOVE)])
    ep = empty.iloc[0]
    assert ep["minutes"] == 10 and not ep["pickup_seen"] and ep["censor_reason"] == "off_duty"
    assert trips.iloc[-1]["is_fake"]
    # the period after the fake trip starts when the driver is available again
    assert empty.iloc[-1]["start_type"] == "after_rest"


def test_pickup_followed_by_parking_is_censored_off_duty():
    # light goes off, then the taxi stays parked with the engine off for 8 min, then drives (audit #4, #13)
    _, _, trips, empty = run([(1, 5, MOVE), (0, 10, MOVE), (1, 8, PARK), (1, 20, MOVE), (0, 3, MOVE)])
    ep = empty.iloc[0]
    assert not ep["pickup_seen"] and ep["censor_reason"] == "off_duty"
    assert not trips.iloc[-1]["is_fake"]          # the later driving part is long enough to be a real trip


def test_slow_cruising_is_not_flagged_as_missed_trip():
    # 150 min cruising at about 22 km/h (55 km): searching, not a hidden trip (audit #27)
    _, _, _, empty = run([(1, 5, MOVE), (0, 150, MOVE), (1, 5, MOVE)], step_deg=0.0034, zigzag=True)
    assert empty.iloc[0]["km_driven"] > 40 and not empty.iloc[0]["possible_missed_trip"]


def test_rest_cuts_period_and_restarts():
    # 5 min driving, 70 min parked engine off, 10 min driving, pickup
    _, _, _, empty = run([(1, 5, MOVE), (0, 5, MOVE), (0, 70, PARK), (0, 10, MOVE), (1, 5, MOVE)])
    assert list(empty["start_type"]) == ["dropoff", "after_rest"]
    first, second = empty.iloc[0], empty.iloc[1]
    assert first["minutes"] == 5 and first["censor_reason"] == "off_duty" and not first["pickup_seen"]
    assert second["pickup_seen"] and second["minutes"] == 10


def test_short_engine_off_wait_is_kept():
    # 39 min parked with the engine off, then a pickup at the same spot: real waiting (audit #37)
    _, _, _, empty = run([(1, 5, MOVE), (0, 39, PARK), (1, 5, MOVE)])
    assert len(empty) == 1 and empty.iloc[0]["pickup_seen"] and empty.iloc[0]["minutes"] == 39


def test_too_long_is_censored_at_180():
    _, _, _, empty = run([(1, 5, MOVE), (0, 200, MOVE), (1, 5, MOVE)], step_deg=0.0001)
    ep = empty.iloc[0]
    assert ep["minutes"] == 180 and ep["censor_reason"] == "too_long" and not ep["pickup_seen"]


def test_long_fast_drive_is_flagged_and_left_out_of_snapshots():
    # 45 min at about 65 km/h while 'empty'
    d, _, _, empty = run([(1, 5, MOVE), (0, 45, MOVE), (1, 5, MOVE)], step_deg=0.01)
    assert empty.iloc[0]["possible_missed_trip"]
    assert len(build_snapshots(d, empty)) == 0


def test_zero_length_period_is_dropped():
    # drop-off is the last record before a long silence (audit #34)
    _, _, _, empty = run([(1, 5, MOVE), (0, 1, MOVE), (1, 5, MOVE)], gaps={6: 1200})
    assert len(empty) == 0


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
            n += 1
    print(f"all {n} tests passed")
