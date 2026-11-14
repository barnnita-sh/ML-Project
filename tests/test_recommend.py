"""Tests for the scoring and recommendation code (made-up data, real classes).
Run:  python tests/test_recommend.py   (must end with: all N tests passed)   or   pytest tests/test_recommend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fake_artifacts import DAY, make_recommender  # noqa: E402

from taxi.recommend import expected_wait  # noqa: E402

REC, HEXES = make_recommender()
CENTRE = str(HEXES[len(HEXES) // 2])


def test_expected_wait_matches_rmean_60():
    S = np.sort(np.random.default_rng(1).random((5, 182)), axis=1)[:, ::-1].copy()
    S[:, 0] = 1
    assert np.allclose(expected_wait(S, np.zeros(5)), S[:, :60].sum(axis=1))
    assert np.allclose(expected_wait(S, np.full(5, 10.0)), S[:, :50].sum(axis=1))
    half = expected_wait(S, np.full(5, 10.5))
    assert np.all((half <= S[:, :50].sum(axis=1)) & (half >= S[:, :49].sum(axis=1)))


def test_top3_shape_and_order():
    r = REC.recommend(hex=CENTRE, time=f"{DAY}T14:20", minutes_empty=22)
    recs = r["recommendations"]
    assert [x["rank"] for x in recs] == [1, 2, 3]
    scores = [x["score"] for x in recs]
    assert scores == sorted(scores)
    for x in recs:
        assert 0 <= x["drive_min"] <= 15
        assert x["score"] >= x["drive_min"] - 0.05          # score = D + a wait that cannot be negative
        assert 0 <= x["p_pickup_15min"] <= 1
        assert x["confidence"] in ("high", "medium", "low")
    assert r["n_candidates"] > 10


def test_staying_has_zero_drive_time():
    r = REC.recommend(hex=CENTRE, time=f"{DAY}T14:20", minutes_empty=22, top_k=500)
    stay = [x for x in r["recommendations"] if x["stay_here"]]
    assert len(stay) == 1 and stay[0]["drive_min"] == 0 and stay[0]["reason"] == "your current area"


def test_latlon_gives_same_answer_as_hex():
    from taxi import geo
    lat, lon = geo.hex_center(np.array([HEXES[len(HEXES) // 2]]))
    a = REC.recommend(lat=float(lat[0]), lon=float(lon[0]), time=f"{DAY}T09:00", minutes_empty=5)
    b = REC.recommend(hex=CENTRE, time=f"{DAY}T09:00", minutes_empty=5)
    assert a["recommendations"] == b["recommendations"]


def test_no_future_data_is_used():
    """Changing the features of the NEXT hour must not change a decision taken in this hour."""
    a = REC.recommend(hex=CENTRE, time=f"{DAY}T14:20", minutes_empty=22)
    h_next = int((pd.Timestamp(f"{DAY}T15:00") - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1))
    f = REC.features.copy()
    m = f.index.get_level_values("hour_id") == h_next
    f.loc[m, "pickups_nbr_1h"] = 999.0
    saved, REC.features = REC.features, f
    try:
        b = REC.recommend(hex=CENTRE, time=f"{DAY}T14:20", minutes_empty=22)
    finally:
        REC.features = saved
    assert a["recommendations"] == b["recommendations"]


def test_errors():
    for kw in ({"hex": "not-a-hex"}, {"hex": CENTRE, "time": "2030-01-01T10:00"}, {}):
        args = {"time": f"{DAY}T10:00", "minutes_empty": 0, **kw}
        try:
            REC.recommend(**args)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


def test_more_minutes_empty_changes_nothing_structurally():
    for e in (0, 5, 15, 30, 45, 120):
        r = REC.recommend(hex=CENTRE, time=f"{DAY}T03:00", minutes_empty=e)
        assert len(r["recommendations"]) == 3


def test_demo_moments_have_no_vehicle_ids():
    m = REC.demo_moments(10)
    assert len(m) == 10 and set(m[0]) == {"time", "hex", "minutes_empty", "minutes_until_end", "pickup_seen"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} tests passed")
