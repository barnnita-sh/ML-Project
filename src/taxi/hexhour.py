"""Hexagon-hour base table: what happened in each hexagon during each clock hour.

One row per (hex, hour_id) with any activity during the day:
  vacant_taxis  distinct taxis that were inside a wait (empty and available) there that hour
  pickups       real pickups there that hour (trips with is_fake = False)
  speed_med     median speed of moving taxis (over MOVING_KMH) there that hour, any state
  n_speed       number of moving pings behind speed_med

hour_id = t // 3600 in the `t` convention (Bangkok wall-clock), so hour of day = hour_id % 24.
Waits flagged possible_missed_trip are not counted as empty taxis.
Time inside a rest is not inside a wait, so resting taxis are not counted as empty.
"""
import numpy as np
import pandas as pd

from . import config
from .geo import to_hex


def build_hexhour(d: pd.DataFrame, trips: pd.DataFrame, empty_today: pd.DataFrame,
                  empty_yesterday, day_start: int, day_end: int) -> pd.DataFrame:
    p = d[(d["t"] >= day_start) & (d["t"] < day_end)][["vid", "t", "lat", "lon", "speed"]].copy()
    p["hex"] = to_hex(p["lat"].to_numpy(), p["lon"].to_numpy())
    p["hour_id"] = p["t"] // 3600

    # ---- empty taxis: pings that fall inside a wait (waits can start the day before)
    w = pd.concat([x for x in (empty_yesterday, empty_today) if x is not None], ignore_index=True)
    w = w[~w["possible_missed_trip"]][["vid", "t_start", "t_end"]].sort_values("t_start")
    q = p[["vid", "t", "hex", "hour_id"]].sort_values("t")
    w["vid"] = w["vid"].astype(str).astype(object)          # same key type on both sides
    q["vid"] = q["vid"].astype(str).astype(object)
    m = pd.merge_asof(q, w, left_on="t", right_on="t_start", by="vid", direction="backward")
    inside = m["t_end"].notna() & (m["t"] < m["t_end"])
    vacant = m[inside].groupby(["hex", "hour_id"])["vid"].nunique().rename("vacant_taxis")

    # ---- real pickups
    real = trips[~trips["is_fake"]]
    pk = pd.DataFrame({"hex": real["hex_pickup"].to_numpy(), "hour_id": (real["t_start"] // 3600).to_numpy()})
    pickups = pk.groupby(["hex", "hour_id"]).size().rename("pickups")

    # ---- traffic speed (moving pings, any taxi state)
    mv = p[p["speed"] > config.MOVING_KMH]
    speed = mv.groupby(["hex", "hour_id"])["speed"].agg(speed_med="median", n_speed="size")

    hh = pd.concat([vacant, pickups, speed], axis=1).reset_index()
    for c in ["vacant_taxis", "pickups", "n_speed"]:
        hh[c] = hh[c].fillna(0).astype("int32")
    hh["speed_med"] = hh["speed_med"].astype("float32")
    hh["hour_id"] = hh["hour_id"].astype("int64")
    return hh
