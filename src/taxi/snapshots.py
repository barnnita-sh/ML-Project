"""Model B rows: a snapshot of each empty taxi every SNAPSHOT_STEP_MIN minutes.

For an empty period from t_start to t_end, snapshots are taken at
t_start + 0, 5, 10, ... minutes while the snapshot time is before t_end
(and at most SNAPSHOT_MAX_MIN minutes in). Each snapshot records where the
taxi was at that moment (its last GPS record at or before the snapshot).

Label: minutes_until_end = minutes from the snapshot to t_end.
  pickup_seen = True  -> it is the exact minutes until the pickup
  pickup_seen = False -> censored: the wait was AT LEAST that long
Periods flagged possible_missed_trip are left out (probably not real waiting).
minutes_empty counts from the start of the period: the drop-off, or the end of a
rest for start_type "after_rest".
Features are NOT added here; they come in a separate step.
"""
import numpy as np
import pandas as pd

from . import config
from .geo import to_hex


def build_snapshots(d: pd.DataFrame, empty: pd.DataFrame) -> pd.DataFrame:
    step = config.SNAPSHOT_STEP_MIN * 60
    max_k = config.SNAPSHOT_MAX_MIN // config.SNAPSHOT_STEP_MIN
    d = d.sort_values(["vid", "t"])
    by_vid = {v: (g["t"].to_numpy(), g["lat"].to_numpy(float), g["lon"].to_numpy(float), g["speed"].to_numpy())
              for v, g in d.groupby("vid", sort=False)}

    rows = []
    for ep in empty[~empty["possible_missed_trip"]].itertuples(index=False):
        t, la, lo, sp = by_vid[ep.vid]
        k = 0
        while k <= max_k:
            st = ep.t_start + k * step
            if st >= ep.t_end:
                break
            idx = np.searchsorted(t, st, side="right") - 1
            rows.append((ep.episode_id, ep.vid, ep.start_type, st, k * config.SNAPSHOT_STEP_MIN, la[idx], lo[idx],
                         sp[idx], (ep.t_end - st) / 60, ep.pickup_seen, ep.censor_reason))
            k += 1

    snaps = pd.DataFrame(rows, columns=["episode_id", "vid", "start_type", "snapshot_t", "minutes_empty", "lat",
                                        "lon", "speed", "minutes_until_end", "pickup_seen", "censor_reason"])
    snaps["hex"] = to_hex(snaps["lat"], snaps["lon"])
    snaps["hour"] = (snaps["snapshot_t"] // 3600) % 24        # Bangkok local hour (see config.py)
    return snaps
