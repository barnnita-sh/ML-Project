"""Load one day of taxi GPS (plus a look-ahead into the next day) and clean it.

Input: eda_output_bangkok/taxi_days/YYYYMMDD.pkl, made by the EDA notebook.
Those files are already filtered to: valid GPS, greater Bangkok, records on
that date, and vehicles with >= MIN_SWITCHES light changes that day.
Columns: vid, t, lat, lon, speed, occ (1 = passenger on board, light off), acc (1 = engine on).
"""
import numpy as np
import pandas as pd

from . import config
from .geo import haversine_km


def day_bounds(day: str):
    """Start and end of a day in the `t` convention (see config.py)."""
    start = int(pd.Timestamp(day).value // 10**9)
    return start, start + 86400


def next_day(day: str) -> str:
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")


def load_day(day: str, days_dir=None) -> pd.DataFrame:
    """One day of records plus the first LOOKAHEAD_SEC of the next day."""
    days_dir = days_dir or config.TAXI_DAYS_DIR
    d = pd.read_pickle(days_dir / f"{day}.pkl")
    _, day_end = day_bounds(day)
    nxt = days_dir / f"{next_day(day)}.pkl"
    if nxt.exists():
        n = pd.read_pickle(nxt)
        d = pd.concat([d, n[n["t"] < day_end + config.LOOKAHEAD_SEC]], ignore_index=True)
    d.attrs["has_next_day"] = nxt.exists()
    return d


def clean(d: pd.DataFrame):
    """Duplicate timestamps, speed jumps and light flicker. Returns (clean_df, stats)."""
    stats = {}
    d = d.sort_values(["vid", "t"]).drop_duplicates(["vid", "t"], keep="first").reset_index(drop=True)
    stats["rows_in"] = len(d)

    # speed jumps: implied speed from the previous record above MAX_SPEED_KMH
    same = d["vid"].eq(d["vid"].shift()).to_numpy()
    dt = d["t"].diff().to_numpy(dtype=float)
    dist = haversine_km(d["lat"].shift(), d["lon"].shift(), d["lat"], d["lon"])
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.where(same & (dt > 0), dist / (dt / 3600), np.nan)
    jump = implied > config.MAX_SPEED_KMH
    stats["speed_jumps_removed"] = int(jump.sum())
    d = d[~jump].reset_index(drop=True)

    # flicker: one record whose light differs from both neighbours, which agree
    sp = d["vid"].eq(d["vid"].shift())
    sn = d["vid"].eq(d["vid"].shift(-1))
    prev, nxt = d["occ"].shift(), d["occ"].shift(-1)
    flick = sp & sn & d["occ"].ne(prev) & prev.eq(nxt)
    d["occ_raw"] = d["occ"]                     # keep the original for the audit
    d.loc[flick, "occ"] = prev[flick].astype("int8")
    stats["flicker_fixed"] = int(flick.sum())
    stats["rows_out"] = len(d)
    return d, stats
