"""Features for Model A (hexagon-hour pickups) and Model B (snapshot waits).

TIMING RULE (no peeking at the future)
--------------------------------------
Every row has a "last completed hour" k and a "current hour" h1 = k + 1.
  * Model B snapshot at time t:  h1 = t // 3600  (the hour t falls in), k = h1 - 1
  * Model A row for hour H:       h1 = H,          k = H - 1
Features only read hexagon-hour data at hours <= k, and completed waits whose end
hour is <= k (so they ended before h1 began, i.e. before the prediction moment).
"Same hour yesterday / last week" means hour h1 - 24 / h1 - 168, which is in the past.

Neighbourhood ("nbr") = the hexagon and its 6 neighbours.
Missing hexagon-hours mean no taxi activity was seen: counts are 0 there.
Hours before the data starts (1 Jan 2025 00:00) give NaN instead of 0.

Feature list: see FEATURES below and docs/features.md.
"""
import numpy as np
import pandas as pd

from .geo import hex_center, neighbours

DATA_START_HOUR = int(pd.Timestamp("2025-01-01").value // 10**9 // 3600)
HISTORY_DAYS = 29          # 4 weeks back + 1 day for the 3-hour window

# Thai public holidays 2025 (main dates), same list as the EDA notebook
HOLIDAYS = {pd.Timestamp(x).date() for x in [
    "2025-01-01", "2025-02-12", "2025-04-07", "2025-04-13", "2025-04-14", "2025-04-15", "2025-04-16",
    "2025-05-01", "2025-05-05", "2025-05-12", "2025-06-03", "2025-07-10", "2025-07-11", "2025-07-28",
    "2025-08-12", "2025-10-13", "2025-10-23", "2025-12-05", "2025-12-10", "2025-12-31"]}

FEATURES = [
    # calendar (of the current hour h1)
    "hour_of_day", "day_of_week", "is_weekend", "is_holiday", "month",
    # place
    "hex_lat", "hex_lon",
    # this hexagon, last completed hour
    "pickups_hex_1h", "vacant_hex_1h",
    # neighbourhood, last completed hour / last 3 hours
    "pickups_nbr_1h", "vacant_nbr_1h", "vacant_per_pickup_nbr_1h", "speed_nbr_1h",
    "pickups_nbr_3h", "vacant_nbr_3h_mean",
    # repeats
    "pickups_nbr_lag24", "pickups_nbr_lag168",
    # 4-week history at the same hour and weekday
    "hist_pickups_hex_4w", "hist_pickups_nbr_4w", "hist_vacant_nbr_4w",
    # completed waits nearby
    "median_wait_nbr_3h", "n_waits_nbr_3h", "hist_wait_nbr_4w", "n_hist_waits_nbr_4w",
    # weather
    "rain_lag_1h",
]


class HexCodes:
    """Stable integer codes for hexagon IDs (strings for H3, integers for the stand-in grid)."""

    def __init__(self):
        self.codes = {}

    def __call__(self, hexes):
        s = pd.Series(np.asarray(hexes))
        new = [h for h in s.unique() if h not in self.codes]
        for h in new:
            self.codes[h] = len(self.codes)
        return s.map(self.codes).to_numpy(np.int64)


def _key(code, hour):
    return np.asarray(code, np.int64) * 1_000_000 + (np.asarray(hour, np.int64) - DATA_START_HOUR + 100_000)


def neighbourhood_sums(hh: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Sum each hexagon-hour into the neighbourhoods it belongs to.
    pairs: (hex, nbr) with hex = neighbourhood centre, nbr = member (itself + 6)."""
    m = hh.merge(pairs, left_on="hex", right_on="nbr", suffixes=("_src", ""))
    m["speed_w"] = m["speed_med"].fillna(0) * m["n_speed"]
    g = m.groupby(["hex", "hour_id"]).agg(pickups=("pickups", "sum"), vacant=("vacant_taxis", "sum"),
                                          speed_w=("speed_w", "sum"), n_speed=("n_speed", "sum")).reset_index()
    g["speed"] = (g["speed_w"] / g["n_speed"]).where(g["n_speed"] > 0)
    return g[["hex", "hour_id", "pickups", "vacant", "speed"]]


def neighbourhood_waits(waits: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Completed waits (hex = where the pickup happened, hour_id = end hour, minutes) copied to
    every neighbourhood that contains that hexagon."""
    m = waits.merge(pairs, left_on="hex", right_on="nbr", suffixes=("_src", ""))
    return m[["hex", "hour_id", "minutes"]]


class FeatureWindow:
    """Holds the recent days of base tables and computes features for query rows."""

    def __init__(self, weather: pd.DataFrame = None):
        self.codes = HexCodes()
        self.pairs = pd.DataFrame(columns=["hex", "nbr"])
        self.known = set()
        self.days = {}          # day -> dict(B=..., N=..., W=...)
        self.rain = None
        if weather is not None:
            self.rain = weather.set_index("hour_id")["precipitation"]

    # ---------------------------------------------------------------- building the window
    def _pairs_for(self, hexes):
        new = [h for h in pd.unique(np.asarray(hexes)) if h not in self.known]
        if new:
            # a member hexagon contributes to its own neighbourhood and its 6 neighbours' ones;
            # the pairs are symmetric, so neighbours(new) gives every centre it belongs to
            p = neighbours(new).rename(columns={"hex": "nbr", "nbr": "hex"})
            self.pairs = pd.concat([self.pairs, p], ignore_index=True)
            self.known.update(new)
        return self.pairs

    def add_day(self, day: str, hexhour: pd.DataFrame, waits: pd.DataFrame):
        """hexhour: base table of that day; waits: completed waits ENDING that day
        (columns hex, hour_id, minutes)."""
        pairs = self._pairs_for(np.r_[hexhour["hex"].to_numpy(), waits["hex"].to_numpy()])
        n = neighbourhood_sums(hexhour, pairs)
        w = neighbourhood_waits(waits, pairs)
        self.days[day] = {
            "B": hexhour.assign(key=_key(self.codes(hexhour["hex"]), hexhour["hour_id"])),
            "N": n.assign(key=_key(self.codes(n["hex"]), n["hour_id"])),
            "W": w.assign(key=_key(self.codes(w["hex"]), w["hour_id"])),
        }

    def drop_before(self, day: str):
        for d in [d for d in self.days if d < day]:
            del self.days[d]

    def _table(self, name):
        return pd.concat([v[name] for _, v in sorted(self.days.items())], ignore_index=True)

    # ---------------------------------------------------------------- queries
    def features(self, hexes, h1) -> pd.DataFrame:
        """Features for query rows: hexagon and current hour h1 (k = h1 - 1)."""
        hexes = np.asarray(hexes)
        h1 = np.asarray(h1, np.int64)
        k = h1 - 1
        code = self.codes(hexes)
        B = self._table("B").set_index("key")
        N = self._table("N").set_index("key")
        W = self._table("W")[["key", "minutes"]]
        out = pd.DataFrame(index=np.arange(len(hexes)))

        def look(tab, col, hour, fill=0.0):
            v = tab[col].reindex(_key(code, hour)).to_numpy(dtype=float)
            v = np.where(np.isnan(v), fill, v)
            return np.where(hour < DATA_START_HOUR, np.nan, v)

        # calendar of the current hour
        dt = pd.to_datetime(h1 * 3600, unit="s")
        out["hour_of_day"] = h1 % 24
        out["day_of_week"] = dt.dayofweek
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["is_holiday"] = pd.Series(dt.date).isin(HOLIDAYS).astype(int).to_numpy()
        out["month"] = dt.month
        lat, lon = hex_center(hexes)
        out["hex_lat"], out["hex_lon"] = lat, lon

        out["pickups_hex_1h"] = look(B, "pickups", k)
        out["vacant_hex_1h"] = look(B, "vacant_taxis", k)
        out["pickups_nbr_1h"] = look(N, "pickups", k)
        out["vacant_nbr_1h"] = look(N, "vacant", k)
        out["vacant_per_pickup_nbr_1h"] = out["vacant_nbr_1h"] / np.maximum(out["pickups_nbr_1h"], 1)
        out["speed_nbr_1h"] = look(N, "speed", k, fill=np.nan)
        p3 = np.vstack([look(N, "pickups", k - i) for i in range(3)])
        v3 = np.vstack([look(N, "vacant", k - i) for i in range(3)])
        out["pickups_nbr_3h"] = p3.sum(axis=0)
        out["vacant_nbr_3h_mean"] = v3.mean(axis=0)
        out["pickups_nbr_lag24"] = look(N, "pickups", h1 - 24)
        out["pickups_nbr_lag168"] = look(N, "pickups", h1 - 168)
        weeks = [h1 - 168 * w for w in range(1, 5)]
        out["hist_pickups_hex_4w"] = np.vstack([look(B, "pickups", x) for x in weeks]).mean(axis=0)
        out["hist_pickups_nbr_4w"] = np.vstack([look(N, "pickups", x) for x in weeks]).mean(axis=0)
        out["hist_vacant_nbr_4w"] = np.vstack([look(N, "vacant", x) for x in weeks]).mean(axis=0)

        def wait_stats(hours_list):
            q = pd.concat([pd.DataFrame({"row": out.index, "key": _key(code, hr)}) for hr in hours_list])
            m = q.merge(W, on="key")
            g = m.groupby("row")["minutes"].agg(["median", "size"])
            med = g["median"].reindex(out.index).to_numpy()
            cnt = g["size"].reindex(out.index).fillna(0).to_numpy()
            start_ok = np.min(np.vstack(hours_list), axis=0) >= DATA_START_HOUR
            return np.where(start_ok, med, np.nan), np.where(start_ok, cnt, np.nan)

        out["median_wait_nbr_3h"], out["n_waits_nbr_3h"] = wait_stats([k, k - 1, k - 2])
        out["hist_wait_nbr_4w"], out["n_hist_waits_nbr_4w"] = wait_stats(weeks)
        out["rain_lag_1h"] = self.rain.reindex(k).to_numpy() if self.rain is not None else np.nan

        return out[FEATURES].astype("float32")


def completed_waits(empty: pd.DataFrame) -> pd.DataFrame:
    """Waits usable as 'recent waits': ended with a real pickup, not a possible missed trip.
    hex = pickup hexagon, hour_id = hour in which the wait ended."""
    e = empty[empty["pickup_seen"] & ~empty["possible_missed_trip"]]
    return pd.DataFrame({"hex": e["hex_end"].to_numpy(), "hour_id": (e["t_end"] // 3600).to_numpy(),
                         "minutes": e["minutes"].to_numpy(np.float32)})


def load_weather(path) -> pd.DataFrame:
    """weather_hourly.csv (Open-Meteo, Bangkok local time) -> hour_id, precipitation."""
    w = pd.read_csv(path)
    # unit-safe: pandas 3 may store parsed times in seconds or microseconds, not nanoseconds
    w["hour_id"] = (pd.to_datetime(w["time"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1)
    return w.groupby("hour_id", as_index=False)["precipitation"].mean()
