"""Recommend the 3 best areas for an empty taxi: score = drive time + expected wait (60-minute clock).

  score(b) = D_b + integral from 0 to (60 - D_b) of S_b(u) du      (lower is better, ties -> shorter drive)

S_b is Model B's survival curve for a taxi arriving in hexagon b already empty for (minutes_empty + D_b).
The integral is the sum of the 1-minute curve (same convention as `rmean_60` in models/survival.py),
with the last partial minute weighted.

Everything the service needs is in one folder (see scripts/make_artifacts.py):
  model_b.joblib, baseline_b.joblib, features.parquet, drive.npz, demo_moments.parquet, manifest.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import geo
from .drive import DriveTable
from .features import FEATURES
from .models.baselines import _group_keys

FINISH_MIN = 60            # the finish line of the score
RADIUS_MIN = 15            # candidates: drive time D <= 15 minutes
MAX_MINUTES_EMPTY = 120    # Model B was trained on snapshots up to 120 minutes empty
SPEED_CLIP = (8.0, 45.0)   # km/h used to turn road km into minutes
MIN_FEATURES_OK = 0.5      # a candidate needs at least this share of non-missing features

REASONS = [  # (feature, direction, text): the candidate must be clearly better than the driver's own hexagon
    ("vacant_per_pickup_nbr_1h", "lower", "fewer empty taxis per pickup nearby"),
    ("median_wait_nbr_3h", "lower", "recent waits nearby were shorter"),
    ("hist_pickups_nbr_4w", "higher", "usually more pickups at this hour"),
    ("pickups_nbr_1h", "higher", "more pickups nearby in the last hour"),
]


def to_t(time) -> int:
    """'2025-11-05T14:00' (Bangkok local time) -> seconds in the repo's `t` convention."""
    return int((pd.Timestamp(time).tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))


def expected_wait(S: np.ndarray, drive_min: np.ndarray) -> np.ndarray:
    """Area under each curve from 0 to (60 - D) minutes. S: (n, HMAX+1) on a 1-minute grid."""
    H = np.clip(FINISH_MIN - drive_min, 0, None)
    h = np.floor(H).astype(int)
    csum = np.cumsum(S, axis=1)
    whole = np.where(h > 0, csum[np.arange(len(S)), np.maximum(h - 1, 0)], 0.0)
    return whole + (H - h) * S[np.arange(len(S)), h]


class Recommender:
    def __init__(self, drive, features, model_b, baseline_b, manifest, moments=None):
        self.drive, self.model_b, self.baseline_b = drive, model_b, baseline_b
        self.manifest, self.moments = manifest, moments
        self.features = features.set_index(["hex", "hour_id"]).sort_index()
        self.hours = sorted(features["hour_id"].unique())

    @classmethod
    def load(cls, folder):
        f = Path(folder)
        manifest = json.loads((f / "manifest.json").read_text())
        if manifest.get("grid") and manifest["grid"] != geo.GRID:
            raise RuntimeError(f"artifacts were built with grid '{manifest['grid']}' but this environment uses "
                               f"'{geo.GRID}': install h3 (pip install h3) in the same way as when training")
        feats = pd.read_parquet(f / "features.parquet")
        feats["hex"] = feats["hex"].astype(str)
        mom = f / "demo_moments.parquet"
        return cls(DriveTable.load(f / "drive.npz"), feats, joblib.load(f / "model_b.joblib"),
                   joblib.load(f / "baseline_b.joblib"), manifest,
                   pd.read_parquet(mom) if mom.exists() else None)

    # ------------------------------------------------------------------ helpers
    def _rows(self, hexes, h1):
        ix = pd.MultiIndex.from_arrays([np.asarray(hexes, dtype=str), np.full(len(hexes), h1)])
        pos = self.features.index.get_indexer(ix)
        return pos

    def _confidence(self, X: pd.DataFrame):
        """high / medium / low from the training data behind each candidate (coverage rule).
        local = enough training snapshots in its ~2.8 km cell, 3-hour band and minutes-empty band
                (the same group the historical-curve baseline needs; at least 300 snapshots);
        recent = 5+ completed waits nearby in the last 3 hours."""
        local = _group_keys(X)[0].isin(self.baseline_b.tables[0].keys()).to_numpy()
        recent = (X["n_waits_nbr_3h"].fillna(0) >= 5).to_numpy()
        pts = local.astype(int) + recent.astype(int)
        return np.array(["low", "medium", "high"])[pts]

    def _reason(self, cand: pd.Series, own: pd.Series):
        out = []
        for col, direction, text in REASONS:
            a, b = cand.get(col), own.get(col)
            if pd.isna(a) or pd.isna(b):
                continue
            better = a <= 0.8 * b if direction == "lower" else a >= 1.25 * b and a > 0
            if better:
                out.append(text)
        return ", ".join(out[:2]) if out else "nearby area with a similar expected total time"

    # ------------------------------------------------------------------ main entry
    def recommend(self, *, minutes_empty, time, hex=None, lat=None, lon=None, top_k=3):
        t = to_t(time)
        h1 = t // 3600
        if h1 not in self.hours:
            raise ValueError("no data for that hour; this demo covers "
                             f"{pd.to_datetime(self.hours[0] * 3600, unit='s'):%Y-%m-%d %H:%M} to "
                             f"{pd.to_datetime(self.hours[-1] * 3600, unit='s'):%Y-%m-%d %H:%M} (Bangkok time)")
        if hex is None:
            if lat is None or lon is None:
                raise ValueError("give a hex, or lat and lon")
            hex = geo.to_hex([lat], [lon])[0]
        hex = str(hex)
        nb = self.drive.neighbours(hex)
        if nb is None:
            raise ValueError("that location is outside the covered area")
        dest, km = nb
        pos = self._rows(dest, h1)
        found = pos >= 0
        dest, km, pos = dest[found], km[found], pos[found]
        own_pos = self._rows([hex], h1)[0]
        own = self.features.iloc[own_pos] if own_pos >= 0 else None

        # driving time: road km / recent taxi speed around the driver (km/h of moving taxis, last hour)
        speed = own["speed_nbr_1h"] if own is not None else np.nan
        if pd.isna(speed):
            speed = self.manifest.get("city_speed_kmh", 25.0)
        speed = float(np.clip(speed, *SPEED_CLIP))
        D = km.astype(float) / speed * 60
        ok = D <= RADIUS_MIN
        dest, D, pos = dest[ok], D[ok], pos[ok]

        X = self.features.iloc[pos][FEATURES].copy()
        enough = X.notna().mean(axis=1).to_numpy() >= MIN_FEATURES_OK
        dest, D, X = dest[enough], D[enough], X[enough]
        base = {"driver_hex": hex, "time": str(pd.Timestamp(time)), "minutes_empty": minutes_empty,
                "n_candidates": int(len(dest)), "model_version": self.manifest.get("model_version"),
                "speed_kmh": round(speed, 1)}
        if len(dest) == 0:
            return {**base, "recommendations": [], "message": "not enough data here"}

        X["minutes_empty"] = np.minimum(minutes_empty + D, MAX_MINUTES_EMPTY)
        S = self.model_b.curve.predict_curve(X[self.model_b.features])
        wait = expected_wait(S, D)
        score = D + wait
        conf = self._confidence(X)
        order = np.lexsort((D, score))[:top_k]       # lowest score, ties -> shorter drive
        lat_c, lon_c = X["hex_lat"].to_numpy(float), X["hex_lon"].to_numpy(float)
        recs = []
        for rank, i in enumerate(order, 1):
            recs.append({"rank": rank, "hex": str(dest[i]), "lat": round(float(lat_c[i]), 5),
                         "lon": round(float(lon_c[i]), 5), "drive_min": round(float(D[i]), 1),
                         "expected_wait_min": round(float(wait[i]), 1), "p_pickup_15min": round(float(1 - S[i, 15]), 2),
                         "score": round(float(score[i]), 1), "confidence": str(conf[i]),
                         "stay_here": bool(str(dest[i]) == hex),
                         "reason": "your current area" if str(dest[i]) == hex else
                         (self._reason(X.iloc[i], own) if own is not None else "short expected total time")})
        return {**base, "recommendations": recs}

    def demo_moments(self, n=30, seed=0):
        """Sample decision moments with what actually happened (no vehicle IDs)."""
        if self.moments is None:
            return []
        m = self.moments.sample(min(n, len(self.moments)), random_state=seed).sort_values("snapshot_t")
        m = m.assign(time=pd.to_datetime(m["snapshot_t"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S"))
        return m[["time", "hex", "minutes_empty", "minutes_until_end", "pickup_seen"]].to_dict("records")
