"""Baselines: what a model has to beat.

Model A - historical average: mean pickups for the same hexagon, hour of day and weekday
          in the training period, shrunk towards hexagon x hour, hexagon and city averages
          when a group has few training rows.
Model B - historical survival curves: Kaplan-Meier curve for the same area (about 2.8 km
          square cell), 3-hour band and minutes-empty band; falls back to coarser groups
          when a group has fewer than MIN_N training rows.
No machine learning: just averages and counting.
"""
import numpy as np
import pandas as pd

from .survival import curve_summaries, km_curve

MIN_N = 300
AREA_DEG = 0.025                                  # about 2.8 km
MINUTE_BANDS = [0, 10, 20, 40, 10_000]


class HistoricalAverage:
    """Mean pickups by hexagon x hour x weekday, each level shrunk towards the coarser one
    (hexagon x hour, then hexagon, then the city) so that groups seen only a few times in
    training do not get extreme values. With K pseudo-rows of shrinkage:
    estimate = (sum of pickups + K x coarser estimate) / (rows + K)."""

    K = 5.0
    LEVELS = (["hex"], ["hex", "hour_of_day"], ["hex", "hour_of_day", "day_of_week"])

    def fit(self, df, target="target_pickups"):
        self.city = float(df[target].mean())
        self.tables = []
        for keys in self.LEVELS:
            self.tables.append((keys, df.groupby(keys)[target].agg(["sum", "size"])))
        return self

    def predict(self, df):
        est = np.full(len(df), self.city)
        for keys, tab in self.tables:
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
            g = tab.reindex(idx)
            s, n = g["sum"].fillna(0).to_numpy(), g["size"].fillna(0).to_numpy()
            est = (s + self.K * est) / (n + self.K)
        return est


def _group_keys(df):
    area = (np.round(df["hex_lat"] / AREA_DEG).astype(int).astype(str) + "_"
            + np.round(df["hex_lon"] / AREA_DEG).astype(int).astype(str))
    hour_band = (df["hour_of_day"] // 3).astype(int).astype(str)
    mband = pd.cut(df["minutes_empty"], MINUTE_BANDS, right=False, labels=False).astype(int).astype(str)
    return [area + "|" + hour_band + "|" + mband, hour_band + "|" + mband, mband]


class HistoricalSurvival:
    def fit(self, df):
        self.tables = []
        T, E = df["minutes_until_end"].to_numpy(), df["pickup_seen"].to_numpy(bool)
        for keys in _group_keys(df):
            curves = {}
            for k, idx in pd.Series(np.arange(len(df))).groupby(keys.to_numpy()).indices.items():
                if len(idx) >= MIN_N:
                    curves[k] = km_curve(T[idx], E[idx])
            self.tables.append(curves)
        self.city = km_curve(T, E)
        return self

    def predict_summaries(self, df):
        """Curve summaries per row, using the finest group that has enough data."""
        keys = _group_keys(df)
        chosen = np.full(len(df), None, dtype=object)
        level = np.full(len(df), 3)
        for lev, (k, curves) in enumerate(zip(keys, self.tables)):
            todo = pd.isna(chosen)
            hit = todo & k.isin(curves.keys()).to_numpy()
            chosen[hit] = k.to_numpy()[hit]
            level[hit] = lev
        labels = np.where(level == 3, "city", level.astype(str) + ":" + chosen.astype(str))
        codes, uniques = pd.factorize(labels)
        curves = np.vstack([self.city if u == "city" else self.tables[int(u[0])][u[2:]] for u in uniques])
        out = {name: v[codes] for name, v in curve_summaries(curves).items()}
        out["fallback_level"] = level
        return out
