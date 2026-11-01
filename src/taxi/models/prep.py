"""Preprocessing for the linear models (Poisson regression, Cox).

Tree models use the raw features (they handle missing values and non-linear shapes
themselves). Linear models need help:
  * counts are log1p-transformed (diminishing effect of each extra pickup or taxi)
  * hour of day and month become sine/cosine pairs (23:00 is next to 00:00)
  * day of week becomes 0/1 columns
  * minutes_empty (Model B) becomes bands plus log1p, so the U-shape can be learned
  * missing medians get a 0/1 "was missing" column and are filled with the training median
  * everything is standardised with training means and standard deviations
"""
import numpy as np
import pandas as pd

LOG_COLS_KEYS = ("pickups", "vacant", "n_waits", "n_hist")
MISSING_FLAG_COLS = ["median_wait_nbr_3h", "hist_wait_nbr_4w", "speed_nbr_1h"]
MINUTE_BANDS = [0, 5, 15, 30, 60, 10_000]


class LinearPrep:
    def __init__(self, features):
        self.features = list(features)

    def _raw(self, df):
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            v = df[c].astype("float64")
            if c == "hour_of_day":
                for k in (1, 2):
                    X[f"hour_sin{k}"] = np.sin(2 * np.pi * k * v / 24)
                    X[f"hour_cos{k}"] = np.cos(2 * np.pi * k * v / 24)
            elif c == "month":
                X["month_sin"] = np.sin(2 * np.pi * v / 12)
                X["month_cos"] = np.cos(2 * np.pi * v / 12)
            elif c == "day_of_week":
                for d in range(1, 7):
                    X[f"dow_{d}"] = (v == d).astype(float)
            elif c == "minutes_empty":
                X["log_minutes_empty"] = np.log1p(v)
                for lo, hi in zip(MINUTE_BANDS[1:-1], MINUTE_BANDS[2:]):
                    X[f"minutes_empty_{lo}_{hi}"] = ((v >= lo) & (v < hi)).astype(float)
            elif any(k in c for k in LOG_COLS_KEYS):
                X[c] = np.log1p(v.clip(lower=0))
            else:
                X[c] = v
            if c in MISSING_FLAG_COLS:
                X[f"{c}_missing"] = v.isna().astype(float)
        return X

    def fit(self, df):
        X = self._raw(df)
        self.medians = X.median()
        X = X.fillna(self.medians)
        self.mean = X.mean()
        self.std = X.std().replace(0, 1).fillna(1)
        self.columns = list(X.columns)
        return self

    def transform(self, df) -> np.ndarray:
        X = self._raw(df)[self.columns].fillna(self.medians)
        return ((X - self.mean) / self.std).to_numpy(np.float64)
