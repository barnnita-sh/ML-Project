"""Splits, feature lists and preprocessing shared by all models."""
import numpy as np
import pandas as pd

from .. import config
from ..features import FEATURES
from ..storage import load_day_table, save_table

# ---------------------------------------------------------------- date splits (inclusive day strings)
WARMUP_END = "20250128"
SPLITS = {"train": ("20250129", "20250915"), "val": ("20250916", "20251109"), "test": ("20251110", "20251231")}
EXCLUDED_DAYS = {"20250829"}                       # low-coverage day
# forward-chaining folds inside training, used only for tuning
FOLDS = [(("20250129", "20250615"), ("20250616", "20250731")),
         (("20250129", "20250731"), ("20250801", "20250915"))]

MODEL_INPUT = "model_input"

# ---------------------------------------------------------------- features
FEATURES_A = list(FEATURES)
FEATURES_B = list(FEATURES) + ["minutes_empty"]

GROUPS = {   # feature groups for the E4 ablation, added in this order
    "calendar_place": ["hour_of_day", "day_of_week", "is_weekend", "is_holiday", "month", "hex_lat", "hex_lon"],
    "recent_demand": ["pickups_hex_1h", "pickups_nbr_1h", "pickups_nbr_3h", "pickups_nbr_lag24", "pickups_nbr_lag168"],
    "competition": ["vacant_hex_1h", "vacant_nbr_1h", "vacant_per_pickup_nbr_1h", "vacant_nbr_3h_mean"],
    "traffic": ["speed_nbr_1h"],
    "recent_waits": ["median_wait_nbr_3h", "n_waits_nbr_3h"],
    "history": ["hist_pickups_hex_4w", "hist_pickups_nbr_4w", "hist_vacant_nbr_4w", "hist_wait_nbr_4w",
                "n_hist_waits_nbr_4w"],
    "weather": ["rain_lag_1h"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES)


def split_of(day: str):
    if day <= WARMUP_END or day in EXCLUDED_DAYS:
        return None
    for name, (a, b) in SPLITS.items():
        if a <= day <= b:
            return name
    return None


def day_end_t(day: str) -> int:
    return int((pd.Timestamp(day) + pd.Timedelta(days=1)).value // 10**9)


def _keep_fraction(keys: pd.Series, frac: float, salt: str) -> np.ndarray:
    """Deterministic sample: the same key is always in or out (no random state to remember)."""
    if frac >= 1:
        return np.ones(len(keys), bool)
    h = pd.util.hash_pandas_object((keys.astype(str) + salt), index=False).to_numpy()
    return (h % 1_000_000) < int(frac * 1_000_000)


def prepare(frac_a: dict, frac_b: dict, days=None, verbose=True):
    """Build model_input/{modelA,modelB}_{train,val,test} from the feature tables."""
    feat = config.OUT_DIR / "features"
    days = days or sorted(f.stem for f in (feat / "snapshots").glob("*") if f.is_file())
    parts = {(m, s): [] for m in ("modelA", "modelB") for s in SPLITS}
    info = {s: {"days": 0} for s in SPLITS}
    for day in days:
        s = split_of(day)
        if s is None:
            continue
        split_end = day_end_t(SPLITS[s][1])
        info[s]["days"] += 1
        b = load_day_table(feat / "snapshots", day)
        b = b[_keep_fraction(b["episode_id"], frac_b[s], "b")]
        # purge: drop rows whose outcome window crosses into the next split (test keeps all)
        if s != "test":
            b = b[b["snapshot_t"] + b["minutes_until_end"] * 60 <= split_end]
        parts[("modelB", s)].append(b.assign(day=day))
        a = load_day_table(feat / "modelA", day)
        a = a[_keep_fraction(a["hex"].astype(str) + a["hour_id"].astype(str), frac_a[s], "a")]
        parts[("modelA", s)].append(a.assign(day=day))
        if verbose:
            print(day, s, len(a), "A rows,", len(b), "B rows")
    out = config.OUT_DIR / MODEL_INPUT
    for (m, s), p in parts.items():
        df = pd.concat(p, ignore_index=True)
        save_table(df, out, f"{m}_{s}")
        info[s][m] = len(df)
    return info


def load_split(model: str, split: str) -> pd.DataFrame:
    df = load_day_table(config.OUT_DIR / MODEL_INPUT, f"{model}_{split}")
    if df is None:
        raise FileNotFoundError(f"run scripts/prepare_model_data.py first ({model}_{split} missing)")
    return df


def in_days(df, a, b):
    return (df["day"] >= a) & (df["day"] <= b)
