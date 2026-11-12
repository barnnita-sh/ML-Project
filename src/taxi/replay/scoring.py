"""60-minute drive-plus-wait scores on the project's minute survival grid."""

import numpy as np
import pandas as pd


def drive_wait_score(survival_curve, drive_min: float, *, horizon: int = 60) -> float:
    """D + sum(S[0:L]), where L = horizon - D, with fractional D rounded up.

    The model uses a one-minute grid; ceil(D) is a conservative discrete
    approximation.  The choice of rounding must be frozen on validation.
    """
    curve = np.asarray(survival_curve, dtype=float)
    if curve.ndim != 1 or len(curve) < horizon:
        raise ValueError("Expected a one-dimensional curve with >= horizon points")
    if not np.all(np.isfinite(curve)) or np.any((curve < 0) | (curve > 1)):
        raise ValueError("Survival probabilities must be finite and in [0, 1]")
    if not np.isfinite(drive_min) or not 0 <= drive_min <= horizon:
        raise ValueError("drive_min must be between 0 and horizon")
    available = max(0, horizon - int(np.ceil(drive_min)))
    return float(drive_min + curve[:available].sum())


def rank_curves(candidates: pd.DataFrame, curves, *, horizon: int = 60) -> pd.DataFrame:
    """Attach Model B scores to candidates; lower is better."""
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[0] != len(candidates):
        raise ValueError("curves must have one row per candidate")
    out = candidates.copy()
    out["score_min"] = [drive_wait_score(curve, d, horizon=horizon)
                        for curve, d in zip(curves, out["drive_min"])]
    return out.sort_values("score_min").reset_index(drop=True)


def model_b_curves(model, candidate_df: pd.DataFrame) -> np.ndarray:
    """Use the ACTUAL existing GBM/Cox wrapper internals for full curves."""
    if hasattr(model, "curve"):  # ModelB_GBM
        return model.curve.predict_curve(candidate_df[model.features])
    if hasattr(model, "est") and hasattr(model, "prep"):  # ModelB_Cox
        return model.est.predict_curve(model.prep.transform(candidate_df))
    raise TypeError("Expected fitted taxi.models.wrappers.ModelB_GBM or ModelB_Cox")



def historical_fallback_curve(model, feature_row):
    """Return a saved historical curve and its fallback level.

    Levels:
        0 = area + hour band + minutes-empty band
        1 = hour band + minutes-empty band
        2 = minutes-empty band only

    Returns (None, None) if no stored group curve is available.
    The citywide curve is intentionally not used.
    """
    from taxi.models.baselines import AREA_DEG, MINUTE_BANDS

    lat = float(feature_row["hex_lat"])
    lon = float(feature_row["hex_lon"])
    hour = int(feature_row["hour_of_day"])
    minutes_empty = float(feature_row["minutes_empty"])

    area = f"{round(lat / AREA_DEG)}_{round(lon / AREA_DEG)}"
    hour_band = hour // 3

    # Match pd.cut(..., right=False) in HistoricalSurvival.
    band = int(np.searchsorted(MINUTE_BANDS[1:], minutes_empty, side="right"))

    keys = [
        f"{area}|{hour_band}|{band}",
        f"{hour_band}|{band}",
        str(band),
    ]

    for level, key in enumerate(keys):
        curve = model.tables[level].get(key)
        if curve is not None:
            return curve, level

    return None, None
