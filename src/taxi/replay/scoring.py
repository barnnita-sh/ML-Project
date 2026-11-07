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
