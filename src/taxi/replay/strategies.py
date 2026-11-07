"""Replay strategies. Every strategy receives the SAME reachable candidates.

Model A and heatmap rank by expected pickups (descending). Model B ranks
by drive-plus-wait (ascending). Stay and random are comparison policies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.replay.scoring import model_b_curves, rank_curves


def _require_candidates(candidates):
    if candidates.empty or not {'hex', 'drive_min'}.issubset(candidates.columns):
        raise ValueError('Expected nonempty candidate table with hex and drive_min')
    if candidates['hex'].duplicated().any():
        raise ValueError('Candidate hexagons must be unique')


def ranked_strategies(candidates: pd.DataFrame, feature_rows: pd.DataFrame,
                      *, origin_hex, model_a, model_b, heatmap,
                      seed: int = 42, random_k: int = 3) -> dict[str, list]:
    """Return ordered candidate hex IDs for each strategy.

    All feature_rows must be aligned row-for-row with candidates. The first
    random_k random hexes are the random strategy's recommendation list.
    """
    _require_candidates(candidates)
    if len(feature_rows) != len(candidates):
        raise ValueError('Candidate/feature row counts do not match')
    if feature_rows['hex'].astype(str).tolist() != candidates['hex'].astype(str).tolist():
        raise ValueError('Candidate and feature rows are not aligned')
    if origin_hex not in set(candidates['hex']):
        raise ValueError('Current hex must be in candidate list')

    a = np.asarray(model_a.predict(feature_rows), dtype=float)
    h = np.asarray(heatmap.predict(feature_rows), dtype=float)
    if a.shape != (len(candidates),) or h.shape != (len(candidates),):
        raise ValueError('Model A/heatmap must return one score per candidate')
    if not np.isfinite(a).all() or not np.isfinite(h).all():
        raise ValueError('Model A/heatmap returned invalid scores')
    curves = model_b_curves(model_b, feature_rows)
    b = rank_curves(candidates, curves)

    def descending(values):
        # Stable tie-break: lower drive time, then original candidate order.
        order = np.lexsort((np.arange(len(values)), candidates['drive_min'].to_numpy(), -values))
        return candidates.iloc[order]['hex'].tolist()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[:min(random_k, len(candidates))]
    return {
        'heatmap': descending(h),
        'model_a': descending(a),
        'model_b': b['hex'].tolist(),
        'stay': [origin_hex],
        'random': candidates.iloc[order]['hex'].tolist(),
    }
