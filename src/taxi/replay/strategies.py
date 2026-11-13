
"""Replay strategies. Every strategy receives the SAME reachable candidates.

Model A and heatmap rank by expected pickups (descending). Model B ranks
by drive-plus-wait (ascending), using historical curves when training
support is insufficient. Stay and random are comparison policies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.replay.scoring import (
    drive_wait_score,
    historical_fallback_curve,
    model_b_curves,
)


def _require_candidates(candidates):
    if candidates.empty or not {'hex', 'drive_min'}.issubset(candidates.columns):
        raise ValueError('Expected nonempty candidate table with hex and drive_min')
    if candidates['hex'].duplicated().any():
        raise ValueError('Candidate hexagons must be unique')


def ranked_strategies(candidates: pd.DataFrame, feature_rows: pd.DataFrame,
                      *, origin_hex, model_a, model_b, heatmap,
                      fallback_b=None, training_coverage=None,
                      snapshot_t=None, min_training_n: int = 30, return_diagnostics: bool = False,
                      seed: int = 42, random_k: int = 3) -> dict[str, list]:
    """Return ordered candidate hex IDs for each strategy.

    Model B:
      - >= min_training_n snapshots: use frozen Model B curve
      - below threshold: use frozen historical survival fallback
      - no supported historical group: exclude from Model B ranking

    Other strategies are unchanged.
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

    if training_coverage is not None:
        if fallback_b is None or snapshot_t is None:
            raise ValueError(
                'fallback_b and snapshot_t are required when training_coverage is used'
            )

    model_b_scores = []
    fallback_used = []
    fallback_levels = []

    for i, candidate in enumerate(candidates.itertuples(index=False)):
        drive_min = float(candidate.drive_min)
        curve = curves[i]

        used_fallback = False
        fallback_level = None
        if training_coverage is not None:
            arrival_hour = int(feature_rows.iloc[i]["hour_of_day"])
            arrival_empty = float(feature_rows.iloc[i]["minutes_empty"])
            training_n = training_coverage(candidate.hex, arrival_hour, arrival_empty)
            if training_n < min_training_n:
                used_fallback = True
                curve, fallback_level = historical_fallback_curve(
                    fallback_b, feature_rows.iloc[i]
                )

        fallback_used.append(used_fallback)
        fallback_levels.append(fallback_level)

        if curve is None:
            model_b_scores.append(np.nan)
        else:
            model_b_scores.append(
                drive_wait_score(curve, drive_min)
            )

    b = candidates.copy()
    b['score_min'] = model_b_scores
    b = b.dropna(subset=['score_min'])
    b = b.sort_values(
        ['score_min', 'drive_min'],
        kind='stable'
    )

    def descending(values):
        # Stable tie-break: lower drive time, then original candidate order.
        order = np.lexsort((
            np.arange(len(values)),
            candidates['drive_min'].to_numpy(),
            -values,
        ))
        return candidates.iloc[order]['hex'].tolist()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[:min(random_k, len(candidates))]

    rankings = {
        'heatmap': descending(h),
        'model_a': descending(a),
        'model_b': b['hex'].tolist(),
        'stay': [origin_hex],
        'random': candidates.iloc[order]['hex'].tolist(),
    }

    if not return_diagnostics:
        return rankings

    diagnostics = pd.DataFrame({
        'hex': candidates['hex'].to_numpy(),
        'fallback_used': fallback_used,
        'fallback_level': fallback_levels,
    })

    return rankings, diagnostics
