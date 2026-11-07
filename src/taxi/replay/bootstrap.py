"""Day-cluster bootstrap confidence intervals for replay policy comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.replay.metrics import paired_differences


def bootstrap_paired_days(results: pd.DataFrame, *, baseline: str = 'heatmap',
                          n_boot: int = 2000, seed: int = 42,
                          confidence: float = 0.95) -> pd.DataFrame:
    """Resample WHOLE calendar days with replacement, then pool decisions.

    Each bootstrap draw samples n_days days, preserving all paired decisions
    from each sampled day. Percentile intervals are descriptive and require
    multiple independent days; a one-day result receives NA bounds.
    """
    if n_boot < 1:
        raise ValueError('n_boot must be >= 1')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1')
    paired = paired_differences(results, baseline=baseline)
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    rows = []
    for strategy, group in paired.groupby('strategy', sort=True):
        by_day = group.groupby('day')['improvement_min'].agg(['sum', 'count'])
        days = len(by_day)
        mean = float(group['improvement_min'].mean())
        if days < 2:
            lower = upper = np.nan
        else:
            draws = rng.integers(0, days, size=(n_boot, days))
            sums = by_day['sum'].to_numpy(dtype=float)
            counts = by_day['count'].to_numpy(dtype=float)
            boot_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
        rows.append({
            'strategy': strategy,
            'baseline': baseline,
            'n_paired_decisions': len(group),
            'n_days': days,
            'mean_improvement_min': mean,
            'ci_lower_min': float(lower),
            'ci_upper_min': float(upper),
            'confidence': confidence,
            'n_boot': n_boot,
            'seed': seed,
        })
    return pd.DataFrame(rows)
