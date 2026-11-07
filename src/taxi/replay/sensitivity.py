"""Compare metrics across separately rebuilt preprocessing variants.

This module does not rebuild snapshots. Supply independently generated replay
results for each variant after the preprocessing owner has rebuilt its data.
Never tune sensitivity choices on held-out test results.
"""
from __future__ import annotations

import pandas as pd

from taxi.replay.metrics import paired_summary, policy_summary


def compare_variants(variant_results: dict[str, pd.DataFrame], *,
                     baseline: str = 'heatmap') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (policy_metrics, paired_metrics) with variant labels."""
    if not variant_results:
        raise ValueError('Provide at least one variant, e.g. {"main": dataframe}')
    policy_parts = []
    paired_parts = []
    for name, results in variant_results.items():
        if not name or not isinstance(results, pd.DataFrame):
            raise ValueError('Each variant must have a nonempty name and DataFrame')
        policy_parts.append(policy_summary(results).assign(variant=name))
        paired_parts.append(paired_summary(results, baseline=baseline).assign(variant=name))
    return (pd.concat(policy_parts, ignore_index=True),
            pd.concat(paired_parts, ignore_index=True))
