"""Evaluation metrics for taxi replay results.

Input: concatenate the one-row-per-policy DataFrames returned by replay_decision.
The optional candidate_outcomes table supplies candidate-level outcome scores
for top-3 hit rate; the Phase 2 engine does not yet emit that table.

All scores are in minutes; LOWER is better. Missing outcomes stay missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ('heatmap', 'model_a', 'model_b', 'stay', 'random')
KEYS = ('episode_id', 'snapshot_t')
REQUIRED = set(KEYS) | {
    'strategy', 'recommended_hex', 'top3_hexes', 'recommended_covered',
    'recommended_score_min', 'best_observed_score_min',
}


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    """Validate replay rows and derive Bangkok local calendar day.

    The upstream project stores local wall-clock timestamps as epoch-like
    seconds; interpreting these with utc=True preserves the encoded date.
    If an explicit `day` column exists, it is used instead.
    """
    missing = REQUIRED - set(results.columns)
    if missing:
        raise ValueError(f'Missing replay columns: {sorted(missing)}')
    out = results.copy()
    if out.duplicated([*KEYS, 'strategy']).any():
        raise ValueError('Duplicate (episode_id, snapshot_t, strategy) replay rows')
    if 'day' not in out:
        out['day'] = pd.to_datetime(out['snapshot_t'], unit='s', utc=True).dt.strftime('%Y%m%d')
    out['day'] = out['day'].astype(str)
    out['recommended_score_min'] = pd.to_numeric(out['recommended_score_min'], errors='coerce')
    out['best_observed_score_min'] = pd.to_numeric(out['best_observed_score_min'], errors='coerce')
    out['recommended_covered'] = out['recommended_covered'].fillna(False).astype(bool)
    out['evaluable'] = out['recommended_covered'] & out['recommended_score_min'].notna()
    out['regret_min'] = np.where(
        out['evaluable'] & out['best_observed_score_min'].notna(),
        (out['recommended_score_min'] - out['best_observed_score_min']).clip(lower=0),
        np.nan,
    )
    return out


def attach_top3_hits(results: pd.DataFrame, candidate_outcomes: pd.DataFrame,
                     *, tolerance_min: float = 2.0) -> pd.DataFrame:
    """Add top3_hit to each policy row using *all* covered candidate outcomes.

    candidate_outcomes columns: episode_id, snapshot_t, hex, score_min, covered.
    A hit means one of the strategy's top three is within tolerance_min of
    the best covered candidate. Returns NA if no candidate is evaluable or
    none of the strategy's top three has an evaluable outcome.
    This does not treat a missing candidate as a failure or an observed zero.
    """
    needed = set(KEYS) | {'hex', 'score_min', 'covered'}
    if needed - set(candidate_outcomes.columns):
        raise ValueError(f'Missing candidate outcome columns: {sorted(needed - set(candidate_outcomes.columns))}')
    if tolerance_min < 0:
        raise ValueError('tolerance_min must be nonnegative')
    if candidate_outcomes.duplicated([*KEYS, 'hex']).any():
        raise ValueError('Duplicate candidate outcome rows')
    out = results.copy()
    lookup = {}
    for key, group in candidate_outcomes.groupby(list(KEYS), sort=False):
        eligible = group.loc[group['covered'].fillna(False) & pd.to_numeric(group['score_min'], errors='coerce').notna()]
        if eligible.empty:
            lookup[key] = None
        else:
            scores = dict(zip(eligible['hex'], eligible['score_min'].astype(float)))
            lookup[key] = (scores, min(scores.values()))
    hits = []
    for row in out.itertuples(index=False):
        key = (row.episode_id, row.snapshot_t)
        value = lookup.get(key)
        if value is None:
            hits.append(pd.NA)
            continue
        scores, best = value
        ranked = row.top3_hexes
        if isinstance(ranked, str):
            raise ValueError('top3_hexes must be a list or tuple, not a string')
        observed_top3 = [h for h in list(ranked)[:3] if h in scores]
        if not observed_top3:
            hits.append(pd.NA)
        else:
            hits.append(any(scores[h] <= best + tolerance_min + 1e-9
                            for h in observed_top3))
    out['top3_hit'] = pd.array(hits, dtype='boolean')
    return out


def policy_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Unpaired descriptive summary; paired comparison is separate below."""
    df = prepare_results(results)
    if 'top3_hit' not in df:
        df['top3_hit'] = pd.array([pd.NA] * len(df), dtype='boolean')
    rows = []
    for strategy, group in df.groupby('strategy', sort=True):
        valid = group.loc[group['evaluable']]
        hit = group['top3_hit'].astype('boolean')
        rows.append({
            'strategy': strategy,
            'n_decisions': len(group),
            'n_evaluable': len(valid),
            'coverage_rate': float(group['evaluable'].mean()) if len(group) else np.nan,
            'recommendation_availability_rate': float(group['recommendation_available'].fillna(False).astype(bool).mean()) if 'recommendation_available' in group else float(group['recommended_hex'].notna().mean()),
            'outcome_evaluability_rate': float(group['evaluable'].mean()) if len(group) else np.nan,
            'mean_score_min': float(valid['recommended_score_min'].mean()) if len(valid) else np.nan,
            'mean_regret_min': float(valid['regret_min'].mean()) if len(valid) else np.nan,
            'top3_hit_rate': float(hit.mean()) if hit.notna().any() else np.nan,
            'n_top3_evaluable': int(hit.notna().sum()),
        })
    return pd.DataFrame(rows)


def paired_differences(results: pd.DataFrame, *, baseline: str = 'heatmap') -> pd.DataFrame:
    """One row per decision and non-baseline strategy, on a COMMON evaluable set.

    improvement_min = baseline score - strategy score; positive is better.
    Decisions with missing outcome for either policy are excluded.
    """
    df = prepare_results(results)
    base = df.loc[(df['strategy'] == baseline) & df['evaluable'],
                  [*KEYS, 'day', 'recommended_score_min']].rename(
                      columns={'recommended_score_min': 'baseline_score_min'})
    other = df.loc[(df['strategy'] != baseline) & df['evaluable'],
                   [*KEYS, 'day', 'strategy', 'recommended_score_min']].copy()
    paired = other.merge(base, on=[*KEYS, 'day'], how='inner', validate='many_to_one')
    paired['improvement_min'] = paired['baseline_score_min'] - paired['recommended_score_min']
    return paired.sort_values(['day', 'snapshot_t', 'episode_id', 'strategy']).reset_index(drop=True)


def paired_summary(results: pd.DataFrame, *, baseline: str = 'heatmap') -> pd.DataFrame:
    """Paired mean improvements, before confidence intervals."""
    paired = paired_differences(results, baseline=baseline)
    if paired.empty:
        return pd.DataFrame(columns=['strategy', 'n_paired_decisions', 'n_days', 'mean_improvement_min'])
    return (paired.groupby('strategy', as_index=False)
            .agg(n_paired_decisions=('improvement_min', 'size'),
                 n_days=('day', 'nunique'),
                 mean_improvement_min=('improvement_min', 'mean')))
