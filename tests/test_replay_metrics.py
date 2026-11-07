"""Phase 3 tests; all fixtures are synthetic and independent of real taxi data."""
import numpy as np
import pandas as pd
import pytest

from taxi.replay.metrics import (prepare_results, attach_top3_hits,
                                 policy_summary, paired_differences, paired_summary)
from taxi.replay.bootstrap import bootstrap_paired_days
from taxi.replay.sensitivity import compare_variants


def sample_results():
    rows = []
    for i, day in enumerate(('2025-10-01', '2025-10-02', '2025-10-03')):
        t = int(pd.Timestamp(f'{day} 10:00:00').timestamp())
        for name, score, first in [('heatmap', 20.0 + i, 'a'),
                                   ('model_b', 16.0 + i, 'b'),
                                   ('stay', 25.0 + i, 'c')]:
            rows.append({'episode_id': f'e{i}', 'snapshot_t': t,
                         'strategy': name, 'recommended_hex': first,
                         'top3_hexes': (first,), 'recommended_covered': True,
                         'recommended_score_min': score,
                         'best_observed_score_min': 15.0 + i})
    return pd.DataFrame(rows)


def test_policy_summary_and_regret():
    out = policy_summary(sample_results()).set_index('strategy')
    assert out.loc['model_b', 'coverage_rate'] == 1
    assert out.loc['model_b', 'mean_regret_min'] == pytest.approx(1.0)
    assert np.isnan(out.loc['model_b', 'top3_hit_rate'])  # no candidate table yet


def test_paired_improvement_positive_means_better():
    out = paired_summary(sample_results()).set_index('strategy')
    assert out.loc['model_b', 'mean_improvement_min'] == pytest.approx(4.0)
    assert out.loc['stay', 'mean_improvement_min'] == pytest.approx(-5.0)
    assert out.loc['model_b', 'n_days'] == 3


def test_missing_outcome_not_silently_imputed():
    df = sample_results()
    mask = (df['strategy'] == 'model_b') & (df['episode_id'] == 'e1')
    df.loc[mask, 'recommended_covered'] = False
    df.loc[mask, 'recommended_score_min'] = np.nan
    paired = paired_differences(df)
    assert len(paired[paired['strategy'] == 'model_b']) == 2
    assert policy_summary(df).set_index('strategy').loc['model_b', 'coverage_rate'] == pytest.approx(2 / 3)


def test_top3_hit_uses_candidate_outcomes():
    df = sample_results()
    candidates = []
    for episode, t in df[['episode_id', 'snapshot_t']].drop_duplicates().itertuples(index=False, name=None):
        for hex_id, score in [('a', 20), ('b', 16), ('c', 25)]:
            candidates.append({'episode_id': episode, 'snapshot_t': t,
                               'hex': hex_id, 'score_min': score, 'covered': True})
    attached = attach_top3_hits(df, pd.DataFrame(candidates), tolerance_min=2)
    hits = attached.groupby('strategy')['top3_hit'].mean()
    assert hits['model_b'] == 1
    assert hits['heatmap'] == 0


def test_bootstrap_reproducible_and_day_clustered():
    df = sample_results()
    a = bootstrap_paired_days(df, n_boot=100, seed=7)
    b = bootstrap_paired_days(df, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    row = a.set_index('strategy').loc['model_b']
    assert row['n_days'] == 3
    assert row['ci_lower_min'] == pytest.approx(4)
    assert row['ci_upper_min'] == pytest.approx(4)


def test_bootstrap_single_day_has_no_interval():
    df = sample_results()
    df = df.loc[df['episode_id'] == 'e0']
    out = bootstrap_paired_days(df)
    assert out['ci_lower_min'].isna().all()


def test_duplicate_policy_decisions_rejected():
    df = sample_results()
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_results(pd.concat([df, df.iloc[:1]], ignore_index=True))


def test_sensitivity_labels_variants():
    df = sample_results()
    policy, paired = compare_variants({'main': df, 'gps_5m': df})
    assert set(policy['variant']) == {'main', 'gps_5m'}
    assert set(paired['variant']) == {'main', 'gps_5m'}
