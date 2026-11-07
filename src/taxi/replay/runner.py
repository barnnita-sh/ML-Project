"""Validation/test replay runner: reuse existing pipeline tables and frozen models.

Validation first. The test split is gated in scripts/run_replay.py.
Comparison references are SAME-SPLIT observed snapshots (retrospective), with
own episode/vehicle removed by outcomes.py. Do not treat this as causal evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from taxi import config
from taxi.features import FeatureWindow, load_weather
from taxi.models.data import EXCLUDED_DAYS, split_of
from taxi.replay.engine import replay_decision
from taxi.replay.metrics import attach_top3_hits, policy_summary, paired_summary
from taxi.replay.bootstrap import bootstrap_paired_days
from taxi.replay.outcomes import prepare_reference
from taxi.replay.splits import replay_days, load_replay_day, decision_snapshots
from taxi.storage import load_day_table, save_table


def load_frozen_models(models_dir=None):
    root = Path(models_dir) if models_dir else config.OUT_DIR / 'models'
    manifest = root / 'final.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Frozen model selection missing: {manifest}')
    final = json.loads(manifest.read_text())
    for key in ('model_a', 'model_b'):
        if key not in final:
            raise ValueError(f'final.json lacks {key}')
    def load(name):
        path = root / f'{name}.joblib'
        if not path.is_file():
            raise FileNotFoundError(path)
        return joblib.load(path)
    return load(final['model_a']), load(final['model_b']), load('baseline_a'), final


def load_snapshot_days(days, folder=None):
    folder = Path(folder) if folder else config.OUT_DIR / 'features' / 'snapshots'
    parts = []
    for day in days:
        tab = load_day_table(folder, day)
        if tab is not None and len(tab):
            parts.append(tab.assign(day=day))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_training_coverage(train: pd.DataFrame):
    """Training-only counts per area, 3-hour band, minutes-empty band.

    Counts rows (snapshots), not independent episodes. Matching bands mirror
    the existing HistoricalSurvival baseline, but threshold here is 30.
    """
    from taxi.models.baselines import MINUTE_BANDS
    from taxi.replay.outcomes import area_key
    if train.empty:
        raise ValueError('Training snapshots required for coverage')
    needed = {'hex', 'snapshot_t', 'minutes_empty'}
    if needed - set(train.columns):
        raise ValueError(f'Training snapshots lack {sorted(needed - set(train.columns))}')
    areas = area_key(train['hex'].to_numpy())
    hours = ((train['snapshot_t'].to_numpy(dtype=np.int64) // 3600) % 24) // 3
    bands = np.searchsorted(MINUTE_BANDS[1:], train['minutes_empty'].to_numpy(float), side='right')
    keys = pd.DataFrame({'area': areas, 'hour_band': hours, 'empty_band': bands})
    counts = keys.value_counts().to_dict()
    def count(hex_id, arrival_hour, minutes_empty):
        area = area_key([hex_id])[0]
        band = int(np.searchsorted(MINUTE_BANDS[1:], float(minutes_empty), side='right'))
        return int(counts.get((area, (int(arrival_hour) % 24) // 3, band), 0))
    return count


def build_feature_window(day: str, root=None):
    """Rebuild FeatureWindow from existing hexhour and completed-wait tables.

    Uses the project's Builder rather than inventing a second feature pipeline.
    The builder loads D-29..D+1, while FeatureWindow.features() reads only
    completed hours, so the next-day load does not leak future features.
    """
    from scripts.build_features import Builder
    # scripts/ is importable when the project root is on sys.path.
    if root is not None:
        raise ValueError('Custom output roots are not supported for feature building; set TAXI_OUT_DIR')
    builder = Builder()
    builder.ensure_window(day)
    if not builder.win.days:
        raise FileNotFoundError('No hexhour history for FeatureWindow; run build_hexhour.py/build_features.py')
    return builder.win


def run_validation(*, split='val', max_days=None, max_decisions=None,
                   speed_kmh=25.0, seed=42, min_episodes=5,
                   empty_tolerance_min=10.0, output_dir=None):
    """Run replay, save per-policy and per-candidate rows and metric summaries.

    max_days/max_decisions are debugging limits, NOT a final evaluation.
    """
    if split not in ('val', 'test'):
        raise ValueError('split must be val or test')
    if speed_kmh <= 0:
        raise ValueError('speed_kmh must be > 0')
    days = replay_days(split)
    if max_days is not None:
        if max_days < 1:
            raise ValueError('max_days must be positive')
        days = days[:max_days]
    if not days:
        raise FileNotFoundError(f'No {split} snapshot files found under pipeline_output/features/snapshots')
    train_days = sorted({p.stem for p in (config.OUT_DIR / 'features' / 'snapshots').glob('*')
                         if p.suffix in ('.pkl', '.parquet') and split_of(p.stem) == 'train'
                         and p.stem not in EXCLUDED_DAYS})
    train = load_snapshot_days(train_days)
    coverage = build_training_coverage(train)
    available_hexes = sorted(train['hex'].dropna().unique().tolist(), key=str)
    model_a, model_b, heatmap, final = load_frozen_models()
    reference = prepare_reference(load_snapshot_days(days))
    # NOTE: reference from same split is retrospective. It uses observed
    # comparison taxis and is not a live forecasting evaluation.
    policies, candidates = [], []
    processed = 0
    for day in days:
        snapshots = decision_snapshots(load_replay_day(day, split))
        if snapshots.empty:
            continue
        win = build_feature_window(day)
        for decision in snapshots.to_dict('records'):
            p, c = replay_decision(
                decision, available_hexes=available_hexes,
                feature_window=win, model_a=model_a, model_b=model_b,
                heatmap=heatmap, comparison_reference=reference,
                speed_kmh=speed_kmh, min_episodes=min_episodes,
                empty_tolerance_min=empty_tolerance_min,
                seed=seed + processed, training_coverage=coverage,
                return_candidate_outcomes=True)
            policies.append(p)
            candidates.append(c)
            processed += 1
            if max_decisions is not None and processed >= max_decisions:
                break
        if max_decisions is not None and processed >= max_decisions:
            break
    if not policies:
        raise ValueError('No replay decisions found in the selected days')
    policies = pd.concat(policies, ignore_index=True)
    candidates = pd.concat(candidates, ignore_index=True)
    policies = attach_top3_hits(policies, candidates)
    out = Path(output_dir) if output_dir else config.OUT_DIR / 'replay' / ('validation' if split == 'val' else 'test')
    out.mkdir(parents=True, exist_ok=True)
    save_table(policies, out, 'policies')
    save_table(candidates, out, 'candidate_outcomes')
    policy_summary(policies).to_csv(out / 'policy_summary.csv', index=False)
    paired_summary(policies).to_csv(out / 'paired_summary.csv', index=False)
    bootstrap_paired_days(policies).to_csv(out / 'bootstrap_95ci.csv', index=False)
    metadata = {'split': split, 'days': days, 'decisions': processed,
                'max_days': max_days, 'max_decisions': max_decisions,
                'speed_kmh': speed_kmh, 'seed': seed,
                'min_episodes': min_episodes,
                'empty_tolerance_min': empty_tolerance_min,
                'model_selection': final,
                'training_coverage_min': 30,
                'comparison_reference': 'same split, other vehicles/episodes',
                'limitations': 'retrospective comparisons, not causal outcomes; approximate drive times'}
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    return out, processed
