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
    return (
        load(final['model_a']),
        load(final['model_b']),
        load(final['baseline_a']),
        load(final['fallback_b']),
        final,
    )


def load_snapshot_days(days, folder=None):
    folder = Path(folder) if folder else config.OUT_DIR / 'features' / 'snapshots'
    parts = []
    for day in days:
        tab = load_day_table(folder, day)
        if tab is not None and len(tab):
            parts.append(tab.assign(day=day))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()



def build_training_coverage(train):
    """Count training snapshots in an H3 neighbourhood and matching bands."""
    import h3
    from taxi.models.baselines import MINUTE_BANDS

    hours = (
        (train["snapshot_t"].to_numpy(dtype=np.int64) // 3600) % 24
    ) // 3

    bands = np.searchsorted(
        MINUTE_BANDS[1:],
        train["minutes_empty"].to_numpy(dtype=float),
        side="right",
    )

    keys = pd.DataFrame({
        "hex": train["hex"].astype(str).to_numpy(),
        "hour_band": hours,
        "empty_band": bands,
    })

    counts = keys.value_counts().to_dict()

    def count(hex_id, arrival_hour, minutes_empty):
        hour_band = (int(arrival_hour) % 24) // 3
        empty_band = int(
            np.searchsorted(
                MINUTE_BANDS[1:],
                float(minutes_empty),
                side="right",
            )
        )

        # The candidate itself plus its six adjacent H3 cells.
        neighbourhood = h3.grid_disk(str(hex_id), 1)

        return int(sum(
            counts.get((neighbour, hour_band, empty_band), 0)
            for neighbour in neighbourhood
        ))

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



def sample_decisions_by_checkpoint(snapshots, *, day, per_day, seed):
    """Deterministic, checkpoint-stratified selection; preserve chronological order."""
    if per_day is None or len(snapshots) <= per_day:
        return snapshots
    if per_day < 1:
        raise ValueError('sample_per_day must be positive')
    import hashlib
    day_seed = int.from_bytes(hashlib.sha256(f'{seed}:{day}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(day_seed)
    groups = {int(k): v.index.to_numpy() for k, v in snapshots.groupby('decision_minute', sort=True)}
    keys = sorted(groups)
    # Equal allocation first; redistribute unused slots when a stratum is small.
    quota = {k: min(len(groups[k]), per_day // len(keys)) for k in keys}
    remaining = per_day - sum(quota.values())
    while remaining:
        eligible = [k for k in keys if quota[k] < len(groups[k])]
        if not eligible:
            break
        for k in eligible:
            if remaining == 0:
                break
            quota[k] += 1
            remaining -= 1
    chosen = np.concatenate([rng.choice(groups[k], size=quota[k], replace=False) for k in keys])
    return snapshots.loc[chosen].sort_values(['snapshot_t', 'episode_id', 'decision_minute']).reset_index(drop=True)


def run_validation(*, split='val', max_days=None, max_decisions=None,
                   speed_kmh=25.0, seed=42, min_episodes=5,
                   empty_tolerance_min=10.0, output_dir=None, sample_per_day=None):
    """Run replay, save per-policy and per-candidate rows and metric summaries.

    max_days/max_decisions are debugging limits, NOT a final evaluation.
    """
    if split == 'test' and (sample_per_day != 100 or seed != 42 or speed_kmh != 25.0
                            or empty_tolerance_min != 10.0 or min_episodes != 5
                            or max_days is not None or max_decisions is not None):
        raise ValueError('Test replay must use the frozen 100/day protocol and parameters')
    if sample_per_day is not None and sample_per_day < 1:
        raise ValueError('sample_per_day must be positive')
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
    model_a, model_b, heatmap, fallback_b, final = load_frozen_models()
    reference = prepare_reference(load_snapshot_days(days))
    # NOTE: reference from same split is retrospective. It uses observed
    # comparison taxis and is not a live forecasting evaluation.
    policies, candidates = [], []
    processed = 0
    checkpoint_root = Path(output_dir) if output_dir else config.OUT_DIR / 'replay' / ('validation_sampled' if split == 'val' and sample_per_day is not None else ('test_sampled' if split == 'test' and sample_per_day is not None else ('validation' if split == 'val' else 'test')))
    if split == 'test' and (checkpoint_root / 'TEST_RUN_DONE').exists():
        raise FileExistsError(f'Test evaluation already completed: {checkpoint_root / "TEST_RUN_DONE"}')
    if sample_per_day is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        (checkpoint_root / 'daily').mkdir(exist_ok=True)
    for day in days:
        snapshots = decision_snapshots(load_replay_day(day, split))
        if snapshots.empty:
            continue
        if sample_per_day is not None:
            snapshots = sample_decisions_by_checkpoint(snapshots, day=day, per_day=sample_per_day, seed=seed)
            daily_dir = checkpoint_root / 'daily'
            p_path = daily_dir / f'{day}_policies.pkl'
            c_path = daily_dir / f'{day}_candidates.pkl'
            if p_path.exists() and c_path.exists():
                saved_p, saved_c = pd.read_pickle(p_path), pd.read_pickle(c_path)
                policies.append(saved_p)
                candidates.append(saved_c)
                processed += len(snapshots)
                print(f'{day}: resumed {len(snapshots)} decisions', flush=True)
                continue
        day_p, day_c = [], []
        win = build_feature_window(day)
        for decision in snapshots.to_dict('records'):
            p, c = replay_decision(
                decision, available_hexes=available_hexes,
                feature_window=win, model_a=model_a, model_b=model_b, fallback_b=fallback_b,
                heatmap=heatmap, comparison_reference=reference,
                speed_kmh=speed_kmh, min_episodes=min_episodes,
                empty_tolerance_min=empty_tolerance_min,
                seed=seed + processed, training_coverage=coverage,
                return_candidate_outcomes=True)
            day_p.append(p)
            day_c.append(c)
            processed += 1
            if max_decisions is not None and processed >= max_decisions:
                break
        if day_p:
            day_p = pd.concat(day_p, ignore_index=True)
            day_c = pd.concat(day_c, ignore_index=True)
            day_p['replay_day'] = day
            day_c['replay_day'] = day
            if sample_per_day is not None and max_decisions is None:
                # Atomic replacement prevents partially written daily checkpoints.
                temp_p = p_path.with_suffix('.tmp')
                temp_c = c_path.with_suffix('.tmp')
                day_p.to_pickle(temp_p)
                day_c.to_pickle(temp_c)
                temp_p.replace(p_path)
                temp_c.replace(c_path)
            policies.append(day_p)
            candidates.append(day_c)
            print(f'{day}: completed {len(snapshots)} decisions; total {processed}', flush=True)
        if max_decisions is not None and processed >= max_decisions:
            break
    if not policies:
        raise ValueError('No replay decisions found in the selected days')
    policies = pd.concat(policies, ignore_index=True)
    candidates = pd.concat(candidates, ignore_index=True)
    policies = attach_top3_hits(policies, candidates)
    out = checkpoint_root
    out.mkdir(parents=True, exist_ok=True)
    save_table(policies, out, 'policies')
    save_table(candidates, out, 'candidate_outcomes')
    policy_summary(policies).to_csv(out / 'policy_summary.csv', index=False)
    paired_summary(policies).to_csv(out / 'paired_summary.csv', index=False)
    bootstrap_paired_days(policies).to_csv(out / 'bootstrap_95ci.csv', index=False)
    metadata = {'split': split, 'days': days, 'decisions': processed,
                'max_days': max_days, 'max_decisions': max_decisions,
                'sample_per_day': sample_per_day, 'sample_method': 'deterministic checkpoint-stratified by day' if sample_per_day else None,
                'population_inference_note': 'Sampled estimates; equal checkpoint allocation is not population proportional. Interpret aggregate metrics as sample/checkpoint-balanced, not population-weighted.' if sample_per_day else None,
                'speed_kmh': speed_kmh, 'seed': seed,
                'min_episodes': min_episodes,
                'empty_tolerance_min': empty_tolerance_min,
                'model_selection': final,
                'training_coverage_min': 30,
                'comparison_reference': 'same split, other vehicles/episodes',
                'limitations': 'retrospective comparisons, not causal outcomes; approximate drive times'}
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    return out, processed
