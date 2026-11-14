"""Extra baselines, the move margin, hit rate and area/time breakdowns from a SAVED replay run.

  python scripts/replay_extras.py --run-dir pipeline_output/replay/validation_sampled
  python scripts/replay_extras.py --run-dir pipeline_output/replay/test_sampled

Reads only the run's saved outputs (policies and candidate_outcomes) and the frozen
training-period models in pipeline_output/models. It does not load validation or test features,
does not re-run the replay and does not touch the test lock. See src/taxi/replay/posthoc.py for
what each added strategy is and why it can be computed afterwards.

Adds the strategies hist_curve and heatmap_near5 (and model_b_m0..m6 when the run saved Model B
scores per candidate, i.e. runs made with the updated engine). Writes into <run-dir>/extras/:
  policies_with_extras.parquet   saved strategies + added ones, with top3_hit
  policy_summary.csv             coverage, mean minutes, regret, top-3 hit rate per strategy
  bootstrap_vs_<baseline>.csv    paired differences with 95% day-bootstrap intervals
  breakdown.csv                  Model B (and hist_curve) vs heat map and vs stay by area and time band
  margin_sweep.csv               Model B with a move margin vs stay, per margin (if available)
  extras_report.md, checks.json  readable summary and the reproduction checks

The margin is a design choice: choose it on VALIDATION only. On test, report only the margin
already chosen on validation.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'src'))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.replay.bootstrap import bootstrap_paired_days  # noqa: E402
from taxi.replay.metrics import attach_top3_hits, paired_differences, policy_summary  # noqa: E402
from taxi.replay.posthoc import KEYS, decision_groups, derive_strategies  # noqa: E402
from taxi.storage import load_tables  # noqa: E402

NAMES = {'model_b': 'Model B', 'heatmap': 'Heat map', 'model_a': 'Model A ranking', 'stay': 'Stay',
         'random': 'Random', 'hist_curve': 'Historical curve (drive + wait)',
         'heatmap_near5': 'Heat map within 5 min'}


def _tuple(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    if isinstance(v, str):
        return tuple(ast.literal_eval(v)) if v.startswith(('(', '[')) else (v,)
    return ()


def load_run(run_dir: Path):
    daily = run_dir / 'daily'
    pfiles = sorted(daily.glob('*_policies.pkl')) if daily.is_dir() else []
    if pfiles:
        pol = pd.concat([pd.read_pickle(f) for f in pfiles], ignore_index=True)
        cand = pd.concat([pd.read_pickle(f.with_name(f.name.replace('_policies', '_candidates'))) for f in pfiles],
                         ignore_index=True)
    else:
        pol = load_tables_named(run_dir, 'policies')
        cand = load_tables_named(run_dir, 'candidate_outcomes')
    pol['top3_hexes'] = pol['top3_hexes'].map(_tuple)
    for df in (pol, cand):
        df['snapshot_t'] = df['snapshot_t'].astype(np.int64)
        df['episode_id'] = df['episode_id'].astype(str)
    cand['score_min'] = pd.to_numeric(cand['score_min'], errors='coerce')
    cand['covered'] = cand['covered'].astype(str).str.lower().isin(['true', '1']) if cand['covered'].dtype == object \
        else cand['covered'].astype(bool)
    if 'model_b_score' in cand:
        cand['model_b_score'] = pd.to_numeric(cand['model_b_score'], errors='coerce')
    return pol, cand


CHECKPOINTS = np.array([0, 5, 15, 30, 45], dtype=float)


def enrich_older_run(pol, cand, meta):
    """Runs made before the engine saved decision_minute and drive_min (Min's 12 Nov runs) lack
    them. Both are recovered exactly: minutes empty from the same snapshot table the replay read
    (joined on episode_id and snapshot_t), drive minutes with the replay's own formula and settings."""
    from taxi.replay.candidates import driving_minutes
    from taxi.storage import load_day_table
    notes = []
    if 'minutes_empty' not in pol.columns or pol['minutes_empty'].isna().all():
        days = sorted(set(pd.to_datetime(pol['snapshot_t'], unit='s').dt.strftime('%Y%m%d')))
        parts = []
        for day in days:
            t = load_day_table(config.OUT_DIR / 'features' / 'snapshots', day)
            if t is not None and len(t):
                parts.append(t[['episode_id', 'snapshot_t', 'minutes_empty']])
        snaps = pd.concat(parts, ignore_index=True)
        snaps['episode_id'] = snaps['episode_id'].astype(str)
        snaps['snapshot_t'] = snaps['snapshot_t'].astype(np.int64)
        snaps = snaps.drop_duplicates(['episode_id', 'snapshot_t'])
        pol = pol.merge(snaps, on=['episode_id', 'snapshot_t'], how='left', validate='many_to_one')
        if pol['minutes_empty'].isna().any():
            raise ValueError(f"{int(pol['minutes_empty'].isna().sum())} policy rows have no matching snapshot")
        notes.append('minutes empty recovered from the snapshot table')
    pol['minutes_empty'] = pd.to_numeric(pol['minutes_empty'])
    if 'decision_minute' not in pol.columns:
        me = pol['minutes_empty'].to_numpy(float)
        pol['decision_minute'] = CHECKPOINTS[np.abs(me[:, None] - CHECKPOINTS[None, :]).argmin(axis=1)]
        notes.append('decision checkpoint derived from minutes empty')
    pol['decision_minute'] = pd.to_numeric(pol['decision_minute'])
    if 'replay_day' not in pol.columns:
        pol['replay_day'] = pd.to_datetime(pol['snapshot_t'], unit='s').dt.strftime('%Y%m%d')
    if 'drive_min' not in cand.columns:
        speed = float(meta.get('speed_kmh', 25.0))
        origin = pol.drop_duplicates(KEYS).set_index(KEYS)['origin_hex']
        drives = np.empty(len(cand))
        for key, idx in cand.groupby(KEYS, sort=False).indices.items():
            drives[idx] = driving_minutes(origin.loc[key], cand['hex'].to_numpy()[idx], speed_kmh=speed)
        cand['drive_min'] = drives
        notes.append(f'drive minutes recomputed (straight line x 1.35 at {speed:g} km/h, as in the run)')
    return pol, cand, notes


def load_tables_named(folder: Path, name: str) -> pd.DataFrame:
    for suffix, read in (('.parquet', pd.read_parquet), ('.pkl', pd.read_pickle)):
        f = folder / f'{name}{suffix}'
        if f.exists():
            return read(f)
    raise FileNotFoundError(f'{name} not found in {folder}')


def boot(policies, baseline, keep):
    sub = policies[policies['strategy'].isin(keep + [baseline])]
    return bootstrap_paired_days(sub, baseline=baseline)


def fmt_ci(r):
    return f"{r['mean_improvement_min']:+.2f} ({r['ci_lower_min']:+.2f} to {r['ci_upper_min']:+.2f}; n = {int(r['n_paired_decisions']):,})"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args(argv)
    run_dir = a.run_dir if a.run_dir.is_absolute() else PROJECT / a.run_dir
    meta = json.loads((run_dir / 'run_metadata.json').read_text()) if (run_dir / 'run_metadata.json').exists() else {}
    split = meta.get('split', 'unknown')
    out = run_dir / 'extras'
    out.mkdir(exist_ok=True)

    pol, cand = load_run(run_dir)
    pol, cand, notes = enrich_older_run(pol, cand, meta)
    for n in notes:
        print('note:', n, flush=True)
    models = config.OUT_DIR / 'models'
    final = json.loads((models / 'final.json').read_text())
    heatmap = joblib.load(models / f"{final['baseline_a']}.joblib")
    fallback_b = joblib.load(models / f"{final['fallback_b']}.joblib")
    print(f'{split}: {pol[KEYS].drop_duplicates().shape[0]:,} decisions, {len(cand):,} candidate rows', flush=True)

    extra, check = derive_strategies(pol, cand, heatmap=heatmap, fallback_b=fallback_b)
    if 'replay_day' in pol:
        extra = extra.drop(columns='replay_day').merge(pol.drop_duplicates(KEYS)[KEYS + ['replay_day']], on=KEYS)
    has_margin = extra['strategy'].str.startswith('model_b_m').any()
    if has_margin:
        m0 = extra[extra['strategy'] == 'model_b_m0'].set_index(KEYS)['recommended_hex']
        saved_b = pol[pol['strategy'] == 'model_b'].set_index(KEYS)['recommended_hex']
        common = m0.index.intersection(saved_b.index)
        check['model_b_first_pick_reproduced_by_m0'] = int((m0.loc[common] == saved_b.loc[common]).sum())
        check['model_b_decisions_checked'] = int(len(common))
    print('checks:', check, flush=True)

    allp = pd.concat([pol, extra[pol.columns.intersection(extra.columns)]], ignore_index=True)
    allp = attach_top3_hits(allp, cand)
    flat = allp.assign(top3_hexes=allp['top3_hexes'].map(str))
    for c in flat.columns:
        if flat[c].dtype == object:
            flat[c] = flat[c].astype(str)
    try:
        flat.to_parquet(out / 'policies_with_extras.parquet', index=False)
    except ImportError:
        flat.to_pickle(out / 'policies_with_extras.pkl')

    summ = policy_summary(allp)
    summ.to_csv(out / 'policy_summary.csv', index=False)
    main_set = ['model_b', 'hist_curve', 'heatmap_near5', 'model_a', 'stay', 'random', 'heatmap']
    bh = boot(allp, 'heatmap', [s for s in main_set if s != 'heatmap'])
    bs = boot(allp, 'stay', ['model_b', 'hist_curve', 'heatmap_near5'] +
              ([f'model_b_m{m}' for m in range(7)] if has_margin else []))
    bc = boot(allp, 'hist_curve', ['model_b'])
    for df, name in ((bh, 'heatmap'), (bs, 'stay'), (bc, 'hist_curve')):
        df.to_csv(out / f'bootstrap_vs_{name}.csv', index=False)

    # breakdown by origin area and time band, paired, with day-bootstrap intervals
    groups = decision_groups(pol)
    rows = []
    for gcol in ('area', 'time_band'):
        for gval in sorted(groups[gcol].dropna().unique()):
            keys = groups.loc[groups[gcol] == gval, KEYS]
            sub = allp.merge(keys, on=KEYS)
            for baseline, others in (('heatmap', ['model_b', 'hist_curve']), ('stay', ['model_b', 'hist_curve']),
                                     ('hist_curve', ['model_b'])):
                b = boot(sub, baseline, others)
                for r in b.to_dict('records'):
                    rows.append({'group_by': gcol, 'group': gval, 'strategy': r['strategy'], 'vs': baseline,
                                 'mean_improvement_min': r['mean_improvement_min'], 'ci_lower_min': r['ci_lower_min'],
                                 'ci_upper_min': r['ci_upper_min'], 'n_paired_decisions': r['n_paired_decisions'],
                                 'n_days': r['n_days']})
    brk = pd.DataFrame(rows)
    brk.to_csv(out / 'breakdown.csv', index=False)

    sweep = pd.DataFrame()
    if has_margin:
        moved = []
        for m in range(7):
            s = allp[allp['strategy'] == f'model_b_m{m}']
            moved.append({'margin_min': m, 'share_recommending_a_move': float((s['recommended_hex'] != s['origin_hex']).mean())})
        sweep = pd.DataFrame(moved).merge(
            bs[bs['strategy'].str.startswith('model_b_m')].assign(margin_min=lambda d: d['strategy'].str[9:].astype(int)),
            on='margin_min')
        sweep.to_csv(out / 'margin_sweep.csv', index=False)

    check['notes'] = notes
    (out / 'checks.json').write_text(json.dumps(check, indent=2, default=float))

    # ---------------------------------------------------------------- readable report
    L = [f'# Replay extras: {split}', '',
         'Computed from the saved replay outputs; no features were reloaded and the replay was not re-run. '
         'Minutes are estimated minutes until the next passenger (drive + wait), lower is better; '
         'paired differences are positive when the strategy is better than the one it is compared with. '
         'Intervals: 95%, resampling whole days 2,000 times.', '',
         f"Check: the heat map's first pick re-derived from the saved candidates matches the saved run in "
         f"{check['heatmap_first_pick_reproduced']:,} of {check['heatmap_decisions_checked']:,} decisions."]
    if has_margin:
        L += [f"Check: Model B with margin 0 matches the saved Model B pick in "
              f"{check['model_b_first_pick_reproduced_by_m0']:,} of {check['model_b_decisions_checked']:,} decisions."]
    L += ['', '## Every strategy', '', '| Strategy | Evaluable decisions | Coverage | Mean minutes | Mean regret | Top-3 hit rate |',
          '| --- | --- | --- | --- | --- | --- |']
    for r in summ.sort_values('mean_score_min').to_dict('records'):
        L.append(f"| {NAMES.get(r['strategy'], r['strategy'])} | {r['n_evaluable']:,} | {r['coverage_rate']:.1%} | "
                 f"{r['mean_score_min']:.2f} | {r['mean_regret_min']:.2f} | {r['top3_hit_rate']:.1%} |")
    L += ['', '## Paired comparisons', '', '| Comparison | Minutes saved (95% CI; n) |', '| --- | --- |']
    for df, base in ((bh, 'heatmap'), (bs, 'stay'), (bc, 'hist_curve')):
        for r in df.to_dict('records'):
            if r['strategy'].startswith('model_b_m'):
                continue
            L.append(f"| {NAMES.get(r['strategy'], r['strategy'])} vs {NAMES[base]} | {fmt_ci(r)} |")
    L += ['', '## By area and time of day (paired, vs heat map / stay / historical curve)', '',
          '| Group | Comparison | Minutes saved (95% CI; n) |', '| --- | --- | --- |']
    for r in brk.to_dict('records'):
        L.append(f"| {r['group']} | {NAMES.get(r['strategy'], r['strategy'])} vs {NAMES[r['vs']]} | {fmt_ci(r)} |")
    if has_margin:
        L += ['', '## Move margin: Model B moves only if it predicts saving at least m minutes', '',
              '| Margin m | Share of decisions told to move | Minutes saved vs stay (95% CI; n) |', '| --- | --- | --- |']
        for r in sweep.to_dict('records'):
            L.append(f"| {r['margin_min']} | {r['share_recommending_a_move']:.1%} | {fmt_ci(r)} |")
        if split == 'test':
            L += ['', '**Test run: do not choose the margin here. Report only the margin chosen on validation.**']
    (out / 'extras_report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
