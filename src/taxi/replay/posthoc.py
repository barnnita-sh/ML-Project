"""Extra replay strategies and breakdowns computed from SAVED replay outputs.

Why post hoc. Every decision in a replay run already saved one row per candidate area
(`candidate_outcomes`: hex, drive minutes and the estimated outcome from comparison taxis).
A strategy that only needs training-period tables to rank those candidates can therefore be
added afterwards without touching any validation or test features and without re-running the
replay. The ranking uses exactly what the original strategies had at decision time:

  hist_curve     the historical survival curve (frozen baseline_b, training data only), scored
                 like Model B: drive + expected wait within 60 minutes. Isolates what Model B
                 adds over a lookup table once both count the drive.
  heatmap_near5  the heat map restricted to areas within 5 minutes' drive (the current hexagon
                 always included): a drive-aware version of the heat map.
  model_b_mX     Model B with a "move only if worth it" rule: stay unless the best move's
                 predicted score is at least X minutes lower than staying. Needs the Model B
                 score per candidate (column model_b_score), which runs made after this change
                 save; older runs do not have it, so these strategies are skipped for them.

The original strategies are re-derived where possible as a check: the heat map ranking is
recomputed from the saved candidates and compared with the saved heat-map picks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.geo import haversine_km, hex_center
from taxi.models.baselines import AREA_DEG, MINUTE_BANDS
from taxi.replay.scoring import drive_wait_score

KEYS = ['episode_id', 'snapshot_t']
CENTRE = (13.7466, 100.5347)            # Siam; central = within 8 km, as in the model reports
NEAR_MIN = 5.0
MARGINS = (0, 1, 2, 3, 4, 5, 6)
TIME_BANDS = [(0, 6, 'night 00-06'), (6, 10, 'morning 06-10'), (10, 16, 'midday 10-16'),
              (16, 20, 'evening 16-20'), (20, 24, 'late 20-24')]


def time_band(hour):
    hour = np.asarray(hour)
    out = np.empty(len(hour), dtype=object)
    for lo, hi, name in TIME_BANDS:
        out[(hour >= lo) & (hour < hi)] = name
    return out


def area_of(lat, lon):
    return np.where(haversine_km(CENTRE[0], CENTRE[1], np.asarray(lat), np.asarray(lon)) <= 8, 'central', 'outer')


def candidate_context(candidates: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    """Add the decision's minutes empty, request hour, weekday and hexagon centre to each candidate row."""
    cols = KEYS + ['origin_hex', 'decision_minute'] + (['minutes_empty'] if 'minutes_empty' in policies else [])
    dec = policies.drop_duplicates(KEYS)[cols].astype({'decision_minute': float})
    dec = dec.rename(columns={'minutes_empty': 'decision_minutes_empty'})
    c = candidates.merge(dec, on=KEYS, how='left', validate='many_to_one')
    if c['decision_minute'].isna().any():
        raise ValueError('candidate rows without a matching decision')
    h1 = c['snapshot_t'].astype(np.int64) // 3600
    c['hour_of_day'] = (h1 % 24).astype(int)
    c['day_of_week'] = pd.to_datetime(h1 * 3600, unit='s').dt.dayofweek.to_numpy()
    uniq = pd.unique(c['hex'])
    lat, lon = hex_center(uniq)
    lat_map, lon_map = dict(zip(uniq, lat)), dict(zip(uniq, lon))
    c['hex_lat'] = c['hex'].map(lat_map).astype(float)
    c['hex_lon'] = c['hex'].map(lon_map).astype(float)
    base = c['decision_minutes_empty'] if 'decision_minutes_empty' in c else c['decision_minute']
    c['minutes_empty'] = base.astype(float) + c['drive_min'].astype(float)
    return c


def hist_curve_scores(fallback_b, ctx: pd.DataFrame) -> np.ndarray:
    """Drive + expected wait from the historical curve; NaN when no group curve exists.

    Same lookup as taxi.replay.scoring.historical_fallback_curve (area cell, 3-hour band,
    minutes-empty band, then coarser groups; never the city-wide curve)."""
    area = (np.round(ctx['hex_lat'] / AREA_DEG).astype(int).astype(str) + '_'
            + np.round(ctx['hex_lon'] / AREA_DEG).astype(int).astype(str)).to_numpy()
    hb = (ctx['hour_of_day'].to_numpy() // 3).astype(str)
    band = np.searchsorted(MINUTE_BANDS[1:], ctx['minutes_empty'].to_numpy(float), side='right').astype(str)
    drive = ctx['drive_min'].to_numpy(float)
    keys = [np.char.add(np.char.add(np.char.add(np.char.add(area.astype(str), '|'), hb), '|'), band),
            np.char.add(np.char.add(hb, '|'), band), band]
    out = np.full(len(ctx), np.nan)
    cache = {}
    for i in range(len(ctx)):
        curve = None
        for level in range(3):
            curve = fallback_b.tables[level].get(str(keys[level][i]))
            if curve is not None:
                break
        if curve is None:
            continue
        ck = (id(curve), float(drive[i]))
        if ck not in cache:
            cache[ck] = drive_wait_score(curve, float(drive[i]))
        out[i] = cache[ck]
    return out


def _policy_row(dec, name, ranking, est, best):
    first = ranking[0] if ranking else None
    o = est.get(first) if first is not None else None
    covered = bool(o is not None and o['covered'] and pd.notna(o['score_min']))
    return {'episode_id': dec['episode_id'], 'snapshot_t': dec['snapshot_t'], 'origin_hex': dec['origin_hex'],
            'decision_minute': dec['decision_minute'], 'strategy': name, 'recommended_hex': first,
            'recommended_drive_min': o['drive_min'] if o is not None else None,
            'recommendation_available': first is not None, 'top3_hexes': tuple(ranking[:3]),
            'n_recommended_candidates': len(ranking), 'n_candidates': dec['n_candidates'],
            'n_covered_candidates': dec['n_covered'], 'recommended_covered': covered,
            'recommended_score_min': float(o['score_min']) if covered else None,
            'best_observed_score_min': best,
            'comparison_episodes': o['comparison_episodes'] if o is not None else 0,
            'training_coverage_checked': True, 'replay_day': dec.get('replay_day')}


def derive_strategies(policies: pd.DataFrame, candidates: pd.DataFrame, *, heatmap, fallback_b,
                      margins=MARGINS, near_min=NEAR_MIN):
    """Return (new policy rows, heat-map reproduction check)."""
    ctx = candidate_context(candidates, policies)
    ctx['hist_score'] = hist_curve_scores(fallback_b, ctx)
    ctx['heat'] = np.asarray(heatmap.predict(ctx[['hex', 'hour_of_day', 'day_of_week']]), float)
    ctx['_order'] = np.arange(len(ctx))
    has_b = 'model_b_score' in ctx.columns and ctx['model_b_score'].notna().any()
    saved_heat = policies.loc[policies['strategy'] == 'heatmap'].set_index(KEYS)['recommended_hex']
    day_of = (policies.drop_duplicates(KEYS).set_index(KEYS)['replay_day']
              if 'replay_day' in policies else None)
    rows, agree, total = [], 0, 0
    for key, g in ctx.groupby(KEYS, sort=False):
        g = g.sort_values('_order')
        hexes = g['hex'].to_numpy()
        drive = g['drive_min'].to_numpy(float)
        origin = g['origin_hex'].iloc[0]
        est = {h: {'covered': bool(cv), 'score_min': s, 'drive_min': float(d), 'comparison_episodes': n}
               for h, cv, s, d, n in zip(hexes, g['covered'], pd.to_numeric(g['score_min'], errors='coerce'),
                                         drive, g['comparison_episodes'])}
        cov = [v['score_min'] for v in est.values() if v['covered'] and pd.notna(v['score_min'])]
        best = float(min(cov)) if cov else None
        dec = {'episode_id': key[0], 'snapshot_t': key[1], 'origin_hex': origin,
               'decision_minute': float(g['decision_minute'].iloc[0]), 'n_candidates': len(g),
               'n_covered': len(cov), 'replay_day': day_of.get(key) if day_of is not None else None}
        pos = np.arange(len(g))
        # heat map, re-derived exactly as strategies.ranked_strategies does it (check only)
        heat = g['heat'].to_numpy()
        heat_rank = list(hexes[np.lexsort((pos, drive, -heat))])
        if key in saved_heat.index:
            total += 1
            agree += int(heat_rank[0] == saved_heat.loc[key])
        # drive-aware heat map
        near = drive <= near_min
        near_rank = list(hexes[near][np.lexsort((pos[near], drive[near], -heat[near]))])
        rows.append(_policy_row(dec, 'heatmap_near5', near_rank, est, best))
        # historical curve scored like Model B
        hs = g['hist_score'].to_numpy()
        ok = np.isfinite(hs)
        hist_rank = list(hexes[ok][np.lexsort((pos[ok], drive[ok], hs[ok]))])
        rows.append(_policy_row(dec, 'hist_curve', hist_rank, est, best))
        # Model B with a move margin
        if has_b:
            bs = g['model_b_score'].to_numpy(float)
            okb = np.isfinite(bs)
            b_rank = list(hexes[okb][np.lexsort((pos[okb], drive[okb], bs[okb]))])
            stay_score = bs[hexes == origin]
            for m in margins:
                r = list(b_rank)
                if r and len(stay_score) and np.isfinite(stay_score[0]) and r[0] != origin:
                    best_b = bs[hexes == r[0]][0]
                    if stay_score[0] - best_b < m:
                        r = [origin] + [h for h in r if h != origin]
                rows.append(_policy_row(dec, f'model_b_m{m}', r, est, best))
    check = {'heatmap_first_pick_reproduced': agree, 'heatmap_decisions_checked': total,
             'share': agree / total if total else None}
    return pd.DataFrame(rows), check


def decision_groups(policies: pd.DataFrame) -> pd.DataFrame:
    """One row per decision: origin area (central/outer) and time band of the request hour."""
    d = policies.drop_duplicates(KEYS)[KEYS + ['origin_hex', 'decision_minute']].copy()
    lat, lon = hex_center(d['origin_hex'].to_numpy())
    d['area'] = area_of(lat, lon)
    d['time_band'] = time_band(((d['snapshot_t'].astype(np.int64) // 3600) % 24).to_numpy())
    return d
