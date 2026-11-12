"""Retrospective comparison-taxi outcomes, with censoring-aware KM estimates.

This estimates outcomes from other observed taxis; it is NOT an observed
counterfactual for a taxi redirected to a different destination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.geo import hex_center
from taxi.models.baselines import AREA_DEG
from taxi.models.survival import km_curve
from taxi.replay.scoring import drive_wait_score

REQUIRED = {'episode_id', 'vid', 'snapshot_t', 'minutes_empty', 'hex',
            'minutes_until_end', 'pickup_seen'}


def area_key(hexes) -> np.ndarray:
    """Use the same approximately 2.8 km area bins as HistoricalSurvival."""
    lat, lon = hex_center(list(hexes))
    return np.array([f'{round(float(a) / AREA_DEG)}_{round(float(b) / AREA_DEG)}'
                     for a, b in zip(lat, lon)], dtype=object)



def prepare_reference(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Prepare same-split comparison data and index it by area and time."""
    missing = REQUIRED - set(snapshots.columns)
    if missing:
        raise ValueError(f'Missing comparison columns: {sorted(missing)}')

    out = snapshots.copy()
    out['_replay_area'] = (
        area_key(out['hex'].to_numpy())
        if len(out)
        else pd.Series(dtype=object)
    )

    # Build the index once, rather than scanning every row per candidate.
    area_index = {}

    for area, group in out.groupby('_replay_area', sort=False):
        ordered = group.sort_values('snapshot_t', kind='stable')
        times = ordered['snapshot_t'].to_numpy(dtype=float)
        area_index[area] = (times, ordered)

    out.attrs['_replay_area_index'] = area_index
    return out


def comparison_taxis(reference: pd.DataFrame, *, destination_hex,
                    arrival_t: float, arrival_minutes_empty: float,
                    own_episode_id, own_vid, time_tolerance_min: float = 15,
                    empty_tolerance_min: float = 10) -> pd.DataFrame:
    """Find comparison taxis using the prebuilt area/time index."""
    if '_replay_area' not in reference:
        raise ValueError('Call prepare_reference() first')

    target_area = area_key([destination_hex])[0]

    area_index = reference.attrs.get('_replay_area_index')

    if area_index is None:
        raise ValueError('Comparison index missing; call prepare_reference()')

    if target_area not in area_index:
        return reference.iloc[0:0].copy()

    times, group = area_index[target_area]

    tolerance_seconds = 60 * time_tolerance_min

    left = np.searchsorted(
        times, arrival_t - tolerance_seconds, side='left'
    )
    right = np.searchsorted(
        times, arrival_t + tolerance_seconds, side='right'
    )

    if left == right:
        return group.iloc[0:0].copy()

    nearby = group.iloc[left:right]

    mask = (
        (nearby['episode_id'].astype(str) != str(own_episode_id))
        & (nearby['vid'].astype(str) != str(own_vid))
        & (
            (nearby['minutes_empty'] - arrival_minutes_empty).abs()
            <= empty_tolerance_min
        )
    )

    found = nearby.loc[mask].copy()

    if found.empty:
        return found

    found['_time_delta'] = (
        found['snapshot_t'] - arrival_t
    ).abs()

    found['_empty_delta'] = (
        found['minutes_empty'] - arrival_minutes_empty
    ).abs()

    found = (
        found.sort_values(
            ['_time_delta', '_empty_delta', 'snapshot_t']
        )
        .drop_duplicates('episode_id')
        .drop(columns=['_time_delta', '_empty_delta'])
    )

    return found



def estimate_candidate(reference: pd.DataFrame, *, destination_hex,
                       drive_min: float, decision_t: float, minutes_empty: float,
                       own_episode_id, own_vid, min_episodes: int = 5,
                       time_tolerance_min: float = 15,
                       empty_tolerance_min: float = 10,
                       horizon: int = 60) -> dict:
    """Estimate candidate drive+wait from comparable observed taxi episodes.

    Returns None for score_min if too few independent comparison episodes.
    The caller must separately enforce any TRAINING coverage threshold.
    """
    peers = comparison_taxis(
        reference, destination_hex=destination_hex,
        arrival_t=decision_t + drive_min * 60,
        arrival_minutes_empty=minutes_empty + drive_min,
        own_episode_id=own_episode_id, own_vid=own_vid,
        time_tolerance_min=time_tolerance_min,
        empty_tolerance_min=empty_tolerance_min)
    n = peers['episode_id'].nunique()
    if n < min_episodes:
        return {'n_episodes': int(n), 'score_min': None, 'covered': False}
    curve = km_curve(peers['minutes_until_end'].to_numpy(float),
                     peers['pickup_seen'].to_numpy(bool))
    return {'n_episodes': int(n), 'score_min': drive_wait_score(curve, drive_min, horizon=horizon),
            'covered': True}
