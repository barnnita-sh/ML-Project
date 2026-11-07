"""One-decision replay orchestration. Does not access held-out test by itself.

The caller supplies frozen models, a populated FeatureWindow, a historical
candidate universe, and an approved comparison-reference table.
"""
from __future__ import annotations

import pandas as pd

from taxi.replay.candidates import reachable_candidates, candidate_features
from taxi.replay.outcomes import estimate_candidate
from taxi.replay.strategies import ranked_strategies


def replay_decision(decision, *, available_hexes, feature_window,
                    model_a, model_b, heatmap, comparison_reference,
                    speed_kmh: float, travel_time_lookup=None,
                    min_episodes: int = 5, empty_tolerance_min: float = 10,
                    seed: int = 42, training_coverage=None,
                    return_candidate_outcomes: bool = False):
    """Replay one snapshot; returns one row per strategy.

    training_coverage: optional callable (hex, arrival_hour, minutes_empty)
    -> training snapshot count. If omitted, only comparison coverage is checked
    and the result is marked training_coverage_checked=False.

    Outcomes are estimated for all candidates once and shared across policies.
    """
    origin = decision['hex']
    candidates = reachable_candidates(
        origin, available_hexes, speed_kmh=speed_kmh,
        travel_time_lookup=travel_time_lookup)
    feats = candidate_features(feature_window, candidates,
                               snapshot_t=int(decision['snapshot_t']),
                               minutes_empty=float(decision['minutes_empty']))
    ranked = ranked_strategies(candidates, feats, origin_hex=origin,
                               model_a=model_a, model_b=model_b,
                               heatmap=heatmap, seed=seed)
    estimated = {}
    for item in candidates.itertuples(index=False):
        result = estimate_candidate(
            comparison_reference, destination_hex=item.hex,
            drive_min=float(item.drive_min), decision_t=float(decision['snapshot_t']),
            minutes_empty=float(decision['minutes_empty']),
            own_episode_id=decision['episode_id'], own_vid=decision['vid'],
            min_episodes=min_episodes, empty_tolerance_min=empty_tolerance_min)
        if training_coverage is not None:
            arrival_hour = int((decision['snapshot_t'] + item.drive_min * 60) // 3600)
            count = training_coverage(item.hex, arrival_hour,
                                      float(decision['minutes_empty']) + item.drive_min)
            result['training_n'] = int(count)
            result['covered'] = result['covered'] and count >= 30
        estimated[item.hex] = result

    covered = {h: r['score_min'] for h, r in estimated.items() if r['covered']}
    best_score = min(covered.values()) if covered else None
    rows = []
    for name, ranking in ranked.items():
        # Do not silently substitute a different area when the recommendation
        # has no outcome evidence; preserve missingness for coverage metrics.
        first = ranking[0]
        outcome = estimated[first]
        rows.append({
            'episode_id': decision['episode_id'], 'vid': decision['vid'],
            'snapshot_t': decision['snapshot_t'], 'origin_hex': origin,
            'strategy': name, 'recommended_hex': first,
            'top3_hexes': tuple(ranking[:3]),
            'n_candidates': len(candidates),
            'n_covered_candidates': len(covered),
            'recommended_covered': bool(outcome['covered']),
            'recommended_score_min': outcome['score_min'] if outcome['covered'] else None,
            'best_observed_score_min': best_score,
            'comparison_episodes': outcome['n_episodes'],
            'training_coverage_checked': training_coverage is not None,
        })
    policy_rows = pd.DataFrame(rows)
    if not return_candidate_outcomes:
        return policy_rows
    candidate_rows = pd.DataFrame([
        {
            'episode_id': decision['episode_id'],
            'snapshot_t': decision['snapshot_t'],
            'hex': h,
            'score_min': r['score_min'] if r['covered'] else None,
            'covered': bool(r['covered']),
            'comparison_episodes': r['n_episodes'],
            'training_n': r.get('training_n'),
        }
        for h, r in estimated.items()
    ])
    return policy_rows, candidate_rows
