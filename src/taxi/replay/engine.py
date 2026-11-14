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
                    model_a, model_b, fallback_b=None, heatmap, comparison_reference,
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
    
    ranked, fallback_diagnostics = ranked_strategies(
        candidates,
        feats,
        origin_hex=origin,
        model_a=model_a,
        model_b=model_b,
        heatmap=heatmap,
        fallback_b=fallback_b,
        training_coverage=training_coverage,
        snapshot_t=int(decision['snapshot_t']),
        seed=seed,
        return_diagnostics=True,
    )


    
    diagnostics_by_hex = fallback_diagnostics.set_index('hex').to_dict('index')

    drive_by_hex = {item.hex: float(item.drive_min) for item in candidates.itertuples(index=False)}
    estimated = {}
    for i, item in enumerate(candidates.itertuples(index=False)):
        result = estimate_candidate(
            comparison_reference,
            destination_hex=item.hex,
            drive_min=float(item.drive_min),
            decision_t=float(decision['snapshot_t']),
            minutes_empty=float(decision['minutes_empty']),
            own_episode_id=decision['episode_id'],
            own_vid=decision['vid'],
            min_episodes=min_episodes,
            empty_tolerance_min=empty_tolerance_min,
        )

        if training_coverage is not None:
            feature_row = feats.iloc[i]

            count = training_coverage(
                item.hex,
                int(feature_row['hour_of_day']),
                float(feature_row['minutes_empty']),
            )

            result['training_n'] = int(count)
            result['training_supported'] = bool(count >= 30)

        diagnostic = diagnostics_by_hex[item.hex]

        result['fallback_used'] = bool(diagnostic['fallback_used'])
        result['fallback_level'] = diagnostic['fallback_level']
        result['model_b_score'] = diagnostic.get('model_b_score')

        estimated[item.hex] = result


    covered = {h: r['score_min'] for h, r in estimated.items() if r['covered']}
    best_score = min(covered.values()) if covered else None
    rows = []
    for name, ranking in ranked.items():
        # Do not silently substitute a different area when the recommendation
        # has no outcome evidence; preserve missingness for coverage metrics.
        
        first = ranking[0] if ranking else None

        if first is None:
            outcome = {
                'covered': False,
                'score_min': None,
                'n_episodes': 0,
            }
        else:
            outcome = estimated[first]

        rows.append({
            'episode_id': decision['episode_id'], 'vid': decision['vid'],
            'snapshot_t': decision['snapshot_t'], 'origin_hex': origin,
            'decision_minute': decision.get('decision_minute', decision['minutes_empty']),
            'strategy': name,
            'recommended_hex': first,
            'recommended_drive_min': drive_by_hex.get(first) if first is not None else None,
            'recommendation_available': first is not None,
            'top3_hexes': tuple(ranking[:3]),
            'n_recommended_candidates': len(ranking),
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
            'drive_min': drive_by_hex[h],
            'score_min': r['score_min'] if r['covered'] else None,
            'covered': bool(r['covered']),
            'comparison_episodes': r['n_episodes'],
            'training_n': r.get('training_n'),
            'training_supported': r.get('training_supported'),
            'fallback_used': r.get('fallback_used', False),
            'fallback_level': r.get('fallback_level'),
            'fallback_curve_found': bool(r.get('fallback_used', False) and r.get('fallback_level') is not None),
            'model_b_score': r.get('model_b_score'),
        }
        for h, r in estimated.items()
    ])
    return policy_rows, candidate_rows
