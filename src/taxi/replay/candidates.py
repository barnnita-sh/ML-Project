"""Reachable destination hexagons and leakage-safe candidate features.

Candidate universe must be supplied from a past/training-only list of valid
hexagons, NOT from future validation/test pickup outcomes.
"""

import numpy as np
import pandas as pd

from taxi.geo import haversine_km, hex_center


def driving_minutes(origin_hex, destinations, *, speed_kmh: float,
                    detour_factor: float = 1.35) -> np.ndarray:
    """Approximate road minutes from hex-centre distance; NOT a road-network ETA."""
    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        raise ValueError("speed_kmh must be positive and finite")
    if not np.isfinite(detour_factor) or detour_factor < 1:
        raise ValueError("detour_factor must be finite and >= 1")
    destinations = np.asarray(list(destinations))
    if not len(destinations):
        return np.array([], dtype=float)
    origin_lat, origin_lon = hex_center([origin_hex])
    lat, lon = hex_center(destinations)
    km = haversine_km(origin_lat[0], origin_lon[0], lat, lon)
    result = km * detour_factor * 60 / speed_kmh
    result[destinations == origin_hex] = 0.0
    return result


def reachable_candidates(origin_hex, available_hexes, *, speed_kmh: float,
                         max_drive_min: float = 15.0, detour_factor: float = 1.35,
                         travel_time_lookup=None) -> pd.DataFrame:
    """Return current hex plus other reachable hexes, sorted by drive minutes.

    travel_time_lookup may be a function (origin_hex, destination_hex) ->
    minutes, allowing an agreed road-time table to replace distance estimates.
    """
    if max_drive_min <= 0:
        raise ValueError("max_drive_min must be positive")
    universe = list(dict.fromkeys([origin_hex, *list(available_hexes)]))
    if travel_time_lookup is None:
        times = driving_minutes(origin_hex, universe, speed_kmh=speed_kmh,
                                detour_factor=detour_factor)
    else:
        times = np.array([0.0 if h == origin_hex else travel_time_lookup(origin_hex, h)
                          for h in universe], dtype=float)
    valid = np.isfinite(times) & (times >= 0) & (times <= max_drive_min)
    result = pd.DataFrame({"hex": np.asarray(universe, dtype=object)[valid],
                           "drive_min": times[valid]})
    return result.sort_values(["drive_min", "hex"], key=lambda col: col.astype(str) if col.name == "hex" else col).reset_index(drop=True)


def candidate_features(feature_window, candidates: pd.DataFrame, *, snapshot_t: int,
                       minutes_empty: float) -> pd.DataFrame:
    """Model-ready candidate features at REQUEST hour (not future arrival hour).

    Project timestamps encode Bangkok wall time as UTC-like epoch seconds;
    never add a timezone offset.  At arrival, minutes_empty includes travel.
    """
    if not {"hex", "drive_min"}.issubset(candidates.columns):
        raise ValueError("candidates requires hex and drive_min")
    if candidates.empty:
        return pd.DataFrame()
    h1 = int(snapshot_t) // 3600
    # Replay-only memoization: static features depend on (window, hex, request hour),
    # not on drive minutes or minutes_empty. Preserve original row order and dtypes.
    hexes = candidates["hex"].to_numpy()
    signature = tuple(feature_window.days.keys())
    cache = getattr(feature_window, "_replay_feature_cache", None)
    if cache is None or cache["signature"] != signature:
        cache = {"signature": signature, "by_hour": {}}
        feature_window._replay_feature_cache = cache
    hour_cache = cache["by_hour"].setdefault(h1, {})
    missing = list(dict.fromkeys(h for h in hexes if h not in hour_cache))
    if missing:
        computed = feature_window.features(
            np.asarray(missing), np.full(len(missing), h1, dtype=np.int64)
        )
        for h, values in zip(missing, computed.to_numpy()):
            hour_cache[h] = values.copy()
        cache["columns"] = computed.columns
    features = pd.DataFrame(
        np.vstack([hour_cache[h] for h in hexes]), columns=cache["columns"]
    ).astype("float32")
    features = features.reset_index(drop=True)
    features.insert(0, "hex", candidates["hex"].to_numpy())
    features.insert(1, "drive_min", candidates["drive_min"].to_numpy(float))
    features["minutes_empty"] = float(minutes_empty) + features["drive_min"]
    return features
