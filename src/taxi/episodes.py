"""Turn cleaned GPS records into trips and empty periods (rules v1.0).

Light switches off (occ 0 -> 1) = pickup.  Light switches on (occ 1 -> 0) = drop-off.
Trip         = pickup  -> next drop-off.
Empty period = drop-off -> next pickup (this is the "wait").

Rules added after the audit (see config.py and docs/audit_findings.md):
  1. Fake pickups and trips (driver resting with the light switched to "occupied"):
     - if the taxi stays parked with the engine off for FAKE_TRIP_STILL_MIN right after
       a "pickup", the empty period before it is censored with reason "off_duty";
     - a whole "trip" under FAKE_TRIP_KM that is mostly parked is marked is_fake, and
       the empty period after it starts as "after_rest".
  2. Rests: a stationary, engine-off stretch of OFFDUTY_STILL_MIN or more cuts the
     empty period (censored "off_duty"). A new period starts when the taxi moves
     again (start_type "after_rest").
  3. Periods longer than MAX_EMPTY_MIN are censored at that length ("too_long").
  4. Fast long drives, or ending far away, while "empty" are flagged possible_missed_trip.
  5. Zero-length periods are dropped.

Only periods that START on the given day are kept; look-ahead records are used to
finish periods that cross midnight.
"""
import numpy as np
import pandas as pd

from . import config
from .geo import haversine_km, to_hex

EMPTY_COLS = ["episode_id", "vid", "start_type", "t_start", "t_end", "minutes", "pickup_seen", "censor_reason",
              "lat_start", "lon_start", "lat_end", "lon_end", "n_records", "km_driven", "far_km",
              "engine_off_share", "longest_still_min", "possible_missed_trip"]
TRIP_COLS = ["trip_id", "vid", "t_start", "t_end", "minutes", "lat_start", "lon_start", "lat_end",
             "lon_end", "km", "slow_min", "engine_off_share", "longest_still_min", "is_fake", "has_gap"]


def _longest_still_min(dur, still):
    best = run = 0.0
    for d, s in zip(dur, still):
        run = run + d if s else 0.0
        best = max(best, run)
    return best / 60


def _segment_stats(t, la, lo, spd, acc, still, a, b):
    """Stats over records a..b (b inclusive; time intervals a..b-1)."""
    dur = np.diff(t[a:b + 1]).astype(float)
    total = dur.sum()
    off = dur[acc[a:b] == 0].sum() / total if total > 0 else np.nan     # time-weighted
    km = float(np.sum(haversine_km(la[a:b], lo[a:b], la[a + 1:b + 1], lo[a + 1:b + 1]))) if b > a else 0.0
    longest = _longest_still_min(dur, still[a:b])
    slow = float(dur[spd[a:b] <= 6].sum() / 60)
    return km, off, longest, slow


def _first_rest(t, still, a, b):
    """First stationary engine-off stretch inside records a..b-1 lasting >= OFFDUTY_STILL_MIN.
    Returns (first record of the rest, first record after it) or None."""
    need = config.OFFDUTY_STILL_MIN * 60
    run, r0 = 0.0, None
    for k in range(a, b):
        if still[k]:
            if run == 0:
                r0 = k
            run += t[k + 1] - t[k]
            if run >= need:
                k2 = k + 1
                while k2 < b and still[k2]:
                    k2 += 1
                return r0, k2
        else:
            run = 0.0
    return None


def _starts_with_rest(t, still, i, j):
    """True if the taxi is parked with the engine off for FAKE_TRIP_STILL_MIN right after
    record i (allowing one moving record first)."""
    k = i if still[i] else i + 1
    if k >= j or not still[k]:
        return False
    run = 0.0
    while k < j and still[k]:
        run += t[k + 1] - t[k]
        k += 1
    return run >= config.FAKE_TRIP_STILL_MIN * 60


def _is_fake_trip(t, la, lo, spd, acc, still, i, j):
    km, off, longest, _ = _segment_stats(t, la, lo, spd, acc, still, i, j)
    return km < config.FAKE_TRIP_KM and (longest >= config.FAKE_TRIP_STILL_MIN or (off == off and off > 0.5))


def build_episodes(d: pd.DataFrame, day_end: int, has_next_day: bool):
    """Return (trips, empty_periods) DataFrames for periods starting before day_end."""
    d = d.sort_values(["vid", "t"]).reset_index(drop=True)
    vids = d["vid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, vids[1:] != vids[:-1]])
    ends = np.r_[starts[1:], len(d)]
    T, LA, LO = d["t"].to_numpy(), d["lat"].to_numpy(float), d["lon"].to_numpy(float)
    SP, AC, OC = d["speed"].to_numpy(), d["acc"].to_numpy(), d["occ"].to_numpy()
    no_pickup_reason = "lookahead_end" if has_next_day else "end_of_data"
    max_sec = config.MAX_EMPTY_MIN * 60

    empty, trips = [], []
    for s, e in zip(starts, ends):
        vid = vids[s]
        t, la, lo, spd, acc, occ = T[s:e], LA[s:e], LO[s:e], SP[s:e], AC[s:e], OC[s:e]
        if len(t) < 2:
            continue
        still = (spd < config.MOVING_KMH) & (acc == 0)
        prev = occ[:-1]
        pick = np.flatnonzero((prev == 0) & (occ[1:] == 1)) + 1
        drop = np.flatnonzero((prev == 1) & (occ[1:] == 0)) + 1
        big_gap = np.flatnonzero(np.diff(t) > config.GAP_SEC)       # gap after record k

        # ---------------- trips (all of them, so look-ahead pickups can be checked for fakes)
        fake, rest_start, after_fake = {}, {}, set()
        for i in pick:
            nxt = drop[drop > i]
            j = nxt[0] if len(nxt) else len(t) - 1
            fake[i] = _is_fake_trip(t, la, lo, spd, acc, still, i, j)
            rest_start[i] = fake[i] or _starts_with_rest(t, still, i, j)
            if fake[i] and len(nxt):
                after_fake.add(j)                  # this "drop-off" is really the end of a rest
            if t[i] >= day_end or not len(nxt):
                continue
            km, off, longest, slow = _segment_stats(t, la, lo, spd, acc, still, i, j)
            has_gap = bool(np.any((big_gap >= i) & (big_gap < j)))
            trips.append((f"{vid}-{t[i]}", vid, t[i], t[j], (t[j] - t[i]) / 60, la[i], lo[i], la[j], lo[j],
                          km, slow, off, longest, fake[i], has_gap))

        # ---------------- empty periods
        def emit(a, b, seen, reason, start_type):
            if b > a and t[a] < day_end and t[b - 0] > t[a]:
                if t[b] - t[a] > max_sec:                       # rule 3: censor very long periods
                    b = a + int(np.searchsorted(t[a:b + 1], t[a] + max_sec, side="right")) - 1
                    seen, reason = False, "too_long"
                km, off, longest, _ = _segment_stats(t, la, lo, spd, acc, still, a, b)
                far = float(haversine_km(la[a], lo[a], la[b], lo[b]))
                kmh = km / ((t[b] - t[a]) / 3600)
                missed = (km > config.MISSED_TRIP_KM and kmh > config.MISSED_TRIP_KMH) or far > config.MISSED_TRIP_FAR_KM
                empty.append((f"{vid}-{t[a]}", vid, start_type, t[a], t[b], (t[b] - t[a]) / 60, seen, reason,
                              la[a], lo[a], la[b], lo[b], b - a + 1, km, far, off, longest, missed))

        for i in drop:
            if t[i] >= day_end:
                break
            nxt = pick[pick > i]
            end, seen, reason = (nxt[0], True, "") if len(nxt) else (len(t) - 1, False, no_pickup_reason)
            g = big_gap[(big_gap >= i) & (big_gap < end)]
            if len(g):
                end, seen, reason = g[0], False, "gap"
            elif seen and rest_start[end]:                        # rule 1: fake pickup
                seen, reason = False, "off_duty"
            a, start_type = i, ("after_rest" if i in after_fake else "dropoff")
            while True:                                           # rule 2: cut at rests
                rest = _first_rest(t, still, a, end)
                if rest is None:
                    emit(a, end, seen, reason, start_type)
                    break
                r0, r1 = rest
                emit(a, r0, False, "off_duty", start_type)
                if r1 >= end:
                    break
                a, start_type = r1, "after_rest"

    empty = pd.DataFrame(empty, columns=EMPTY_COLS)
    trips = pd.DataFrame(trips, columns=TRIP_COLS)
    empty = empty[empty["minutes"] > 0].reset_index(drop=True)            # rule 5
    empty["hex_start"] = to_hex(empty["lat_start"], empty["lon_start"])
    empty["hex_end"] = to_hex(empty["lat_end"], empty["lon_end"])         # pickup hexagon when pickup_seen
    trips["hex_pickup"] = to_hex(trips["lat_start"], trips["lon_start"])
    trips["hex_dropoff"] = to_hex(trips["lat_end"], trips["lon_end"])
    return trips, empty
