# Features and training tables

Task 2 of the plan (features). Status: **built for all 365 days with real H3 hexagons**
and checked with `scripts/verify_outputs.py` (`docs/verification_report.md`). Code: `src/taxi/hexhour.py`, `src/taxi/features.py`.
Coverage: `docs/coverage_report.md`. Updated EDA: `docs/eda_update_v1.md`.

## How to build

Run after `build_days.py` (Task 1):

```bash
python scripts/build_hexhour.py  --start 20250101 --end 20251231   # about 5 min
python scripts/build_features.py --start 20250101 --end 20251231   # about 10 min
python tests/test_features.py                                      # 7 timing tests
python scripts/check_build.py --features                           # completeness check
python scripts/coverage_report.py                                  # docs/coverage_report.md
python scripts/eda_update.py                                       # docs/eda_update_v1.md + figures
python scripts/verify_outputs.py                                   # deep check, docs/verification_report.md
```

Both builders skip days already built; add `--force` after any change. Features for day D
need days D-29 to D+1, so after rebuilding some days, rebuild the features of the 30 days
that follow them too (or simply rebuild the year with `--force`).

## The tables

| Table | One row per | Rows (2025) | Used for |
| --- | --- | --- | --- |
| `pipeline_output/hexhour/` | Hexagon-hour with any taxi activity | about 50,000 a day | Base counts behind the features |
| `pipeline_output/features/modelA/` | Hexagon-hour with an empty taxi present | 12,738,845 | Model A: predict `target_pickups` |
| `pipeline_output/features/snapshots/` | Waiting taxi every 5 minutes | 29,996,060 | Model B: predict `minutes_until_end` (censored when `pickup_seen` is False) |

**Hexagon-hour base table (`hexhour/`):**

| Column | Meaning |
| --- | --- |
| `hex`, `hour_id` | Hexagon and clock hour (`hour_id = t // 3600`; hour of day = `hour_id % 24`) |
| `vacant_taxis` | Distinct taxis inside a wait there that hour (resting taxis and possible missed trips not counted) |
| `pickups` | Real pickups there that hour (fake trips excluded) |
| `speed_med`, `n_speed` | Median speed of moving taxis (over 5 km/h, any state) and number of pings behind it |

**Model A table extra columns:** `target_pickups` (the label), `exposure_vacant_taxis`
(empty taxis present during the target hour: known only afterwards, so use it for weighting
or analysis, **never as a feature**).

**Model B table extra columns:** `episode_id`, `vid`, `start_type`, `snapshot_t`,
`minutes_empty`, `hex`, `speed` (the taxi's own speed), label `minutes_until_end`,
`pickup_seen`, `censor_reason`.

## Timing rule

Every row has a **current hour h1** and a **last completed hour k = h1 - 1**.

- Model B snapshot at time t: h1 = the hour t falls in. At 14:20, k is 13:00 to 14:00.
- Model A row for hour H: h1 = H, so k = H - 1.

Features read hexagon-hour data only at hours up to k, and completed waits only if they
ended in hour k or earlier, so everything used finished before the prediction moment.
"Same hour yesterday" and the 4-week history look at h1 minus 24 hours or whole weeks, which
are in the past. `tests/test_features.py` checks this (for example, 50 pickups put in the
current hour must not appear in any feature), and `scripts/verify_outputs.py` re-computes
features by hand for 320 random snapshots from the trips and waits tables.

## Feature dictionary

"nbr" = neighbourhood: the hexagon plus its 6 neighbours (about 5.2 km²).
Counts are 0 where no activity was seen; NaN where the hours needed are before 1 January 2025.

| Feature | Group | How it is computed | Model |
| --- | --- | --- | --- |
| `hour_of_day` | Calendar | Hour of h1, 0 to 23 | A, B |
| `day_of_week` | Calendar | Monday 0 to Sunday 6 | A, B |
| `is_weekend` | Calendar | 1 on Saturday and Sunday | A, B |
| `is_holiday` | Calendar | 1 on a Thai public holiday (list in `features.py`) | A, B |
| `month` | Calendar | 1 to 12 | A, B |
| `hex_lat`, `hex_lon` | Place | Centre of the hexagon | A, B |
| `pickups_hex_1h` | This hexagon | Real pickups in the hexagon during hour k | A, B |
| `vacant_hex_1h` | This hexagon | Empty taxis in the hexagon during hour k | A, B |
| `pickups_nbr_1h` | Neighbourhood | Real pickups in the 7 hexagons during hour k | A, B |
| `vacant_nbr_1h` | Neighbourhood | Empty taxis summed over the 7 hexagons during hour k (a taxi crossing two hexagons counts in both) | A, B |
| `vacant_per_pickup_nbr_1h` | Competition | `vacant_nbr_1h` ÷ max(`pickups_nbr_1h`, 1) | A, B |
| `speed_nbr_1h` | Traffic | Moving-taxi speed in the 7 hexagons during hour k (weighted by pings); NaN if no moving taxi | A, B |
| `pickups_nbr_3h` | Neighbourhood | Real pickups in the 7 hexagons over hours k-2 to k | A, B |
| `vacant_nbr_3h_mean` | Neighbourhood | Mean of `vacant_nbr` over hours k-2 to k | A, B |
| `pickups_nbr_lag24` | Repeats | Neighbourhood pickups in hour h1 - 24 (same hour yesterday) | A, B |
| `pickups_nbr_lag168` | Repeats | Neighbourhood pickups in hour h1 - 168 (same hour last week) | A, B |
| `hist_pickups_hex_4w` | History | Mean hexagon pickups at the same hour and weekday over the previous 4 weeks | A, B |
| `hist_pickups_nbr_4w` | History | Same, neighbourhood | A, B |
| `hist_vacant_nbr_4w` | History | Mean neighbourhood empty taxis, same hour and weekday, previous 4 weeks | A, B |
| `median_wait_nbr_3h` | Recent waits | Median length of completed waits whose pickup was in the 7 hexagons and that ended in hours k-2 to k; NaN if none | A, B |
| `n_waits_nbr_3h` | Recent waits | How many waits the median above is based on | A, B |
| `hist_wait_nbr_4w` | History | Median of completed waits ending in the neighbourhood at the same hour and weekday over the previous 4 weeks | A, B |
| `n_hist_waits_nbr_4w` | History | How many waits that median is based on | A, B |
| `rain_lag_1h` | Weather | Rainfall (mm) in hour k from Open-Meteo for central Bangkok; optional | A, B |
| `minutes_empty` | Taxi state | Minutes since the wait started (already a column of the snapshot table) | B only |

Not built yet (optional, from the plan): `predicted_pickups` (Model A output as a Model B
input, must be out-of-fold), place features from OpenStreetMap.

## Notes for modelling

- Leave out 29 August 2025 and the January warm-up (1 to 28 Jan) from training rows; the
  warm-up has no 4-week history by design.
- Missing values: LightGBM handles NaN itself. For Cox, fill NaN (for example with the
  training median) and add a 0/1 "was missing" column for `median_wait_nbr_3h`,
  `hist_wait_nbr_4w` and `speed_nbr_1h`.
- Snapshots of one wait share a pickup: keep each `episode_id` in one split, and sample by
  wait or by day rather than by row when sub-sampling.
- Model A rows exist only where an empty taxi was present; a zero there is a real zero.
