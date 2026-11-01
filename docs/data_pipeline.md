# Data pipeline: trips, empty periods, snapshots (rules v1.0)

Task 1 of the plan: turn the cleaned daily GPS files into the tables every later step
uses. Status: **done**. Rules frozen as v1.0 after the audit (`docs/audit_findings.md`);
full year built (`docs/build_summary_v1.0.md`).

## Setup (once, on your own computer)

```bash
cd ~/DSAI/ML-LAB/Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # includes h3 and pyarrow
python tests/test_episodes.py          # 12 checks, ends with "all 12 tests passed"
```

Input: `eda_output_bangkok/taxi_days/YYYYMMDD.pkl` (made by the EDA notebook, 365 files,
about 16 GB, not in git). If it lives elsewhere: `export TAXI_DATA_DIR=/path/to/eda_output_bangkok`.

## Commands

| Command | What it does |
| --- | --- |
| `python scripts/build_days.py --day 20250115` | Build one day |
| `python scripts/build_days.py --start 20250101 --end 20251231` | Build the year (about 10 min; skips days already built) |
| add `--force` | Rebuild days that already exist (after any rule change) |
| `python scripts/summarize_build.py` | Write `docs/build_summary_<rules>.md` from the built tables |
| `python scripts/check_audit_rules.py` | Show what the current rules do to the 50 audited periods |
| `python scripts/make_audit_sample.py --seed 7` | Draw a new audit sample (never overwrites `audit/audit_sheet.csv`) |

**Important:** the year was first built where `h3` and `pyarrow` were not available, so
`pipeline_output/` currently holds `.pkl` files with the stand-in hexagon grid. On your
own computer, after installing the requirements, rebuild once with h3:

```bash
python scripts/build_days.py --start 20250101 --end 20251231 --force
```

That writes `.parquet` files with real H3 IDs. Then move the old `.pkl` files out of
`pipeline_output/trips`, `empty` and `snapshots` (loading prefers parquet if both exist).

## The rules, in order

1. **Clean** (`clean.py`): load the day plus 3 hours of the next day; drop duplicate
   timestamps; drop points implying over 150 km/h; fix single-ping light flicker.
2. **Trips and waits** (`episodes.py`): light off = pickup, light on = drop-off.
3. **Censor** a wait (we stop watching, "at least this long") when: GPS silent over 10 min
   (`gap`); a rest or fake pickup (`off_duty`); over 180 min (`too_long`); data ends.
4. **Rests:** 45+ min parked with the engine off cuts the wait; a new wait starts when the
   taxi moves (`start_type = after_rest`).
5. **Fake pickups / trips:** parked engine off 5+ min right after a "pickup" = not a pickup;
   a "trip" under 0.5 km mostly parked = `is_fake`.
6. **Possible missed trips:** over 40 km at over 40 km/h, or ending over 30 km away.
   Kept in `empty`, left out of `snapshots`.
7. **Zero-length waits** are dropped.
8. **Snapshots** (`snapshots.py`): every 5 min from the start of each wait, up to 120 min.

All numbers live in `src/taxi/config.py`.

## Output tables (`pipeline_output/`, one file per day)

**empty/** one row per wait

| Column | Meaning |
| --- | --- |
| episode_id | `vid-t_start`, unique |
| start_type | `dropoff`, or `after_rest` (the driver became available again after a rest) |
| t_start, t_end | Seconds, Bangkok wall-clock stored as if UTC (do not add 7 hours) |
| minutes | Length of the wait |
| pickup_seen | True = ended with a real pickup; False = censored |
| censor_reason | `off_duty`, `too_long`, `gap`, `lookahead_end`, `end_of_data`, or empty |
| km_driven, far_km | km driven during the wait; straight-line km start to end |
| engine_off_share | Share of the wait's time with the engine off |
| longest_still_min | Longest stretch parked with the engine off |
| possible_missed_trip | Probably a passenger trip without the light; left out of training |
| hex_start, hex_end | Hexagon at the start and at the end (pickup hexagon if pickup_seen) |

**trips/** one row per trip: times, km, slow minutes, engine-off share,
`is_fake` (do not count as a pickup), `has_gap`, pickup and drop-off hexagons.

**snapshots/** Model B rows: one per waiting taxi every 5 minutes

| Column | Meaning |
| --- | --- |
| episode_id, vid, start_type | Which wait and taxi |
| snapshot_t, hour | Moment of the snapshot and its Bangkok hour |
| minutes_empty | Minutes since the wait started (0, 5, 10, ...) |
| lat, lon, hex, speed | Where the taxi was at that moment |
| minutes_until_end | The label: minutes to the pickup (or to censoring) |
| pickup_seen, censor_reason | False = censored: the wait was at least minutes_until_end |

**manifest/** one JSON per day: counts, rules version, grid used, seconds taken.

Features are added in the next task (feature tables), not here.

## Known limits

- Records whose timestamp falls on a different date than their file were already dropped
  in the EDA step (0.4%); moving them needs the raw archives.
- The taxi rule (at least 4 light switches a day) was applied in the EDA step.
- 29 August 2025 (low coverage) is built like any day; leave it out at modelling time.
- An `after_rest` wait that starts after midnight is not created (small loss).
