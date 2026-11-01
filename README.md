# Bangkok Taxi Area Recommender

Machine Learning (AT82.03) group project, Asian Institute of Technology.

When a street-hail taxi in Bangkok is empty, which nearby area gets it its next
passenger soonest, counting the drive there? We learn this from one year (2025) of
Bangkok taxi GPS data published by the iTIC Foundation
([Longdo Traffic open data](https://traffic.longdo.com/opendata/probe-data/), CC BY 4.0).

This README covers the parts that are finished: the **data pipeline and data audit**
(raw GPS pings to clean trips, waits and snapshot rows) and the **features** (the
Model A and Model B training tables). Models and the app come next.

## Contents

1. [What the pipeline does](#1-what-the-pipeline-does)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Input data](#4-input-data)
5. [Run the pipeline](#5-run-the-pipeline)
6. [Run the data audit](#6-run-the-data-audit)
6b. [Build the features](#6b-build-the-features)
7. [Outputs](#7-outputs)
8. [The rules (v1.0)](#8-the-rules-v10)
9. [Changing a rule](#9-changing-a-rule)
10. [Troubleshooting](#10-troubleshooting)

## 1. What the pipeline does

```
EDA notebook  ->  eda_output_bangkok/taxi_days/YYYYMMDD.pkl   (taxi GPS pings, one file per day)
                         |
   clean.py         load the day + 3 h of the next day, remove bad pings, fix light flicker
                         |
   episodes.py      light off = pickup, light on = drop-off  ->  trips and waits,
                    with censoring, rests, fake pickups and missed-trip flags
                         |
   snapshots.py     one row per waiting taxi every 5 min  ->  Model B training rows
                         |
   pipeline_output/{trips, empty, snapshots, manifest}/YYYYMMDD.*
```

The **audit** checks the rules by eye: it samples 50 waits, draws a picture of each GPS
trace, and the team labels what really happened. The labels were used to set and
freeze the rules (version v1.0).

## 2. Repository layout

| Path | What it is |
| --- | --- |
| `src/taxi/config.py` | **Every rule and threshold**, folder paths, rules version |
| `src/taxi/clean.py` | Load a day (+ 3 h look-ahead), remove duplicates and GPS jumps, fix light flicker |
| `src/taxi/episodes.py` | Build trips and waits; censoring, rests, fake pickups, missed-trip flag |
| `src/taxi/snapshots.py` | Build Model B rows (a snapshot every 5 min of each wait) |
| `src/taxi/audit.py` | Pick audit samples, draw audit pictures, make the labelling sheet |
| `src/taxi/geo.py` | H3 hexagons (stand-in grid if `h3` is missing), distances |
| `src/taxi/storage.py` | Save and load tables (Parquet, or pickle without `pyarrow`) |
| `scripts/build_days.py` | Run the pipeline for one day or a date range |
| `scripts/make_audit_sample.py` | Draw an audit sample: pictures + labelling sheet |
| `scripts/check_audit_rules.py` | Show what the current rules do to the 50 audited waits |
| `scripts/summarize_build.py` | Summarise a full build into `docs/build_summary_<rules>.md` |
| `src/taxi/hexhour.py` | Hexagon-hour base table: empty taxis, real pickups, traffic speed |
| `src/taxi/features.py` | All features, with the no-future timing rule |
| `scripts/build_hexhour.py` | Build the hexagon-hour tables |
| `scripts/build_features.py` | Build the Model A and Model B training tables |
| `scripts/check_build.py` | Check a build is complete, uses H3 and one file format |
| `scripts/coverage_report.py` | Write `docs/coverage_report.md` |
| `scripts/eda_update.py` | Write `docs/eda_update_v1.md` and `docs/figures/` |
| `scripts/verify_outputs.py` | Deep check of every table (counts, labels, H3, features re-computed by hand) |
| `tests/test_episodes.py` | 12 small made-up traces with known answers |
| `tests/test_features.py` | 7 timing tests: no feature may use the future |
| `audit/audit_sheet.csv` | The labelled audit (50 waits) that froze rules v1.0 |
| `audit/claude_labels.csv` | The same 50 waits with label, confidence and reason |
| `audit/rules_check_v1.0.csv` | What rules v1.0 do to each audited wait |
| `docs/data_pipeline.md` | Detailed pipeline guide, every output column |
| `docs/audit_findings.md` | What the audit found and which rule each finding led to |
| `docs/build_summary_v1.0.md` | Full-year numbers under rules v1.0 |
| `docs/features.md` | Feature dictionary, training tables, timing rule |
| `docs/coverage_report.md` | How often features have real data behind them |
| `docs/eda_update_v1.md` | Key EDA numbers and charts recomputed with the cleaned data |
| `EDA-Bangkok-all.ipynb` | Exploratory data analysis (also makes `taxi_days/`) |

Not in git (too big, see `.gitignore`): `eda_output_bangkok/`, `pipeline_output/`,
`audit/plots/`, `data/`.

## 3. Setup

Python 3.10 or newer. From the project folder:

```bash
cd ~/DSAI/ML-LAB/Project              # or wherever you cloned the repo
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt       # pandas, numpy, h3, pyarrow, matplotlib, pytest
```

Run `source venv/bin/activate` every time you open a new terminal for this project.

Check the install:

```bash
python -c "import h3, pyarrow; print('h3', h3.__version__, '| pyarrow', pyarrow.__version__)"
python tests/test_episodes.py         # must end with: all 12 tests passed
```

Without `h3` the code still runs but uses a stand-in hexagon grid and prints a warning.
Without `pyarrow` it saves `.pkl` instead of `.parquet`. Build final outputs with both installed.

## 4. Input data

The pipeline reads cleaned daily GPS files made by the EDA notebook:

```
eda_output_bangkok/taxi_days/20250101.pkl ... 20251231.pkl   (365 files, about 16 GB)
```

Each file holds one day of pings from taxis only, already filtered to valid GPS,
greater Bangkok, and vehicles whose light switches at least 4 times that day.
Columns: `vid`, `t`, `lat`, `lon`, `speed`, `occ` (1 = passenger on board), `acc` (1 = engine on).

**To make these files:** download the 12 monthly 2025 archives from the iTIC page
above into `~/Downloads`, then run `EDA-Bangkok-all.ipynb` (step 1 writes `taxi_days/`).

If the folder is somewhere else:

```bash
export TAXI_DATA_DIR=/path/to/eda_output_bangkok
export TAXI_OUT_DIR=/path/to/pipeline_output        # optional, where outputs go
```

**Time convention:** `t` is Bangkok wall-clock time stored as seconds as if it were UTC.
`pd.to_datetime(t, unit="s")` gives Bangkok time directly. Never add 7 hours.

## 5. Run the pipeline

**One day** (about 2 seconds), to check everything works:

```bash
python scripts/build_days.py --day 20250115
```

Expected output, roughly:

```
grid=h3 rules=v1.0 format=parquet out=.../pipeline_output
20250115 {'taxis': 1260, 'trips': ..., 'empty_periods': ..., 'empty_censored_share': ..., 'fake_trips': ..., 'possible_missed_trip': ..., 'snapshots': ..., 'seconds': 1.9}
```

**The whole year** (about 10 minutes):

```bash
python scripts/build_days.py --start 20250101 --end 20251231
```

- Days already built are skipped, so if it stops you can run the same command again.
- Add `--force` to rebuild days that already exist (needed after any rule change).

**Summarise the build:**

```bash
python scripts/summarize_build.py      # writes docs/build_summary_<rules>.md
```

**Look at the results** in a notebook:

```python
import sys; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from taxi.storage import load_tables

day = pd.read_parquet("pipeline_output/empty/20250115.parquet")      # one day of waits
snaps = load_tables(Path("pipeline_output/snapshots"))                # all days (about 30 million rows, needs RAM)
```

## 6. Run the data audit

The audit that froze rules v1.0 is already done and labelled (`audit/audit_sheet.csv`).
These steps show how to reproduce it, check the rules against it, or run a new one.

### 6.1 Draw a sample

```bash
python scripts/make_audit_sample.py              # seed 42
python scripts/make_audit_sample.py --seed 7     # a different sample
```

- Builds waits for 9 training-period days (one per month, January to mid-September;
  never validation or test days, and never 29 August).
- Picks 50 waits in 7 groups so unusual cases are covered: night (8), very short under
  3 min (7), long 60+ min (8), cut by a rest or over 180 min (7), ended by a GPS gap (6),
  mostly engine off (7), random (7).
- Writes `audit/runs/sheet_<rules>_seed<seed>.csv` and one picture per wait in
  `audit/plots/<rules>_seed<seed>/`. It never overwrites `audit/audit_sheet.csv`.

### 6.2 Read the pictures

Each picture has two parts.

- **Left, map:** kilometres around the drop-off (black triangle at 0, 0; right = east,
  up = north). Green dots = GPS pings during the wait, grey = 10 minutes before and
  after, blue square = where the wait ended. Pings at one spot overlap, so check the
  count in the corner box (pings, km driven, furthest distance, longest engine-off stop).
- **Right, three timelines** on one clock (minutes from the drop-off; the pale yellow
  band is the wait): speed, engine (olive on, grey off), light (green empty, red
  occupied; x = flicker fixed).

### 6.3 Label

Two people label the same 50 waits **separately** (columns `label_A` and `label_B`),
then agree on `agreed_label` and write the reason in `notes`.

| Label | Use when the trace shows |
| --- | --- |
| `searching` | Moving around the city at normal speeds, then a pickup |
| `waiting_at_spot` | Stopped in one place (engine on or off), often picked up at that same spot |
| `off_duty` | Long stop with the engine off, often at night or for hours |
| `light_error_or_missed_trip` | Fast or far driving while "empty", or light changes that do not fit the movement |
| `gps_problem` | Jumps, frozen position, missing stretches |
| `unsure` | Cannot tell |

Tip: a light switched to "occupied" while the taxi stays parked with the engine off is
a driver resting, not a passenger.

### 6.4 Check the rules against the labels

```bash
python scripts/check_audit_rules.py
```

Rebuilds each audited day with the current rules and prints, for each of the 50
labelled waits, what the rules now do: kept with a pickup, cut at a rest, censored
(and why), flagged as a possible missed trip, or dropped. Writes
`audit/rules_check_<rules>.csv`. Off-duty waits should be cut or censored; real waits
should be kept.

### 6.5 Freeze the rules and build

1. Change thresholds in `src/taxi/config.py` if the labels show a rule is wrong.
2. Bump `RULES_VERSION` (for example `"v1.1"`).
3. `python tests/test_episodes.py` (add a test for any new rule).
4. `python scripts/check_audit_rules.py`
5. `python scripts/build_days.py --start 20250101 --end 20251231 --force`
6. `python scripts/summarize_build.py`
7. Record the change in `docs/audit_findings.md`.

What the v1.0 audit found and why each rule exists: `docs/audit_findings.md`.

## 6b. Build the features

Run after the pipeline (section 5). Details and the full feature dictionary: `docs/features.md`.

```bash
python scripts/build_hexhour.py  --start 20250101 --end 20251231   # hexagon-hour base table, about 5 min
python scripts/build_features.py --start 20250101 --end 20251231   # training tables, about 10 min
python tests/test_features.py                                      # must end with: all 7 tests passed
python scripts/check_build.py --features                           # every table complete, H3, one format
python scripts/coverage_report.py                                  # docs/coverage_report.md
python scripts/eda_update.py                                       # docs/eda_update_v1.md + figures
python scripts/verify_outputs.py                                   # deep check of every table, docs/verification_report.md
```

| Output | One row per | Label |
| --- | --- | --- |
| `pipeline_output/features/modelA/` | Hexagon-hour with an empty taxi present (12.7 million) | `target_pickups` |
| `pipeline_output/features/snapshots/` | Waiting taxi every 5 min (30.0 million) | `minutes_until_end`, censored when `pickup_seen` is False |

**Timing rule:** a row at current hour h1 only uses data up to the last completed hour
(h1 - 1) and waits that ended by then. At 14:20, "last hour" means 13:00 to 14:00.

Features for a day need the 29 days before it and the day after, so after rebuilding some
days, rebuild features with `--force` for the following month too.

## 7. Outputs

`pipeline_output/` has one file per day in each folder.

| Folder | One row per | Key columns |
| --- | --- | --- |
| `trips/` | Trip (pickup to drop-off) | times, `km`, `slow_min`, `is_fake` (never count as a pickup), `has_gap`, pickup and drop-off hexagons |
| `empty/` | Wait (drop-off to next pickup) | `start_type`, `minutes`, `pickup_seen`, `censor_reason`, `km_driven`, `far_km`, `engine_off_share`, `longest_still_min`, `possible_missed_trip`, `hex_start`, `hex_end` |
| `snapshots/` | Waiting taxi every 5 min (Model B rows) | `minutes_empty`, `lat`, `lon`, `hex`, `hour`, `speed`, label `minutes_until_end`, `pickup_seen`, `censor_reason` |
| `manifest/` | Day (JSON) | counts, rules version, grid used, seconds taken |
| `hexhour/` | Hexagon-hour | `vacant_taxis`, `pickups` (real), `speed_med`, `n_speed` |
| `features/modelA/`, `features/snapshots/` | See section 6b | features listed in `docs/features.md` |

`pickup_seen = False` means censored: the wait lasted at least `minutes_until_end`.
Every column is explained in `docs/data_pipeline.md`.

**Full year, rules v1.0:** 4.96 million real trips, 389,000 fake trips removed (7.3%),
5.65 million waits (15.6% censored), median completed wait 18 minutes, 30.0 million
snapshot rows. Details: `docs/build_summary_v1.0.md`.

## 8. The rules (v1.0)

All numbers are in `src/taxi/config.py`.

| Rule | Value |
| --- | --- |
| Remove GPS jumps | Faster than 150 km/h between pings |
| Fix light flicker | A single ping differing from both neighbours |
| Look-ahead into next day | 3 hours |
| GPS gap ends a wait (censored `gap`) | Silence over 10 minutes |
| Rest cuts a wait (censored `off_duty`), new wait `after_rest` when moving again | 45+ minutes parked with the engine off |
| Fake pickup (censored `off_duty`) | 5+ minutes parked with the engine off right after the light goes off |
| Fake trip (`is_fake`) | Under 0.5 km and mostly parked |
| Very long wait (censored `too_long`) | Over 180 minutes, cut at 180 |
| Possible missed trip (left out of snapshots) | Over 40 km at over 40 km/h, or ending over 30 km away |
| Zero-length waits | Dropped |
| Snapshots | Every 5 minutes, up to 120 minutes into a wait |

## 9. Changing a rule

1. Edit the value in `src/taxi/config.py` and bump `RULES_VERSION`.
2. Add or update a test in `tests/test_episodes.py`; run it.
3. Run `scripts/check_audit_rules.py` to see the effect on the audited waits.
4. Rebuild with `--force`, then run `scripts/summarize_build.py`.
5. Note the change and the reason in `docs/audit_findings.md`.

Each output's manifest records the rules version that made it, so mixed builds are easy to spot.

## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| `command not found: python` | Use `python3`, or activate the environment: `source venv/bin/activate` |
| `ModuleNotFoundError: taxi` | Run scripts from the project folder, not from inside `scripts/` |
| Warning `h3 is not installed` | `pip install h3` with the environment active |
| Output files end in `.pkl` | `pip install pyarrow`, then rebuild with `--force` |
| Both `.pkl` and `.parquet` in an output folder | Loading prefers Parquet; move the old `.pkl` files out |
| `no input file, skipped` | That day is not in `taxi_days/`, or `TAXI_DATA_DIR` points to the wrong folder |
| `already built (use --force to rebuild)` | Add `--force` |
| A test fails | A rule change broke known behaviour; fix it before building |
