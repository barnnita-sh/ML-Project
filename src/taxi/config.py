"""All paths and thresholds for the data pipeline, in one place.

Every number here matches Section 16 of the team document
("Bangkok Taxi Area Recommender: Where the Project Stands").
Change a value here, bump RULES_VERSION, and rebuild the outputs.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Cleaned per-day GPS files made by the EDA notebook (one .pkl per day).
# Override with:  export TAXI_DATA_DIR=/path/to/eda_output_bangkok
DATA_DIR = Path(os.environ.get("TAXI_DATA_DIR", PROJECT_DIR / "eda_output_bangkok"))
TAXI_DAYS_DIR = DATA_DIR / "taxi_days"

# Where this pipeline writes its tables. Not committed to git.
OUT_DIR = Path(os.environ.get("TAXI_OUT_DIR", PROJECT_DIR / "pipeline_output"))
AUDIT_DIR = PROJECT_DIR / "audit"

# Bump this whenever a rule below changes, then rebuild.
# v1.0 = rules frozen after the 50-period audit (audit/claude_labels.csv, docs/audit_findings.md)
RULES_VERSION = "v1.0"

# ---------------------------------------------------------------- time
# IMPORTANT: column `t` in taxi_days is Bangkok wall-clock time stored as
# seconds since 1970 *as if it were UTC*. So:
#   pd.to_datetime(t, unit="s")  -> Bangkok local time (do NOT add 7 hours)
#   (t // 3600) % 24             -> Bangkok local hour
#   midnight of a day            -> pd.Timestamp("20250115").value // 10**9

# ---------------------------------------------------------------- area
BBOX = dict(lat_min=13.45, lat_max=14.15, lon_min=100.25, lon_max=100.95)
H3_RES = 8                      # 0.74 km2 hexagons

# ---------------------------------------------------------------- cleaning
MIN_SWITCHES = 4                # light changes per day for a vehicle to count as a taxi (applied in the EDA step)
MAX_SPEED_KMH = 150             # a point implying a faster jump from the previous one is a GPS error
MOVING_KMH = 5                  # below this speed a taxi counts as stationary

# ---------------------------------------------------------------- episodes
GAP_SEC = 600                   # silence longer than this ends a period as censored (kept after audit)
LOOKAHEAD_SEC = 3 * 3600        # read this much of the next day so periods crossing midnight are complete

# Rest: a stretch stationary with the engine off lasting at least this long is a rest,
# not waiting. The empty period is cut (censored, reason "off_duty") where the rest
# starts, and a new period ("after_rest") starts when the taxi moves again.
# Audit: waits with the engine off up to 39 min ended in pickups; rests were 54+ min.
OFFDUTY_STILL_MIN = 45

# Drivers switch the light to "occupied" when they park to rest ("not available").
# Fake pickup: right after the light goes off, the taxi stays parked with the engine
#   off for FAKE_TRIP_STILL_MIN or more. The empty period before it is censored
#   (reason "off_duty") instead of counted as ending in a pickup.  Audit: #4, #13, #19, #41.
# Fake trip: the whole "trip" is shorter than FAKE_TRIP_KM and mostly parked. It is
#   marked is_fake and must not be counted as a pickup.  Audit: #11, #12, #33.
FAKE_TRIP_KM = 0.5
FAKE_TRIP_STILL_MIN = 5

# Very long periods are censored at this length ("too_long") instead of being deleted.
MAX_EMPTY_MIN = 180

# Probable unrecorded passenger trip: fast long drives, or ending far away, while the
# light says empty. Flagged, kept in the empty table, left out of the snapshot table.
# Audit: #22, #31, #35. (Slow cruising over many km, like #18 or #27, is NOT flagged.)
MISSED_TRIP_KM = 40             # more than this many km driven ...
MISSED_TRIP_KMH = 40            # ... at an average above this speed, or
MISSED_TRIP_FAR_KM = 30         # ending more than this many km (straight line) from the start

# ---------------------------------------------------------------- snapshots (Model B rows)
SNAPSHOT_STEP_MIN = 5           # one snapshot of each empty taxi every 5 minutes
SNAPSHOT_MAX_MIN = 120          # stop taking snapshots after 120 minutes empty

# ---------------------------------------------------------------- audit
AUDIT_N = 50                    # traces labelled by BOTH labellers, then compared
AUDIT_CONTEXT_MIN = 10          # minutes of GPS shown before and after each period
