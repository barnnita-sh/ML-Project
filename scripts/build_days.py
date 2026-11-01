"""Build trips, empty periods and Model B snapshots for one or more days.

Examples
  python scripts/build_days.py --day 20250115
  python scripts/build_days.py --start 20250101 --end 20250131
  python scripts/build_days.py --start 20250101 --end 20251231 --force

Writes to pipeline_output/ (or $TAXI_OUT_DIR):
  trips/YYYYMMDD.parquet      one row per trip
  empty/YYYYMMDD.parquet      one row per empty period
  snapshots/YYYYMMDD.parquet  one row per snapshot (Model B rows, no features yet)
  manifest/YYYYMMDD.json      counts, rules version, grid used
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config, geo  # noqa: E402
from taxi.clean import clean, day_bounds, load_day  # noqa: E402
from taxi.episodes import build_episodes  # noqa: E402
from taxi.snapshots import build_snapshots  # noqa: E402
from taxi.storage import FORMAT, save_json, save_table  # noqa: E402


def build_one(day: str, force: bool = False) -> dict:
    out = config.OUT_DIR
    if not force and (out / "snapshots" / f"{day}.{FORMAT}").exists():
        return {"day": day, "skipped": True}
    t0 = time.time()
    raw = load_day(day)
    d, stats = clean(raw)
    _, day_end = day_bounds(day)
    trips, empty = build_episodes(d, day_end, raw.attrs["has_next_day"])
    snaps = build_snapshots(d, empty)

    save_table(trips, out / "trips", day)
    save_table(empty, out / "empty", day)
    save_table(snaps, out / "snapshots", day)
    info = {
        "day": day, "rules_version": config.RULES_VERSION, "grid": geo.GRID,
        "taxis": int(raw.loc[raw["t"] < day_end, "vid"].nunique()), **stats,
        "trips": len(trips), "empty_periods": len(empty),
        "empty_censored_share": round(float((~empty["pickup_seen"]).mean()), 4),
        "censor_reasons": empty["censor_reason"].value_counts().to_dict(),
        "start_types": empty["start_type"].value_counts().to_dict(),
        "possible_missed_trip": int(empty["possible_missed_trip"].sum()),
        "fake_trips": int(trips["is_fake"].sum()),
        "snapshots": len(snaps), "seconds": round(time.time() - t0, 1),
    }
    save_json(info, out / "manifest" / f"{day}.json")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.day:
        days = [a.day]
    elif a.start and a.end:
        days = [x.strftime("%Y%m%d") for x in pd.date_range(a.start, a.end)]
    else:
        p.error("give --day or --start and --end")
    print(f"grid={geo.GRID} rules={config.RULES_VERSION} format={FORMAT} out={config.OUT_DIR}")
    for day in days:
        if not (config.TAXI_DAYS_DIR / f"{day}.pkl").exists():
            print(day, "no input file, skipped")
            continue
        info = build_one(day, a.force)
        if info.get("skipped"):
            print(day, "already built (use --force to rebuild)")
        else:
            print(day, {k: info[k] for k in ["taxis", "trips", "empty_periods", "empty_censored_share",
                                             "fake_trips", "possible_missed_trip", "snapshots", "seconds"]})


if __name__ == "__main__":
    main()
