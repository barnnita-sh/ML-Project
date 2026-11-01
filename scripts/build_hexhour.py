"""Build the hexagon-hour base table for one or more days (run after build_days.py).

  python scripts/build_hexhour.py --start 20250101 --end 20251231
  python scripts/build_hexhour.py --day 20250115 --force

Reads taxi_days/ (GPS pings) and pipeline_output/{trips,empty}/, writes
pipeline_output/hexhour/YYYYMMDD.*  (see src/taxi/hexhour.py for the columns).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config, geo  # noqa: E402
from taxi.clean import clean, day_bounds, load_day  # noqa: E402
from taxi.hexhour import build_hexhour  # noqa: E402
from taxi.storage import FORMAT, load_day_table, save_table  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    days = [a.day] if a.day else [x.strftime("%Y%m%d") for x in pd.date_range(a.start, a.end)]
    out = config.OUT_DIR
    print(f"grid={geo.GRID} format={FORMAT} out={out / 'hexhour'}")
    for day in days:
        if not a.force and (out / "hexhour" / f"{day}.{FORMAT}").exists():
            continue
        trips = load_day_table(out / "trips", day)
        empty = load_day_table(out / "empty", day)
        if trips is None or empty is None:
            print(day, "pipeline tables missing, run build_days.py first")
            continue
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        t0 = time.time()
        d, _ = clean(load_day(day))
        start, end = day_bounds(day)
        hh = build_hexhour(d, trips, empty, load_day_table(out / "empty", prev), start, end)
        save_table(hh, out / "hexhour", day)
        print(day, f"{len(hh):,} hexagon-hours, {int(hh['pickups'].sum()):,} pickups, "
                   f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
