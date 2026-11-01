"""Build the Model A and Model B training tables with features (run after build_hexhour.py).

  python scripts/build_features.py --start 20250101 --end 20251231
  python scripts/build_features.py --day 20250115 --force

Writes
  pipeline_output/features/snapshots/YYYYMMDD.*   Model B: every snapshot + label + features
  pipeline_output/features/modelA/YYYYMMDD.*      Model A: every hexagon-hour with an empty
                                                  taxi present + pickups (target) + features
Features use only data from before the prediction moment (see src/taxi/features.py).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config, geo  # noqa: E402
from taxi.clean import day_bounds  # noqa: E402
from taxi.features import HISTORY_DAYS, FeatureWindow, completed_waits, load_weather  # noqa: E402
from taxi.storage import FORMAT, load_day_table, save_table  # noqa: E402

SNAP_COLS = ["episode_id", "vid", "start_type", "snapshot_t", "minutes_empty", "hex", "speed",
             "minutes_until_end", "pickup_seen", "censor_reason"]


def shift(day, n):
    return (pd.Timestamp(day) + pd.Timedelta(days=n)).strftime("%Y%m%d")


class Builder:
    def __init__(self):
        wpath = config.DATA_DIR / "weather_hourly.csv"
        self.win = FeatureWindow(load_weather(wpath) if wpath.exists() else None)
        self.waits_src = {}     # source day -> completed waits of waits that STARTED that day

    def _waits_from(self, day):
        if day not in self.waits_src:
            e = load_day_table(config.OUT_DIR / "empty", day)
            self.waits_src[day] = completed_waits(e) if e is not None else None
        return self.waits_src[day]

    def _waits_ending(self, day):
        start, end = day_bounds(day)
        parts = [w for w in (self._waits_from(shift(day, -1)), self._waits_from(day)) if w is not None]
        w = pd.concat(parts, ignore_index=True)
        return w[(w["hour_id"] >= start // 3600) & (w["hour_id"] < end // 3600)]

    def ensure_window(self, day):
        for i in range(-HISTORY_DAYS, 2):              # D-29 .. D+1
            d = shift(day, i)
            if d in self.win.days:
                continue
            hh = load_day_table(config.OUT_DIR / "hexhour", d)
            if hh is None:
                continue
            self.win.add_day(d, hh, self._waits_ending(d))
        self.win.drop_before(shift(day, -HISTORY_DAYS))
        for d in [d for d in self.waits_src if d < shift(day, -HISTORY_DAYS - 1)]:
            del self.waits_src[d]

    def build(self, day):
        self.ensure_window(day)
        out = config.OUT_DIR / "features"
        snaps = load_day_table(config.OUT_DIR / "snapshots", day)
        f = self.win.features(snaps["hex"].to_numpy(), (snaps["snapshot_t"] // 3600).to_numpy())
        snap_tab = pd.concat([snaps[SNAP_COLS].reset_index(drop=True), f], axis=1)
        save_table(snap_tab, out / "snapshots", day)

        hh = load_day_table(config.OUT_DIR / "hexhour", day)
        rows = hh[hh["vacant_taxis"] > 0].reset_index(drop=True)
        f = self.win.features(rows["hex"].to_numpy(), rows["hour_id"].to_numpy())
        a_tab = pd.concat([rows[["hex", "hour_id"]],
                           rows[["pickups"]].rename(columns={"pickups": "target_pickups"}),
                           rows[["vacant_taxis"]].rename(columns={"vacant_taxis": "exposure_vacant_taxis"}),
                           f], axis=1)
        save_table(a_tab, out / "modelA", day)
        return len(snap_tab), len(a_tab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    days = [a.day] if a.day else [x.strftime("%Y%m%d") for x in pd.date_range(a.start, a.end)]
    out = config.OUT_DIR / "features"
    print(f"grid={geo.GRID} format={FORMAT} out={out}")
    b = Builder()
    for day in days:
        done = (out / "snapshots" / f"{day}.{FORMAT}").exists() and (out / "modelA" / f"{day}.{FORMAT}").exists()
        if done and not a.force:
            continue
        if load_day_table(config.OUT_DIR / "snapshots", day) is None:
            print(day, "snapshots missing, run build_days.py first")
            continue
        t0 = time.time()
        n_s, n_a = b.build(day)
        print(day, f"{n_s:,} snapshot rows, {n_a:,} Model A rows, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
