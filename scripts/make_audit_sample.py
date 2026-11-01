"""Make the audit sample: 50 empty periods spread over the year, one picture each,
and a labelling sheet that both labellers fill in.

  python scripts/make_audit_sample.py              # 9 training-period days, one per month
  python scripts/make_audit_sample.py --seed 7     # a different sample

Writes audit/runs/sheet_<rules>_seed<seed>.csv and audit/plots/<rules>_seed<seed>/*.png.
(The labelled audit that froze rules v1.0 is audit/audit_sheet.csv; it is never overwritten.)
Only training-period days (before 16 September) are used, so the audit never
looks at validation or test data.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.audit import labelling_sheet, plot_episode, sample_for_audit  # noqa: E402
from taxi.clean import clean, day_bounds, load_day  # noqa: E402
from taxi.episodes import build_episodes  # noqa: E402

TRAIN_MONTHS = range(1, 10)     # Jan..Sep; September days are picked before the 16th


def pick_days(seed: int):
    rng = np.random.default_rng(seed)
    days = []
    for m in TRAIN_MONTHS:
        last = 15 if m == 9 else pd.Period(f"2025-{m:02d}").days_in_month
        first = 8 if m == 1 else 1             # skip the first week of January
        day = f"2025{m:02d}{rng.integers(first, last + 1):02d}"
        if day != "20250829" and (config.TAXI_DAYS_DIR / f"{day}.pkl").exists():
            days.append(day)
    return days


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    days = pick_days(a.seed)
    print("audit days:", days)

    cleaned, empties = {}, []
    for day in days:
        raw = load_day(day)
        d, _ = clean(raw)
        _, day_end = day_bounds(day)
        _, empty = build_episodes(d, day_end, raw.attrs["has_next_day"])
        cleaned[day] = d
        empties.append(empty.assign(day=day))
        print(day, len(empty), "empty periods")
    empty = pd.concat(empties, ignore_index=True)

    sample = sample_for_audit(empty, a.seed)
    run = f"{config.RULES_VERSION}_seed{a.seed}"
    sheet = labelling_sheet(sample, plot_dir=f"plots/{run}")
    out = config.AUDIT_DIR
    (out / "plots" / run).mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for ep, row in zip(sample.itertuples(index=False), sheet.itertuples(index=False)):
        plot_episode(cleaned[ep.day], ep, out / row.plot_file)
    dst = out / "runs" / f"sheet_{run}.csv"
    sheet.to_csv(dst, index=False)
    print(f"wrote {len(sheet)} periods to {dst}")
    print(sheet["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
