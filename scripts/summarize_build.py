"""Summarise a full build: counts, censoring, waits, and the effect of each rule.

  python scripts/summarize_build.py

Reads pipeline_output/{manifest,empty,trips,snapshots} and writes
docs/build_summary_<RULES_VERSION>.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.storage import load_tables  # noqa: E402

COLS_E = ["minutes", "pickup_seen", "censor_reason", "start_type", "possible_missed_trip", "engine_off_share"]


def main():
    out = config.OUT_DIR
    man = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / "manifest").glob("*.json"))])
    e = load_tables(out / "empty")[COLS_E]
    t = load_tables(out / "trips")[["minutes", "km", "is_fake"]]
    n_snap = int(man["snapshots"].sum())

    real = t[~t["is_fake"]]
    seen = e[e["pickup_seen"] & ~e["possible_missed_trip"]]
    q = seen["minutes"].quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist()
    reasons = e["censor_reason"].replace("", "none (pickup seen)").value_counts()
    starts = e["start_type"].value_counts()

    lines = [
        f"# Build summary, rules {config.RULES_VERSION}",
        "",
        f"Days built: {len(man)} ({man['day'].min()} to {man['day'].max()}). Grid: {', '.join(man['grid'].unique())}.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Taxis per day (median) | {int(man['taxis'].median()):,} |",
        f"| GPS records after cleaning | {int(man['rows_out'].sum()):,} |",
        f"| Speed jumps removed | {int(man['speed_jumps_removed'].sum()):,} |",
        f"| Light flicker fixed | {int(man['flicker_fixed'].sum()):,} |",
        f"| Trips, all | {len(t):,} |",
        f"| Trips marked fake (resting with light 'occupied') | {int(t['is_fake'].sum()):,} ({t['is_fake'].mean():.1%}) |",
        f"| Real trips | {len(real):,} (median {real['minutes'].median():.1f} min, {real['km'].median():.1f} km) |",
        f"| Empty periods | {len(e):,} |",
        f"| ... starting at a drop-off / after a rest | {int(starts.get('dropoff', 0)):,} / {int(starts.get('after_rest', 0)):,} |",
        f"| ... flagged possible missed trip (left out of snapshots) | {int(e['possible_missed_trip'].sum()):,} ({e['possible_missed_trip'].mean():.1%}) |",
        f"| ... censored | {(~e['pickup_seen']).mean():.1%} |",
        f"| Wait until pickup, completed and not flagged (25% / median / 75% / 90%) | {q[0]} / {q[1]} / {q[2]} / {q[3]} min |",
        f"| Model B snapshot rows | {n_snap:,} |",
        "",
        "## Why periods ended",
        "",
        "| Reason | Periods | Share |",
        "| --- | --- | --- |",
    ]
    for k, v in reasons.items():
        lines.append(f"| {k} | {v:,} | {v / len(e):.1%} |")
    lines += [
        "",
        "Reasons: `off_duty` = a rest (45+ min parked, engine off) or a fake pickup cut the period;",
        "`too_long` = censored at 180 min; `gap` = GPS silence over 10 min;",
        "`lookahead_end` / `end_of_data` = data ran out.",
        "",
        "Censoring is much higher than the 2% in the original EDA because rests and fake pickups",
        "are no longer counted as waits that ended in a passenger. Survival models handle censored rows.",
    ]
    dst = config.PROJECT_DIR / "docs" / f"build_summary_{config.RULES_VERSION}.md"
    dst.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
