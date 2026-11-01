"""Check what the current rules do to each of the 50 audited periods.

  python scripts/check_audit_rules.py

For every period in audit/audit_sheet.csv (built with the old v0.1 rules) it rebuilds
that day with the current rules and reports what became of it: kept as is, cut at a
rest, censored as off duty, flagged as a possible missed trip, or dropped.
Writes audit/rules_check_<RULES_VERSION>.csv and prints a summary by agreed label.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.clean import clean, day_bounds, load_day  # noqa: E402
from taxi.episodes import build_episodes  # noqa: E402


def describe(new: pd.DataFrame) -> str:
    if new.empty:
        return "dropped"
    parts = []
    for r in new.itertuples(index=False):
        end = "pickup" if r.pickup_seen else f"censored:{r.censor_reason}"
        flag = " +missed_trip_flag" if r.possible_missed_trip else ""
        parts.append(f"{r.start_type} {r.minutes:.0f}min {end}{flag}")
    return " | ".join(parts)


def main():
    sheet = pd.read_csv(config.AUDIT_DIR / "audit_sheet.csv", dtype={"day": str})
    sheet["vid"] = sheet["episode_id"].str.rsplit("-", n=1).str[0]
    sheet["t_start"] = sheet["episode_id"].str.rsplit("-", n=1).str[1].astype("int64")
    sheet["t_end_old"] = sheet["t_start"] + (sheet["minutes"] * 60).round().astype("int64")

    rows = []
    for day, g in sheet.groupby("day"):
        raw = load_day(day)
        d, _ = clean(raw)
        _, day_end = day_bounds(day)
        _, empty = build_episodes(d, day_end, raw.attrs["has_next_day"])
        for r in g.itertuples(index=False):
            new = empty[(empty["vid"] == r.vid) & empty["t_start"].between(r.t_start, r.t_end_old)]
            usable = new[(~new["possible_missed_trip"])]
            rows.append({"episode_id": r.episode_id, "agreed_label": r.agreed_label, "old_minutes": r.minutes,
                         "new_periods": describe(new),
                         "minutes_kept_for_training": round(float(usable["minutes"].sum()), 1)})
    out = pd.DataFrame(rows)
    order = sheet[["episode_id"]].reset_index().rename(columns={"index": "n"})
    out = order.merge(out, on="episode_id")
    out["n"] += 1
    dst = config.AUDIT_DIR / f"rules_check_{config.RULES_VERSION}.csv"
    out.to_csv(dst, index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(out[["n", "agreed_label", "old_minutes", "new_periods"]].to_string(index=False))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
