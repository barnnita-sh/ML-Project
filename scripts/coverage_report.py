"""Coverage report: how often the features have real information behind them.

  python scripts/coverage_report.py

Reads pipeline_output/features/{snapshots,modelA}/ day by day (low memory) and writes
docs/coverage_report.md plus docs/coverage_by_hour.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.geo import haversine_km  # noqa: E402

CENTRE = (13.7466, 100.5347)     # Siam
CENTRAL_KM = 8
PERIODS = [("warm-up", "20250101", "20250128"), ("train", "20250129", "20250915"),
           ("validation", "20250916", "20251109"), ("test", "20251110", "20251231")]

CHECKS = {   # name -> function of the feature table returning a boolean array
    "pickup in hexagon, last hour": lambda f: f["pickups_hex_1h"] > 0,
    "pickup in neighbourhood, last hour": lambda f: f["pickups_nbr_1h"] > 0,
    "pickup in neighbourhood, last 3 hours": lambda f: f["pickups_nbr_3h"] > 0,
    "any completed wait nearby, last 3 hours": lambda f: f["n_waits_nbr_3h"] >= 1,
    "5+ completed waits nearby, last 3 hours": lambda f: f["n_waits_nbr_3h"] >= 5,
    "4-week pickup history > 0 (neighbourhood)": lambda f: f["hist_pickups_nbr_4w"] > 0,
    "taxi speed known (neighbourhood, last hour)": lambda f: f["speed_nbr_1h"].notna(),
}


def period_of(day):
    for name, a, b in PERIODS:
        if a <= day <= b:
            return name
    return "other"


def main():
    folder = config.OUT_DIR / "features" / "snapshots"
    files = sorted(folder.glob("*.parquet")) or sorted(folder.glob("*.pkl"))
    read = pd.read_parquet if files[0].suffix == ".parquet" else pd.read_pickle
    cols = ["hour_of_day", "hex_lat", "hex_lon", "pickups_hex_1h", "pickups_nbr_1h", "pickups_nbr_3h",
            "n_waits_nbr_3h", "hist_pickups_nbr_4w", "speed_nbr_1h"]
    parts = []
    for f in files:
        x = read(f)[cols]
        g = pd.DataFrame({
            "period": period_of(f.stem),
            "hour": x["hour_of_day"].astype(int),
            "area": np.where(haversine_km(CENTRE[0], CENTRE[1], x["hex_lat"], x["hex_lon"]) <= CENTRAL_KM,
                             "central", "outer"),
        })
        for name, fn in CHECKS.items():
            g[name] = fn(x).to_numpy()
        parts.append(g.groupby(["period", "hour", "area"]).agg(["sum", "size"]))
    agg = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
    names = list(CHECKS)
    n = agg[(names[0], "size")]
    sums = pd.DataFrame({k: agg[(k, "sum")] for k in names})
    sums["snapshots"] = n

    def shares(df):
        tot = df["snapshots"].sum()
        return {k: df[k].sum() / tot for k in names}, int(tot)

    lines = ["# Feature coverage report", "",
             f"Model B snapshot rows: {int(n.sum()):,}. A feature has 'coverage' when there is real "
             "activity behind it; a zero is still a valid value, but a model can learn little from "
             "mostly-zero features. Central = within 8 km of Siam.", ""]
    # overall by area (excluding warm-up)
    body = sums.drop(index="warm-up", level=0, errors="ignore")
    lines += ["## Overall (January warm-up excluded)", "", "| Check | All | Central | Outer |", "| --- | --- | --- | --- |"]
    all_s, tot = shares(body)
    c_s, c_n = shares(body.xs("central", level=2))
    o_s, o_n = shares(body.xs("outer", level=2))
    for k in names:
        lines.append(f"| {k} | {all_s[k]:.0%} | {c_s[k]:.0%} | {o_s[k]:.0%} |")
    lines.append(f"| snapshots | {tot:,} | {c_n:,} | {o_n:,} |")

    lines += ["", "## By period", "", "| Check | " + " | ".join(p for p, _, _ in PERIODS) + " |",
              "| --- |" + " --- |" * len(PERIODS)]
    per = {p: shares(sums.xs(p, level=0))[0] for p, _, _ in PERIODS if p in sums.index.get_level_values(0)}
    for k in names:
        lines.append(f"| {k} | " + " | ".join(f"{per[p][k]:.0%}" if p in per else "" for p, _, _ in PERIODS) + " |")
    lines += ["", "In the warm-up month the 4-week history and some lags are missing by design (NaN)."]

    byh = body.groupby(level=1).sum()
    tab = pd.DataFrame({k: byh[k] / byh["snapshots"] for k in names})
    tab["snapshots"] = byh["snapshots"]
    tab.round(3).to_csv(config.PROJECT_DIR / "docs" / "coverage_by_hour.csv")
    pick = ["pickup in hexagon, last hour", "pickup in neighbourhood, last hour",
            "pickup in neighbourhood, last 3 hours", "5+ completed waits nearby, last 3 hours"]
    lines += ["", "## By hour of day (January excluded)", "",
              "| Hour | Snapshots | " + " | ".join(pick) + " |", "| --- | --- |" + " --- |" * len(pick)]
    for h, r in tab.iterrows():
        lines.append(f"| {h:02d}:00 | {int(r['snapshots']):,} | " + " | ".join(f"{r[k]:.0%}" for k in pick) + " |")

    lines += ["", "## What this means", "",
              "- Single-hexagon, single-hour pickup counts are mostly zero; the neighbourhood and "
              "3-hour versions carry most of the signal, as planned.",
              "- Night hours and outer areas have the least coverage; expect weaker predictions there "
              "and use the coverage rule (Section 10 of the team document) to fall back or say "
              "'not enough data'.",
              "- Full table by hour: `docs/coverage_by_hour.csv`."]

    a_folder = config.OUT_DIR / "features" / "modelA"
    a_files = sorted(a_folder.glob("*.parquet")) or sorted(a_folder.glob("*.pkl"))
    tgt = pd.concat([read(f)[["target_pickups"]] for f in a_files])["target_pickups"]
    lines += ["", "## Model A rows", "",
              f"{len(tgt):,} hexagon-hours with an empty taxi present; {(tgt > 0).mean():.0%} have at "
              f"least one pickup; mean {tgt.mean():.2f}, max {int(tgt.max())} pickups."]

    dst = config.PROJECT_DIR / "docs" / "coverage_report.md"
    dst.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
