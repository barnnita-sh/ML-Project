"""Deep check of everything the pipeline and feature steps produced.

  python scripts/verify_outputs.py

Goes beyond check_build.py (which only checks files exist): it opens the tables and checks
counts, value ranges, hexagon IDs, labels, feature completeness, the rain feature, and
re-computes features for random snapshots by hand to confirm nothing uses the future.
Prints PASS / FAIL per check and writes docs/verification_report.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.features import FEATURES  # noqa: E402
from taxi.geo import neighbours  # noqa: E402
from taxi.storage import load_day_table  # noqa: E402

OUT = config.OUT_DIR
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def days():
    return sorted(f.stem for f in (OUT / "manifest").glob("*.json"))


def main():
    ds = days()
    man = pd.DataFrame([json.loads((OUT / "manifest" / f"{d}.json").read_text()) for d in ds])
    check("365 days built", len(ds) == 365, f"{len(ds)} days")
    check("all days built with rules v1.0 and H3",
          set(man["rules_version"]) == {config.RULES_VERSION} and set(man["grid"]) == {"h3"},
          f"rules {set(man['rules_version'])}, grid {set(man['grid'])}")

    # ---------------------------------------------------------------- per-day consistency (all days)
    bad = {k: [] for k in ["counts", "hexhour_dup", "hexhour_pickups", "snap_rows", "modelA_rows",
                           "modelA_target", "labels", "calendar", "features_cols"]}
    nan_sum, n_rows, rain_rows, rows_train = None, 0, 0, 0
    first_hex = None
    for d, m in zip(ds, man.itertuples(index=False)):
        trips = load_day_table(OUT / "trips", d)
        empty = load_day_table(OUT / "empty", d)
        snaps = load_day_table(OUT / "snapshots", d)
        hh = load_day_table(OUT / "hexhour", d)
        fs = load_day_table(OUT / "features" / "snapshots", d)
        fa = load_day_table(OUT / "features" / "modelA", d)
        if first_hex is None:
            first_hex = str(snaps["hex"].iloc[0])
        if (len(trips), len(empty), len(snaps)) != (m.trips, m.empty_periods, m.snapshots):
            bad["counts"].append(d)
        if hh.duplicated(["hex", "hour_id"]).any():
            bad["hexhour_dup"].append(d)
        if int(hh["pickups"].sum()) != int((~trips["is_fake"]).sum()):
            bad["hexhour_pickups"].append(d)
        if len(fs) != len(snaps):
            bad["snap_rows"].append(d)
        if len(fa) != int((hh["vacant_taxis"] > 0).sum()):
            bad["modelA_rows"].append(d)
        j = fa.merge(hh, on=["hex", "hour_id"])
        if not (j["target_pickups"] == j["pickups"]).all():
            bad["modelA_target"].append(d)
        if (fs["minutes_until_end"] <= 0).any() or (fs["minutes_empty"] % 5 != 0).any() \
                or (fs["minutes_empty"] > config.SNAPSHOT_MAX_MIN).any():
            bad["labels"].append(d)
        if not (fs["hour_of_day"].to_numpy() == (fs["snapshot_t"] // 3600 % 24).to_numpy()).all():
            bad["calendar"].append(d)
        if [c for c in FEATURES if c not in fs.columns or c not in fa.columns]:
            bad["features_cols"].append(d)
        if "20250129" <= d <= "20250915" and d != "20250829":
            s = fs[FEATURES].isna().sum()
            nan_sum = s if nan_sum is None else nan_sum + s
            rows_train += len(fs)
        rain_rows += int((fs["rain_lag_1h"] > 0.1).sum())
        n_rows += len(fs)

    for k, label in [("counts", "table row counts match the manifests"),
                     ("hexhour_dup", "hexagon-hour table has one row per hexagon and hour"),
                     ("hexhour_pickups", "hexagon-hour pickups = real trips of the day"),
                     ("snap_rows", "Model B feature table has one row per snapshot"),
                     ("modelA_rows", "Model A table = hexagon-hours with an empty taxi"),
                     ("modelA_target", "Model A target = hexagon-hour pickups"),
                     ("labels", "snapshot labels positive, minutes_empty in 5-min steps up to 120"),
                     ("calendar", "hour_of_day matches the snapshot time"),
                     ("features_cols", "all 25 features present in both tables")]:
        check(label, not bad[k], f"failing days: {bad[k][:5]}" if bad[k] else "all days")

    try:
        import h3
        ok = h3.is_valid_cell(first_hex) and h3.get_resolution(first_hex) == config.H3_RES
    except Exception:
        ok = False
    check("hexagon IDs are valid H3 resolution 8", ok, f"example {first_hex}")

    nan_share = (nan_sum / rows_train).round(4)
    expected_nan = {"median_wait_nbr_3h", "hist_wait_nbr_4w", "speed_nbr_1h", "rain_lag_1h"}
    other = nan_share.drop(labels=list(expected_nan))
    check("no missing values in count features (training period)", (other == 0).all(),
          f"max {other.max():.2%}" if len(other) else "")
    check("rain feature is filled from the weather file", rain_rows > 0,
          f"{rain_rows:,} snapshot rows had rain > 0.1 mm in the last hour")

    # ---------------------------------------------------------------- recompute features by hand
    rng = np.random.default_rng(7)
    sample_days = sorted(str(x) for x in rng.choice([d for d in ds if "20250205" <= d <= "20251230"], 8, replace=False))
    total = match = 0
    for d in sample_days:
        prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        fs = load_day_table(OUT / "features" / "snapshots", d)
        tr = pd.concat([load_day_table(OUT / "trips", x) for x in (prev, d)])
        em = pd.concat([load_day_table(OUT / "empty", x) for x in (prev, d)])
        tr = tr[~tr["is_fake"]]
        th = tr["t_start"] // 3600
        em = em[em["pickup_seen"] & ~em["possible_missed_trip"]]
        eh = em["t_end"] // 3600
        for i in rng.choice(len(fs), 40, replace=False):
            r = fs.iloc[i]
            k = r["snapshot_t"] // 3600 - 1
            nb = set(neighbours([r["hex"]])["nbr"])
            p_nbr = int(((th == k) & tr["hex_pickup"].isin(nb)).sum())
            p_hex = int(((th == k) & (tr["hex_pickup"] == r["hex"])).sum())
            w3 = em[(eh >= k - 2) & (eh <= k) & em["hex_end"].isin(nb)]
            ok = (p_nbr == r["pickups_nbr_1h"] and p_hex == r["pickups_hex_1h"]
                  and len(w3) == r["n_waits_nbr_3h"] and (w3["t_end"] < r["snapshot_t"]).all())
            total += 1
            match += ok
    check("hand re-computation of features for random snapshots", match == total,
          f"{match} of {total} matched (days {', '.join(sample_days)})")

    lines = ["# Verification report", "",
             f"Run on {pd.Timestamp.now():%Y-%m-%d %H:%M}. {sum(ok for _, ok, _ in results)} of "
             f"{len(results)} checks passed.", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {det} |" for n, ok, det in results]
    lines += ["", "## Missing-value share per feature (training period)", "",
              "Expected: only the recent and historical wait medians, speed and rain can be missing.", "",
              "| Feature | Missing |", "| --- | --- |"]
    lines += [f"| {f} | {v:.2%} |" for f, v in nan_share.items()]
    lines += ["", f"Model B rows checked: {n_rows:,}."]
    dst = config.PROJECT_DIR / "docs" / "verification_report.md"
    dst.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {dst}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
