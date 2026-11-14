"""Build the small `artifacts/` folder that the API and the Docker image serve from.

Run ONCE, on the machine that has the pipeline tables (hexhour, empty, trips and snapshots for the 31 days up to --day, plus models):

  python scripts/make_artifacts.py --day 20251105

--day must be a VALIDATION day (16 Sep to 9 Nov 2025): the demo must not touch the test period
before the team's one-time test run. Writes (all small):
  artifacts/model_b.joblib      the frozen Model B (as named in pipeline_output/models/final.json)
  artifacts/baseline_b.joblib   historical survival curves (coverage rule)
  artifacts/features.parquet    the 25 features of every known hexagon for every hour of --day
  artifacts/drive.npz           road km between hexagons (straight line x detour factor from our trips)
  artifacts/demo_moments.parquet  sample decision moments with the real outcome (NO vehicle IDs)
  artifacts/manifest.json       model version, rules version, training cutoff, detour, grid
  artifacts/versions.txt        library versions of this machine (the Docker build pins to them)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build_features import Builder  # noqa: E402

from taxi import config, geo  # noqa: E402
from taxi.clean import day_bounds  # noqa: E402
from taxi.drive import build_drive_pairs, detour_factor, save_drive  # noqa: E402
from taxi.features import FEATURES  # noqa: E402
from taxi.storage import FORMAT, load_day_table  # noqa: E402

LIBS = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "h3", "pyarrow"]


def sample_trips(n_days):
    """Trips of up to n_days days, spread over the days that exist (for the detour factor).
    One day of trips is already enough for a stable median; any day is fine (it is only a distance ratio)."""
    files = sorted(f.stem for f in (config.OUT_DIR / "trips").glob("*") if f.is_file())
    if not files:
        sys.exit(f"no trip tables in {config.OUT_DIR / 'trips'}: build some (build_days.py) or pass --detour 1.35")
    pick = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(n_days, len(files))).astype(int))]
    return pd.concat([load_day_table(config.OUT_DIR / "trips", d) for d in pick], ignore_index=True), pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="20251105")
    ap.add_argument("--out", default=str(config.PROJECT_DIR / "artifacts"))
    ap.add_argument("--models-dir", default=str(config.OUT_DIR / "models"),
                    help="folder with final.json and the .joblib files (read only; never modified)")
    ap.add_argument("--trip-days", type=int, default=12)
    ap.add_argument("--max-road-km", type=float, default=10.0)
    ap.add_argument("--detour", type=float, help="override the measured detour factor")
    ap.add_argument("--moments", type=int, default=400)
    a = ap.parse_args()
    if not "20250916" <= a.day <= "20251109":
        sys.exit("--day must be a validation day (20250916 to 20251109); do not use test days before the test run")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = Path(a.models_dir)
    final = json.loads((models / "final.json").read_text())

    # 1. the frozen models
    shutil.copy(models / f"{final['model_b']}.joblib", out / "model_b.joblib")
    shutil.copy(models / f"{final['fallback_b']}.joblib", out / "baseline_b.joblib")
    shutil.copy(models / "final.json", out / "final.json")
    print("models copied:", final["model_b"], "+", final["fallback_b"])

    # 2. features of every known hexagon for each hour of the day (same code as the training tables)
    b = Builder()
    b.ensure_window(a.day)
    hexes = pd.unique(np.concatenate([v["B"]["hex"].to_numpy() for v in b.win.days.values()]))
    h0 = day_bounds(a.day)[0] // 3600
    hours = np.arange(h0, h0 + 24)
    f = b.win.features(np.repeat(hexes, 24), np.tile(hours, len(hexes)))
    f.insert(0, "hour_id", np.tile(hours, len(hexes)).astype("int64"))
    f.insert(0, "hex", np.repeat([str(h) for h in hexes], 24))
    f.to_parquet(out / "features.parquet", index=False)
    print(f"features: {len(hexes):,} hexagons x 24 hours")

    # 3. driving distances
    if a.detour:
        detour, how = a.detour, "given with --detour"
    else:
        trips, days_used = sample_trips(a.trip_days)
        detour, how = detour_factor(trips), f"measured on {len(trips):,} trips of {len(days_used)} days"
    ids, o, d, km = build_drive_pairs(hexes, detour, a.max_road_km)
    save_drive(out / "drive.npz", ids, o, d, km, detour, a.max_road_km)
    print(f"detour factor {detour:.3f} ({how}); {len(o):,} hexagon pairs")

    # 4. demo moments (decision moments + what really happened; vehicle and episode IDs are dropped)
    snaps = load_day_table(config.OUT_DIR / "snapshots", a.day)     # labels only; features are not needed here
    if snaps is None:
        sys.exit(f"no snapshot table for {a.day} in {config.OUT_DIR / 'snapshots'} (run build_days.py --day {a.day})")
    snaps = snaps[snaps["minutes_empty"].isin([0, 5, 15, 30, 45])]
    snaps = snaps.sample(min(a.moments, len(snaps)), random_state=0)
    snaps = snaps[["snapshot_t", "hex", "minutes_empty", "minutes_until_end", "pickup_seen"]].copy()
    snaps["hex"] = snaps["hex"].astype(str)
    snaps.to_parquet(out / "demo_moments.parquet", index=False)

    # 5. manifest + library versions
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    (out / "versions.txt").write_text("".join(f"{k}=={v}\n" for k, v in versions.items()))
    manifest = {"model_version": f"{final['model_b']}-{config.RULES_VERSION}", "model_file": final["model_b"],
                "rules_version": config.RULES_VERSION, "training_cutoff": "2025-09-15", "demo_day": a.day,
                "grid": geo.GRID, "storage_format": FORMAT, "detour_factor": float(detour),
                "city_speed_kmh": float(np.nanmedian(f["speed_nbr_1h"])), "python": sys.version.split()[0],
                "built_at": datetime.now().isoformat(timespec="seconds"), "n_hexagons": int(len(hexes)),
                "validation": final.get("validation", {}).get("model_b", {}).get(final["model_b"], {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"done: {out}  ({size:.1f} MB). Next: git add artifacts && docker build -t taxi-api .")


if __name__ == "__main__":
    main()
