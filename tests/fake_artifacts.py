"""Small made-up artifacts (real classes, invented data) so the recommender and the API can be
tested without the 16 GB of GPS data or the real trained models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import geo  # noqa: E402
from taxi.drive import DriveTable, build_drive_pairs  # noqa: E402
from taxi.features import FEATURES  # noqa: E402
from taxi.models.baselines import HistoricalSurvival  # noqa: E402
from taxi.models.wrappers import ModelB_GBM  # noqa: E402
from taxi.recommend import Recommender  # noqa: E402

DAY = "2025-11-05"
DAY_HOUR0 = int(pd.Timestamp(DAY).value // 10**9 // 3600)


def make_features(rng, hexes, hours):
    lat, lon = geo.hex_center(hexes)
    n = len(hexes) * len(hours)
    df = pd.DataFrame({"hex": np.repeat([str(h) for h in hexes], len(hours)), "hour_id": np.tile(hours, len(hexes))})
    lat, lon = np.repeat(lat, len(hours)), np.repeat(lon, len(hours))
    hod = df["hour_id"] % 24
    pk = rng.poisson(3 + 4 * np.exp(-((lat - 13.75) ** 2 + (lon - 100.5) ** 2) / 0.001), n)
    vac = rng.poisson(6, n)
    f = {"hour_of_day": hod, "day_of_week": 2, "is_weekend": 0, "is_holiday": 0, "month": 11,
         "hex_lat": lat, "hex_lon": lon, "pickups_hex_1h": rng.poisson(1, n), "vacant_hex_1h": rng.poisson(2, n),
         "pickups_nbr_1h": pk, "vacant_nbr_1h": vac, "vacant_per_pickup_nbr_1h": vac / np.maximum(pk, 1),
         "speed_nbr_1h": rng.normal(24, 4, n), "pickups_nbr_3h": pk * 3, "vacant_nbr_3h_mean": vac,
         "pickups_nbr_lag24": pk, "pickups_nbr_lag168": pk, "hist_pickups_hex_4w": pk / 7,
         "hist_pickups_nbr_4w": pk, "hist_vacant_nbr_4w": vac, "median_wait_nbr_3h": rng.normal(18, 5, n),
         "n_waits_nbr_3h": rng.poisson(8, n), "hist_wait_nbr_4w": rng.normal(18, 4, n),
         "n_hist_waits_nbr_4w": rng.poisson(30, n), "rain_lag_1h": 0.0}
    return pd.concat([df, pd.DataFrame(f)[FEATURES].astype("float32")], axis=1)


def make_recommender(seed=0, folder=None):
    rng = np.random.default_rng(seed)
    pts = [(la, lo) for la in np.arange(13.70, 13.80, 0.007) for lo in np.arange(100.45, 100.55, 0.007)]
    hexes = pd.unique(geo.to_hex([p[0] for p in pts], [p[1] for p in pts]))
    hours = np.arange(DAY_HOUR0, DAY_HOUR0 + 24)
    feats = make_features(rng, hexes, hours)

    # training rows: busier surroundings -> shorter waits; censored rows included
    n = 30_000
    tr = feats.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    tr["minutes_empty"] = rng.choice([0, 5, 15, 30, 45], n)
    wait = rng.lognormal(2.6 - 0.04 * tr["pickups_nbr_1h"].to_numpy() + 0.02 * tr["vacant_nbr_1h"].to_numpy(), 0.7)
    tr["minutes_until_end"] = np.minimum(wait, 180)
    tr["pickup_seen"] = (rng.random(n) > 0.15)
    tr["episode_id"] = np.arange(n)
    model = ModelB_GBM(params={"max_leaf_nodes": 15, "min_samples_leaf": 100}).fit(tr)
    base = HistoricalSurvival().fit(tr)

    ids, o, d, km = build_drive_pairs(hexes, detour=1.35, max_road_km=10.0)
    drive = DriveTable(ids, o, d, km, 1.35, 10.0)
    manifest = {"model_version": "fake-model-b", "grid": geo.GRID, "city_speed_kmh": 24.0}
    moments = pd.DataFrame({"snapshot_t": hours[rng.integers(0, 24, 50)] * 3600 + 600, "hex": [str(h) for h in rng.choice(hexes, 50)],
                            "minutes_empty": rng.choice([0, 5, 15], 50), "minutes_until_end": rng.random(50) * 30,
                            "pickup_seen": rng.random(50) > 0.1})
    rec = Recommender(drive, feats, model, base, manifest, moments)
    if folder is not None:
        import joblib
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "model_b.joblib")
        joblib.dump(base, folder / "baseline_b.joblib")
        np.savez_compressed(folder / "drive.npz", hexes=ids, o=o, d=d, km=km, detour=1.35, max_road_km=10.0)
        feats.to_parquet(folder / "features.parquet", index=False)
        moments.to_parquet(folder / "demo_moments.parquet", index=False)
        (folder / "manifest.json").write_text(__import__("json").dumps(manifest))
    return rec, hexes
