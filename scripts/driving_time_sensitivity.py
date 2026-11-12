"""Fixed-recommendation, penalty-only driving-time sensitivity on replay policies.

Does NOT change historical peer matching, the waiting horizon, or model rankings.
Usage:
 python scripts/driving_time_sensitivity.py --policies pipeline_output/replay/driving_sensitivity_validation/policies.parquet --out pipeline_output/replay/driving_sensitivity_validation/driving_time_sensitivity.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def analyze(policies: pd.DataFrame, *, n_boot=2000, seed=42):
    required = {"episode_id", "snapshot_t", "replay_day", "strategy", "recommended_score_min",
                "recommended_drive_min", "recommended_covered"}
    missing = required - set(policies.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Generate fresh replay rows using patched engine.py.")
    p = policies.copy()
    p = p[p["strategy"].isin(["model_b", "stay"]) & p["recommended_covered"].fillna(False)].copy()
    p["day"] = p["replay_day"].astype(str)
    # A snapshot can share an episode with another checkpoint, so include timestamp.
    key = ["replay_day", "episode_id", "snapshot_t"]
    b = p.loc[p.strategy == "model_b", key + ["day", "recommended_score_min", "recommended_drive_min"]]
    s = p.loc[p.strategy == "stay", key + ["recommended_score_min", "recommended_drive_min"]]
    pairs = b.merge(s, on=key, suffixes=("_b", "_stay"), validate="one_to_one")
    if pairs.empty:
        raise ValueError("No jointly evaluable Model B/Stay decisions")
    for col in ["recommended_score_min_b", "recommended_score_min_stay",
                "recommended_drive_min_b", "recommended_drive_min_stay"]:
        pairs[col] = pd.to_numeric(pairs[col], errors="raise")
    pairs = pairs.dropna()
    if pairs.empty:
        raise ValueError("No complete paired outcomes with driving times")
    rng = np.random.default_rng(seed)
    days = pairs["day"].unique()
    out = []
    for alpha in (1.0, 0.75, 0.5):
        score_b = pairs.recommended_score_min_b - (1-alpha)*pairs.recommended_drive_min_b
        score_stay = pairs.recommended_score_min_stay - (1-alpha)*pairs.recommended_drive_min_stay
        diff = score_b - score_stay  # positive: Stay better
        daily = pd.DataFrame({"day": pairs.day, "diff": diff}).groupby("day")["diff"].agg(["sum", "count"])
        sums = daily["sum"].to_numpy()
        counts = daily["count"].to_numpy()
        sampled = rng.integers(0, len(days), size=(n_boot, len(days)))
        draws = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        out.append({"drive_penalty_fraction": alpha, "n_paired_decisions": len(pairs),
                    "n_days": len(days), "model_b_minus_stay_min": diff.mean(),
                    "ci_95_low": np.quantile(draws, .025),
                    "ci_95_high": np.quantile(draws, .975)})
    return pd.DataFrame(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.policies.suffix == ".parquet":
        p = pd.read_parquet(args.policies)
    elif args.policies.suffix in (".pkl", ".pickle"):
        p = pd.read_pickle(args.policies)
    else:
        p = pd.read_csv(args.policies)
    result = analyze(p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print("Positive Model B minus Stay means Stay is better. Validation-only, fixed recommendations.")

if __name__ == "__main__":
    main()
