"""Model A feature-group check on the FULL training sample (follow-up to E4).

  python scripts/check_model_a_features.py

Why: E4 in evaluate_validation.py trains on 400,000 rows. There, calendar and place
alone scored better than all 25 features, while the frozen model (all features, about
2 million rows) scored better than both. This script repeats the test at full size so the
team can decide the final Model A features on solid numbers:

  1. cumulative: add one group at a time (as E4)
  2. drop-one: all features except one group

Same tuned settings as the frozen model; scored on validation only (test untouched).
Differences under about 0.003 are within the run-to-run noise of early stopping.

Writes docs/models_feature_check.md
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taxi import config  # noqa: E402
from taxi.models.data import GROUPS, load_split  # noqa: E402
from taxi.models.metrics import poisson_deviance  # noqa: E402
from taxi.models.wrappers import ModelA_GBM  # noqa: E402

MODELS = config.OUT_DIR / "models"


def main():
    params = json.loads((MODELS / "tuning.json").read_text())["model_a_gbm"]["best"]["setting"]
    tr, va = load_split("modelA", "train"), load_split("modelA", "val")
    y = va["target_pickups"].to_numpy()
    print(f"{len(tr):,} training rows, {len(va):,} validation rows; settings {params}")

    def score(feats):
        t0 = time.time()
        d = poisson_deviance(y, ModelA_GBM(params=params, features=feats).fit(tr).predict(va))
        print(f"  {len(feats):2d} features -> {d:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return d

    all_feats = sum(GROUPS.values(), [])
    cum, used = [], []
    print("cumulative")
    for g, cols in GROUPS.items():
        used = used + cols
        cum.append((g, len(used), score(used)))
    full = cum[-1][2]
    print("drop one group")
    drop = []
    for g, cols in GROUPS.items():
        if g == "calendar_place":
            continue
        feats = [c for c in all_feats if c not in cols]
        d = score(feats)
        drop.append((g, len(feats), d, d - full))

    lines = ["# Model A feature-group check (full training sample)", "",
             f"Boosted trees (Poisson) with the frozen settings, {len(tr):,} training rows, scored on "
             f"{len(va):,} validation hexagon-hours. Poisson deviance, lower is better. Differences under "
             "about 0.003 are noise.", "",
             "## Adding one group at a time", "", "| Added group | Features | Deviance |", "| --- | --- | --- |"]
    lines += [f"| {g.replace('_', ' ')} | {n} | {d:.4f} |" for g, n, d in cum]
    lines += ["", "## Leaving one group out (all others kept)", "",
              "Positive change = the model is worse without the group, so the group helps.", "",
              "| Left out | Features | Deviance | Change vs all features |", "| --- | --- | --- | --- |"]
    lines += [f"| {g.replace('_', ' ')} | {n} | {d:.4f} | {c:+.4f} |" for g, n, d, c in drop]
    out = config.PROJECT_DIR / "docs" / "models_feature_check.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
