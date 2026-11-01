"""Train the baselines, Model A and Model B (training period only).

  python scripts/train_models.py                 # everything
  python scripts/train_models.py --what a        # only Model A (baselines | a | b | all)
  python scripts/train_models.py --trials 5      # smaller tuning budget for a quick run

Tuning: each learned model gets the same budget (--trials settings, default 10), scored on
two forward-chaining folds inside the training period (fit on earlier months, score on the
next block). Model A is scored by Poisson deviance (lower is better), Model B by C-index
(higher is better). The best setting is then refitted on the whole training period.
Validation and test data are not touched here.

Writes pipeline_output/models/*.joblib and pipeline_output/models/tuning.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from taxi import config  # noqa: E402
from taxi.models.baselines import HistoricalAverage, HistoricalSurvival  # noqa: E402
from taxi.models.data import FOLDS, in_days, load_split  # noqa: E402
from taxi.models.metrics import concordance_index, poisson_deviance  # noqa: E402
from taxi.models.wrappers import ModelA_GBM, ModelA_GLM, ModelB_Cox, ModelB_GBM, sample_params  # noqa: E402

MODELS = config.OUT_DIR / "models"


def folds(df, cap, seed=0):
    rng = np.random.default_rng(seed)
    for (fa, fb), (ea, eb) in FOLDS:
        fit, ev = df[in_days(df, fa, fb)], df[in_days(df, ea, eb)]
        if len(fit) > cap:
            fit = fit.iloc[np.sort(rng.choice(len(fit), cap, replace=False))]
        if len(ev) > cap // 2:
            ev = ev.iloc[np.sort(rng.choice(len(ev), cap // 2, replace=False))]
        yield fit, ev


def tune(make, settings, df, score, cap, label):
    log = []
    for i, s in enumerate(settings):
        t0 = time.time()
        vals = [score(make(s).fit(fit), ev) for fit, ev in folds(df, cap)]
        log.append({"setting": s, "fold_scores": vals, "mean": float(np.mean(vals)), "seconds": round(time.time() - t0, 1)})
        print(f"  {label} trial {i + 1}/{len(settings)}: {s} -> {np.mean(vals):.4f} ({log[-1]['seconds']}s)", flush=True)
    return log


def score_a(model, ev):
    return poisson_deviance(ev["target_pickups"], model.predict(ev))


def score_b(model, ev):
    return concordance_index(ev["minutes_until_end"], ev["pickup_seen"], model.predict_time(ev))


def save(obj, name):
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, MODELS / f"{name}.joblib")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", default="all", choices=["all", "baselines", "a", "b"])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--tune-rows", type=int, default=600_000, help="max rows per fold fit while tuning")
    a = p.parse_args()
    tpath = MODELS / "tuning.json"
    tuning = json.loads(tpath.read_text()) if tpath.exists() else {}

    if a.what in ("all", "baselines"):
        print("baselines")
        save(HistoricalAverage().fit(load_split("modelA", "train")), "baseline_a")
        save(HistoricalSurvival().fit(load_split("modelB", "train")), "baseline_b")

    if a.what in ("all", "a"):
        tr = load_split("modelA", "train")
        print(f"Model A: {len(tr):,} training rows")
        alphas = list(np.logspace(-5, 1, a.trials))
        log = tune(lambda s: ModelA_GLM(alpha=s), alphas, tr, score_a, a.tune_rows, "GLM")
        best = min(log, key=lambda r: r["mean"])
        save(ModelA_GLM(alpha=best["setting"]).fit(tr), "model_a_glm")
        tuning["model_a_glm"] = {"metric": "poisson_deviance (lower is better)", "best": best, "trials": log}
        log = tune(lambda s: ModelA_GBM(params=s), sample_params(a.trials, seed=1), tr, score_a, a.tune_rows, "GBM")
        best = min(log, key=lambda r: r["mean"])
        save(ModelA_GBM(params=best["setting"]).fit(tr), "model_a_gbm")
        tuning["model_a_gbm"] = {"metric": "poisson_deviance (lower is better)", "best": best, "trials": log}
        tpath.write_text(json.dumps(tuning, indent=2, default=float))

    if a.what in ("all", "b"):
        tr = load_split("modelB", "train")
        print(f"Model B: {len(tr):,} training rows ({tr['pickup_seen'].mean():.1%} with a pickup seen)")
        l2s = list(np.logspace(-4, 1, a.trials))
        log = tune(lambda s: ModelB_Cox(l2=s), l2s, tr, score_b, a.tune_rows, "Cox")
        best = max(log, key=lambda r: r["mean"])
        save(ModelB_Cox(l2=best["setting"]).fit(tr), "model_b_cox")
        tuning["model_b_cox"] = {"metric": "c_index (higher is better)", "best": best, "trials": log}
        log = tune(lambda s: ModelB_GBM(params=s), sample_params(a.trials, seed=2), tr, score_b, a.tune_rows, "GBM")
        best = max(log, key=lambda r: r["mean"])
        save(ModelB_GBM(params=best["setting"]).fit(tr), "model_b_gbm")
        tuning["model_b_gbm"] = {"metric": "c_index (higher is better)", "best": best, "trials": log}
        tpath.write_text(json.dumps(tuning, indent=2, default=float))
    print("done; models in", MODELS)


if __name__ == "__main__":
    main()
