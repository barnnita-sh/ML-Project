"""ONE-TIME test-period evaluation of the frozen models (experiment E1, plus E3 repeated on test).

  python scripts/evaluate_test.py --confirm

Only run this after the models, features and thresholds are frozen (evaluate_validation.py
wrote pipeline_output/models/final.json) and the team has agreed. It refuses to run twice:
after a run it writes pipeline_output/models/TEST_RUN_DONE. Re-running anyway needs
--force and must be reported, because looking at test results more than once and then
changing the model turns the test into a second validation set.

Nothing is trained or tuned here. Every saved model (baselines and both candidates per model)
is only asked for predictions on the test period, so the report shows all of them next to their
validation numbers. The frozen choice in final.json is not revisited.

Writes
  docs/models_test.md                      the report (validation next to test)
  docs/figures/test_a_calibration.png      Model A calibration on test
  docs/figures/test_b_calibration.png      Model B calibration on test
  pipeline_output/models/test_results.json all numbers, machine-readable
  pipeline_output/models/TEST_RUN_DONE     the lock
The replay comparison with the heat map (E2) is a separate script (scripts/run_replay.py).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi import config  # noqa: E402
from taxi.geo import haversine_km  # noqa: E402
from taxi.models.data import SPLITS, load_split  # noqa: E402
from taxi.models.metrics import (calibration_table, censoring_curve, concordance_index, count_metrics,  # noqa: E402
                                 poisson_deviance, survival_metrics)

MODELS = config.OUT_DIR / "models"
DOCS = config.PROJECT_DIR / "docs"
FIG = DOCS / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CENTRE = (13.7466, 100.5347)
A_MODELS = ["baseline_a", "model_a_glm", "model_a_gbm"]
B_MODELS = ["baseline_b", "model_b_cox", "model_b_gbm"]
NAMES = {"baseline_a": "Historical average", "model_a_glm": "Poisson regression", "model_a_gbm": "Boosted trees (Poisson)",
         "baseline_b": "Historical survival curves", "model_b_cox": "Cox model", "model_b_gbm": "Boosted trees (log time)"}
C_SEEDS = [0, 1, 2, 3, 4]      # C-index is computed on 20,000-row subsamples; 5 seeds show how stable it is


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return f"figures/{name}"


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return out


def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def area(df):
    return np.where(haversine_km(CENTRE[0], CENTRE[1], df["hex_lat"], df["hex_lon"]) <= 8, "central", "outer")


def groups_for(df):
    hours = df["hour_of_day"].to_numpy()
    g = {"all": np.ones(len(df), bool),
         "quiet hours (01:00-05:00)": (hours >= 1) & (hours <= 5),
         "rush hours (07-09, 16-19)": np.isin(hours, [7, 8, 9, 16, 17, 18, 19]),
         "weekday": df["is_weekend"].to_numpy() == 0,
         "weekend": df["is_weekend"].to_numpy() == 1,
         "public holiday": df["is_holiday"].to_numpy() == 1,
         "central (<= 8 km)": area(df) == "central",
         "outer (> 8 km)": area(df) == "outer"}
    if "minutes_empty" in df:
        me = df["minutes_empty"].to_numpy()
        g["empty under 15 min"] = me < 15
        g["empty 30 min or more"] = me >= 30
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true", help="I understand this is the one-time test run")
    p.add_argument("--force", action="store_true", help="run again although a test run was already done")
    a = p.parse_args()
    lock = MODELS / "TEST_RUN_DONE"
    if not a.confirm:
        sys.exit("Refusing: this is the one-time test run. Add --confirm when the team has frozen the models.")
    if lock.exists() and not a.force:
        sys.exit(f"Refusing: the test run was already done ({lock.read_text().strip()}).")
    final = json.loads((MODELS / "final.json").read_text())
    val = final["validation"]
    fa, fb = final["model_a"], final["model_b"]
    started = datetime.now()
    out = {"run_at": f"{started:%Y-%m-%d %H:%M}", "forced": bool(a.force and lock.exists()),
           "frozen": {"model_a": fa, "model_b": fb}, "model_a": {}, "model_b": {}}

    lines = ["# Test-period results (one-time run)", "",
             f"Run on {started:%Y-%m-%d %H:%M}. Test period {SPLITS['test'][0]} to {SPLITS['test'][1]}; "
             f"models trained on {SPLITS['train'][0]} to {SPLITS['train'][1]} and frozen on validation "
             f"({SPLITS['val'][0]} to {SPLITS['val'][1]}). Nothing was retrained or retuned for this run. "
             f"Frozen choices: Model A = {NAMES[fa]}, Model B = {NAMES[fb]}.", ""]
    if out["forced"]:
        lines += ["**Note: this was a forced re-run (--force). Report it.**", ""]

    # ================================================================ Model A
    ta = load_split("modelA", "test")
    ya = ta["target_pickups"].to_numpy()
    pa = {k: joblib.load(MODELS / f"{k}.joblib").predict(ta) for k in A_MODELS}
    ma = {k: count_metrics(ya, v) for k, v in pa.items()}
    out["model_a"]["overall"] = ma
    rows = [{"Model": NAMES[k] + (" (frozen)" if k == fa else ""),
             "Deviance, validation": fmt(val["model_a"][k]["poisson_deviance"]),
             "Deviance, test": fmt(m["poisson_deviance"]), "MAE, test": fmt(m["mae"]), "RMSE, test": fmt(m["rmse"]),
             "Mean predicted, test": fmt(m["mean_predicted"])} for k, m in ma.items()]
    gain = 1 - ma[fa]["poisson_deviance"] / ma["baseline_a"]["poisson_deviance"]
    lines += ["## E1, Model A: pickups per hexagon-hour", "",
              f"{len(ta):,} test hexagon-hours, mean {ya.mean():.3f} pickups. Poisson deviance decides; lower is better.", ""]
    lines += table(rows, ["Model", "Deviance, validation", "Deviance, test", "MAE, test", "RMSE, test",
                          "Mean predicted, test"])
    lines += ["", f"Frozen Model A vs historical average on test: deviance {gain:.1%} lower "
              f"({'beats' if gain > 0 else 'does not beat'} the baseline).", ""]
    rows = []
    for g, m in groups_for(ta).items():
        if m.sum() == 0:
            continue
        rows.append({"Group": g, "Rows": f"{m.sum():,}", **{NAMES[k]: fmt(poisson_deviance(ya[m], pa[k][m])) for k in pa}})
    out["model_a"]["by_group"] = rows
    lines += ["Poisson deviance by group (test):", ""] + table(rows, ["Group", "Rows"] + [NAMES[k] for k in pa]) + [""]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, c in [("baseline_a", ORANGE), (fa, BLUE)]:
        q = pd.qcut(pa[k], 10, duplicates="drop")
        g = pd.DataFrame({"p": pa[k], "y": ya}).groupby(q, observed=True).mean()
        ax.plot(g["p"], g["y"], marker="o", color=c, lw=2, label=NAMES[k])
    mx = max(ya.mean() * 6, 1)
    ax.plot([0, mx], [0, mx], color=INK2, lw=1, ls="--")
    style(ax, "Model A on test: predicted vs actual pickups", "mean predicted", "mean actual")
    ax.legend(frameon=False, fontsize=9)
    lines += [f"![Model A calibration on test]({save(fig, 'test_a_calibration.png')})", ""]
    del ta, pa

    # ================================================================ Model B
    tb = load_split("modelB", "test")
    T, E = tb["minutes_until_end"].to_numpy(), tb["pickup_seen"].to_numpy(bool)
    G = censoring_curve(T, E)
    sb = {k: joblib.load(MODELS / f"{k}.joblib").predict_summaries(tb) for k in B_MODELS}
    mb = {k: survival_metrics(T, E, s, G) for k, s in sb.items()}
    for k in sb:
        cs = [concordance_index(T, E, sb[k]["rmean_120"], seed=s) for s in C_SEEDS]
        mb[k]["c_index_seeds"] = cs
    out["model_b"]["overall"] = mb
    rows = [{"Model": NAMES[k] + (" (frozen)" if k == fb else ""),
             "C-index, validation": fmt(val["model_b"][k]["c_index"]),
             "C-index, test": fmt(m["c_index"]),
             "Test range over 5 subsamples": f"{min(m['c_index_seeds']):.3f} to {max(m['c_index_seeds']):.3f}",
             "Brier at 15 min, test": fmt(m["brier_15"]),
             "Calibration error at 15 min, test": fmt(m["calibration_error_15"]),
             "MAE completed (min), test": fmt(m["mae_completed_min"], 1),
             "Mean predicted P(15)": fmt(m["mean_p_pickup_15"]), "Observed P(15)": fmt(m["observed_p_pickup_15"])}
            for k, m in mb.items()]
    lines += ["## E1, Model B: minutes until the next pickup (censored)", "",
              f"{len(tb):,} test snapshots, {E.mean():.1%} with a pickup seen. C-index decides (higher is better, "
              "0.5 = coin flip); it is computed on a 20,000-row subsample, and the range over 5 different subsamples "
              "shows how much that choice moves it. Brier score and calibration error: lower is better.", ""]
    lines += table(rows, ["Model", "C-index, validation", "C-index, test", "Test range over 5 subsamples",
                          "Brier at 15 min, test", "Calibration error at 15 min, test", "MAE completed (min), test",
                          "Mean predicted P(15)", "Observed P(15)"])
    better = mb[fb]["c_index"] > mb["baseline_b"]["c_index"] and min(mb[fb]["c_index_seeds"]) > max(mb["baseline_b"]["c_index_seeds"])
    lines += ["", f"Frozen Model B vs historical survival curves on test: C-index {mb[fb]['c_index']:.3f} vs "
              f"{mb['baseline_b']['c_index']:.3f} ({'beats the baseline on every subsample' if better else 'not a clear win: check the subsample ranges'}).", ""]

    rows = []
    for g, m in groups_for(tb).items():
        if m.sum() < 200:
            continue
        r = {"Group": g, "Snapshots": f"{m.sum():,}", "Censored": f"{1 - E[m].mean():.0%}"}
        for k in sb:
            r[NAMES[k]] = fmt(concordance_index(T[m], E[m], sb[k]["rmean_120"][m]))
        rows.append(r)
    out["model_b"]["by_group"] = rows
    lines += ["## E3 on test: C-index by group", "",
              "Same groups as the validation report, plus weekday, weekend and public holidays (E5). "
              "Groups with fewer than 200 snapshots are left out.", ""]
    lines += table(rows, ["Group", "Snapshots", "Censored"] + [NAMES[k] for k in sb]) + [""]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, c in [("baseline_b", ORANGE), ("model_b_cox", AQUA), ("model_b_gbm", BLUE)]:
        cal = calibration_table(T, E, sb[k]["p_pickup_15"])
        ax.plot([r[0] for r in cal], [r[1] for r in cal], marker="o", color=c, lw=2, label=NAMES[k])
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    style(ax, "Model B on test: predicted vs observed chance of a pickup within 15 min",
          "predicted (decile mean)", "observed (Kaplan-Meier)")
    ax.legend(frameon=False, fontsize=9)
    lines += [f"![Model B calibration on test]({save(fig, 'test_b_calibration.png')})", ""]

    # ================================================================ write
    (DOCS / "models_test.md").write_text("\n".join(lines) + "\n")
    (MODELS / "test_results.json").write_text(json.dumps(out, indent=2, default=float))
    lock.write_text(f"test run on {started:%Y-%m-%d %H:%M}\n")
    print("\n".join(lines))
    print(f"\nWrote docs/models_test.md, docs/figures/test_*_calibration.png, {MODELS / 'test_results.json'}, {lock}")


if __name__ == "__main__":
    main()
