"""Evaluate all models on the validation period, run experiments E1, E3, E4, and freeze the
final Model A and Model B.

  python scripts/evaluate_validation.py

Needs prepare_model_data.py and train_models.py first. Writes
  docs/models_validation.md, docs/figures/models_*.png
  pipeline_output/models/final.json (which models are frozen) and docs/model_card.md
The test period is not used here.
"""
import json
import sys
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
from taxi.models.data import FEATURES_A, FEATURES_B, GROUPS, SPLITS, load_split  # noqa: E402
from taxi.models.metrics import (calibration_table, censoring_curve, concordance_index, count_metrics,  # noqa: E402
                                 poisson_deviance, survival_metrics)
from taxi.models.wrappers import ModelA_GBM, ModelB_GBM  # noqa: E402

MODELS = config.OUT_DIR / "models"
DOCS = config.PROJECT_DIR / "docs"
FIG = DOCS / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CENTRE = (13.7466, 100.5347)
NAMES = {"baseline_a": "Historical average", "model_a_glm": "Poisson regression", "model_a_gbm": "Boosted trees (Poisson)",
         "baseline_b": "Historical survival curves", "model_b_cox": "Cox model", "model_b_gbm": "Boosted trees (log time)"}


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


def load(name):
    return joblib.load(MODELS / f"{name}.joblib")


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return out


def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def area(df):
    return np.where(haversine_km(CENTRE[0], CENTRE[1], df["hex_lat"], df["hex_lon"]) <= 8, "central", "outer")


def main():
    tuning = json.loads((MODELS / "tuning.json").read_text())
    lines = ["# Model validation report", "",
             f"Validation period {SPLITS['val'][0]} to {SPLITS['val'][1]}; models trained on "
             f"{SPLITS['train'][0]} to {SPLITS['train'][1]} (29 August excluded). The test period is untouched.", ""]

    # ================================================================ Model A
    va = load_split("modelA", "val")
    tra = load_split("modelA", "train")
    pa = {k: load(k).predict(va) for k in ["baseline_a", "model_a_glm", "model_a_gbm"]}
    ya = va["target_pickups"].to_numpy()
    ma = {k: count_metrics(ya, v) for k, v in pa.items()}
    rows = [{"Model": NAMES[k], "Poisson deviance": fmt(m["poisson_deviance"]), "MAE": fmt(m["mae"]),
             "RMSE": fmt(m["rmse"]), "MAE, zero hours": fmt(m["mae_zero_hours"]),
             "MAE, non-zero hours": fmt(m["mae_nonzero_hours"]), "Mean predicted": fmt(m["mean_predicted"], 3)}
            for k, m in ma.items()]
    final_a = min(["model_a_glm", "model_a_gbm"], key=lambda k: ma[k]["poisson_deviance"])
    base_dev = ma["baseline_a"]["poisson_deviance"]
    lines += ["## Model A: pickups per hexagon-hour", "",
              f"{len(va):,} validation hexagon-hours (mean {ya.mean():.3f} pickups, {np.mean(ya > 0):.0%} non-zero). "
              "Lower is better for every column except the mean, which should match the actual mean "
              f"({ya.mean():.3f}).", ""]
    lines += table(rows, ["Model", "Poisson deviance", "MAE", "RMSE", "MAE, zero hours", "MAE, non-zero hours",
                          "Mean predicted"])
    lines += ["", f"**Frozen Model A: {NAMES[final_a]}** (lowest validation Poisson deviance; "
              f"{(1 - ma[final_a]['poisson_deviance'] / base_dev):.1%} lower than the historical average).", ""]
    ar = area(va)
    sub = []
    for g in ["central", "outer"]:
        m = ar == g
        sub.append({"Area": g, "Rows": f"{m.sum():,}",
                    **{NAMES[k]: fmt(poisson_deviance(ya[m], pa[k][m])) for k in pa}})
    lines += ["Poisson deviance by area:", ""] + table(sub, ["Area", "Rows"] + [NAMES[k] for k in pa]) + [""]

    # calibration of Model A: predicted vs actual by decile
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, c in [("baseline_a", ORANGE), (final_a, BLUE)]:
        q = pd.qcut(pa[k], 10, duplicates="drop")
        g = pd.DataFrame({"p": pa[k], "y": ya}).groupby(q, observed=True).mean()
        ax.plot(g["p"], g["y"], marker="o", color=c, lw=2, label=NAMES[k])
    mx = max(ya.mean() * 6, 1)
    ax.plot([0, mx], [0, mx], color=INK2, lw=1, ls="--")
    style(ax, "Model A: predicted vs actual pickups (deciles of prediction)", "mean predicted", "mean actual")
    ax.legend(frameon=False, fontsize=9)
    fa = save(fig, "models_a_calibration.png")
    lines += [f"![Model A calibration]({fa})", ""]

    # ================================================================ Model B
    vb = load_split("modelB", "val")
    trb = load_split("modelB", "train")
    T, E = vb["minutes_until_end"].to_numpy(), vb["pickup_seen"].to_numpy(bool)
    G = censoring_curve(T, E)
    sb = {"baseline_b": load("baseline_b").predict_summaries(vb)}
    for k in ["model_b_cox", "model_b_gbm"]:
        sb[k] = load(k).predict_summaries(vb)
    mb = {k: survival_metrics(T, E, s, G) for k, s in sb.items()}
    rows = [{"Model": NAMES[k], "C-index": fmt(m["c_index"]), "Brier at 15 min": fmt(m["brier_15"]),
             "Calibration error at 15 min": fmt(m["calibration_error_15"]),
             "MAE, completed waits (min)": fmt(m["mae_completed_min"], 1),
             "Mean predicted P(pickup in 15 min)": fmt(m["mean_p_pickup_15"]),
             "Observed": fmt(m["observed_p_pickup_15"])} for k, m in mb.items()]
    final_b = max(["model_b_cox", "model_b_gbm"], key=lambda k: (round(mb[k]["c_index"], 3), -mb[k]["brier_15"]))
    lines += ["## Model B: minutes until the next pickup (censored)", "",
              f"{len(vb):,} validation snapshots, {E.mean():.1%} with a pickup seen. C-index: higher is better "
              "(0.5 = coin flip). Brier score and calibration error: lower is better. The observed chance of a "
              "pickup within 15 minutes is the Kaplan-Meier value for all validation snapshots.", ""]
    lines += table(rows, ["Model", "C-index", "Brier at 15 min", "Calibration error at 15 min",
                          "MAE, completed waits (min)", "Mean predicted P(pickup in 15 min)", "Observed"])
    lines += ["", f"**Frozen Model B: {NAMES[final_b]}** (highest validation C-index; it gives a full survival curve, "
              "so the app can compute the 60-minute score from it). The historical survival curves stay as the "
              "fallback for areas with too little data.", ""]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, c in [("baseline_b", ORANGE), ("model_b_cox", AQUA), ("model_b_gbm", BLUE)]:
        cal = calibration_table(T, E, sb[k]["p_pickup_15"])
        ax.plot([r[0] for r in cal], [r[1] for r in cal], marker="o", color=c, lw=2, label=NAMES[k])
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    style(ax, "Model B: predicted vs observed chance of a pickup within 15 min",
          "predicted (decile mean)", "observed (Kaplan-Meier)")
    ax.legend(frameon=False, fontsize=9)
    fb = save(fig, "models_b_calibration.png")
    lines += [f"![Model B calibration]({fb})", ""]

    # ---------------------------------------------------------------- E3: subgroups
    hours = vb["hour_of_day"].to_numpy()
    groups = {"all": np.ones(len(vb), bool), "quiet hours (01:00-05:00)": (hours >= 1) & (hours <= 5),
              "rush hours (07-09, 16-19)": np.isin(hours, [7, 8, 9, 16, 17, 18, 19]),
              "central (<= 8 km)": area(vb) == "central", "outer (> 8 km)": area(vb) == "outer",
              "empty under 15 min": vb["minutes_empty"].to_numpy() < 15,
              "empty 30 min or more": vb["minutes_empty"].to_numpy() >= 30}
    rows = []
    for g, m in groups.items():
        r = {"Group": g, "Snapshots": f"{m.sum():,}", "Censored": f"{1 - E[m].mean():.0%}"}
        for k in sb:
            r[NAMES[k]] = fmt(concordance_index(T[m], E[m], sb[k]["rmean_120"][m]))
        rows.append(r)
    lines += ["## E3: is the result robust across groups and censoring?", "",
              "C-index by group (higher is better). Censoring is higher at night and in outer areas, where "
              "handling censored waits matters most.", ""]
    lines += table(rows, ["Group", "Snapshots", "Censored"] + [NAMES[k] for k in sb]) + [""]

    # ---------------------------------------------------------------- E4: feature groups
    rng = np.random.default_rng(0)
    tra_s = tra.iloc[rng.choice(len(tra), min(len(tra), 400_000), replace=False)]
    trb_s = trb.iloc[rng.choice(len(trb), min(len(trb), 400_000), replace=False)]
    pa_best = tuning["model_a_gbm"]["best"]["setting"]
    pb_best = tuning["model_b_gbm"]["best"]["setting"]
    rows, used = [], []
    for gname, cols in GROUPS.items():
        used = used + cols
        ma_g = ModelA_GBM(params=pa_best, features=used).fit(tra_s)
        fb_ = used + ["minutes_empty"]
        mb_g = ModelB_GBM(params=pb_best, features=fb_).fit(trb_s)
        rows.append({"Added group": gname.replace("_", " "), "Features": len(used),
                     "Model A deviance": fmt(poisson_deviance(ya, ma_g.predict(va))),
                     "Model B C-index": fmt(concordance_index(T, E, mb_g.predict_time(vb)))})
        print("E4", gname, rows[-1], flush=True)
    lines += ["## E4: which feature groups add value?", "",
              "Boosted trees with the tuned settings, trained on 400,000 training rows, adding one group at a time "
              "(calendar and place first). Model B always also has minutes empty.", ""]
    lines += table(rows, ["Added group", "Features", "Model A deviance", "Model B C-index"]) + [""]

    # ---------------------------------------------------------------- feature importance (permutation)
    def perm_importance(score_fn, df, feats, n, higher_better):
        d = df.iloc[rng.choice(len(df), min(len(df), n), replace=False)].copy()
        base = score_fn(d)
        imp = {}
        for f in feats:
            x = d[f].copy()
            d[f] = rng.permutation(x.to_numpy())
            s = score_fn(d)
            imp[f] = (base - s) if higher_better else (s - base)
            d[f] = x
        return pd.Series(imp).sort_values(ascending=False)

    gbm_a, gbm_b = load("model_a_gbm"), load("model_b_gbm")
    imp_a = perm_importance(lambda d: poisson_deviance(d["target_pickups"], gbm_a.predict(d)), va, FEATURES_A, 60_000, False)
    imp_b = perm_importance(lambda d: concordance_index(d["minutes_until_end"], d["pickup_seen"], gbm_b.predict_time(d),
                                                        max_n=15_000), vb, FEATURES_B, 15_000, True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, imp, ttl, xl in [(axes[0], imp_a, "Model A (boosted trees)", "increase in Poisson deviance"),
                             (axes[1], imp_b, "Model B (boosted trees)", "drop in C-index")]:
        top = imp.head(12)[::-1]
        ax.barh(top.index, top.values, color=BLUE, height=0.65)
        style(ax, ttl, xl, "")
        ax.grid(axis="y", visible=False)
    fi = save(fig, "models_feature_importance.png")
    lines += ["## Feature importance", "",
              "Permutation importance on validation rows: how much worse the model gets when one feature is "
              "shuffled. These are associations the model uses, not causes.", "", f"![Feature importance]({fi})", ""]

    # ---------------------------------------------------------------- tuning summary
    lines += ["## Tuning", "", "Random search with the same budget for each learned model, scored on two "
              "forward-chaining folds inside the training period.", ""]
    rows = [{"Model": NAMES[k], "Metric": v["metric"], "Trials": len(v["trials"]), "Best fold mean": fmt(v["best"]["mean"], 4),
             "Best setting": str(v["best"]["setting"]).replace("|", "/")} for k, v in tuning.items()]
    lines += table(rows, ["Model", "Metric", "Trials", "Best fold mean", "Best setting"]) + [""]

    # ---------------------------------------------------------------- freeze
    final = {"model_a": final_a, "model_b": final_b, "fallback_b": "baseline_b", "baseline_a": "baseline_a",
             "validation": {"model_a": ma, "model_b": mb}, "features_a": FEATURES_A, "features_b": FEATURES_B,
             "tuning": {k: tuning[k]["best"]["setting"] for k in tuning}}
    (MODELS / "final.json").write_text(json.dumps(final, indent=2, default=float))
    card = ["# Model card", "",
            f"- **Model A** (pickups per hexagon-hour): {NAMES[final_a]}, setting {final['tuning'][final_a]}.",
            f"  Validation Poisson deviance {ma[final_a]['poisson_deviance']:.3f} (historical average {base_dev:.3f}).",
            f"- **Model B** (minutes until pickup): {NAMES[final_b]}, setting {final['tuning'][final_b]}.",
            f"  Validation C-index {mb[final_b]['c_index']:.3f} (historical curves {mb['baseline_b']['c_index']:.3f}), "
            f"Brier at 15 min {mb[final_b]['brier_15']:.3f}.",
            "- **Fallback** for Model B: historical survival curves by area, 3-hour band and minutes-empty band.",
            "- **Trained on** 29 Jan to 15 Sep 2025 (29 Aug excluded), sampled as in `model_input/info.json`.",
            "- **Inputs**: the features in `docs/features.md`; Model B also uses minutes empty.",
            "- **Not for**: claims about real driver time savings; areas and hours with little data (use the coverage rule).",
            "- Files: `pipeline_output/models/*.joblib`, choice in `pipeline_output/models/final.json`."]
    (DOCS / "model_card.md").write_text("\n".join(card) + "\n")
    (DOCS / "models_validation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
