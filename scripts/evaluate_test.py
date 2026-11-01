"""ONE-TIME test-period evaluation of the frozen models (week 3 of the plan).

  python scripts/evaluate_test.py --confirm

Only run this after the models, features and thresholds are frozen (evaluate_validation.py
wrote pipeline_output/models/final.json) and the team has agreed. It refuses to run twice:
after a run it writes pipeline_output/models/TEST_RUN_DONE. Re-running anyway needs
--force and must be reported, because looking at test results more than once and then
changing the model turns the test into a second validation set.

Writes docs/models_test.md with the same metrics as the validation report (Model A and
Model B against their baselines). The replay comparison with the heat map (E2) is a
separate script (evaluation task).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402

from taxi import config  # noqa: E402
from taxi.models.data import load_split  # noqa: E402
from taxi.models.metrics import censoring_curve, count_metrics, survival_metrics  # noqa: E402

MODELS = config.OUT_DIR / "models"


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

    ta = load_split("modelA", "test")
    y = ta["target_pickups"].to_numpy()
    res_a = {k: count_metrics(y, joblib.load(MODELS / f"{k}.joblib").predict(ta)) for k in ("baseline_a", final["model_a"])}

    tb = load_split("modelB", "test")
    T, E = tb["minutes_until_end"].to_numpy(), tb["pickup_seen"].to_numpy(bool)
    G = censoring_curve(T, E)
    res_b = {k: survival_metrics(T, E, joblib.load(MODELS / f"{k}.joblib").predict_summaries(tb), G)
             for k in ("baseline_b", final["model_b"])}

    lines = ["# Test-period results (one-time run)", "", f"Run on {datetime.now():%Y-%m-%d %H:%M}.", "",
             "## Model A", "", "| Model | Poisson deviance | MAE | RMSE |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {m['poisson_deviance']:.3f} | {m['mae']:.3f} | {m['rmse']:.3f} |" for k, m in res_a.items()]
    lines += ["", "## Model B", "", "| Model | C-index | Brier at 15 min | Calibration error at 15 min | MAE completed (min) |",
              "| --- | --- | --- | --- | --- |"]
    lines += [f"| {k} | {m['c_index']:.3f} | {m['brier_15']:.3f} | {m['calibration_error_15']:.3f} | "
              f"{m['mae_completed_min']:.1f} |" for k, m in res_b.items()]
    (config.PROJECT_DIR / "docs" / "models_test.md").write_text("\n".join(lines) + "\n")
    lock.write_text(f"test run on {datetime.now():%Y-%m-%d %H:%M}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
