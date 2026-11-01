"""Checks of the survival maths, metrics and baselines on small made-up data with known answers.
Run:  python tests/test_models.py   (or  python -m pytest tests)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taxi.models.baselines import HistoricalAverage  # noqa: E402
from taxi.models.metrics import brier_at, censoring_curve, concordance_index, poisson_deviance  # noqa: E402
from taxi.models.survival import CoxModel, ResidualCurveModel, km_curve  # noqa: E402


def test_kaplan_meier_by_hand():
    # pickups at 1, 2, 3 minutes; one taxi censored at 2
    S = km_curve([1, 2, 2, 3], [1, 1, 0, 1])
    assert S[0] == 1 and np.isclose(S[1], 3 / 4)
    assert np.isclose(S[2], 3 / 4 * 2 / 3)            # 3 at risk at minute 2, 1 pickup
    assert np.isclose(S[3], 0.0)                     # last one at risk picks up


def test_concordance_index():
    T = np.array([5, 10, 15, 20, 25.0])
    E = np.ones(5, bool)
    assert concordance_index(T, E, T) == 1.0
    assert concordance_index(T, E, -T) == 0.0
    assert concordance_index(T, E, np.ones(5)) == 0.5


def test_cox_recovers_known_effect():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=(n, 1))
    T = rng.exponential(20 / np.exp(0.7 * x[:, 0]))   # higher x -> faster pickup (hazard x e^0.7)
    C = rng.exponential(60, n)                       # random censoring
    m = CoxModel(l2=1e-6).fit(x, np.minimum(T, C), T <= C)
    assert abs(m.coef_[0] - 0.7) < 0.06, m.coef_
    S = m.predict_curve(np.array([[0.0]]))[0]
    assert abs(S[20] - np.exp(-1)) < 0.05            # baseline median-ish check: S(20) for x=0


def test_residual_curve_model():
    class Perfect:
        def fit(self, X, y): return self
        def predict(self, X): return np.log1p(np.asarray(X)[:, 0])
    X = np.array([[10.0], [30.0]])
    m = ResidualCurveModel(Perfect()).fit(X, [10, 30], X, [10, 30], [True, True])   # zero errors
    S = m.predict_curve(X)
    assert S[0, 9] == 1 and S[0, 10] == 0 and S[1, 29] == 1 and S[1, 30] == 0


def test_brier_perfect_and_poisson_deviance():
    T = np.array([5, 30.0]); E = np.array([True, True])
    G = censoring_curve(T, E)
    assert brier_at(T, E, np.array([0.0, 1.0]), 15, G) == 0.0         # right about who is still waiting
    assert brier_at(T, E, np.array([1.0, 0.0]), 15, G) == 1.0
    assert poisson_deviance([0, 2], [0, 2]) < 1e-6


def test_historical_average_fallback():
    df = pd.DataFrame({"hex": ["a", "a", "b"], "hour_of_day": [1, 1, 2], "day_of_week": [0, 1, 0],
                       "target_pickups": [2, 4, 6]})
    m = HistoricalAverage().fit(df)
    q = pd.DataFrame({"hex": ["a", "a", "c"], "hour_of_day": [1, 5, 1], "day_of_week": [0, 0, 0]})
    p = m.predict(q)
    assert p[2] == 4                                  # unseen hexagon -> city mean
    assert 2 < p[0] < p[1] < 4                        # seen group (mean 2) shrunk towards hexagon/city (4)


def test_residual_errors_use_censored_waits():
    from taxi.models.survival import km_continuous
    v, s = km_continuous([1, 2, 3, 4], [1, 0, 1, 1])
    assert list(v) == [1, 3, 4] and np.allclose(s, [0.75, 0.375, 0.0])


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
            n += 1
    print(f"all {n} tests passed")
