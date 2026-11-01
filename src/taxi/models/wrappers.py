"""Model A and Model B wrappers: each bundles its preprocessing and estimator so that
fit(df) / predict(df) work directly on the training tables.

Model A (pickups per hexagon-hour):
  GLM  - Poisson regression (scikit-learn PoissonRegressor) on LinearPrep features
  GBM  - histogram gradient-boosted trees with Poisson loss (scikit-learn
         HistGradientBoostingRegressor, the same algorithm family as LightGBM); uses
         LightGBM itself if it is installed and use_lightgbm=True
Model B (minutes until pickup, censored):
  GBM  - boosted trees on log(1 + minutes) of completed waits, turned into a survival curve
         with the spread of its errors (survival.ResidualCurveModel)
  Cox  - Cox proportional hazards with L2 penalty (survival.CoxModel) on LinearPrep features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from .data import FEATURES_A, FEATURES_B
from .prep import LinearPrep
from .survival import CoxModel, ResidualCurveModel, predict_summaries

GBM_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [20, 50, 100, 200, 500],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "max_features": [0.5, 0.8, 1.0],
}


def sample_params(n, seed=0):
    """n different random settings from GBM_SPACE (random search)."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        s = {k: v[rng.integers(len(v))] for k, v in GBM_SPACE.items()}
        key = tuple(s.values())
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _gbm(loss, params, seed=0):
    return HistGradientBoostingRegressor(loss=loss, max_iter=500, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=seed, **params)


class ModelA_GLM:
    name = "model_a_glm"

    def __init__(self, alpha=1e-3, features=FEATURES_A):
        self.alpha, self.features = alpha, list(features)

    def fit(self, df):
        self.prep = LinearPrep(self.features).fit(df)
        self.est = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(self.prep.transform(df), df["target_pickups"])
        return self

    def predict(self, df):
        return self.est.predict(self.prep.transform(df))


class ModelA_GBM:
    name = "model_a_gbm"

    def __init__(self, params=None, features=FEATURES_A, seed=0):
        self.params, self.features, self.seed = params or {}, list(features), seed

    def fit(self, df):
        self.est = _gbm("poisson", self.params, self.seed).fit(df[self.features], df["target_pickups"])
        return self

    def predict(self, df):
        return self.est.predict(df[self.features])


class ModelB_GBM:
    name = "model_b_gbm"

    def __init__(self, params=None, features=FEATURES_B, seed=0):
        self.params, self.features, self.seed = params or {}, list(features), seed

    def fit(self, df):
        # hold out 10% of waits (all their snapshots) to measure the error distribution,
        # including censored ones; fit the regressor on completed waits of the rest
        ep = df["episode_id"].astype(str)
        hold = (pd.util.hash_pandas_object(ep + "hold", index=False).to_numpy() % 10) == 0
        fit = df[~hold & df["pickup_seen"].to_numpy(bool)]
        h = df[hold]
        self.curve = ResidualCurveModel(_gbm("squared_error", self.params, self.seed)).fit(
            fit[self.features], fit["minutes_until_end"].to_numpy(), h[self.features],
            h["minutes_until_end"].to_numpy(), h["pickup_seen"].to_numpy(bool))
        return self

    def predict_time(self, df):
        """Predicted typical further wait in minutes (used for ranking / C-index)."""
        return np.expm1(self.curve.predict_log(df[self.features]))

    def predict_summaries(self, df):
        return predict_summaries(self.curve, df[self.features])


class ModelB_Cox:
    name = "model_b_cox"

    def __init__(self, l2=0.1, features=FEATURES_B):
        self.l2, self.features = l2, list(features)

    def fit(self, df):
        self.prep = LinearPrep(self.features).fit(df)
        self.est = CoxModel(l2=self.l2).fit(self.prep.transform(df), df["minutes_until_end"].to_numpy(),
                                            df["pickup_seen"].to_numpy(bool))
        return self

    def predict_time(self, df):
        return -self.est.risk(self.prep.transform(df))       # higher risk = sooner pickup

    def predict_summaries(self, df):
        return predict_summaries(self.est, self.prep.transform(df))
