"""Survival curves for Model B: Kaplan-Meier, Cox, and a curve from a time-regression model.

All curves are on a 1-minute grid t = 0, 1, ..., HMAX (minutes after the snapshot):
S[t] = chance the taxi is still waiting t minutes later.
Times are counted in whole minutes, rounded up (a pickup after 4.2 minutes counts at 5).
"""
import numpy as np
from scipy import optimize, sparse

HMAX = 181                      # labels are at most 180 minutes (waits are censored at 180)
GRID = np.arange(HMAX + 1)


def time_index(minutes):
    return np.clip(np.ceil(np.asarray(minutes, float)).astype(int), 0, HMAX)


# ---------------------------------------------------------------- Kaplan-Meier
def km_curve(minutes, event):
    """Kaplan-Meier 'still waiting' curve on GRID. Censored rows count as at risk until they leave."""
    t = time_index(minutes)
    e = np.asarray(event, bool)
    d = np.bincount(t[e], minlength=HMAX + 1).astype(float)
    leave = np.bincount(t, minlength=HMAX + 1)
    at_risk = len(t) - np.r_[0, np.cumsum(leave)[:-1]]
    hazard = np.divide(d, at_risk, out=np.zeros_like(d), where=at_risk > 0)
    return np.cumprod(1 - hazard)


# ---------------------------------------------------------------- curve summaries
def curve_summaries(S):
    """S: (n, HMAX+1). Returns the numbers every evaluation and the app need."""
    below = S <= 0.5
    median = np.where(below.any(axis=1), below.argmax(axis=1), HMAX).astype(float)
    return {
        "p_pickup_15": 1 - S[:, 15],              # chance of a pickup within 15 minutes
        "s_15": S[:, 15],
        "median": median,                          # predicted median further wait
        "rmean_60": S[:, :60].sum(axis=1),         # expected wait counted up to 60 minutes
        "rmean_120": S[:, :120].sum(axis=1),
    }


def predict_summaries(model, X, chunk=50_000):
    out = {}
    for i in range(0, len(X), chunk):
        part = curve_summaries(model.predict_curve(X[i:i + chunk]))
        for k, v in part.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- Cox proportional hazards
class CoxModel:
    """Cox model with an L2 penalty, Breslow handling of ties, fitted with L-BFGS.
    X must already be numeric and standardised (see prep.LinearPrep)."""

    def __init__(self, l2=0.1, max_iter=200):
        self.l2 = l2
        self.max_iter = max_iter

    def fit(self, X, minutes, event):
        X = np.asarray(X, np.float64)
        t = time_index(minutes)
        e = np.asarray(event, bool)
        n, p = X.shape
        M = sparse.csr_matrix((np.ones(n), (t, np.arange(n))), shape=(HMAX + 1, n))  # time x row
        d = np.bincount(t[e], minlength=HMAX + 1).astype(float)
        xe = X[e].sum(axis=0)
        n_ev = max(e.sum(), 1)

        def loss_grad(b):
            eta = X @ b
            c = eta.max()
            w = np.exp(eta - c)
            r = np.cumsum((M @ w)[::-1])[::-1]                     # sum of w over rows with t >= time
            rx = np.cumsum((M @ (w[:, None] * X))[::-1], axis=0)[::-1]
            ok = d > 0
            ll = eta[e].sum() - (d[ok] * (np.log(r[ok]) + c)).sum()
            g = xe - (d[ok, None] * rx[ok] / r[ok, None]).sum(axis=0)
            loss = -ll / n_ev + 0.5 * self.l2 * b @ b
            grad = -g / n_ev + self.l2 * b
            return loss, grad

        res = optimize.minimize(loss_grad, np.zeros(p), jac=True, method="L-BFGS-B",
                                options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.converged_ = bool(res.success)
        # Breslow baseline cumulative hazard
        eta = X @ self.coef_
        c = eta.max()
        r = np.cumsum((M @ np.exp(eta - c))[::-1])[::-1]
        h0 = np.divide(d, r * np.exp(c), out=np.zeros_like(d), where=r > 0)
        self.H0 = np.cumsum(h0)
        return self

    def risk(self, X):
        return np.asarray(X, np.float64) @ self.coef_

    def predict_curve(self, X):
        return np.exp(-np.outer(np.exp(self.risk(X)), self.H0))


# ---------------------------------------------------------------- curve from a time regression
def km_continuous(values, event):
    """Kaplan-Meier on continuous values: returns (sorted event values, survival just after each)."""
    v = np.asarray(values, float)
    e = np.asarray(event, bool)
    order = np.argsort(v, kind="mergesort")
    v, e = v[order], e[order]
    at_risk = len(v) - np.arange(len(v))
    factor = np.where(e, 1 - 1 / at_risk, 1.0)
    surv = np.cumprod(factor)
    return v[e], surv[e]


class ResidualCurveModel:
    """Wraps a regressor trained on log(1 + minutes) of completed waits.

    Curve: S(t) = P(log(1+T) > log(1+t)) where log(1+T) = prediction + error. The error
    distribution is estimated with Kaplan-Meier on held-out waits INCLUDING censored ones
    (a censored wait gives an error that is at least as large as observed). This corrects
    the optimism of a regressor that only saw completed waits."""

    def __init__(self, regressor, max_resid=300_000, seed=0):
        self.regressor = regressor
        self.max_resid = max_resid
        self.seed = seed

    def fit(self, X, minutes, X_hold, minutes_hold, event_hold=None):
        self.regressor.fit(X, np.log1p(minutes))
        r = np.log1p(np.asarray(minutes_hold, float)) - self.regressor.predict(X_hold)
        e = np.ones(len(r), bool) if event_hold is None else np.asarray(event_hold, bool)
        if len(r) > self.max_resid:
            idx = np.random.default_rng(self.seed).choice(len(r), self.max_resid, replace=False)
            r, e = r[idx], e[idx]
        self.r_values, self.r_surv = km_continuous(r, e)
        return self

    def predict_log(self, X):
        return self.regressor.predict(X)

    def predict_curve(self, X):
        mu = self.predict_log(X)
        z = np.log1p(GRID)[None, :] - mu[:, None]
        k = np.searchsorted(self.r_values, z.ravel(), side="right").reshape(z.shape)
        S = np.where(k > 0, self.r_surv[np.maximum(k - 1, 0)], 1.0)
        S[:, 0] = 1.0
        return S
