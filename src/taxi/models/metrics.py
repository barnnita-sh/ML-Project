"""Evaluation metrics for Model A (counts) and Model B (censored waits)."""
import numpy as np

from .survival import km_curve, time_index


# ---------------------------------------------------------------- Model A
def poisson_deviance(y, mu):
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-9, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def count_metrics(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    z = y == 0
    return {
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "poisson_deviance": poisson_deviance(y, mu),
        "mae_zero_hours": float(np.mean(np.abs(y[z] - mu[z]))) if z.any() else np.nan,
        "mae_nonzero_hours": float(np.mean(np.abs(y[~z] - mu[~z]))) if (~z).any() else np.nan,
        "mean_predicted": float(mu.mean()),
        "mean_actual": float(y.mean()),
    }


# ---------------------------------------------------------------- Model B
def concordance_index(minutes, event, predicted_time, max_n=20_000, seed=0, chunk=500):
    """Harrell's C-index on a random subsample: among usable pairs (the shorter time is a
    real pickup), how often the row predicted to wait less really waited less.
    0.5 = coin flip, 1 = perfect. Censored rows are used correctly."""
    T = np.asarray(minutes, float)
    E = np.asarray(event, bool)
    P = np.asarray(predicted_time, float)
    if len(T) > max_n:
        idx = np.random.default_rng(seed).choice(len(T), max_n, replace=False)
        T, E, P = T[idx], E[idx], P[idx]
    conc = ties = total = 0.0
    ev = np.flatnonzero(E)
    for s in range(0, len(ev), chunk):
        i = ev[s:s + chunk]
        usable = T[None, :] > T[i, None]                 # j waited longer than i, and i is a real pickup
        pi, pj = P[i, None], P[None, :]
        conc += np.sum(usable & (pi < pj))
        ties += np.sum(usable & (pi == pj))
        total += np.sum(usable)
    return float((conc + 0.5 * ties) / total) if total else np.nan


def censoring_curve(minutes, event):
    """G(t): chance of NOT yet being censored at t (Kaplan-Meier with the roles swapped)."""
    return km_curve(minutes, ~np.asarray(event, bool))


def brier_at(minutes, event, s_t, t, G):
    """Censoring-adjusted (IPCW) Brier score of the predicted 'still waiting at t' s_t."""
    T = np.asarray(minutes, float)
    E = np.asarray(event, bool)
    ti = time_index(T)
    g_before = np.where(ti > 0, G[np.maximum(ti - 1, 0)], 1.0)
    w_event = np.where((T <= t) & E, 1 / np.clip(g_before, 1e-6, None), 0.0)
    w_alive = np.where(T > t, 1 / max(G[t], 1e-6), 0.0)
    return float(np.mean(w_event * s_t ** 2 + w_alive * (1 - s_t) ** 2))


def calibration_table(minutes, event, p_pickup, t=15, bins=10):
    """Group rows by predicted chance of a pickup within t minutes; compare with the
    Kaplan-Meier observed chance in each group."""
    p = np.asarray(p_pickup, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() < 50:
            continue
        S = km_curve(np.asarray(minutes)[m], np.asarray(event)[m])
        rows.append((float(p[m].mean()), float(1 - S[t]), int(m.sum())))
    return rows


def survival_metrics(minutes, event, summ, G):
    """summ: output of survival.predict_summaries or a baseline's equivalent."""
    T = np.asarray(minutes, float)
    E = np.asarray(event, bool)
    cal = calibration_table(T, E, summ["p_pickup_15"])
    return {
        "c_index": concordance_index(T, E, summ["rmean_120"]),
        "mae_completed_min": float(np.mean(np.abs(summ["median"][E] - T[E]))),
        "brier_15": brier_at(T, E, summ["s_15"], 15, G),
        "calibration_error_15": float(np.mean([abs(a - b) for a, b, _ in cal])) if cal else np.nan,
        "mean_p_pickup_15": float(np.mean(summ["p_pickup_15"])),
        "observed_p_pickup_15": float(1 - km_curve(T, E)[15]),
    }
