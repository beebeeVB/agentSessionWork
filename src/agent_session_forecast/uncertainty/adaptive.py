"""Adaptive Conformal Inference (ACI): do intervals survive drift over time?

Trains on the early part of the week, then streams the later sessions one by one,
nudging the miscoverage level after every outcome.
"""
import numpy as np

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm
from .conformal import conformal_q


def run_aci(df, k, alpha=cfg.ALPHA, gamma=0.02, refresh=10):
    """Adaptive Conformal Inference vs static split-conformal under time drift.

    Train on first 40% of time, calibrate on next 20%, then STREAM the last 40%.
    ACI nudges alpha_t after every outcome so long-run miscoverage tracks alpha.
    """
    out = features_at_k(df, k)
    if out is None:
        return None
    order = np.argsort(out["start"])
    X = out["X"].iloc[order].reset_index(drop=True)
    y = out["y"][order]
    n = len(y)
    if n < 600:
        return None
    i1, i2 = int(.4 * n), int(.6 * n)
    m = new_gbm().fit(X.iloc[:i1], y[:i1])
    cal = np.abs(y[i1:i2] - m.predict(X.iloc[i1:i2]))
    q_static = conformal_q(cal, alpha)
    sc = np.abs(y[i2:] - m.predict(X.iloc[i2:]))
    err_static = (sc > q_static).astype(float)
    a_t, pool, q_t = alpha, list(cal), q_static
    err_aci = np.zeros(len(sc))
    for t in range(len(sc)):
        if t % refresh == 0:
            level = 1 - a_t
            q_t = np.inf if level >= 1 else (0.0 if level <= 0 else
                                              np.quantile(pool, level, method="higher"))
        err_aci[t] = float(sc[t] > q_t)
        a_t += gamma * (alpha - err_aci[t])
        pool.append(sc[t])
    return dict(err_static=err_static, err_aci=err_aci, n=len(sc),
                cov_static=1 - err_static.mean(), cov_aci=1 - err_aci.mean())
