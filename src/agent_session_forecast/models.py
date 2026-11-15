"""The models being compared, plus the train/test split helpers.

Baselines (global median, per-harness median), gradient boosting, gradient
boosting with account-history features, the quantile / scale / classifier
variants used for uncertainty, and a temporal splitter.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as cfg
from .features import add_history


_GBM_KW = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
               categorical_features=[0, 1], random_state=0)


def new_gbm():
    return HistGradientBoostingRegressor(**_GBM_KW)


def new_gbm_q(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **_GBM_KW)


def new_scale_gbm():
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=8,
                                         min_samples_leaf=30, categorical_features=[0, 1],
                                         random_state=0)


def new_clf():
    kw = dict(_GBM_KW); kw["max_iter"] = 150
    return HistGradientBoostingClassifier(**kw)


def baseline_harness_predict(Xtr, ytr, Xte):
    """Survival-conditional per-harness median (train rows are already n > k)."""
    med = pd.Series(ytr).groupby(Xtr["harness"].values).median()
    glob = float(np.median(ytr))
    return np.array([med.get(h, glob) for h in Xte["harness"].values])


class TemporalSplit:
    """One split: train on the earliest `frac` of sessions, test on the rest."""
    def __init__(self, t, frac=0.7):
        self.t, self.frac = np.asarray(t), frac

    def split(self, X, y=None, groups=None):
        cut = np.quantile(self.t, self.frac)
        yield np.where(self.t <= cut)[0], np.where(self.t > cut)[0]


def oof_predictions(out, splitter, split_groups):
    """Out-of-fold predictions for every model. NaN where a row is never tested."""
    X, y, groups = out["X"], out["y"], out["groups"]
    P = {m: np.full(len(y), np.nan) for m in cfg.MODELS}
    for tr, te in splitter.split(X, y, split_groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        P["baseline_global"][te] = float(np.median(y[tr]))
        P["baseline_harness"][te] = baseline_harness_predict(Xtr, y[tr], Xte)
        P["gbm"][te] = new_gbm().fit(Xtr, y[tr]).predict(Xte)
        Htr, Hte = add_history(Xtr, y[tr], groups[tr], Xte, groups[te])
        P["gbm_hist"][te] = new_gbm().fit(Htr, y[tr]).predict(Hte)
    return P
