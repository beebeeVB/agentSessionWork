"""A friendlier question: will this session be LONG?

Predicts P(total requests > T | first k requests) and scores it with AUC (averaged
within folds, see fold_auc for why) and Brier skill against a per-harness base rate.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from .. import config as cfg
from ..features import features_at_k
from ..models import new_clf


def fold_auc(y, p, fold):
    """Mean of WITHIN-fold AUCs. Pooling out-of-fold scores across folds is biased
    (under account-grouped CV it can fall below 0.5 with no real signal, because
    folds differ in base rate and the training mean moves opposite to the test mean)."""
    vals = [roc_auc_score(y[fold == f], p[fold == f]) for f in np.unique(fold)
            if len(set(y[fold == f])) == 2]
    return float(np.mean(vals)) if vals else np.nan


def boot_auc(y, p, groups, fold, B=100):
    point = fold_auc(y, p, fold)
    idx_by = list(pd.Series(np.arange(len(y))).groupby(groups).indices.values())
    vals = []
    for _ in range(B):
        pick = cfg.RNG.integers(0, len(idx_by), len(idx_by))
        ii = np.concatenate([idx_by[j] for j in pick])
        v = fold_auc(y[ii], p[ii], fold[ii])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return point, np.nan, np.nan
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def evaluate_long_sessions(df, ks=(1, 3, 5, 12), Ts=(20, 50)):
    """P(total requests > T | first k requests), random vs account-grouped."""
    rows = []
    for k in ks:
        out = features_at_k(df, k)
        if out is None:
            continue
        X, groups, n = out["X"], out["groups"], out["n"]
        for T in Ts:
            if k >= T:
                continue
            yb = (n > T).astype(int)
            if yb.sum() < 30 or (1 - yb).sum() < 30 or len(set(groups)) < 6:
                continue
            designs = {"random": (KFold(5, shuffle=True, random_state=0), None),
                       "grouped": (GroupKFold(5), groups)}
            for dname, (splitter, sg) in designs.items():
                pg, pb = np.zeros(len(yb)), np.zeros(len(yb))
                fold = np.zeros(len(yb), dtype=int)
                for fi, (tr, te) in enumerate(splitter.split(X, yb, sg)):
                    fold[te] = fi
                    if len(set(yb[tr])) < 2:
                        pg[te] = pb[te] = yb[tr].mean(); continue
                    clf = new_clf().fit(X.iloc[tr], yb[tr])
                    pg[te] = clf.predict_proba(X.iloc[te])[:, 1]
                    rate = pd.Series(yb[tr]).groupby(X.iloc[tr]["harness"].values).mean()
                    gl = yb[tr].mean()
                    pb[te] = [rate.get(h, gl) for h in X.iloc[te]["harness"].values]
                auc, lo, hi = boot_auc(yb, pg, groups, fold)
                auc_b = fold_auc(yb, pb, fold)
                bs_g, bs_b = brier_score_loss(yb, pg), brier_score_loss(yb, pb)
                rows.append(dict(k=k, T=T, design=dname, n=len(yb), pos_rate=yb.mean(),
                                 auc_gbm=auc, auc_lo=lo, auc_hi=hi, auc_baseline=auc_b,
                                 brier_gbm=bs_g, brier_baseline=bs_b,
                                 brier_skill=1 - bs_g / bs_b))
    return pd.DataFrame(rows)
