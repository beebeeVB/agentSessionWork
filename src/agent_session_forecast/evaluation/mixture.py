"""Latent-class mixture-of-experts forecaster (``--mixture``).

Idea: sessions may come in a few behavioural types (quick one-shot, exploratory, long grind) with
very different length distributions; one regression forced through all of them blurs that.

Inside every training fold (so nothing about the test accounts is used):
  1. standardise the early-behaviour features and reduce to 8 principal components;
  2. fit Gaussian mixtures (diagonal covariance) with each candidate number of classes in
     ``cfg.MIX_CLASSES`` and keep the one with the lowest BIC;
  3. train one gradient-boosting expert per class on the sessions whose most likely class it is
     (classes with fewer than 60 sessions fall back to the global model);
  4. predict with the posterior-probability-weighted average of the experts (soft gating).

``mixture_profiles.csv`` describes the classes of a mixture fitted on all the data at one k
(size, length distribution, harness mix) so the classes can be interpreted, not just scored.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm, prep_linear
from .compare import evaluate_predictors, fmt_paired

MIN_CLASS = 60


def fit_mixture(Ztr):
    best = None
    for c in cfg.MIX_CLASSES:
        if c * 30 > len(Ztr):
            continue
        gm = GaussianMixture(c, covariance_type="diag", random_state=0, n_init=2).fit(Ztr)
        b = gm.bic(Ztr)
        if best is None or b < best[0]:
            best = (b, c, gm)
    return best


def pred_mixture(out, tr, te, ctx):
    X, y = out["X"], out["y"]
    A_tr, A_te = prep_linear(X.iloc[tr], X.iloc[te])
    pca = PCA(min(8, A_tr.shape[1]), random_state=0).fit(A_tr)
    Ztr, Zte = pca.transform(A_tr), pca.transform(A_te)
    best = fit_mixture(Ztr)
    base_model = new_gbm().fit(X.iloc[tr], y[tr])
    base = base_model.predict(X.iloc[te])
    if best is None:
        return base
    _, c, gm = best
    R_tr, R_te = gm.predict_proba(Ztr), gm.predict_proba(Zte)
    hard = R_tr.argmax(1)
    pred = np.zeros(len(te))
    for j in range(c):
        idx = np.where(hard == j)[0]
        p = base if len(idx) < MIN_CLASS else new_gbm().fit(X.iloc[tr[idx]], y[tr[idx]]).predict(X.iloc[te])
        pred += R_te[:, j] * p
    return pred


def profiles(df, k):
    out = features_at_k(df, k)
    if out is None or len(out["y"]) < 200:
        return pd.DataFrame()
    A, _ = prep_linear(out["X"], out["X"])
    Z = PCA(min(8, A.shape[1]), random_state=0).fit_transform(A)
    best = fit_mixture(Z)
    if best is None:
        return pd.DataFrame()
    _, c, gm = best
    cls = gm.predict(Z)
    d, n = out["d"], out["n"]
    rows = []
    for j in range(c):
        m = cls == j
        if not m.any():
            continue
        row = dict(k=k, n_classes=c, cls=j, size=int(m.sum()), share=float(m.mean()),
                   median_n_requests=float(np.median(n[m])), p90_n_requests=float(np.quantile(n[m], 0.9)),
                   mean_log_length=float(out["y"][m].mean()))
        mix = d.loc[m, "harness"].fillna("?").value_counts(normalize=True)
        for h, v in mix.head(4).items():
            row[f"harness_{h}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("median_n_requests").reset_index(drop=True)


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    res, paired, store = evaluate_predictors(df, {"mixture_experts": pred_mixture})
    kp = 5 if 5 in cfg.EXTRA_KS else cfg.EXTRA_KS[-1]
    prof = profiles(df, kp)
    tables = {"mixture_forecast": res, "mixture_paired": paired, "mixture_profiles": prof}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", "LATENT-CLASS MIXTURE OF EXPERTS (classes by BIC inside each training fold):"]
    L += fmt_paired(paired, designs=("grouped", "random"), refs=("gbm",))
    if len(prof):
        L += [f"  class profiles (k={kp}, mixture fitted on all data; descriptive):",
              prof.round(3).to_string(index=False)]
    return dict(tables=tables, lines=L, payload=dict(res=res, paired=paired, prof=prof))
