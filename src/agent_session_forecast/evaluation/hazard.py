"""Landmark discrete-time hazard / survival model (``--hazard``).

Instead of regressing log(1 + n_requests) directly, model *how long the session keeps going*:
at landmark k (we have seen k requests and the session is still alive) the hazard
``h(j | x_k)`` is the probability that the session ends exactly j requests later, given that it
lasted j-1 more. Stacking the person-period rows (one row per session per future step j) turns
this into an ordinary classification problem. From the hazards:

* survival  S(j) = prod_{i<=j} (1 - h(i))              -> P(session lasts more than j more requests)
* median remaining length  min{j : S(j) <= 0.5}        -> a point forecast comparable to the others
* the full remaining-length distribution                -> useful for scheduling decisions

Right-censored sessions (cut off at the week boundary) contribute hazard terms for the steps we
actually saw and no terminal event, so they bias nothing. The horizon is capped at
``cfg.HAZARD_J`` requests; a median beyond the cap is reported as J + 1 (a documented limitation).

Two hazard models are compared: gradient boosting and a plain logistic regression.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .. import config as cfg
from ..features import features_at_k
from ..models import prep_linear
from .classification import fold_auc
from .compare import evaluate_predictors, fmt_paired


def person_period(r, cens, J):
    """(row index, horizon j, event flag) for sessions with remaining lengths r >= 1."""
    r, cens = np.asarray(r), np.asarray(cens, bool)
    m = np.where(cens & (r <= J), r - 1, np.minimum(r, J))     # censored end-step carries no event
    m = np.maximum(m, 0)
    rep = np.repeat(np.arange(len(r)), m)
    j = np.concatenate([np.arange(1, mm + 1) for mm in m]) if m.sum() else np.array([], int)
    ev = ((j == r[rep]) & ~cens[rep]).astype(int)
    return rep, j, ev


def _stack(X, rep, j):
    Z = X.iloc[rep].reset_index(drop=True).copy()
    Z["j"] = j
    Z["logj"] = np.log(j)
    return Z


def _fit_predict_survival(kind, X, r, cens, tr, te, J):
    """Fit the hazard model on training sessions; return survival matrix (len(te) x J)."""
    rep, j, ev = person_period(r[tr], cens[tr], J)
    Ztr = _stack(X.iloc[tr], rep, j)
    rep_te = np.repeat(np.arange(len(te)), J)
    Zte = _stack(X.iloc[te], rep_te, np.tile(np.arange(1, J + 1), len(te)))
    if ev.sum() < 5:                                    # (almost) no events: constant hazard
        h = np.full(len(te) * J, max(ev.mean(), 1e-3))
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, max_leaf_nodes=15,
                                             min_samples_leaf=40, categorical_features=[0, 1],
                                             random_state=0).fit(Ztr, ev)
        h = clf.predict_proba(Zte)[:, 1]
    else:
        A, B = prep_linear(Ztr, Zte)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(A, ev)
        h = clf.predict_proba(B)[:, 1]
    h = np.clip(h.reshape(len(te), J), 1e-6, 1 - 1e-6)
    return np.cumprod(1 - h, axis=1)


def median_remaining(S):
    """Smallest j with S(j) <= 0.5; J + 1 if the survival curve never gets that low."""
    J = S.shape[1]
    hit = S <= 0.5
    return np.where(hit.any(1), hit.argmax(1) + 1, J + 1)


def make_predictor(kind, surv):
    def fn(out, tr, te, ctx):
        k, J = ctx["k"], cfg.HAZARD_J
        r = out["n"] - k
        S = _fit_predict_survival(kind, out["X"], r, out["cens_right"], tr, te, J)
        key = (k, ctx["design"], f"hazard_{kind}")
        if key not in surv:
            surv[key] = dict(S=np.full((len(out["y"]), J), np.nan), fold=np.full(len(out["y"]), -1))
        surv[key]["S"][te] = S
        surv[key]["fold"][te] = ctx["fold"]
        return np.log1p(k + median_remaining(S))
    return fn


def _calibration(df, surv, horizons=(5, 10, 20), nbins=8):
    cal, sco, curve, scat = [], [], [], []
    for (k, design, model), v in surv.items():
        if design != "grouped":
            continue
        out = features_at_k(df, k)
        S, fold = v["S"], v["fold"]
        ok = np.isfinite(S).all(1)
        r, cr = out["n"] - k, out["cens_right"]
        J = S.shape[1]
        for j in range(1, J + 1):                         # survival curve: predicted vs observed
            known = ok & ~(cr & (r <= j))
            if known.sum() >= 30 and j in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40):
                curve.append(dict(k=k, model=model, j=j, pred=S[known, j - 1].mean(),
                                  obs=float((r[known] > j).mean()), n=int(known.sum())))
        for h in horizons:
            if h > J:
                continue
            known = ok & ~(cr & (r <= h))
            if known.sum() < 60:
                continue
            p, o, fo = S[known, h - 1], (r[known] > h).astype(int), fold[known]
            clim = o.mean()
            bs, bc = float(np.mean((p - o) ** 2)), float(np.mean((clim - o) ** 2))
            sco.append(dict(k=k, model=model, horizon=h, n=int(known.sum()), prevalence=clim,
                            auc=fold_auc(o, p, fo) if len(set(o)) == 2 else np.nan,
                            brier=bs, brier_skill_vs_climatology=1 - bs / bc if bc > 0 else np.nan))
            if h == 10:
                q = pd.qcut(p, min(nbins, max(2, len(p) // 40)), duplicates="drop")
                t = pd.DataFrame({"p": p, "o": o, "b": q}).groupby("b", observed=True).agg(
                    pred=("p", "mean"), obs=("o", "mean"), n=("o", "size")).reset_index(drop=True)
                t.insert(0, "horizon", h); t.insert(0, "model", model); t.insert(0, "k", k)
                cal.append(t)
        if model == "hazard_gbm":
            m = ok & ~cr
            scat.append(pd.DataFrame(dict(k=k, pred_remaining=median_remaining(S[m]), true_remaining=r[m])))
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return cat(cal), pd.DataFrame(sco), pd.DataFrame(curve), cat(scat)


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    surv = {}
    preds = {"hazard_gbm": make_predictor("gbm", surv), "hazard_logit": make_predictor("logit", surv)}
    res, paired, store = evaluate_predictors(df, preds)
    cal, scores, curve, scat = _calibration(df, surv)
    tables = {"hazard_forecast": res, "hazard_paired": paired, "hazard_horizon_scores": scores,
              "hazard_calibration": cal, "hazard_survival_curve": curve}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", "HAZARD MODEL (landmark discrete-time survival; remaining-length median as point forecast):",
         f"  horizon capped at J={cfg.HAZARD_J} requests ahead; a median beyond the cap is J+1"]
    L += fmt_paired(paired, designs=("grouped", "random"), refs=("baseline_harness",))
    if len(scores):
        s = scores[scores.model == "hazard_gbm"][["k", "horizon", "prevalence", "auc", "brier_skill_vs_climatology"]]
        L += ["  P(lasts > h more requests), hazard_gbm, grouped CV:", s.round(3).to_string(index=False)]
    return dict(tables=tables, lines=L, payload=dict(res=res, paired=paired, cal=cal, scores=scores,
                                                     curve=curve, scat=scat))
