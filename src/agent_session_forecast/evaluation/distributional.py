"""Distributional forecasts: scoring the whole predictive distribution (``--distributional``).

Every method returns a predicted quantile for each level in ``cfg.QUANTILES`` and is scored on
the same grid, under the three designs, on identical folds:

* ``resid_quantiles``  - GBM mean + the empirical quantiles of its out-of-fold residuals
                         (the honest, distribution-free baseline);
* ``qgbm``             - one gradient-boosted quantile model per level (crossings are sorted);
* ``lognormal``        - GBM mean on the log scale + one global residual sigma;
* ``lognormal_het``    - same, but sigma depends on the features (a second model of |residual|).

Scores (lower is better):
* pinball loss averaged over the grid, and the grid CRPS ~ 2 x mean pinball over the levels;
* interval score of the central interval spanned by the lowest and highest level (it rewards
  narrow intervals and punishes misses);
* quantile calibration: the share of outcomes below each predicted level (should equal the level).

The grid is coarse on purpose (5 levels by default), so CRPS here is an approximation: use it to
compare methods against each other, not as an absolute number.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm, new_gbm_q, new_scale_gbm
from .bootstrap import paired_boot
from .compare import iter_folds


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def _oof_resid(X, y, tr):
    oof = np.zeros(len(tr))
    for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
        oof[b] = new_gbm().fit(X.iloc[tr[a]], y[tr[a]]).predict(X.iloc[tr[b]])
    return y[tr] - oof


def predict_quantiles(method, out, tr, te, cache=None):
    """(len(te), len(QUANTILES)) predicted quantiles on the log scale.

    ``cache`` (a dict) lets the methods of one fold share the GBM mean and its OOF residuals.
    """
    X, y, taus = out["X"], out["y"], np.array(cfg.QUANTILES)
    if method == "qgbm":
        Q = np.column_stack([new_gbm_q(t).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]) for t in taus])
    else:
        if cache is not None and "mu" in cache:
            mu, res = cache["mu"], cache["res"]
        else:
            mu = new_gbm().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            res = _oof_resid(X, y, tr)
            if cache is not None:
                cache["mu"], cache["res"] = mu, res
        if method == "resid_quantiles":
            Q = mu[:, None] + np.quantile(res, taus)[None, :]
        elif method == "lognormal":
            Q = mu[:, None] + res.std() * norm.ppf(taus)[None, :]
        elif method == "lognormal_het":
            sm = new_scale_gbm().fit(X.iloc[tr], np.abs(res))
            sd = np.maximum(sm.predict(X.iloc[te]) * np.sqrt(np.pi / 2), 0.05)   # E|N(0,s)| = s*sqrt(2/pi)
            Q = mu[:, None] + sd[:, None] * norm.ppf(taus)[None, :]
        else:
            raise ValueError(method)
    return np.sort(Q, axis=1)                                   # remove quantile crossing


METHODS = ["resid_quantiles", "qgbm", "lognormal", "lognormal_het"]


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    taus = np.array(cfg.QUANTILES)
    alpha = 1 - (taus[-1] - taus[0])
    rows, cal_rows, prow = [], [], []
    for k in cfg.EXTRA_KS:
        out = features_at_k(df, k)
        if out is None or len(out["y"]) < 100 or len(set(out["groups"])) < 6:
            continue
        y, groups = out["y"], out["groups"]
        print(f"    k={k}: n={len(y)}, methods: {', '.join(METHODS)}")
        scores = {}
        for design in cfg.DESIGNS:
            Qm = {m: np.full((len(y), len(taus)), np.nan) for m in METHODS}
            for f, tr, te in iter_folds(out, design):
                cache = {}
                for m in METHODS:
                    try:
                        Qm[m][te] = predict_quantiles(m, out, tr, te, cache)
                    except Exception as e:
                        print(f"    [warn] {m} failed ({type(e).__name__}: {e})")
            for m, Q in Qm.items():
                ok = np.isfinite(Q).all(1)
                if ok.sum() < 50:
                    continue
                pb = np.column_stack([pinball(y[ok], Q[ok, i], t) for i, t in enumerate(taus)])
                crps = 2 * pb.mean(1)
                lo, hi = Q[ok, 0], Q[ok, -1]
                iscore = (hi - lo) + (2 / alpha) * np.maximum(lo - y[ok], 0) + (2 / alpha) * np.maximum(y[ok] - hi, 0)
                c_full, i_full = np.full(len(y), np.nan), np.full(len(y), np.nan)
                c_full[ok], i_full[ok] = crps, iscore
                scores[(design, m)] = (c_full, i_full)
                rows.append(dict(k=k, design=design, method=m, n=int(ok.sum()),
                                 pinball=float(pb.mean()), crps_grid=float(crps.mean()),
                                 interval_score=float(iscore.mean()),
                                 coverage_central=float(((y[ok] >= lo) & (y[ok] <= hi)).mean()),
                                 width_factor=float(np.median(np.exp(hi - lo)))))
                for i, t in enumerate(taus):
                    cal_rows.append(dict(k=k, design=design, method=m, level=t,
                                         observed=float((y[ok] <= Q[ok, i]).mean())))
        for design in cfg.DESIGNS:                              # paired vs the residual-quantile baseline
            for m in METHODS[1:]:
                a, b = scores.get((design, m)), scores.get((design, "resid_quantiles"))
                if a is None or b is None:
                    continue
                for si, sname in ((0, "crps_grid"), (1, "interval_score")):
                    r = paired_boot(a[si], b[si], groups)
                    if r:
                        prow.append(dict(k=k, design=design, method=m, vs="resid_quantiles",
                                         score=sname, **r))
    res, cal, paired = pd.DataFrame(rows), pd.DataFrame(cal_rows), pd.DataFrame(prow)
    tables = {"dist_scores": res, "dist_calibration": cal, "dist_paired": paired}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", f"DISTRIBUTIONAL FORECASTS (quantile grid {[round(float(t), 2) for t in taus]}; lower is better):"]
    if len(res):
        g = res[res.design == "grouped"].pivot_table(index="k", columns="method", values="crps_grid")
        L += ["  grid CRPS, account-grouped CV:", g.round(3).to_string(),
              "  coverage of the central interval, account-grouped CV (target "
              f"{taus[-1] - taus[0]:.0%}):",
              res[res.design == "grouped"].pivot_table(index="k", columns="method",
                                                       values="coverage_central").round(3).to_string()]
    if len(paired):
        p = paired[(paired.design == "grouped") & (paired.score == "crps_grid")]
        for m in sorted(p.method.unique()):
            L.append(f"  CRPS skill of {m} vs resid_quantiles (grouped):")
            for _, r in p[p.method == m].sort_values("k").iterrows():
                L.append(f"    k={int(r.k):>2}: {r.skill:+.1%} [{r.skill_lo:+.1%}, {r.skill_hi:+.1%}] p={r.p_boot:.3f}")
    return dict(tables=tables, lines=L, payload=dict(res=res, cal=cal, paired=paired))
