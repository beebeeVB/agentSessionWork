"""Hierarchical (mixed-effects) forecasting with partial pooling by account (``--hierarchical``).

Model for one landmark k:

    y_ij = mu_harness(i) + u_account(i) + e_ij ,   u ~ N(0, s_u^2),  e ~ N(0, s_e^2)

``y`` is log(1 + n_requests). Variance components come from one-way ANOVA moments on the
residuals from the harness means (deterministic, no optimiser, works on any machine); if
``statsmodels`` is installed, a REML ``MixedLM`` is also fitted on the full data as a cross-check
and both sets of variance components are reported.

The empirical-Bayes prediction for a *seen* account shrinks its mean residual toward zero by

    B_g = n_g s_u^2 / (n_g s_u^2 + s_e^2)             (more sessions -> less shrinkage)

and a *brand-new* account gets the population prediction (u = 0). That is exactly the behaviour
the ad hoc "account history" feature in ``gbm_hist`` tries to imitate; here it is the model.

Two predictors are scored under the three designs next to ``gbm_hist``:

* ``hier_eb``  - harness mean + account random effect (a pure mixed model);
* ``gbm_re``   - the gradient-boosting prediction + a shrunken account offset estimated from
                 out-of-fold residuals (partial pooling on top of the strong model).

A random *slope* on k is not fitted: each landmark is its own model, so k is constant inside one.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm
from ..profiling import icc_oneway
from .compare import evaluate_predictors, fmt_paired, pred_gbm_hist


def fit_variance_components(r, groups):
    """ANOVA-moment estimates (s_u^2, s_e^2) for residuals r grouped by account."""
    s = pd.DataFrame({"r": r, "g": groups})
    gm = s.groupby("g")["r"].agg(["sum", "count", "mean"])
    n, N = gm["count"].values.astype(float), float(len(s))
    k = len(gm)
    if k < 2 or N <= k:
        return 0.0, float(np.var(r)) if len(r) else 1.0
    ssw = float(((s["r"] - s["g"].map(gm["mean"])) ** 2).sum())
    ssb = float((n * (gm["mean"].values - r.mean()) ** 2).sum())
    msw, msb = ssw / (N - k), ssb / (k - 1)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return max((msb - msw) / n0, 0.0), max(msw, 1e-9)


def account_effects(r, groups, su2, se2):
    """Empirical-Bayes account effects: dict account -> (u_hat, shrinkage B, posterior sd, n)."""
    s = pd.DataFrame({"r": r, "g": groups}).groupby("g")["r"].agg(["mean", "count"])
    n = s["count"].values.astype(float)
    B = n * su2 / (n * su2 + se2) if su2 > 0 else np.zeros(len(n))
    u = B * s["mean"].values
    sd = np.sqrt(su2 * se2 / np.maximum(n * su2 + se2, 1e-12))
    return {g: (u[i], B[i], sd[i], int(n[i])) for i, g in enumerate(s.index)}


def pred_hier_eb(out, tr, te, ctx):
    X, y, g = out["X"], out["y"], out["groups"]
    h_tr, h_te = X["harness"].values[tr], X["harness"].values[te]
    glob = float(y[tr].mean())
    mu = pd.Series(y[tr]).groupby(h_tr).mean()
    r = y[tr] - np.array([mu.get(h, glob) for h in h_tr])
    su2, se2 = fit_variance_components(r, g[tr])
    eff = account_effects(r, g[tr], su2, se2)
    return np.array([mu.get(h, glob) for h in h_te]) + np.array([eff.get(a, (0.0,))[0] for a in g[te]])


def pred_gbm_re(out, tr, te, ctx):
    """GBM + shrunken account offset from out-of-fold residuals of the training sessions."""
    X, y, g = out["X"], out["y"], out["groups"]
    oof = np.zeros(len(tr))
    for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
        oof[b] = new_gbm().fit(X.iloc[tr[a]], y[tr[a]]).predict(X.iloc[tr[b]])
    r = y[tr] - oof
    su2, se2 = fit_variance_components(r, g[tr])
    eff = account_effects(r, g[tr], su2, se2)
    base = new_gbm().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return base + np.array([eff.get(a, (0.0,))[0] for a in g[te]])


def variance_table(df, ks):
    rows, eff_rows = [], None
    for k in ks:
        out = features_at_k(df, k)
        if out is None or len(set(out["groups"])) < 6:
            continue
        X, y, g = out["X"], out["y"], out["groups"]
        glob = float(y.mean())
        mu = pd.Series(y).groupby(X["harness"].values).mean()
        r = y - np.array([mu.get(h, glob) for h in X["harness"].values])
        su2, se2 = fit_variance_components(r, g)
        rows.append(dict(k=k, method="anova_eb", sigma_u2=su2, sigma_e2=se2,
                         icc_model=su2 / (su2 + se2) if su2 + se2 > 0 else np.nan,
                         icc_raw_profile=icc_oneway(y, g), n=len(y), accounts=len(set(g))))
        try:                                            # optional cross-check with REML
            import statsmodels.formula.api as smf
            d = pd.DataFrame(dict(y=y, h=X["harness"].values.astype(str), g=g))
            m = smf.mixedlm("y ~ C(h)", d, groups=d["g"]).fit(reml=True)
            su, se = float(m.cov_re.iloc[0, 0]), float(m.scale)
            rows.append(dict(k=k, method="mixedlm_reml", sigma_u2=su, sigma_e2=se,
                             icc_model=su / (su + se), icc_raw_profile=icc_oneway(y, g),
                             n=len(y), accounts=len(set(g))))
        except Exception:
            pass
        if eff_rows is None and (k >= 3 or k == ks[-1]):
            eff = account_effects(r, g, su2, se2)
            raw = pd.Series(r).groupby(g).mean()
            eff_rows = pd.DataFrame([dict(k=k, account=a, n=v[3], raw_mean_resid=float(raw[a]),
                                          effect_eb=v[0], shrinkage=v[1], post_sd=v[2])
                                     for a, v in eff.items()]).sort_values("effect_eb")
    return pd.DataFrame(rows), (eff_rows if eff_rows is not None else pd.DataFrame())


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    preds = {"hier_eb": pred_hier_eb, "gbm_re": pred_gbm_re, "gbm_hist": pred_gbm_hist}
    res, paired, store = evaluate_predictors(df, preds)
    var, eff = variance_table(df, cfg.EXTRA_KS)
    tables = {"hier_forecast": res, "hier_paired": paired, "hier_variance_components": var,
              "hier_random_effects": eff}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", "HIERARCHICAL MODEL (harness fixed effect + account random intercept, partial pooling):"]
    if len(var):
        L += ["  variance components (log-length scale):",
              var[["k", "method", "sigma_u2", "sigma_e2", "icc_model", "icc_raw_profile"]].round(3)
              .to_string(index=False)]
    L += fmt_paired(paired, designs=("grouped", "random"), refs=("gbm",))
    return dict(tables=tables, lines=L, payload=dict(res=res, paired=paired, var=var, eff=eff))
