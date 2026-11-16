"""Do domain-shift methods shrink the cost of meeting a NEW account? (``--shift``)

The repo already measures the gap  MAE(account-grouped) - MAE(random)  for the plain GBM. This
module trains the same GBM in ways designed to generalise to unseen groups and measures whether
the gap shrinks, with a paired account-level bootstrap:

* ``gbm_balanced``  importance weighting: every ACCOUNT gets equal total weight (1 / n_g per
                    session), i.e. train for "a random new account", not "a random session";
* ``gbm_sqrt``      the milder version, weight 1 / sqrt(n_g);
* ``gbm_dro``       group distributionally robust optimisation (Sagawa et al. 2020): groups =
                    accounts with >= 15 training sessions (the rest pooled); a few rounds of
                    exponentiated-gradient reweighting towards the groups with the highest
                    held-out loss, losses measured out-of-fold so they are not memorised;
* ``gbm_noidentity`` drops the quarter of features that most strongly separate accounts (ANOVA
                    F-statistic across training accounts): an approximation to invariance, not
                    adversarial training.

The headline numbers are in ``shift_gap.csv``: for each method, the gap (grouped - random) and how
it differs from the plain GBM's gap (negative = the method closes part of the gap).

CAUTION: a smaller gap is NOT the same as a better model. The gap shrinks if the method makes
RANDOM-split accuracy worse, even when it does nothing for new accounts. So ``shift_gap.csv`` also
carries ``newacct_skill_vs_gbm`` (paired, account-grouped design: 1 - MAE(method)/MAE(gbm); positive
= better on a new account). That column is the one that says whether a method actually helps.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .. import config as cfg
from ..models import new_gbm
from .bootstrap import paired_boot
from .compare import evaluate_predictors


def _fit_w(X, y, w, light=False):
    """GBM fit with sample weights.

    cfg.WEIGHT_MODE == "exact"   : sklearn ``sample_weight`` (exact, but slow for absolute-error loss);
    cfg.WEIGHT_MODE == "resample": weighted bootstrap (rows drawn with probability ~ weight; fast,
                                   approximate; used by --fast).
    ``light`` = fewer trees, for the inner DRO loop only.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from ..models import _GBM_KW
    kw = dict(_GBM_KW)
    if light:
        kw["max_iter"] = 50
    m = HistGradientBoostingRegressor(**kw)
    if cfg.WEIGHT_MODE == "resample":
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=len(y), replace=True, p=np.asarray(w, float) / np.sum(w))
        return m.fit(X.iloc[idx], np.asarray(y)[idx])
    return m.fit(X, y, sample_weight=w)


def pred_balanced(power):
    def fn(out, tr, te, ctx):
        X, y, g = out["X"], out["y"], out["groups"]
        n_g = pd.Series(g[tr]).map(pd.Series(g[tr]).value_counts()).values.astype(float)
        w = 1.0 / n_g ** power
        return _fit_w(X.iloc[tr], y[tr], w / w.mean()).predict(X.iloc[te])
    return fn


def pred_dro(out, tr, te, ctx, min_n=15, eta=1.0):
    X, y, g = out["X"], out["y"], out["groups"]
    gt = pd.Series(g[tr])
    big = gt.map(gt.value_counts()) >= min_n
    grp = np.where(big, gt, "__rare__")
    labels = np.unique(grp)
    if len(labels) < 3:
        return new_gbm().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    q = np.full(len(labels), 1.0 / len(labels))
    cnt = pd.Series(grp).value_counts().reindex(labels).values.astype(float)
    wavg = None
    for _ in range(cfg.DRO_ROUNDS):
        w = (q / cnt)[np.searchsorted(labels, grp)]
        oof = np.zeros(len(tr))
        for a, b in KFold(3, shuffle=True, random_state=0).split(tr):
            oof[b] = _fit_w(X.iloc[tr[a]], y[tr[a]], w[a] / w[a].mean(), light=True).predict(X.iloc[tr[b]])
        loss = pd.Series(np.abs(y[tr] - oof)).groupby(grp).mean().reindex(labels).values
        q = q * np.exp(eta * (loss - loss.mean()))
        q /= q.sum()
        wavg = q if wavg is None else wavg + q
    q = wavg / wavg.sum()
    w = (q / cnt)[np.searchsorted(labels, grp)]
    return _fit_w(X.iloc[tr], y[tr], w / w.mean()).predict(X.iloc[te])


def pred_noidentity(out, tr, te, ctx, drop_frac=0.25):
    X, y, g = out["X"], out["y"], out["groups"]
    Xt, gt = X.iloc[tr], g[tr]
    cols, F = [], []
    labels = pd.Series(gt)
    for c in X.columns:
        if c in ("harness", "model0"):
            continue
        v = pd.Series(Xt[c].to_numpy(float))
        grp = v.groupby(labels.values)
        n_i, m_i = grp.size(), grp.mean()
        k, N = len(n_i), len(v)
        if k < 2 or N <= k:
            continue
        ssb = (n_i * (m_i - v.mean()) ** 2).sum()
        ssw = ((v.values - labels.map(m_i).values) ** 2).sum()
        cols.append(c); F.append((ssb / (k - 1)) / max(ssw / (N - k), 1e-12))
    drop = set(np.array(cols)[np.argsort(F)[::-1][:int(np.ceil(drop_frac * len(cols)))]]) if cols else set()
    keep = [c for c in X.columns if c not in drop]
    cat = [i for i, c in enumerate(keep) if c in ("harness", "model0")]
    from sklearn.ensemble import HistGradientBoostingRegressor
    from ..models import _GBM_KW
    kw = dict(_GBM_KW); kw["categorical_features"] = cat
    return HistGradientBoostingRegressor(**kw).fit(Xt[keep], y[tr]).predict(X.iloc[te][keep])


def gap_table(store):
    """Paired gap (grouped - random) per method, and the difference from the plain GBM's gap."""
    rows = []
    for k, s in store.items():
        E, g = s["err"], s["groups"]
        models = sorted({m for (_, m) in E})
        gap = {m: E[("grouped", m)] - E[("random", m)] for m in models
               if ("grouped", m) in E and ("random", m) in E}
        for m, v in gap.items():
            ok = np.isfinite(v)
            r_self = paired_boot(E[("grouped", m)], E[("random", m)], g)
            rec = dict(k=k, method=m, n=int(ok.sum()),
                       mae_grouped=float(np.nanmean(E[("grouped", m)])), mae_random=float(np.nanmean(E[("random", m)])),
                       gap=r_self["diff"] if r_self else np.nan,
                       gap_lo=r_self["diff_lo"] if r_self else np.nan, gap_hi=r_self["diff_hi"] if r_self else np.nan)
            if m != "gbm" and "gbm" in gap:
                r = paired_boot(v, gap["gbm"], g)
                if r:
                    rec.update(gap_change_vs_gbm=r["diff"], gap_change_lo=r["diff_lo"],
                               gap_change_hi=r["diff_hi"], gap_change_p=r["p_boot"])
                if ("grouped", "gbm") in E:
                    r2 = paired_boot(E[("grouped", m)], E[("grouped", "gbm")], g)
                    if r2:
                        rec.update(newacct_skill_vs_gbm=r2["skill"], newacct_skill_lo=r2["skill_lo"],
                                   newacct_skill_hi=r2["skill_hi"], newacct_skill_p=r2["p_boot"])
            rows.append(rec)
    return pd.DataFrame(rows)


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    preds = {"gbm_balanced": pred_balanced(1.0), "gbm_sqrt": pred_balanced(0.5),
             "gbm_dro": pred_dro, "gbm_noidentity": pred_noidentity}
    res, paired, store = evaluate_predictors(df, preds, designs=("random", "grouped"))
    gap = gap_table(store)
    tables = {"shift_forecast": res, "shift_paired": paired, "shift_gap": gap}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", "DOMAIN-SHIFT METHODS (gap = MAE grouped - MAE random; gap change < 0 = a smaller gap).",
         "  A smaller gap is not a better model: the decisive column is 'new-account skill vs gbm'",
         "  (account-grouped MAE, paired; + = better than the plain GBM on a NEW account):"]
    if len(gap):
        for _, r in gap.sort_values(["method", "k"]).iterrows():
            if "gap_change_vs_gbm" in r and pd.notna(r.get("gap_change_vs_gbm")):
                sk = ""
                if pd.notna(r.get("newacct_skill_vs_gbm")):
                    flag = ("" if r.newacct_skill_lo <= 0 <= r.newacct_skill_hi else
                            ("  <- better on new accounts" if r.newacct_skill_vs_gbm > 0 else "  <- WORSE on new accounts"))
                    sk = (f" | new-account skill vs gbm {r.newacct_skill_vs_gbm:+.1%} "
                          f"[{r.newacct_skill_lo:+.1%}, {r.newacct_skill_hi:+.1%}]{flag}")
                L.append(f"  k={int(r.k):>2} {r.method:<15} gap {r.gap:+.3f} (change {r.gap_change_vs_gbm:+.3f}, "
                         f"p={r.gap_change_p:.3f}){sk}")
            else:
                L.append(f"  k={int(r.k):>2} {r.method:<15} gap {r.gap:+.3f} [{r.gap_lo:+.3f}, {r.gap_hi:+.3f}]")
    return dict(tables=tables, lines=L, payload=dict(res=res, paired=paired, gap=gap))
