"""One shared evaluation harness for every new point forecaster.

A *predictor* is a function ``fn(out, tr, te, ctx) -> predictions for rows te`` where

* ``out``  is the dict from ``features_at_k`` (features X, target y, groups, start, ``d`` = the
           sessions' raw arrays),
* ``tr, te`` are integer index arrays (the training rows and the rows to predict),
* ``ctx``  is ``dict(design=..., k=..., fold=...)``.

``evaluate_predictors`` runs every predictor under the three designs (random / account-grouped /
temporal) on EXACTLY the same folds as the references (``gbm`` and ``baseline_harness``), so all
comparisons are paired, and puts cluster-bootstrap CIs on everything.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .. import config as cfg
from ..features import add_history, features_at_k
from ..models import TemporalSplit, baseline_harness_predict, new_gbm
from .bootstrap import cluster_boot_mae, paired_boot

REFS = ("gbm", "baseline_harness")


def iter_folds(out, design):
    """Yield (fold_id, train_idx, test_idx) for one design. Same folds everywhere."""
    y, groups = out["y"], out["groups"]
    if design == "random":
        sp, sg = KFold(5, shuffle=True, random_state=0), None
    elif design == "grouped":
        sp, sg = GroupKFold(5), groups
    elif design == "temporal":
        sp, sg = TemporalSplit(out["start"]), None
    else:
        raise ValueError(design)
    for f, (tr, te) in enumerate(sp.split(out["X"], y, sg)):
        yield f, tr, te


# ---- reference predictors ------------------------------------------------------------
def ref_gbm(out, tr, te, ctx):
    X, y = out["X"], out["y"]
    return new_gbm().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])


def ref_baseline_harness(out, tr, te, ctx):
    X, y = out["X"], out["y"]
    return baseline_harness_predict(X.iloc[tr], y[tr], X.iloc[te])


def pred_gbm_hist(out, tr, te, ctx):
    """The original account-history model (for comparison with the hierarchical model)."""
    X, y, g = out["X"], out["y"], out["groups"]
    Htr, Hte = add_history(X.iloc[tr], y[tr], g[tr], X.iloc[te], g[te])
    return new_gbm().fit(Htr, y[tr]).predict(Hte)


_REF_FNS = {"gbm": ref_gbm, "baseline_harness": ref_baseline_harness}


def evaluate_predictors(df, predictors, ks=None, designs=None, min_n=100, min_groups=6, verbose=True):
    """Score ``predictors`` (dict name -> fn) plus the two references.

    Returns (results, paired, store):
      results : k, design, model, n, accounts, mae_log, ci_lo, ci_hi
      paired  : k, design, model, vs, diff/skill (+CI, p)   [model vs each reference; negative
                diff / positive skill = the model is better]
      store   : {k: {"groups": ..., "err": {(design, model): per-session |error|}}}
    """
    ks = ks or cfg.EXTRA_KS
    designs = designs or cfg.DESIGNS
    allp = dict(_REF_FNS)
    allp.update(predictors)
    rows, prow, store = [], [], {}
    failed = set()
    for k in ks:
        out = features_at_k(df, k)
        if out is None:
            continue
        y, groups = out["y"], out["groups"]
        if len(y) < min_n or len(set(groups)) < min_groups:
            if verbose:
                print(f"    k={k}: skipped (n={len(y)}, accounts={len(set(groups))})")
            continue
        if verbose:
            print(f"    k={k}: n={len(y)} sessions, {len(set(groups))} accounts, "
                  f"models: {', '.join(allp)}")
        store[k] = dict(groups=groups, err={})
        for design in designs:
            P = {m: np.full(len(y), np.nan) for m in allp}
            for f, tr, te in iter_folds(out, design):
                ctx = dict(design=design, k=k, fold=f)
                for m, fn in allp.items():
                    if m in failed:
                        continue
                    try:
                        P[m][te] = fn(out, tr, te, ctx)
                    except Exception as e:     # one broken predictor must not lose the others
                        failed.add(m)
                        print(f"    [warn] predictor '{m}' failed ({type(e).__name__}: {e}); skipped")
            for m, p in P.items():
                cov = np.isfinite(p)
                if cov.sum() < 50:
                    continue
                e_full = np.full(len(y), np.nan)
                e_full[cov] = np.abs(y[cov] - p[cov])
                store[k]["err"][(design, m)] = e_full
                mae, lo, hi = cluster_boot_mae(e_full[cov], groups[cov])
                rows.append(dict(k=k, design=design, model=m, n=int(cov.sum()),
                                 accounts=len(set(groups[cov])), mae_log=mae, ci_lo=lo, ci_hi=hi))
        E = store[k]["err"]
        for design in designs:
            for m in predictors:
                for ref in REFS:
                    if (design, m) in E and (design, ref) in E:
                        r = paired_boot(E[(design, m)], E[(design, ref)], groups)
                        if r:
                            prow.append(dict(k=k, design=design, model=m, vs=ref, **r))
    return pd.DataFrame(rows), pd.DataFrame(prow), store


def fmt_paired(paired, designs=("grouped",), refs=("baseline_harness", "gbm")):
    """Plain-text lines: skill of each model over each reference (95% CI, p)."""
    L = []
    if paired is None or not len(paired):
        return ["  (no paired results)"]
    for design in designs:
        for ref in refs:
            s = paired[(paired.design == design) & (paired.vs == ref)]
            for m in sorted(s.model.unique()):
                L.append(f"  [{design}] {m} vs {ref}:")
                for _, r in s[s.model == m].sort_values("k").iterrows():
                    flag = "" if r.skill_lo <= 0 <= r.skill_hi else ("  <- better" if r.skill > 0 else "  <- worse")
                    L.append(f"    k={int(r.k):>2}: skill {r.skill:+.1%} [{r.skill_lo:+.1%}, {r.skill_hi:+.1%}] "
                             f"p={r.p_boot:.3f}{flag}")
    return L
