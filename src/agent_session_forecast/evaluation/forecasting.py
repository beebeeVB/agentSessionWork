"""Main experiment: predict log(1 + total requests) from the first k requests.

Runs every model under three designs (random folds, account-grouped folds,
temporal split) and then compares them with paired bootstraps.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .. import config as cfg
from ..features import features_at_k
from ..models import TemporalSplit, oof_predictions
from .bootstrap import cluster_boot_mae, paired_boot, r2


def evaluate_forecasts(df):
    rows, store = [], {}
    for k in cfg.K_LIST:
        out = features_at_k(df, k)
        if out is None:
            continue
        y, groups = out["y"], out["groups"]
        n_groups = len(set(groups))
        if len(y) < 100 or n_groups < 6:
            print(f"  k={k}: skipped (n={len(y)}, accounts={n_groups})")
            continue
        print(f"  k={k}: n={len(y)} sessions, {n_groups} accounts")
        designs = {
            "random": (KFold(5, shuffle=True, random_state=0), None),
            "grouped": (GroupKFold(5), groups),
            "temporal": (TemporalSplit(out["start"]), None),
        }
        store[k] = dict(groups=groups, err={})
        for dname, (splitter, sg) in designs.items():
            P = oof_predictions(out, splitter, sg)
            for mname, p in P.items():
                cov = np.isfinite(p)
                if cov.sum() < 50:
                    continue
                e_full = np.full(len(y), np.nan)
                e_full[cov] = np.abs(y[cov] - p[cov])
                store[k]["err"][(dname, mname)] = e_full
                e = e_full[cov]
                mae, lo, hi = cluster_boot_mae(e, groups[cov])
                macro = pd.Series(e).groupby(groups[cov]).mean().mean()
                med_req = float(np.median(np.abs(np.expm1(y[cov]) - np.expm1(p[cov]))))
                rows.append(dict(k=k, design=dname, model=mname, n=int(cov.sum()),
                                 accounts=len(set(groups[cov])), mae_log=mae, ci_lo=lo, ci_hi=hi,
                                 macro_mae_log=macro, medae_requests=med_req,
                                 r2=r2(y[cov], p[cov])))
    return pd.DataFrame(rows), store


def paired_analysis(store):
    rows = []
    comps = [("gbm", "baseline_harness", "gbm_vs_baseline_harness"),
             ("baseline_harness", "baseline_global", "harness_vs_global"),
             ("gbm_hist", "gbm", "history_vs_gbm")]
    for k, s in store.items():
        E, g = s["err"], s["groups"]
        for model in cfg.MODELS:
            if ("grouped", model) in E and ("random", model) in E:
                r = paired_boot(E[("grouped", model)], E[("random", model)], g)
                if r:
                    rows.append(dict(k=k, comparison="grouped_minus_random", design="-", model=model, **r))
        for design in cfg.DESIGNS:
            for a, b, name in comps:
                if (design, a) in E and (design, b) in E:
                    r = paired_boot(E[(design, a)], E[(design, b)], g)
                    if r:
                        rows.append(dict(k=k, comparison=name, design=design, model=a, **r))
    return pd.DataFrame(rows)
