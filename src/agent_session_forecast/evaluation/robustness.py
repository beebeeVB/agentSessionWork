"""Is the headline result driven by a few big accounts? (``--robustness``)

The grouped analyses rest on few *effective* accounts (1/HHI is far smaller than the account
count when a handful of accounts produce most sessions). This module re-runs the two headline
quantities after removing or capping the heaviest accounts:

* the new-account gap     MAE(account-grouped) - MAE(random)          for the GBM;
* the GBM's skill over the per-harness median, under both designs.

Scenarios (each re-runs the whole cross-validation on the reduced data):

  all         every session (the reference)
  drop_top1   without the single heaviest account
  drop_top3   without the 3 heaviest accounts
  drop_top10  without the 10 heaviest accounts
  cap50       every account keeps at most 50 sessions (random but fixed subsample)

If the claims are robust, the gap stays positive and the skill stays positive across scenarios;
``robustness_stability`` states the largest change from the reference.
"""
import os

import numpy as np
import pandas as pd

from .. import config as cfg
from .bootstrap import paired_boot
from .compare import evaluate_predictors

CAP = 50


def scenarios(df):
    """dict name -> reduced DataFrame."""
    top = df.groupby("user_id").size().sort_values(ascending=False).index.tolist()
    out = {"all": df}
    for n in (1, 3, 10):
        if len(top) > n + 5:
            out[f"drop_top{n}"] = df[~df.user_id.isin(top[:n])].reset_index(drop=True)
    rng = np.random.default_rng(0)
    idx = []
    for _, ix in df.groupby("user_id").indices.items():
        idx.extend(rng.choice(ix, size=CAP, replace=False) if len(ix) > CAP else ix)
    capped = df.iloc[np.sort(np.array(idx))].reset_index(drop=True)
    if len(capped) < len(df):
        out[f"cap{CAP}"] = capped
    return out


def _eff(df):
    share = df.groupby("user_id").size() / len(df)
    return float(1.0 / (share ** 2).sum())


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for name, d in scenarios(df).items():
        print(f"    scenario {name}: {len(d)} sessions, {d.user_id.nunique()} accounts")
        res, _, store = evaluate_predictors(d, {}, designs=("random", "grouped"), verbose=False)
        for k, s in store.items():
            E, g = s["err"], s["groups"]
            if not all(key in E for key in (("grouped", "gbm"), ("random", "gbm"),
                                            ("grouped", "baseline_harness"), ("random", "baseline_harness"))):
                continue
            gap = paired_boot(E[("grouped", "gbm")], E[("random", "gbm")], g)
            sk_g = paired_boot(E[("grouped", "gbm")], E[("grouped", "baseline_harness")], g)
            sk_r = paired_boot(E[("random", "gbm")], E[("random", "baseline_harness")], g)
            if not (gap and sk_g and sk_r):
                continue
            rows.append(dict(scenario=name, k=k, n_sessions=gap["n"], accounts=gap["accounts"],
                             effective_accounts=_eff(d),
                             gap=gap["diff"], gap_lo=gap["diff_lo"], gap_hi=gap["diff_hi"],
                             skill_grouped=sk_g["skill"], skill_grouped_lo=sk_g["skill_lo"],
                             skill_grouped_hi=sk_g["skill_hi"],
                             skill_random=sk_r["skill"], skill_random_lo=sk_r["skill_lo"],
                             skill_random_hi=sk_r["skill_hi"]))
    tab = pd.DataFrame(rows)
    L = ["", "ROBUSTNESS TO HEAVY ACCOUNTS (drop / cap the biggest accounts, re-run the whole CV):",
         "  gap = MAE(grouped) - MAE(random) for the GBM; skill = 1 - MAE(gbm)/MAE(per-harness median)"]
    stab = pd.DataFrame()
    if len(tab):
        tab.to_csv(os.path.join(outdir, "robustness.csv"), index=False)
        for k in sorted(tab.k.unique()):
            L.append(f"  k={int(k)}:")
            for _, r in tab[tab.k == k].iterrows():
                L.append(f"    {r.scenario:<11} acc={int(r.accounts):>3} eff={r.effective_accounts:>5.1f}  "
                         f"gap {r.gap:+.3f} [{r.gap_lo:+.3f}, {r.gap_hi:+.3f}] | "
                         f"skill new-account {r.skill_grouped:+.1%} [{r.skill_grouped_lo:+.1%}, {r.skill_grouped_hi:+.1%}]"
                         f", random {r.skill_random:+.1%}")
        ref = tab[tab.scenario == "all"].set_index("k")
        srows = []
        for sc in [s for s in tab.scenario.unique() if s != "all"]:
            t = tab[tab.scenario == sc].set_index("k")
            common = t.index.intersection(ref.index)
            if not len(common):
                continue
            srows.append(dict(scenario=sc,
                              max_abs_gap_change=float((t.loc[common, "gap"] - ref.loc[common, "gap"]).abs().max()),
                              max_abs_skill_change=float((t.loc[common, "skill_grouped"]
                                                          - ref.loc[common, "skill_grouped"]).abs().max()),
                              gap_still_positive=bool((t.loc[common, "gap_lo"] > 0).all()),
                              skill_still_positive=bool((t.loc[common, "skill_grouped_lo"] > 0).all()),
                              skill_sign_kept=bool((t.loc[common, "skill_grouped"] > 0).all())))
        stab = pd.DataFrame(srows)
        L.append("  STABILITY vs the full data:")
        for _, r in stab.iterrows():
            L.append(f"    {r.scenario:<11} max |change| gap {r.max_abs_gap_change:.3f}, new-account skill "
                     f"{r.max_abs_skill_change:.1%}; gap CI excludes 0 at every k: {'yes' if r.gap_still_positive else 'NO'}; "
                     f"skill CI excludes 0 at every k: {'yes' if r.skill_still_positive else 'NO'}")
        stab.to_csv(os.path.join(outdir, "robustness_stability.csv"), index=False)
    else:
        L.append("  (skipped: not enough sessions or accounts)")
    return dict(tables={"robustness": tab, "robustness_stability": stab}, lines=L,
                payload=dict(tab=tab, stab=stab))
