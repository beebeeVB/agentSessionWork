"""The two dashboard images.

dashboard.png  - profile of the data, forecast error, and interval coverage
dashboard2.png - is the signal real? paired tests, robustness, drift
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as cfg
from ..profiling import gini


def make_dashboard(df, per_user, res, conf, conf_ph, k_conf, out):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Agent session length: profile, early forecasting, uncertainty", fontsize=15)

    a = ax[0, 0]
    bins = np.logspace(0, np.log10(max(df.n_requests.max(), 10)), 40)
    a.hist(df.n_requests, bins=bins, color="#4C72B0", label="all sessions")
    if df.cens_right.any():
        a.hist(df[df.cens_right].n_requests, bins=bins, color="#C44E52", label="ends at week end")
    a.set_xscale("log"); a.set_xlabel("requests per session (log)"); a.set_ylabel("sessions")
    a.set_title("Session length (heavy-tailed)"); a.legend()

    a = ax[0, 1]
    top = df.harness.value_counts().index[:6]
    a.boxplot([np.log10(df[df.harness == h].n_requests) for h in top], showfliers=False)
    a.set_xticks(range(1, len(top) + 1)); a.set_xticklabels(list(top))
    a.set_ylabel("log10(requests per session)"); a.set_title("Length by harness")
    a.tick_params(axis="x", rotation=30)

    a = ax[0, 2]
    v = np.sort(per_user.values)
    cum = np.insert(np.cumsum(v) / v.sum(), 0, 0)
    a.plot(np.linspace(0, 1, len(cum)), cum, color="#55A868", lw=2)
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set_xlabel("share of accounts"); a.set_ylabel("share of sessions")
    a.set_title(f"Account concentration (Gini {gini(v):.2f})")

    a = ax[1, 0]
    if res is not None and len(res):
        style = {("random", "gbm"): ("#4C72B0", "-", "GBM, random"),
                 ("grouped", "gbm"): ("#C44E52", "-", "GBM, account-grouped"),
                 ("temporal", "gbm"): ("#DD8452", "-", "GBM, temporal"),
                 ("random", "baseline_harness"): ("#4C72B0", ":", "baseline, random"),
                 ("grouped", "baseline_harness"): ("#C44E52", ":", "baseline, grouped"),
                 ("temporal", "baseline_harness"): ("#DD8452", ":", "baseline, temporal")}
        for (d, mo), (c, ls, lab) in style.items():
            s = res[(res.design == d) & (res.model == mo)].sort_values("k")
            if len(s):
                a.plot(s.k, s.mae_log, color=c, ls=ls, marker="o", ms=4, label=lab)
                if mo == "gbm" and d == "grouped":
                    a.fill_between(s.k, s.ci_lo, s.ci_hi, color=c, alpha=.15)
        a.set_xlabel("requests observed (k)"); a.set_ylabel("MAE on log(1+requests)")
        a.set_title("Early forecasting error (band: cluster-bootstrap 95% CI)")
        a.legend(fontsize=7)

    a = ax[1, 1]
    if conf is not None:
        meths = [m for m in cfg.CONF_METHODS if m in set(conf.method)]
        g = conf.groupby("method")
        means = [g.coverage.mean()[m] for m in meths]
        lo = [np.percentile(conf[conf.method == m].coverage, 10) for m in meths]
        hi = [np.percentile(conf[conf.method == m].coverage, 90) for m in meths]
        cols = {"split": "#55A868", "mondrian": "#8172B3", "normalized": "#64B5CD",
                "cqr": "#CCB974", "random_calib": "#C44E52"}
        a.bar(meths, means, color=[cols[m] for m in meths],
              yerr=[np.array(means) - lo, np.array(hi) - means], capsize=5)
        for i, m in enumerate(meths):
            a.text(i, 0.04, f"x{g.width_factor.median()[m]:.1f}\nwide", ha="center", fontsize=8, color="white")
        a.axhline(1 - cfg.ALPHA, color="k", ls="--", label=f"target {1-cfg.ALPHA:.0%}")
        a.set_ylim(0, 1.2); a.set_ylabel("coverage on NEW accounts")
        a.set_title(f"Conformal methods, k={k_conf} (bars: 10-90% over splits)")
        a.tick_params(axis="x", labelsize=8); a.legend(loc="upper right")

    a = ax[1, 2]
    if conf_ph is not None:
        sub = conf_ph[conf_ph.method.isin(["split", "mondrian"])]
        pv = sub.groupby(["h", "method"]).apply(
            lambda t: np.average(t["mean"], weights=t["count"])).unstack("method")
        pv = pv.sort_values("split") if "split" in pv else pv
        yy = np.arange(len(pv))
        if "split" in pv:
            a.barh(yy - .2, pv["split"], height=.4, color="#55A868", label="split")
        if "mondrian" in pv:
            a.barh(yy + .2, pv["mondrian"], height=.4, color="#8172B3", label="mondrian")
        a.set_yticks(yy); a.set_yticklabels(pv.index)
        a.axvline(1 - cfg.ALPHA, color="k", ls="--")
        a.set_xlim(0, 1.3); a.set_xlabel("coverage"); a.legend(fontsize=8, loc="lower right")
        a.set_title("Coverage by harness: global vs Mondrian")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out, "dashboard.png")
    plt.savefig(path, dpi=130); plt.close()
    return path


def make_dashboard2(paired, long_res, imp, conf, aci, k_conf, out):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Is the signal real? Paired tests, robustness, drift", fontsize=15)

    a = ax[0, 0]
    if paired is not None and len(paired):
        for mo, c in [("gbm", "#4C72B0"), ("baseline_harness", "#999999")]:
            s = paired[(paired.comparison == "grouped_minus_random") & (paired.model == mo)].sort_values("k")
            if len(s):
                a.errorbar(s.k, s["diff"], yerr=[s["diff"] - s.diff_lo, s.diff_hi - s["diff"]],
                           marker="o", capsize=3, color=c, label=mo)
        a.axhline(0, color="k", lw=1)
        a.set_xlabel("k"); a.set_ylabel("MAE(grouped) - MAE(random)")
        a.set_title("Cost of holding out accounts (paired 95% CI)"); a.legend(fontsize=8)

    a = ax[0, 1]
    if paired is not None and len(paired):
        spec = [("gbm_vs_baseline_harness", "grouped", "#C44E52", "GBM vs harness-median (grouped)"),
                ("gbm_vs_baseline_harness", "random", "#4C72B0", "GBM vs harness-median (random)"),
                ("history_vs_gbm", "random", "#55A868", "+account history vs GBM (random)"),
                ("harness_vs_global", "grouped", "#8172B3", "harness-median vs global (grouped)")]
        for comp, des, c, lab in spec:
            s = paired[(paired.comparison == comp) & (paired.design == des)].sort_values("k")
            if len(s):
                a.plot(s.k, s.skill, marker="o", ms=4, color=c, label=lab)
                a.fill_between(s.k, s.skill_lo, s.skill_hi, color=c, alpha=.12)
        a.axhline(0, color="k", lw=1)
        a.set_xlabel("k"); a.set_ylabel("skill = 1 - MAE_model / MAE_reference")
        a.set_title("Skill vs reference (above 0 = better; paired 95% CI)"); a.legend(fontsize=7)

    a = ax[0, 2]
    if conf is not None:
        meths = [m for m in cfg.CONF_METHODS if m in set(conf.method)]
        g = conf.groupby("method")
        x = np.arange(len(meths))
        a.bar(x - .2, [g.worst_harness_cov.mean()[m] for m in meths], .4, color="#C44E52",
              label="worst harness")
        a.bar(x + .2, [g.acct_cov_p10.mean()[m] for m in meths], .4, color="#8172B3",
              label="10th pct account")
        a.axhline(1 - cfg.ALPHA, color="k", ls="--")
        a.set_xticks(x); a.set_xticklabels(meths, fontsize=8); a.set_ylim(0, 1.05)
        a.set_ylabel("coverage"); a.set_title(f"Conditional coverage, k={k_conf}"); a.legend(fontsize=8)

    a = ax[1, 0]
    if long_res is not None and len(long_res):
        for (T, d), s in long_res.groupby(["T", "design"]):
            s = s.sort_values("k")
            a.errorbar(s.k, s.auc_gbm, yerr=[s.auc_gbm - s.auc_lo, s.auc_hi - s.auc_gbm],
                       marker="o", capsize=3, ls="-" if d == "grouped" else "--",
                       label=f"P(>{T}) {d}")
        a.axhline(0.5, color="k", lw=1)
        a.set_xlabel("k"); a.set_ylabel("AUC"); a.set_ylim(0.4, 1.0)
        a.set_title("Will the session be long? (AUC, account-bootstrap CI)"); a.legend(fontsize=7)

    a = ax[1, 1]
    if imp is not None:
        t = imp.head(10)[::-1]
        a.barh(t.index, t.values, color="#4C72B0")
        a.axvline(0, color="k", lw=1)
        a.set_xlabel("MAE increase when shuffled (held-out accounts)")
        a.set_title("Which features carry signal on new accounts")

    a = ax[1, 2]
    if aci is not None:
        w = 200
        for key, lab, c in [("err_static", f"static (cov {aci['cov_static']:.1%})", "#C44E52"),
                            ("err_aci", f"ACI (cov {aci['cov_aci']:.1%})", "#55A868")]:
            cov = 1 - pd.Series(aci[key]).rolling(w, min_periods=w // 2).mean()
            a.plot(cov.values, color=c, label=lab)
        a.axhline(1 - cfg.ALPHA, color="k", ls="--")
        a.set_ylim(0.5, 1.02); a.set_xlabel("stream position (time order)")
        a.set_ylabel(f"rolling coverage ({w})"); a.set_title("Drift: static vs adaptive intervals")
        a.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out, "dashboard2.png")
    plt.savefig(path, dpi=130); plt.close()
    return path
