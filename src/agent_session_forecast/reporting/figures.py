"""One PNG (200 dpi) and one vector PDF per figure, plus ``dashboard3.png`` for the extra analyses.

Every figure is a function ``fn(ctx, axes) -> bool`` that draws into the axes it is given and
returns False (drawing nothing) if its inputs are missing, so a figure whose analysis did not run
is silently skipped. ``ctx`` is a dict with the pieces of the run (see ``pipeline.run``):
``df, per_user, res, paired, long_res, imp, conf, aci, k_conf, extras``.

Colours are the Okabe-Ito colour-blind-safe palette; lines also differ in marker/linestyle so the
figures survive greyscale printing.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as cfg

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F", "#000000"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
DESIGN_COL = {"random": PAL[0], "grouped": PAL[3], "temporal": PAL[2]}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 10.5, "legend.fontsize": 8})


def _P(ctx, name):
    """payload of an extra analysis, or None."""
    R = ctx.get("extras", {}).get(name)
    return R["payload"] if R else None


def _err(ax, x, d, lo, hi, **kw):
    ax.errorbar(x, d, yerr=[np.maximum(np.asarray(d) - np.asarray(lo), 0),
                            np.maximum(np.asarray(hi) - np.asarray(d), 0)], capsize=3, **kw)


# ---------------------------------------------------------------- 1-6: the base study
def fig_error_vs_k(ctx, axes):
    res = ctx.get("res")
    if res is None or not len(res):
        return False
    ax = axes[0]
    for des in cfg.DESIGNS:
        for mo, ls, a in (("gbm", "-", 0.18), ("baseline_harness", "--", 0.0)):
            s = res[(res.design == des) & (res.model == mo)].sort_values("k")
            if not len(s):
                continue
            ax.plot(s.k, s.mae_log, ls, marker="o", color=DESIGN_COL[des], label=f"{mo} ({des})", ms=4)
            if a:
                ax.fill_between(s.k, s.ci_lo, s.ci_hi, color=DESIGN_COL[des], alpha=a)
    ax.set_xlabel("requests seen so far (k)"); ax.set_ylabel("MAE of log(1 + requests)  (lower = better)")
    ax.set_title("Forecast error vs k under three evaluation designs\n(band = account-cluster bootstrap 95% CI)")
    ax.legend(ncol=2)
    return True


def fig_gap_vs_k(ctx, axes):
    p = ctx.get("paired")
    if p is None or not len(p):
        return False
    ax, drew = axes[0], False
    for i, mo in enumerate(("gbm", "baseline_harness")):
        s = p[(p.comparison == "grouped_minus_random") & (p.model == mo)].sort_values("k")
        if len(s):
            _err(ax, s.k, s["diff"], s.diff_lo, s.diff_hi, marker=MARK[i], color=PAL[i], label=mo)
            drew = True
    if not drew:
        return False
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("k"); ax.set_ylabel("MAE(account-grouped) - MAE(random)")
    ax.set_title("Cost of meeting a NEW account (paired 95% CI)\nabove 0 = random CV was too optimistic")
    ax.legend()
    return True


def fig_skill_vs_harness(ctx, axes):
    p = ctx.get("paired")
    if p is None or not len(p):
        return False
    ax = axes[0]
    s0 = p[p.comparison == "gbm_vs_baseline_harness"]
    if not len(s0):
        return False
    for i, des in enumerate(cfg.DESIGNS):
        s = s0[s0.design == des].sort_values("k")
        if len(s):
            _err(ax, s.k, s.skill, s.skill_lo, s.skill_hi, marker=MARK[i], color=DESIGN_COL[des], label=f"GBM ({des})")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("k"); ax.set_ylabel("skill vs per-harness median  (+ = better)")
    ax.set_title("Does the GBM beat the harness-median baseline? (paired 95% CI)")
    ax.legend()
    if len(axes) > 1:
        a2, i = axes[1], 0
        for name, E in ctx.get("extras", {}).items():
            pp = E["payload"].get("paired")
            if pp is None or not len(pp) or "vs" not in pp or "model" not in pp or "skill" not in pp:
                continue
            for m in sorted(pp[(pp.design == "grouped") & (pp.vs == "baseline_harness")].model.unique()):
                s = pp[(pp.design == "grouped") & (pp.vs == "baseline_harness") & (pp.model == m)].sort_values("k")
                _err(a2, s.k + 0.08 * i, s.skill, s.skill_lo, s.skill_hi, marker=MARK[i % 8], color=PAL[i % 8],
                     label=m, lw=1, ms=4)
                i += 1
        a2.axhline(0, color="k", lw=1)
        a2.set_xlabel("k"); a2.set_title("New models vs harness median (account-grouped)")
        if i:
            a2.legend(ncol=2)
    return True


def fig_concentration(ctx, axes):
    pu = ctx.get("per_user")
    if pu is None or not len(pu):
        return False
    v = np.sort(pu.values.astype(float))
    cum = np.r_[0, np.cumsum(v) / v.sum()]
    x = np.linspace(0, 1, len(cum))
    share = v / v.sum()
    gini = 1 - 2 * (getattr(np, "trapezoid", None) or np.trapz)(cum, x)
    eff = 1 / (share ** 2).sum()
    ax = axes[0]
    ax.plot(x, cum, color=PAL[0], lw=2, label="accounts")
    ax.plot([0, 1], [0, 1], color="k", ls=":", label="perfect equality")
    ax.set_xlabel("share of accounts (smallest first)"); ax.set_ylabel("share of sessions")
    ax.set_title(f"Who generates the sessions? Gini {gini:.2f}; effective number of accounts {eff:.1f} of {len(v)}")
    ax.legend()
    return True


def fig_conformal(ctx, axes):
    conf = ctx.get("conf")
    if conf is None or not len(conf):
        return False
    order = [m for m in cfg.CONF_METHODS + cfg.GROUP_CONF_METHODS if m in set(conf.method)]
    g = conf.groupby("method")
    mean, sd = g[["coverage", "width_factor", "worst_harness_cov", "acct_cov_p10"]].mean(), \
        g[["coverage", "width_factor"]].std().fillna(0)
    ax = axes[0]
    for i, m in enumerate(order):
        ax.errorbar(mean.loc[m, "width_factor"], mean.loc[m, "coverage"], xerr=sd.loc[m, "width_factor"],
                    yerr=sd.loc[m, "coverage"], marker=MARK[i % 8], color=PAL[i % 8], capsize=3, label=m)
    ax.axhline(1 - cfg.ALPHA, color="k", ls="--", lw=1)
    ax.set_xlabel("interval width factor (x multiplicative; smaller = sharper)")
    ax.set_ylabel("overall coverage on unseen accounts")
    ax.set_title(f"Coverage vs sharpness (target {1 - cfg.ALPHA:.0%}); bars = sd over account re-splits")
    ax.legend(ncol=2)
    if len(axes) > 1:
        a2 = axes[1]
        x, w = np.arange(len(order)), 0.38
        a2.bar(x - w / 2, mean.loc[order, "worst_harness_cov"], w, color=PAL[1], label="worst harness")
        a2.bar(x + w / 2, mean.loc[order, "acct_cov_p10"], w, color=PAL[2], label="10th-percentile account")
        a2.axhline(1 - cfg.ALPHA, color="k", ls="--", lw=1)
        a2.set_xticks(x); a2.set_xticklabels(order, rotation=30, ha="right")
        a2.set_ylim(max(0.4, float(mean.loc[order, ["worst_harness_cov", "acct_cov_p10"]].min().min()) - 0.1), 1.0)
        a2.set_ylabel("coverage"); a2.set_title("Coverage where it is hardest (subgroup / weak accounts)")
        a2.legend()
    return True


def fig_aci(ctx, axes):
    aci = ctx.get("aci")
    if not aci:
        return False
    ax = axes[0]
    w = max(50, len(aci["err_static"]) // 10)
    for i, (key, lab) in enumerate((("err_static", "static split conformal"), ("err_aci", "adaptive (ACI)"))):
        cov = 1 - pd.Series(aci[key]).rolling(w, min_periods=w // 2).mean()
        ax.plot(cov.values, color=PAL[3 if i == 0 else 2], ls="-" if i else "--", label=lab)
    ax.axhline(1 - cfg.ALPHA, color="k", ls=":", lw=1)
    ax.set_xlabel("streamed sessions (time order)"); ax.set_ylabel(f"rolling coverage (window {w})")
    ax.set_title("Do intervals keep their coverage as the week drifts?")
    ax.legend()
    return True


def fig_weekly_drift(ctx, axes):
    df = ctx.get("df")
    if df is None or "week" not in df or df.week.nunique() < 2:
        return False
    g = df.groupby("week")
    wk = sorted(df.week.unique())
    a1 = axes[0]
    a1.plot(range(len(wk)), g.n_requests.median().reindex(wk), marker="o", color=PAL[0], label="median")
    a1.plot(range(len(wk)), g.n_requests.quantile(0.9).reindex(wk), marker="s", color=PAL[3], label="90th percentile")
    a1.set_yscale("log"); a1.set_xticks(range(len(wk)))
    a1.set_xticklabels([w.split("_")[0][5:] for w in wk], rotation=60, ha="right")
    a1.set_ylabel("requests per session"); a1.set_title("Session length by week (week start, MM-DD)")
    a1.legend()
    if len(axes) > 1:
        a2 = axes[1]
        mix = pd.crosstab(df.week, df.harness.fillna("?"), normalize="index").reindex(wk)
        bottom = np.zeros(len(wk))
        for i, h in enumerate(mix.columns[:8]):
            a2.bar(range(len(wk)), mix[h].values, bottom=bottom, color=PAL[i % 8], label=h)
            bottom += mix[h].values
        a2.set_xticks(range(len(wk))); a2.set_xticklabels([w.split("_")[0][5:] for w in wk], rotation=60, ha="right")
        a2.set_ylabel("share of sessions"); a2.set_title("Harness mix by week")
        a2.set_ylim(0, 1.3); a2.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a2.legend(ncol=3, fontsize=7, loc="upper center")
    return True


# ---------------------------------------------------------------- 7-13: the extra analyses
def fig_hazard(ctx, axes):
    P = _P(ctx, "hazard")
    if not P or P["curve"] is None or not len(P["curve"]):
        return False
    c = P["curve"]
    k = c.k.max()
    ax = axes[0]
    for i, m in enumerate(sorted(c.model.unique())):
        s = c[(c.k == k) & (c.model == m)].sort_values("j")
        ax.plot(s.j, s.pred, marker=MARK[i], color=PAL[i], label=f"{m} predicted")
    s = c[(c.k == k) & (c.model == c.model.iloc[0])].sort_values("j")
    ax.plot(s.j, s.obs, "k--", marker="x", label="observed")
    ax.set_xlabel("j: requests beyond k"); ax.set_ylabel("P(session lasts more than j more requests)")
    ax.set_title(f"Survival-curve calibration, k={int(k)} (account-grouped CV)")
    ax.legend()
    if len(axes) > 1:
        sc = P["scat"]
        a2 = axes[1]
        if sc is not None and len(sc):
            s = sc[sc.k == k]
            g = s.groupby("pred_remaining").true_remaining
            q = pd.DataFrame({"med": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75), "n": g.size()})
            q = q[q.n >= 5]
            _err(a2, q.index, q.med, q.lo, q.hi, marker="o", color=PAL[0])
            a2.plot([1, cfg.HAZARD_J], [1, cfg.HAZARD_J], "k:", label="y = x")
            a2.set_yscale("log"); a2.set_xlabel("predicted median remaining requests")
            a2.set_ylabel("actual remaining requests (median, IQR)")
            a2.set_title(f"Predicted vs actual remaining length, k={int(k)}")
            a2.legend()
    return True


def fig_hierarchical(ctx, axes):
    P = _P(ctx, "hierarchical")
    if not P or P["eff"] is None or not len(P["eff"]):
        return False
    e = P["eff"].sort_values("effect_eb").reset_index(drop=True)
    ax = axes[0]
    ax.scatter(e.index, e.raw_mean_resid, s=10, color=PAL[6], alpha=0.5, label="raw account mean residual")
    ax.errorbar(e.index, e.effect_eb, yerr=1.96 * e.post_sd, fmt="o", ms=3, color=PAL[0], lw=0.8,
                label="shrunken effect (95% interval)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("accounts, sorted by estimated effect"); ax.set_ylabel("effect on log(1 + requests)")
    ax.set_title(f"Account random effects at k={int(e.k.iloc[0])}: shrinkage pulls small accounts to 0")
    ax.legend()
    if len(axes) > 1 and P["var"] is not None and len(P["var"]):
        v = P["var"]
        a2 = axes[1]
        ks = sorted(v.k.unique())
        x = np.arange(len(ks))
        a2.bar(x - 0.2, [v[(v.k == k) & (v.method == "anova_eb")].icc_model.mean() for k in ks], 0.4,
               color=PAL[0], label="ICC from the model (harness-adjusted)")
        a2.bar(x + 0.2, [v[v.k == k].icc_raw_profile.mean() for k in ks], 0.4, color=PAL[1],
               label="raw ICC (profiling)")
        a2.set_xticks(x); a2.set_xticklabels([f"k={k}" for k in ks])
        a2.set_ylabel("share of variance due to the account"); a2.set_title("Variance components: how much is 'who'?")
        a2.legend()
    return True


def fig_distributional(ctx, axes):
    P = _P(ctx, "distributional")
    if not P or P["cal"] is None or not len(P["cal"]):
        return False
    cal, res = P["cal"], P["res"]
    k = cal.k.max()
    ax = axes[0]
    for i, m in enumerate(sorted(cal.method.unique())):
        s = cal[(cal.k == k) & (cal.design == "grouped") & (cal.method == m)].sort_values("level")
        ax.plot(s.level, s.observed, marker=MARK[i], color=PAL[i], label=m)
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("share of outcomes below the predicted quantile")
    ax.set_title(f"Quantile calibration on unseen accounts, k={int(k)}")
    ax.legend()
    if len(axes) > 1:
        a2 = axes[1]
        g = res[res.design == "grouped"].pivot_table(index="method", columns="k", values="crps_grid")
        x, ks = np.arange(len(g)), list(g.columns)
        for i, kk in enumerate(ks):
            a2.bar(x + (i - (len(ks) - 1) / 2) * 0.8 / len(ks), g[kk], 0.8 / len(ks), color=PAL[i], label=f"k={kk}")
        a2.set_xticks(x); a2.set_xticklabels(g.index, rotation=20, ha="right")
        a2.set_ylabel("grid CRPS (lower = better)"); a2.set_title("Distributional accuracy, account-grouped CV")
        a2.legend()
    return True


def fig_shift(ctx, axes):
    P = _P(ctx, "shift")
    if not P or P["gap"] is None or "gap_change_vs_gbm" not in P["gap"]:
        return False
    g = P["gap"].dropna(subset=["gap_change_vs_gbm"])
    g = g[~g.method.isin(["baseline_harness"])]
    if not len(g):
        return False
    ax = axes[0]
    ms, ks = sorted(g.method.unique()), sorted(g.k.unique())
    for i, k in enumerate(ks):
        s = g[g.k == k].set_index("method").reindex(ms)
        y = np.arange(len(ms)) + (i - (len(ks) - 1) / 2) * 0.3
        ax.errorbar(s.gap_change_vs_gbm, y, xerr=[s.gap_change_vs_gbm - s.gap_change_lo,
                                                  s.gap_change_hi - s.gap_change_vs_gbm],
                    fmt=MARK[i], color=PAL[i], capsize=3, label=f"k={k}")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms)
    ax.set_xlabel("change in the new-account gap vs plain GBM (log-MAE; < 0 = gap closed)")
    ax.set_title("Do domain-shift methods help on unseen accounts? (paired 95% CI)")
    ax.legend()
    return True


def fig_process(ctx, axes):
    P = _P(ctx, "process")
    if not P or P["P"] is None:
        return False
    M = P["P"].fillna(0)
    ax = axes[0]
    ax.grid(False)
    ax.imshow(M.values, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(M.values))))
    ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M.values[i, j] > 0.3 else "black")
    ax.set_xlabel("next request's state"); ax.set_ylabel("current request's state")
    ax.set_title("Transition probabilities between request states")
    if len(axes) > 1:
        t = P["tab"]
        a2 = axes[1]
        a2.bar(range(len(t)), t.p_end_given_state, color=PAL[3])
        a2.set_xticks(range(len(t))); a2.set_xticklabels(t.state, rotation=45, ha="right")
        a2.set_ylabel("P(session ends right after this request)")
        a2.set_title("Which states precede the end of a session?")
    return True


def fig_decision(ctx, axes):
    P = _P(ctx, "decision")
    if not P or P["res"] is None or not len(P["res"]):
        return False
    r = P["res"]
    r = r[r.policy != "fixed_ttl"]
    ax = axes[0]
    pols, ks = list(dict.fromkeys(r.policy)), sorted(r.k.unique())
    for i, k in enumerate(ks):
        s = r[r.k == k].set_index("policy").reindex(pols)
        x = np.arange(len(pols)) + (i - (len(ks) - 1) / 2) * 0.8 / len(ks)
        ax.bar(x, s.saving_vs_fixed, 0.8 / len(ks), color=PAL[i], label=f"k={k}",
               yerr=[np.maximum(s.saving_vs_fixed - s.saving_lo, 0), np.maximum(s.saving_hi - s.saving_vs_fixed, 0)],
               capsize=2)
    ax.axhline(0, color="k", lw=1)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=15, ha="right")
    ax.set_ylabel("saving vs best fixed TTL  (+ = cheaper)")
    ax.set_title(f"KV-cache retention cost by policy (HOLD_RATE={cfg.HOLD_RATE}); oracle = lower bound")
    ax.legend()
    return True


def fig_mixture(ctx, axes):
    P = _P(ctx, "mixture")
    if not P or P["prof"] is None or not len(P["prof"]):
        return False
    p = P["prof"]
    ax = axes[0]
    x = np.arange(len(p))
    ax.bar(x - 0.2, p.median_n_requests, 0.4, color=PAL[0], label="median requests")
    ax.bar(x + 0.2, p.p90_n_requests, 0.4, color=PAL[1], label="90th percentile")
    for i, sh in enumerate(p.share):
        ax.text(i, 0, f"{sh:.0%}\nof sessions", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"class {c}" for c in p.cls])
    ax.set_ylabel("requests per session"); ax.set_title(f"Latent session classes at k={int(p.k.iloc[0])}")
    ax.legend()
    if len(axes) > 1:
        a2, bottom = axes[1], np.zeros(len(p))
        hc = [c for c in p.columns if c.startswith("harness_")]
        for i, c in enumerate(hc):
            v = p[c].fillna(0).values
            a2.bar(x, v, bottom=bottom, color=PAL[i % 8], label=c.replace("harness_", ""))
            bottom += v
        a2.set_xticks(x); a2.set_xticklabels([f"class {c}" for c in p.cls])
        a2.set_ylabel("share of the class"); a2.set_title("Harness mix of each class"); a2.legend(fontsize=7)
    return True


def fig_extras_vs_gbm(ctx, axes):
    rows = []
    for name, E in ctx.get("extras", {}).items():
        pp = E["payload"].get("paired")
        if pp is not None and len(pp) and "vs" in pp and "skill" in pp and "model" in pp:
            rows.append(pp[pp.vs == "gbm"])
    if not rows:
        return False
    d = pd.concat(rows, ignore_index=True)
    d = d[d.model != "baseline_harness"].drop_duplicates(["k", "design", "model"])
    for ax, des in zip(axes, ("grouped", "random")):
        for i, m in enumerate(sorted(d.model.unique())):
            s = d[(d.design == des) & (d.model == m)].sort_values("k")
            if len(s):
                _err(ax, s.k + 0.08 * i, s.skill, s.skill_lo, s.skill_hi, marker=MARK[i % 8], color=PAL[i % 8],
                     label=m, lw=1, ms=4)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("k"); ax.set_title(f"Skill vs plain GBM, {des} CV (+ = better; paired 95% CI)")
    axes[0].set_ylabel("skill vs GBM"); axes[0].legend(ncol=2, fontsize=7)
    return True


# name, n axes, function, needs the base study (False = depends only on the extras)
FIGS = [("fig01_error_vs_k", 1, fig_error_vs_k, True),
        ("fig02_gap_vs_k", 1, fig_gap_vs_k, True),
        ("fig03_skill_vs_harness", 2, fig_skill_vs_harness, True),
        ("fig04_account_concentration", 1, fig_concentration, True),
        ("fig05_conformal", 2, fig_conformal, True),
        ("fig06_aci_drift", 1, fig_aci, True),
        ("fig07_hazard", 2, fig_hazard, False),
        ("fig08_hierarchical", 2, fig_hierarchical, False),
        ("fig09_distributional", 2, fig_distributional, False),
        ("fig10_shift", 1, fig_shift, False),
        ("fig11_process", 2, fig_process, False),
        ("fig12_decision", 1, fig_decision, False),
        ("fig13_mixture", 2, fig_mixture, False),
        ("fig14_weekly_drift", 2, fig_weekly_drift, True),
        ("fig15_extras_vs_gbm", 2, fig_extras_vs_gbm, False)]


def _try(fn, ctx, n):
    """Draw into a throwaway figure; return the figure if it drew something, else None."""
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.6), squeeze=False)
    axes = list(axes[0])
    try:
        ok = fn(ctx, axes)
    except Exception as e:
        print(f"  [figure skipped: {fn.__name__} -> {type(e).__name__}: {e}]")
        ok = False
    if not ok:
        plt.close(fig)
        return None
    for a in axes:
        if not a.has_data() and not a.patches and not a.images:
            a.set_visible(False)
    fig.tight_layout()
    return fig


def make_all(ctx, outdir):
    """Save every figure that has data as PNG + PDF. Returns the list of names saved."""
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for name, n, fn, _ in FIGS:
        fig = _try(fn, ctx, n)
        if fig is None:
            continue
        fig.savefig(os.path.join(outdir, name + ".png"), dpi=200)
        fig.savefig(os.path.join(outdir, name + ".pdf"))
        plt.close(fig)
        saved.append(name)
    print(f"  saved {len(saved)} figure(s) to {outdir}/")
    return saved


def make_dashboard3(extras, out):
    """One page collecting the panels of every extra analysis that ran."""
    if not extras:
        return None
    ctx = dict(extras=extras)
    specs = [(name, n, fn) for name, n, fn, base in FIGS if not base]
    panels = []
    for name, n, fn in specs:
        probe = _try(fn, ctx, n)
        if probe is not None:
            plt.close(probe)
            panels.append((name, n, fn))
    if not panels:
        return None
    total = sum(n for _, n, _ in panels)
    cols = 3
    rows = int(np.ceil(total / cols))
    fig = plt.figure(figsize=(6.2 * cols, 4.6 * rows))
    gs = fig.add_gridspec(rows, cols)
    slot = 0
    for name, n, fn in panels:
        axes = [fig.add_subplot(gs[(slot + i) // cols, (slot + i) % cols]) for i in range(n)]
        try:
            fn(ctx, axes)
        except Exception as e:
            print(f"  [dashboard3 panel skipped: {name} -> {e}]")
        slot += n
    fig.suptitle("Extra analyses (new methods)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out, "dashboard3.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
