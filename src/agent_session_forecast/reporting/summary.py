"""Turns result tables into the plain-text lines of summary.txt."""
from .. import config as cfg


def forecast_tables(res):
    """MAE tables: all designs/models, account-macro MAE, error in raw requests."""
    lines = ["", "Forecast MAE on log(1+requests) [lower = better]:"]
    lines.append(res.pivot_table(index="k", columns=["design", "model"], values="mae_log").round(3).to_string())
    gbm = res[res.model == "gbm"]
    lines += ["", "Account-macro MAE (every account weighted equally), GBM:"]
    lines.append(gbm.pivot(index="k", columns="design", values="macro_mae_log").round(3).to_string())
    lines += ["", "Median absolute error in REQUESTS, GBM:"]
    lines.append(gbm.pivot(index="k", columns="design", values="medae_requests").round(1).to_string())
    return lines


def _fmt_rows(paired, comp, model=None, design=None, col="diff"):
    s = paired[paired.comparison == comp]
    if model is not None:
        s = s[s.model == model]
    if design is not None:
        s = s[s.design == design]
    rows = []
    for _, r in s.sort_values("k").iterrows():
        if col == "diff":
            rows.append(f"  k={int(r.k):>2}: {r['diff']:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]  p={r.p_boot:.3f}")
        else:
            rows.append(f"  k={int(r.k):>2}: skill {r.skill:+.1%} [{r.skill_lo:+.1%}, {r.skill_hi:+.1%}]  p={r.p_boot:.3f}")
    return rows


def paired_tables(paired):
    """The paired-bootstrap comparisons, each with its 95% CI and p-value."""
    L = []
    L += ["", "PAIRED: MAE(grouped) - MAE(random), GBM  (positive = new accounts are harder):"]
    L += _fmt_rows(paired, "grouped_minus_random", model="gbm")
    L += ["", "PAIRED: GBM skill over per-harness median, GROUPED eval:"]
    L += _fmt_rows(paired, "gbm_vs_baseline_harness", design="grouped", col="skill")
    L += ["", "PAIRED: GBM skill over per-harness median, RANDOM eval:"]
    L += _fmt_rows(paired, "gbm_vs_baseline_harness", design="random", col="skill")
    L += ["", "PAIRED: skill from knowing the harness at all (harness median vs global median), grouped:"]
    L += _fmt_rows(paired, "harness_vs_global", design="grouped", col="skill")
    L += ["", "PAIRED: skill from account-history features over plain GBM, RANDOM (warm accounts):"]
    L += _fmt_rows(paired, "history_vs_gbm", design="random", col="skill")
    L += ["", "PAIRED: same, TEMPORAL (future sessions, mix of warm/new accounts):"]
    L += _fmt_rows(paired, "history_vs_gbm", design="temporal", col="skill")
    return L


def long_session_table(long_res):
    cols = ["k", "T", "design", "pos_rate", "auc_gbm", "auc_baseline", "brier_skill"]
    return ["", "P(session > T requests | first k): AUC (GBM / harness-rate baseline) and Brier skill:",
            long_res[cols].round(3).to_string(index=False)]


def importance_table(imp, k):
    return ["", f"Top features by grouped permutation importance (k={k}):", imp.head(8).round(4).to_string()]


def conformal_table(conf, k):
    L = ["", f"Conformal ({1 - cfg.ALPHA:.0%} target) on NEW accounts, k={k}, "
             f"mean over {conf.rep.nunique()} splits:"]
    agg = conf.groupby("method")[["coverage", "width_factor", "worst_harness_cov", "acct_cov_p10"]].mean()
    agg = agg.reindex([m for m in cfg.CONF_METHODS if m in agg.index]).round(3)
    agg.columns = ["coverage", "width(x, median)", "worst_harness", "acct_p10"]
    L.append(agg.to_string())
    seen = conf.loc[conf.method == "random_calib", "cov_seen"].mean()
    L.append(f"  random_calib on SEEN accounts (what it looks like in-house): {seen:.1%}")
    return L


def aci_lines(aci, k):
    return ["", f"Drift test (train 40% / calibrate 20% / stream 40% in time order, k={k}):",
            f"  static split-conformal coverage {aci['cov_static']:.1%} | ACI coverage {aci['cov_aci']:.1%} "
            f"(target {1 - cfg.ALPHA:.0%}, {aci['n']} streamed sessions)"]


CAVEATS = [
    "",
    "Caveats: target is requests per session; only sessions with > k requests are scored;",
    "censoring flags are a heuristic (UTC week bounds; try --drop-censored); sub-agent requests not counted;",
    "read grouped results with the effective number of accounts above in mind.",
]


def group_conformal_table(conf, k):
    """Group-aware conformal methods next to the five originals (same splits)."""
    from .. import config as cfg
    allm = cfg.CONF_METHODS + cfg.GROUP_CONF_METHODS
    t = conf.groupby("method")[["coverage", "width_factor", "worst_harness_cov", "acct_cov_p10"]].mean()
    t = t.reindex([m for m in allm if m in t.index])
    return ["", f"GROUP-AWARE CONFORMAL (k={k}; mean over account re-splits; target coverage {1 - cfg.ALPHA:.0%}):",
            "  loao / cv_plus_group use train+calibration accounts together; weighted uses the split-conformal split.",
            t.round(3).to_string()]
