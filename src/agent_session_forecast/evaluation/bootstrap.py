"""Resampling tools that respect the fact that sessions cluster inside accounts.

* cluster_boot_mae  - MAE with a CI from resampling whole accounts
* paired_boot       - paired comparison of two error vectors on the same sessions
"""
import numpy as np
import pandas as pd

from .. import config as cfg


def cluster_boot_mae(abs_err, groups, B=None):
    """MAE with a cluster bootstrap CI (resample ACCOUNTS, not sessions)."""
    B = B or cfg.BOOT
    s = pd.DataFrame({"e": abs_err, "g": groups}).groupby("g")["e"].agg(["sum", "count"])
    sums, cnts = s["sum"].values, s["count"].values
    idx = cfg.RNG.integers(0, len(s), size=(B, len(s)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return abs_err.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def paired_boot(ea, eb, groups, B=None):
    """Paired cluster bootstrap of mean(ea) - mean(eb) and skill = 1 - sum(ea)/sum(eb).

    Sessions are matched (same rows, two error vectors); accounts are resampled.
    p_boot is a two-sided bootstrap p-value for diff == 0.
    """
    B = B or cfg.PAIR_BOOT
    m = np.isfinite(ea) & np.isfinite(eb)
    if m.sum() < 50:
        return None
    t = pd.DataFrame({"a": ea[m], "b": eb[m], "g": groups[m]}).groupby("g").agg(
        sa=("a", "sum"), sb=("b", "sum"), n=("a", "size"))
    sa, sb, n = t.sa.values, t.sb.values, t.n.values
    idx = cfg.RNG.integers(0, len(t), size=(B, len(t)))
    N = n[idx].sum(1)
    da, db = sa[idx].sum(1) / N, sb[idx].sum(1) / N
    diff, skill = da - db, 1 - da / db
    return dict(
        diff=sa.sum() / n.sum() - sb.sum() / n.sum(),
        diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5),
        skill=1 - sa.sum() / sb.sum(),
        skill_lo=np.percentile(skill, 2.5), skill_hi=np.percentile(skill, 97.5),
        p_boot=1.0 if not diff.any() else min(1.0, 2 * min((diff > 0).mean(), (diff < 0).mean())),
        n=int(n.sum()), accounts=len(t))
