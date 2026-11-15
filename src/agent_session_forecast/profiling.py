"""Descriptive statistics: who produces the sessions, and how long are they?

Covers the length distribution, per-harness differences, how concentrated the
data is in a few accounts (Gini, effective number of accounts, ICC) and the
week-boundary censoring flags.
"""
import os

import numpy as np
import pandas as pd


def flag_censoring(df, bounds, tol_ms=2 * 3600 * 1000):
    """Heuristic: sessions ending near the week end / starting near the week start."""
    df = df.copy()
    df["cens_right"] = False
    df["cens_left"] = False
    for (ws, we) in bounds:
        in_week = (df["start_ms"] >= ws - tol_ms) & (df["start_ms"] < we)
        df.loc[in_week & (df["end_ms"] >= we - tol_ms), "cens_right"] = True
        df.loc[in_week & (df["start_ms"] <= ws + tol_ms), "cens_left"] = True
    return df


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def profile(df, out):
    L = []
    per_user = df.groupby("user_id").size().sort_values(ascending=False)
    shares = per_user / per_user.sum()
    eff_accounts = 1.0 / (shares ** 2).sum()
    L.append(f"Sessions: {len(df):,} | accounts: {per_user.size} | harnesses: {df.harness.nunique()}")
    L.append(f"Requests per session: median {df.n_requests.median():.0f}, "
             f"mean {df.n_requests.mean():.1f}, p90 {df.n_requests.quantile(.9):.0f}, "
             f"p99 {df.n_requests.quantile(.99):.0f}, max {df.n_requests.max()}")
    L.append(f"Single-request sessions: {(df.n_requests == 1).mean():.1%}")
    L.append(f"Account concentration: top-1 = {shares.iloc[0]:.1%} of sessions, "
             f"top-3 = {shares.iloc[:3].sum():.1%}, Gini = {gini(per_user.values):.2f}")
    L.append(f"Effective number of independent accounts (1/HHI): {eff_accounts:.1f} "
             f"(of {per_user.size})  <- this limits grouped analysis")
    L.append(f"Possibly right-censored: {df.cens_right.mean():.1%}; "
             f"left-truncated: {df.cens_left.mean():.1%}")
    L.append(f"Observed start range: {pd.to_datetime(df.start_ms.min(), unit='ms')} -> "
             f"{pd.to_datetime(df.start_ms.max(), unit='ms')} (sanity-check the UTC assumption)")

    y = np.log1p(df.n_requests)
    grp = df.assign(y=y).groupby("user_id")["y"]
    n_i, m_i = grp.size(), grp.mean()
    k, N = len(n_i), len(df)
    if k > 1 and N > k:
        ssb = (n_i * (m_i - y.mean()) ** 2).sum()
        ssw = ((y - df["user_id"].map(m_i)) ** 2).sum()
        msb, msw = ssb / (k - 1), ssw / (N - k)
        n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
        icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))
        L.append(f"ICC of log session length across accounts: {icc:.2f} "
                 f"(share of variance explained by WHO the account is)")

    h = (df.groupby("harness").n_requests.agg(["count", "median", "mean"])
           .sort_values("count", ascending=False))
    h.to_csv(os.path.join(out, "results_harness.csv"))
    return L, per_user
