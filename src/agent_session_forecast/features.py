"""Turning the first k requests of a session into model features.

Everything here is computed from ONLY the first k requests, never later ones, so
nothing leaks from the future. Also holds the account-history encoder used by the
"warm account" model.
"""
import numpy as np
import pandas as pd

from . import config as cfg


def set_category_maps(df):
    cfg.HARN.clear(); cfg.MOD.clear()
    cfg.HARN.update({h: i for i, h in enumerate(sorted(df.harness.dropna().astype(str).unique()))})
    cfg.MOD.update({m: i for i, m in enumerate(sorted(df.model0.dropna().astype(str).unique()))})


def features_at_k(df, k, strict=True):
    """Features from ONLY the first k requests.

    strict=True  : sessions with n > k only (the usual forecasting setup: at least one more
                   request is still to come).
    strict=False : sessions with n >= k (also sessions that END at request k). Needed by the
                   hazard / decision analyses, which must see the sessions that stop.
    """
    d = df[(df.n_requests > k) if strict else (df.n_requests >= k)].reset_index(drop=True)
    if d.empty:
        return None
    A = lambda col: np.array([a[:k] for a in d[col]], dtype=float)
    comp, prm, cac = A("a_completion"), A("a_prompt"), A("a_cached")
    ttft, st, nt, ne = A("a_ttft"), A("a_start"), A("a_ntools"), A("a_nerr")
    elapsed = (st[:, -1] - st[:, 0]) / 1000.0
    X = pd.DataFrame({
        "harness": d.harness.astype(str).map(cfg.HARN).fillna(-1).astype(int).values,
        "model0": d.model0.astype(str).map(cfg.MOD).fillna(-1).astype(int).values,
        "cum_completion": comp.sum(1),
        "mean_completion": comp.mean(1),
        "last_completion": comp[:, -1],
        "prompt_first": prm[:, 0],
        "prompt_last": prm[:, -1],
        "prompt_growth": prm[:, -1] - prm[:, 0],
        "cached_frac": cac.sum(1) / np.maximum(prm.sum(1), 1),
        "mean_ttft": ttft.mean(1),
        "elapsed_s": elapsed,
        "gap_mean_s": elapsed / max(k - 1, 1),
        "tool_calls": nt.sum(1),
        "frac_tool_reqs": (nt > 0).mean(1),
        "last_ntools": nt[:, -1],
        "tool_errors": ne.sum(1),
        "err_rate": ne.sum(1) / np.maximum(nt.sum(1), 1),
        "tooldefs0": d.tooldefs0.values,
        "system0": d.system0.values,
        "n_tooldefs0": d.n_tooldefs0.values,
    })
    if cfg.FEATURE_SET == "rich" and "a_user" in d.columns:
        X = pd.concat([X, rich_features(d, k)], axis=1)
    return dict(
        d=d, cens_right=d.cens_right.values if "cens_right" in d else np.zeros(len(d), bool),
        X=X,
        y=np.log1p(d.n_requests.values.astype(float)),
        n=d.n_requests.values,
        groups=d.user_id.astype(str).values,
        harness=d.harness.astype(str).values,
        start=d.start_ms.values.astype(float),
        cens=(d.cens_right | d.cens_left).values,
    )


RICH_COLS = ["user_trigger_frac", "mutation_count", "finish_length_count", "result_tokens_sum",
             "tool_latency_mean", "subagent_count", "n_tool_cats"]


def rich_features(d, k):
    """Extra features from the first k requests only: who triggered each step, prompt
    mutations (e.g. context compaction), truncated outputs, tool result sizes / latency,
    sub-agent spawns, and how many distinct tool categories were used."""
    A = lambda col: np.array([a[:k] for a in d[col]], dtype=float)
    U, M, Ln, R, Lt, S = A("a_user"), A("a_mut"), A("a_lenfin"), A("a_restok"), A("a_lat"), A("a_nsub")
    nt = A("a_ntools")
    ncat = [len({c for cs in a[:k] for c in cs.split("|") if c}) for a in d["a_cats"]]
    return pd.DataFrame({
        "user_trigger_frac": U.mean(1), "mutation_count": M.sum(1), "finish_length_count": Ln.sum(1),
        "result_tokens_sum": R.sum(1), "tool_latency_mean": Lt.sum(1) / np.maximum(nt.sum(1), 1),
        "subagent_count": S.sum(1), "n_tool_cats": np.array(ncat, dtype=float),
    })


def add_history(Xtr, ytr, gtr, Xte, gte, prior=3.0):
    """Leave-one-out target encoding of an account's mean log length.

    Train rows use all OTHER training sessions of the same account (no leakage).
    Test rows use the account's training sessions; unseen accounts get NaN.
    """
    mu = float(np.mean(ytr))
    s = pd.DataFrame({"g": gtr, "y": ytr}).groupby("g")["y"].agg(["sum", "count"])
    tsum = pd.Series(gtr).map(s["sum"]).values.astype(float)
    tcnt = pd.Series(gtr).map(s["count"]).values.astype(float)
    ls, lc = tsum - ytr, tcnt - 1
    h_tr = np.where(lc > 0, (ls + prior * mu) / (lc + prior), np.nan)
    # "History dropout": hide the history of a random ~20% of training rows so the
    # model also learns what to do for NEW accounts (otherwise it leans on a feature
    # that is always missing at test time under account-grouped evaluation).
    hide = cfg.RNG.random(len(gtr)) < 0.2   # row-level: lower variance than hiding whole big accounts
    h_tr = np.where(hide, np.nan, h_tr)
    lc = np.where(hide, 0.0, lc)
    esum = pd.Series(gte).map(s["sum"]).values.astype(float)
    ecnt = pd.Series(gte).map(s["count"]).fillna(0).values.astype(float)
    h_te = np.where(ecnt > 0, (np.nan_to_num(esum) + prior * mu) / (ecnt + prior), np.nan)
    A, B = Xtr.copy(), Xte.copy()
    A["hist_mean"], A["hist_n"] = h_tr, lc
    B["hist_mean"], B["hist_n"] = h_te, ecnt
    return A, B
