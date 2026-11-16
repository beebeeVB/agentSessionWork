"""Decision-focused evaluation: does a better forecast save KV-cache cost? (``--decision``)

Forecast accuracy is a proxy. What a serving system actually decides is *how long to keep a
session's prompt prefix cached* after request k. This module scores forecasts by the cost of
that decision under a deliberately simple, fully documented cost model.

Cost model (everything in "tokens of prefill recomputed", scaled by the prefix size P = the
prompt tokens of request k, so big sessions matter more, as they do in reality):

* holding the prefix costs ``HOLD_RATE * P`` per second (``cfg.HOLD_RATE``);
* a miss (the next request arrives after the prefix was evicted) costs ``P``;
* a retention time tau is chosen from ``cfg.TTL_GRID_S``; if the next request arrives after
  ``g`` seconds then: hit iff g <= tau (cost ``HOLD_RATE*P*g``), else miss (cost ``HOLD_RATE*P*tau + P``);
* if the session ENDS at request k, holding is wasted: ``HOLD_RATE*P*tau``.

With the default HOLD_RATE = 0.01 the break-even gap is 100 s. These numbers are assumptions, not
measurements: change ``cfg.HOLD_RATE`` / ``cfg.TTL_GRID_S`` and re-run to see how conclusions move.

Policies (account-grouped 5-fold CV, so the model has never seen the account):
  evict        tau = 0
  fixed_ttl    the single best tau on the training accounts
  harness_rate P(continue) = per-harness continuation rate in training
  hazard_step  P(continue | features of the first k requests) from a gradient-boosting classifier
               (the discrete-time hazard at step k)
  oracle       knows the true gap: the best tau for each session (a lower bound, not a policy)

For the three forecast-driven policies the gap distribution (given the session continues) is the
empirical gap distribution of the training accounts, and tau minimises the expected cost.
Sessions that end at k are only included when the session is not right-censored.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .. import config as cfg
from ..features import features_at_k
from ..models import new_clf
from .bootstrap import paired_boot

POLICIES = ["evict", "fixed_ttl", "harness_rate", "hazard_step", "oracle"]


def _gaps(d, k):
    """Seconds between the end of request k and the start of request k+1 (inf if no k+1)."""
    ends = np.array([a[k - 1] if a[k - 1] > 0 else s[k - 1] for a, s in zip(d["a_end"], d["a_start"])], float)
    nxt = np.array([s[k] if len(s) > k else np.nan for s in d["a_start"]], float)
    g = np.where(np.isfinite(nxt), np.maximum((nxt - ends) / 1000.0, 0.0), np.inf)
    return g


def _cost_matrix(g, P, tau):
    """(n, len(tau)) cost per session for each retention time."""
    h = cfg.HOLD_RATE
    G, T = g[:, None], np.asarray(tau, float)[None, :]
    hit = G <= T
    cont = np.isfinite(G)
    c = np.where(cont, np.where(hit, h * G, h * T + 1.0), h * T)
    c = np.where(cont & (T == 0), 1.0, c)                # tau=0: a continuing session always misses
    return P[:, None] * c


def _expected(Gtr, tau):
    """A_j = E[cost per unit P | continues], B_j = cost if it ends, for each tau_j."""
    h = cfg.HOLD_RATE
    T = np.asarray(tau, float)
    A = np.array([np.mean(h * np.minimum(Gtr, t) + (Gtr > t)) for t in T])
    A[T == 0] = 1.0
    return A, h * T


SWEEP_HOLD_RATES = [0.002, 0.005, 0.01, 0.02, 0.05]     # break-even gap 500 s ... 20 s


def evaluate_k(df, k, hold_rates=None):
    """Score the policies at request k. ``hold_rates`` = list of HOLD_RATE values to evaluate
    (default: just cfg.HOLD_RATE). The forecasting models do not depend on the cost model, so
    they are fitted once per fold and only the decision step is redone for each rate."""
    out = features_at_k(df, k, strict=False)
    if out is None or k >= cfg.KMAX:
        return None
    d = out["d"]
    keep = ~((out["n"] == k) & out["cens_right"])
    if keep.sum() < 150 or np.nanstd(np.array([s[0] for s in d["a_start"]], float)) == 0:
        return None
    X = out["X"][keep].reset_index(drop=True)
    d = d[keep].reset_index(drop=True)
    g = _gaps(d, k)
    P = np.array([p[k - 1] for p in d["a_prompt"]], float)
    cont = np.isfinite(g)
    groups = out["groups"][keep]
    if len(set(groups)) < 6 or cont.sum() < 50 or (~cont).sum() < 10:
        return None
    tau = cfg.TTL_GRID_S
    harness = X["harness"].values
    folds = list(GroupKFold(5).split(X, cont, groups))
    p_h = np.zeros(len(g))
    p_x = np.zeros(len(g))
    for tr, te in folds:                                   # cost-model independent: fit once
        rate = pd.Series(cont[tr].astype(float)).groupby(harness[tr]).mean()
        p_h[te] = [rate.get(h, cont[tr].mean()) for h in harness[te]]
        p_x[te] = new_clf().fit(X.iloc[tr], cont[tr].astype(int)).predict_proba(X.iloc[te])[:, 1]
    rates = list(hold_rates) if hold_rates is not None else [cfg.HOLD_RATE]
    base_rate = cfg.HOLD_RATE
    parts = []
    try:
        for hr in rates:
            cfg.HOLD_RATE = hr                             # _cost_matrix/_expected read it
            C = _cost_matrix(g, P, tau)
            choice = {p: np.zeros(len(g), int) for p in POLICIES}
            for tr, te in folds:
                A, B = _expected(g[tr][cont[tr]], tau)
                choice["fixed_ttl"][te] = int(np.argmin(C[tr].sum(0)))
                choice["harness_rate"][te] = np.argmin(p_h[te][:, None] * A + (1 - p_h[te][:, None]) * B, axis=1)
                choice["hazard_step"][te] = np.argmin(p_x[te][:, None] * A + (1 - p_x[te][:, None]) * B, axis=1)
            choice["oracle"] = C.argmin(1)
            cost = {p: C[np.arange(len(g)), choice[p]] for p in POLICIES}
            held = {p: np.asarray(tau)[choice[p]] for p in POLICIES}
            hit = {p: float(np.mean(g[cont] <= held[p][cont])) for p in POLICIES}
            ref = cost["fixed_ttl"]
            for p in POLICIES:
                r = paired_boot(cost[p], ref, groups) if p != "fixed_ttl" else None
                parts.append(dict(k=k, hold_rate=hr, break_even_s=1.0 / hr, policy=p, n=len(g),
                                  accounts=len(set(groups)),
                                  mean_cost=float(cost[p].mean()), relative_to_fixed=float(cost[p].sum() / ref.sum()),
                                  hit_rate=hit[p], mean_hold_s=float(held[p].mean()),
                                  fixed_ttl_s=float(np.asarray(tau)[choice["fixed_ttl"]].mean()),
                                  saving_vs_fixed=r["skill"] if r else 0.0, saving_lo=r["skill_lo"] if r else 0.0,
                                  saving_hi=r["skill_hi"] if r else 0.0, p_boot=r["p_boot"] if r else np.nan))
    finally:
        cfg.HOLD_RATE = base_rate
    return pd.DataFrame(parts)


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    rates = sorted(set(SWEEP_HOLD_RATES) | {cfg.HOLD_RATE})
    parts = []
    for k in cfg.EXTRA_KS:
        r = evaluate_k(df, k, hold_rates=rates)
        if r is None:
            print(f"    k={k}: skipped (not enough sessions, or no usable timestamps)")
        else:
            print(f"    k={k}: n={r.n.iloc[0]} sessions")
            parts.append(r)
    sens = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res = sens[np.isclose(sens.hold_rate, cfg.HOLD_RATE)].reset_index(drop=True) if len(sens) else sens
    if len(res):
        res.to_csv(os.path.join(outdir, "decision_costs.csv"), index=False)
        sens.to_csv(os.path.join(outdir, "decision_sensitivity.csv"), index=False)
    L = ["", f"DECISION-FOCUSED EVALUATION (KV-cache retention; HOLD_RATE={cfg.HOLD_RATE}, "
             f"break-even gap {1 / cfg.HOLD_RATE:.0f}s; TTL grid {cfg.TTL_GRID_S}):",
         "  saving = 1 - cost/cost(fixed_ttl); positive = cheaper than the best fixed TTL. Account-grouped CV.",
         "  The cost model is an ASSUMPTION (see the sensitivity table below)."]
    if not len(res):
        L.append("  (skipped: not enough usable sessions or timestamps)")
    for _, r in res.iterrows():
        if r.policy in ("fixed_ttl",):
            continue
        sig = "" if r.saving_lo <= 0 <= r.saving_hi else ("  <- cheaper" if r.saving_vs_fixed > 0 else "  <- costlier")
        L.append(f"  k={int(r.k):>2} {r.policy:<13} saving {r.saving_vs_fixed:+.1%} [{r.saving_lo:+.1%}, {r.saving_hi:+.1%}]"
                 f"  hit-rate {r.hit_rate:.2f}{sig}")
    if len(sens):
        L += ["", "  SENSITIVITY of the hazard_step saving to the cost model (break-even gap = 1/HOLD_RATE seconds;",
              "  a hit costs HOLD_RATE per second held, a miss costs one prefix recompute):"]
        for k in sorted(sens.k.unique()):
            for _, r in sens[(sens.k == k) & (sens.policy == "hazard_step")].sort_values("hold_rate").iterrows():
                sig = "" if r.saving_lo <= 0 <= r.saving_hi else ("  <- cheaper" if r.saving_vs_fixed > 0 else "  <- costlier")
                L.append(f"    k={int(k):>2} break-even {r.break_even_s:>5.0f}s  hazard_step saving "
                         f"{r.saving_vs_fixed:+.1%} [{r.saving_lo:+.1%}, {r.saving_hi:+.1%}]{sig}")
        hz = sens[sens.policy == "hazard_step"]
        L.append(f"  -> hazard_step is significantly cheaper than the best fixed TTL in "
                 f"{int((hz.saving_lo > 0).sum())} of {len(hz)} (k, cost-model) cells and significantly costlier in "
                 f"{int((hz.saving_hi < 0).sum())}.")
    return dict(tables={"decision_costs": res, "decision_sensitivity": sens}, lines=L,
                payload=dict(res=res, sens=sens))
