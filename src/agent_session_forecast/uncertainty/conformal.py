"""Prediction intervals with a coverage guarantee, tested on NEW accounts.

Five methods are compared: plain split-conformal, Mondrian (per-harness),
normalized, CQR, and the "trap" of calibrating on random sessions of seen accounts.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm, new_gbm_q, new_scale_gbm


def conformal_q(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")


def interval_metrics(y, lo, hi, harness, groups):
    cov = (y >= lo) & (y <= hi)
    ph = pd.DataFrame({"h": harness, "c": cov}).groupby("h")["c"].agg(["mean", "count"])
    ok = ph[ph["count"] >= 30]
    pa = pd.DataFrame({"g": groups, "c": cov}).groupby("g")["c"].agg(["mean", "count"])
    pa = pa[pa["count"] >= 10]
    met = dict(coverage=float(cov.mean()), width=float(np.mean(hi - lo)),
               width_factor=float(np.median(np.exp(hi - lo))),
               worst_harness_cov=float(ok["mean"].min()) if len(ok) else np.nan,
               acct_cov_p10=float(np.percentile(pa["mean"], 10)) if len(pa) >= 5 else np.nan)
    return met, ph


def run_conformal(df, k, alpha=cfg.ALPHA, reps=None):
    """Compare interval methods on UNSEEN accounts (repeat over account splits)."""
    reps = reps or cfg.CONF_REPS
    out = features_at_k(df, k)
    if out is None:
        return None, None
    X, y, groups, harness = out["X"], out["y"], out["groups"], out["harness"]
    users = np.array(sorted(set(groups)))
    if len(users) < 12:
        return None, None
    rec, ph_all = [], []
    for r in range(reps):
        perm = cfg.RNG.permutation(users)
        n = len(perm)
        tr_u, ca_u, te_u = perm[: int(.5 * n)], perm[int(.5 * n): int(.75 * n)], perm[int(.75 * n):]
        tr, ca, te = (np.where(np.isin(groups, u))[0] for u in (tr_u, ca_u, te_u))
        if len(te) < 20 or len(tr) < 50 or len(ca) < 20:
            continue
        Xtr, Xca, Xte = X.iloc[tr], X.iloc[ca], X.iloc[te]
        ytr, yca, yte = y[tr], y[ca], y[te]
        h_ca, h_te = harness[ca], harness[te]
        m = new_gbm().fit(Xtr, ytr)
        p_ca, p_te = m.predict(Xca), m.predict(Xte)
        res_ca = np.abs(yca - p_ca)
        iv = {}

        # split-conformal, one global width
        q = conformal_q(res_ca, alpha)
        iv["split"] = (p_te - q, p_te + q)

        # Mondrian: one quantile per harness (fallback to global if too few calib rows)
        qh = {h: (conformal_q(res_ca[h_ca == h], alpha) if (h_ca == h).sum() >= 30 else q)
              for h in set(h_te)}
        q_te = np.array([qh[h] for h in h_te])
        iv["mondrian"] = (p_te - q_te, p_te + q_te)

        # normalized: width scales with a learned error-size model (fit on OOF residuals)
        try:
            oof = np.zeros(len(tr))
            for a, b in GroupKFold(3).split(Xtr, ytr, groups[tr]):
                oof[b] = new_gbm().fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b])
            sm = new_scale_gbm().fit(Xtr, np.abs(ytr - oof))
            s_ca = np.maximum(sm.predict(Xca), 0.05)
            s_te = np.maximum(sm.predict(Xte), 0.05)
            qn = conformal_q(res_ca / s_ca, alpha)
            iv["normalized"] = (p_te - qn * s_te, p_te + qn * s_te)
        except Exception:
            pass

        # CQR: conformalized quantile regression
        mlo, mhi = new_gbm_q(alpha / 2).fit(Xtr, ytr), new_gbm_q(1 - alpha / 2).fit(Xtr, ytr)
        l_ca, u_ca = mlo.predict(Xca), mhi.predict(Xca)
        l_ca, u_ca = np.minimum(l_ca, u_ca), np.maximum(l_ca, u_ca)
        qc = conformal_q(np.maximum(l_ca - yca, yca - u_ca), alpha)
        l_te, u_te = mlo.predict(Xte), mhi.predict(Xte)
        l_te, u_te = np.minimum(l_te, u_te), np.maximum(l_te, u_te)
        iv["cqr"] = (l_te - qc, u_te + qc)

        # the trap: calibrate on random sessions of SEEN accounts
        seen = cfg.RNG.permutation(np.concatenate([tr, ca]))
        a_, b_ = int(.6 * len(seen)), int(.8 * len(seen))
        i_tr, i_ca, i_st = seen[:a_], seen[a_:b_], seen[b_:]
        mB = new_gbm().fit(X.iloc[i_tr], y[i_tr])
        qB = conformal_q(np.abs(y[i_ca] - mB.predict(X.iloc[i_ca])), alpha)
        pB = mB.predict(Xte)
        iv["random_calib"] = (pB - qB, pB + qB)
        cov_seen = float((np.abs(y[i_st] - mB.predict(X.iloc[i_st])) <= qB).mean())

        for name, (lo, hi) in iv.items():
            met, ph = interval_metrics(yte, lo, hi, h_te, groups[te])
            met.update(rep=r, method=name, cov_seen=cov_seen if name == "random_calib" else np.nan)
            rec.append(met)
            ph = ph.reset_index(); ph["rep"] = r; ph["method"] = name
            ph_all.append(ph)
    if not rec:
        return None, None
    return pd.DataFrame(rec), pd.concat(ph_all)
