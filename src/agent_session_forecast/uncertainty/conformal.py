"""Prediction intervals with a coverage guarantee, tested on NEW accounts.

Five methods are compared: plain split-conformal, Mondrian (per-harness),
normalized, CQR, and the "trap" of calibrating on random sessions of seen accounts.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

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


def group_intervals(X, y, groups, tr, ca, te, alpha=cfg.ALPHA):
    """Group-aware conformal methods (``--group-conformal``), tested on unseen accounts.

    The ordinary methods assume calibration residuals look like test residuals. They do not,
    because a calibration account's residual is small when the model has seen the account and
    large for a stranger. These three repair that:

    * ``loao``          leave-accounts-out: calibration residuals are out-of-fold with WHOLE
                        ACCOUNTS held out, so each one is a "stranger" residual; the final model
                        is fitted on everything (train + calibration accounts).
    * ``cv_plus_group`` CV+ (Barber et al. 2021) with account-level folds: an interval per test
                        point from the K fold models and the out-of-fold residuals.
    * ``weighted``      weighted split-conformal (Tibshirani et al. 2019): calibration residuals are
                        reweighted by a classifier-estimated density ratio calibration-vs-test
                        accounts, correcting for covariate shift between the account groups.
                        Weights are clipped to [0.05, 20]; an infinite quantile is capped at the
                        largest calibration residual (a small coverage deficit is possible).

    ``loao`` and ``cv_plus_group`` use train + calibration accounts together (they need no
    separate calibration set); ``weighted`` uses the same train/calibration split as ``split``.
    """
    out = {}
    pool = np.concatenate([tr, ca])
    Xp, yp, gp = X.iloc[pool], y[pool], groups[pool]
    Xte = X.iloc[te]
    n_folds = min(5, len(set(gp)))
    folds = list(GroupKFold(n_folds).split(Xp, yp, gp))
    mus = np.zeros((len(folds), len(te)), dtype=np.float32)
    oof, fold_of = np.zeros(len(pool)), np.zeros(len(pool), int)
    for f, (a, b) in enumerate(folds):
        m = new_gbm().fit(Xp.iloc[a], yp[a])
        oof[b], fold_of[b], mus[f] = m.predict(Xp.iloc[b]), f, m.predict(Xte)
    R = np.abs(yp - oof)
    p_full = new_gbm().fit(Xp, yp).predict(Xte)
    q = conformal_q(R, alpha)
    out["loao"] = (p_full - q, p_full + q)

    n = len(R)
    M = mus[fold_of]                                         # (n_pool, n_test)
    k_hi, k_lo = min(int(np.ceil((1 - alpha) * (n + 1))), n), max(int(np.floor(alpha * (n + 1))), 1)
    hi = np.partition(M + R[:, None].astype(np.float32), k_hi - 1, axis=0)[k_hi - 1]
    lo = np.partition(M - R[:, None].astype(np.float32), k_lo - 1, axis=0)[k_lo - 1]
    out["cv_plus_group"] = (lo.astype(float), hi.astype(float))

    # weighted split conformal
    m = new_gbm().fit(X.iloc[tr], y[tr])
    res_ca, p_te = np.abs(y[ca] - m.predict(X.iloc[ca])), m.predict(Xte)
    Z = pd.concat([X.iloc[ca], Xte], ignore_index=True)
    lab = np.r_[np.zeros(len(ca)), np.ones(len(te))].astype(int)
    clf = HistGradientBoostingClassifier(max_iter=40, max_depth=3, min_samples_leaf=50,
                                         categorical_features=[0, 1], random_state=0)
    pr = cross_val_predict(clf, Z, lab, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1]
    pr = np.clip(pr, 1e-3, 1 - 1e-3)
    w_all = np.clip(pr / (1 - pr) * (len(ca) / len(te)), 0.05, 20.0)
    w_ca, w_te = w_all[:len(ca)], w_all[len(ca):]
    order = np.argsort(res_ca)
    sres, cw = res_ca[order], np.cumsum(w_ca[order])
    need = (1 - alpha) * (cw[-1] + w_te)
    idx = np.searchsorted(cw, need, side="left")
    qw = sres[np.minimum(idx, len(sres) - 1)]                # capped at the largest residual
    out["weighted"] = (p_te - qw, p_te + qw)
    return out


def run_conformal(df, k, alpha=cfg.ALPHA, reps=None, group_methods=False):
    """Compare interval methods on UNSEEN accounts (repeat over account splits).

    group_methods=True also evaluates the group-aware methods on the SAME splits.
    """
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

        if group_methods:
            try:
                iv.update(group_intervals(X, y, groups, tr, ca, te, alpha))
            except Exception as e:
                print(f"    [warn] group-aware conformal failed on split {r} ({type(e).__name__}: {e})")

        for name, (lo, hi) in iv.items():
            met, ph = interval_metrics(yte, lo, hi, h_te, groups[te])
            met.update(rep=r, method=name, cov_seen=cov_seen if name == "random_calib" else np.nan)
            rec.append(met)
            ph = ph.reset_index(); ph["rep"] = r; ph["method"] = name
            ph_all.append(ph)
    if not rec:
        return None, None
    return pd.DataFrame(rec), pd.concat(ph_all)
