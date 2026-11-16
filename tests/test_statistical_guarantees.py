"""Do the statistical methods actually do what they claim? (simulation / brute-force checks)

These tests do not look at the code's plumbing; they simulate data where the right answer is
known, or re-implement a method the slow, obvious way, and check that the package agrees:

1. hazard model      : recovers a known constant-in-time hazard, with and without censoring;
2. conformal         : the finite-sample coverage guarantee, the failure under shift that the
                       group-aware methods exist to repair, and their coverage on a simulation
                       with a real account effect;
3. CV+ (group folds) : the vectorised implementation equals a naive double loop from the definition.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from agent_session_forecast import config as cfg
from agent_session_forecast.evaluation.hazard import _fit_predict_survival, median_remaining, person_period
from agent_session_forecast.models import new_gbm
from agent_session_forecast.uncertainty.conformal import conformal_q, group_intervals

# --------------------------------------------------------------------------------------------
# 1. hazard model recovers a known hazard
# --------------------------------------------------------------------------------------------
H = {0: 0.05, 1: 0.15}            # true per-step ending probability for harness 0 / 1


def _simulate_hazard_data(n, cens_rate, seed):
    rng = np.random.default_rng(seed)
    harness = rng.integers(0, 2, n)
    p = np.where(harness == 1, H[1], H[0])
    T = rng.geometric(p)                                   # remaining requests, support 1, 2, ...
    if cens_rate:
        C = rng.geometric(cens_rate, n)                    # independent censoring time
        r, cens = np.minimum(T, C), C < T
    else:
        r, cens = T, np.zeros(n, bool)
    X = pd.DataFrame({"harness": harness, "model0": np.zeros(n, int),
                      "noise": rng.normal(size=n)})
    return X, r, cens, harness


@pytest.mark.parametrize("kind", ["logit", "gbm"])
@pytest.mark.parametrize("cens_rate", [0.0, 0.04])
def test_hazard_model_recovers_the_true_survival_curve(kind, cens_rate):
    """With S(j) = (1-h)^j known, predicted survival must be close, including under censoring.

    The censored case is the point: dropping censored sessions, or treating the censoring step
    as an event, would bias the curve; the person-period construction should not."""
    J, n = 40, 6000
    X, r, cens, harness = _simulate_hazard_data(n, cens_rate, seed=3)
    idx = np.arange(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    S = _fit_predict_survival(kind, X, r, cens, tr, te, J)
    assert S.shape == (len(te), J) and np.isfinite(S).all()
    assert (np.diff(S, axis=1) <= 1e-9).all(), "survival must be non-increasing"
    tol = 0.05
    for g in (0, 1):
        truth = (1 - H[g]) ** np.arange(1, J + 1)
        got = S[harness[te] == g].mean(0)
        for j in (1, 3, 5, 10, 20):
            assert abs(got[j - 1] - truth[j - 1]) < tol, (kind, cens_rate, g, j, got[j - 1], truth[j - 1])


def test_hazard_median_matches_the_geometric_median():
    """Median of Geometric(p) is ceil(log(0.5)/log(1-p)); the model's median should agree closely."""
    J, n = 40, 8000
    X, r, cens, harness = _simulate_hazard_data(n, 0.0, seed=5)
    idx = np.arange(n)
    S = _fit_predict_survival("logit", X, r, cens, idx[: n // 2], idx[n // 2:], J)
    med = median_remaining(S)
    for g in (0, 1):
        true_med = int(np.ceil(np.log(0.5) / np.log(1 - H[g])))
        got = np.median(med[harness[n // 2:] == g])
        assert abs(got - true_med) <= 1, (g, got, true_med)


def test_naive_handling_of_censoring_would_be_biased():
    """Guards the test above: with heavy censoring, treating censored ends as events (the naive
    shortcut) gives a clearly worse survival curve than person_period does. If this ever fails,
    the censoring test is no longer discriminating."""
    J, n = 40, 6000
    X, r, cens, harness = _simulate_hazard_data(n, 0.08, seed=9)
    idx = np.arange(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    good = _fit_predict_survival("logit", X, r, cens, tr, te, J)
    naive = _fit_predict_survival("logit", X, r, np.zeros(n, bool), tr, te, J)   # censoring ignored
    truth = np.where(harness[te] == 1, 1 - H[1], 1 - H[0])[:, None] ** np.arange(1, J + 1)
    err = lambda S: np.abs(S - truth).mean()
    assert err(good) < err(naive)


def test_person_period_event_counts_match_the_data():
    """Total events = number of uncensored sessions whose end falls within the horizon."""
    rng = np.random.default_rng(0)
    r = rng.integers(1, 30, 500)
    cens = rng.random(500) < 0.2
    rep, j, ev = person_period(r, cens, 20)
    assert ev.sum() == int(((~cens) & (r <= 20)).sum())
    assert len(rep) == len(j) == len(ev)


# --------------------------------------------------------------------------------------------
# 2. conformal coverage
# --------------------------------------------------------------------------------------------
def test_split_conformal_has_exact_finite_sample_coverage():
    """With n calibration scores, exchangeable data, coverage = ceil((n+1)(1-a))/(n+1).
    For n = 99 and a = 0.1 that is exactly 0.90; the simulated mean must be inside a tight band."""
    rng = np.random.default_rng(0)
    n, alpha, reps = 99, 0.1, 20000
    cal = rng.exponential(size=(reps, n))
    new = rng.exponential(size=reps)
    q = np.array([conformal_q(c, alpha) for c in cal[:4000]])
    cov = (new[:4000] <= q).mean()
    assert 0.885 < cov < 0.915, cov
    assert cov > 0.88          # never meaningfully below the nominal level


def test_calibrating_on_the_wrong_population_undercovers():
    """The failure that motivates the group-aware methods: calibration residuals from a
    population whose errors are SMALLER than the test population's give coverage far below 90%."""
    rng = np.random.default_rng(1)
    q = conformal_q(np.abs(rng.normal(0, 1.0, 2000)), 0.1)     # 'seen account' residuals, sd 1
    new_accounts = np.abs(rng.normal(0, 2.0, 20000))            # strangers: sd 2
    assert (new_accounts <= q).mean() < 0.75
    # while calibrating on the right population restores it
    q2 = conformal_q(np.abs(rng.normal(0, 2.0, 2000)), 0.1)
    assert 0.87 < (new_accounts <= q2).mean() < 0.93


def _simulate_accounts(n_acc, per, seed, sd_acc=1.0, sd_noise=0.5, fingerprint=False):
    """y = f(x) + account effect + noise.

    fingerprint=False: features carry NO account identity, so even a seen account's effect cannot
                       be learned.
    fingerprint=True : one feature is a per-account constant (like the real tool-definition /
                       system-prompt fingerprints), so a model can MEMORISE a seen account's effect
                       and its residuals there are small, while a stranger's are large. This is the
                       situation the group-aware conformal methods are built for."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_acc), per)
    u = rng.normal(0, sd_acc, n_acc)[g]
    x1, x2 = rng.normal(size=len(g)), rng.normal(size=len(g))
    X = pd.DataFrame({"harness": rng.integers(0, 3, len(g)), "model0": np.zeros(len(g), int),
                      "x1": x1, "x2": x2})
    if fingerprint:
        X["fp"] = rng.permutation(n_acc)[g].astype(float)
    y = 1.0 + 0.8 * x1 - 0.3 * x2 + X["harness"].values * 0.2 + u + rng.normal(0, sd_noise, len(g))
    return X, y, g.astype(str)


def _split_accounts(g, seed, n_tr, n_ca):
    accts = np.array(sorted(set(g)))
    rng = np.random.default_rng(seed)
    rng.shuffle(accts)
    part = {a: i for i, a in enumerate(accts)}
    idx = np.array([part[a] for a in g])
    return (np.where(idx < n_tr)[0], np.where((idx >= n_tr) & (idx < n_tr + n_ca))[0],
            np.where(idx >= n_tr + n_ca)[0])


def test_group_aware_conformal_beats_random_calibration_on_new_accounts():
    """Accounts differ by a random effect AND the features contain an account fingerprint, so a
    model memorises seen accounts. Then:
      * 'trap' (calibrate on random sessions of seen accounts) badly under-covers strangers;
      * loao / cv_plus_group get close to the 90% target;
      * weighted is in between or better (it only reweights, so it is the weakest here).
    Averaged over 6 account splits to keep the check stable."""
    cov = {"loao": [], "cv_plus_group": [], "weighted": [], "trap": []}
    for seed in range(6):
        X, y, g = _simulate_accounts(60, 40, seed=seed, fingerprint=True)
        tr, ca, te = _split_accounts(g, seed, 30, 15)
        iv = group_intervals(X, y, g, tr, ca, te, alpha=0.1)
        for m, (lo, hi) in iv.items():
            cov[m].append(float(((y[te] >= lo) & (y[te] <= hi)).mean()))
        rng = np.random.default_rng(seed)
        seen = rng.permutation(np.concatenate([tr, ca]))
        a, b = int(0.6 * len(seen)), int(0.8 * len(seen))
        m = new_gbm().fit(X.iloc[seen[:a]], y[seen[:a]])
        q = conformal_q(np.abs(y[seen[a:b]] - m.predict(X.iloc[seen[a:b]])), 0.1)
        cov["trap"].append(float((np.abs(y[te] - m.predict(X.iloc[te])) <= q).mean()))
    mean = {k: float(np.mean(v)) for k, v in cov.items()}
    assert mean["trap"] < 0.70, mean                       # the trap is badly overconfident
    for m in ("loao", "cv_plus_group"):
        assert 0.84 < mean[m] < 0.98, mean                 # close to the 90% target
        assert mean[m] > mean["trap"] + 0.15, mean
    assert mean["weighted"] > mean["trap"] + 0.10, mean


def test_random_calibration_is_fine_when_accounts_cannot_be_memorised():
    """The converse, so the test above is not just a rigged simulation: with NO account
    fingerprint the model cannot recognise seen accounts, calibration residuals already contain
    the account effect, and random calibration is roughly right (coverage not far below 90%)."""
    cov = []
    for seed in range(6):
        X, y, g = _simulate_accounts(60, 40, seed=seed, fingerprint=False)
        tr, ca, te = _split_accounts(g, seed, 30, 15)
        rng = np.random.default_rng(seed)
        seen = rng.permutation(np.concatenate([tr, ca]))
        a, b = int(0.6 * len(seen)), int(0.8 * len(seen))
        m = new_gbm().fit(X.iloc[seen[:a]], y[seen[:a]])
        q = conformal_q(np.abs(y[seen[a:b]] - m.predict(X.iloc[seen[a:b]])), 0.1)
        cov.append(float((np.abs(y[te] - m.predict(X.iloc[te])) <= q).mean()))
    assert np.mean(cov) > 0.80, cov


# --------------------------------------------------------------------------------------------
# 3. CV+ against a brute-force implementation
# --------------------------------------------------------------------------------------------
def _cv_plus_bruteforce(X, y, groups, tr, ca, te, alpha):
    """CV+ straight from Barber et al. (2021), eq. (3): for each test point t,
         upper = ceil((1-a)(n+1))-th smallest of { mu_{-k(i)}(x_t) + R_i },
         lower = floor(a(n+1))-th smallest  of { mu_{-k(i)}(x_t) - R_i },
       with R_i the leave-fold-out residual of calibration point i. Plain Python loops."""
    pool = np.concatenate([tr, ca])
    Xp, yp, gp = X.iloc[pool], y[pool], groups[pool]
    n_folds = min(5, len(set(gp)))
    folds = list(GroupKFold(n_folds).split(Xp, yp, gp))
    models = [new_gbm().fit(Xp.iloc[a], yp[a]) for a, _ in folds]
    fold_of = np.empty(len(pool), int)
    for f, (_, b) in enumerate(folds):
        fold_of[b] = f
    n = len(pool)
    R = np.array([abs(yp[i] - models[fold_of[i]].predict(Xp.iloc[[i]])[0]) for i in range(n)])
    mu_te = np.array([m.predict(X.iloc[te]) for m in models])            # (n_folds, n_test)
    k_hi = min(int(np.ceil((1 - alpha) * (n + 1))), n)
    k_lo = max(int(np.floor(alpha * (n + 1))), 1)
    lo, hi = np.empty(len(te)), np.empty(len(te))
    for t in range(len(te)):
        up = sorted(mu_te[fold_of[i], t] + R[i] for i in range(n))
        dn = sorted(mu_te[fold_of[i], t] - R[i] for i in range(n))
        hi[t], lo[t] = up[k_hi - 1], dn[k_lo - 1]
    return lo, hi


def test_cv_plus_group_matches_the_brute_force_definition():
    X, y, g = _simulate_accounts(20, 12, seed=11)
    accts = np.array(sorted(set(g)))
    part = {a: i for i, a in enumerate(accts)}
    idx = np.array([part[a] for a in g])
    tr, ca, te = (np.where(idx < 10)[0], np.where((idx >= 10) & (idx < 15))[0], np.where(idx >= 15)[0])
    te = te[:25]                                    # keep the brute-force loop quick
    fast = group_intervals(X, y, g, tr, ca, te, alpha=0.1)["cv_plus_group"]
    lo, hi = _cv_plus_bruteforce(X, y, g, tr, ca, te, 0.1)
    np.testing.assert_allclose(fast[0], lo, atol=1e-4)
    np.testing.assert_allclose(fast[1], hi, atol=1e-4)
    assert (fast[1] >= fast[0]).all()


def test_cv_plus_is_at_least_as_wide_as_loao_in_most_cases():
    """Sanity relation: CV+ takes order statistics of mu+R over fold models, loao uses a single
    full-data model and the same residuals; on average their widths should be of the same order."""
    X, y, g = _simulate_accounts(30, 15, seed=2)
    accts = np.array(sorted(set(g)))
    part = {a: i for i, a in enumerate(accts)}
    idx = np.array([part[a] for a in g])
    tr, ca, te = (np.where(idx < 15)[0], np.where((idx >= 15) & (idx < 22))[0], np.where(idx >= 22)[0])
    iv = group_intervals(X, y, g, tr, ca, te)
    w_cv = np.mean(iv["cv_plus_group"][1] - iv["cv_plus_group"][0])
    w_lo = np.mean(iv["loao"][1] - iv["loao"][0])
    assert 0.5 < w_cv / w_lo < 2.0
