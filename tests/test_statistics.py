"""The statistical building blocks behave the way their docstrings promise."""
import numpy as np

from agent_session_forecast import config as cfg
from agent_session_forecast.evaluation.bootstrap import paired_boot
from agent_session_forecast.evaluation.classification import fold_auc
from agent_session_forecast.models import TemporalSplit
from agent_session_forecast.profiling import gini
from agent_session_forecast.uncertainty.conformal import conformal_q


def test_gini_extremes():
    assert gini([5, 5, 5, 5]) == 0
    assert gini([0, 0, 0, 100]) > 0.7   # one account has everything


def test_conformal_q_covers_at_least_the_target():
    # The guarantee is on AVERAGE over calibration draws (a single draw can land a bit
    # under), so average the coverage of many independent calibration sets.
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(300):
        calib = np.abs(rng.normal(size=100))
        test = np.abs(rng.normal(size=2000))
        cover.append((test <= conformal_q(calib, 0.10)).mean())
    assert np.mean(cover) >= 0.895   # >= 90% in theory, tiny slack for simulation noise


def test_paired_boot_sees_a_real_difference_and_not_a_fake_one():
    cfg.RNG = np.random.default_rng(0)
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 20)
    base = rng.uniform(0.5, 1.5, size=600)

    worse = paired_boot(base + 0.3, base, groups, B=400)
    assert worse["diff_lo"] > 0 and worse["p_boot"] < 0.05

    same = paired_boot(base + rng.normal(0, 0.05, 600), base, groups, B=400)
    assert same["diff_lo"] < 0 < same["diff_hi"]


def test_fold_auc_is_not_fooled_by_fold_base_rates():
    # two folds with different base rates and a score that only encodes the fold:
    # pooled AUC would look informative, within-fold AUC correctly says 0.5
    y = np.array([0, 1] * 50 + [0, 0, 0, 1] * 25)
    fold = np.array([0] * 100 + [1] * 100)
    p = np.where(fold == 0, 0.5, 0.25)
    assert abs(fold_auc(y, p, fold) - 0.5) < 1e-9


def test_temporal_split_never_trains_on_the_future():
    t = np.random.default_rng(0).uniform(0, 100, 500)
    tr, te = next(TemporalSplit(t).split(None))
    assert t[tr].max() <= t[te].min()
    assert len(set(tr) & set(te)) == 0
