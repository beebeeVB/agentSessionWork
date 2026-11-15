"""The features must only ever see the first k requests, and history must not leak."""
import numpy as np
import pandas as pd

from agent_session_forecast import config as cfg
from agent_session_forecast.features import add_history, features_at_k


def test_features_ignore_requests_after_k(df):
    k = 3
    before = features_at_k(df, k)["X"]

    tampered = df.copy()
    for col in ["a_completion", "a_prompt", "a_cached", "a_ttft", "a_start", "a_ntools", "a_nerr"]:
        # wreck everything AFTER the first k requests; features must not change
        tampered[col] = [a[:k] + [10**9] * (len(a) - k) for a in tampered[col]]
    after = features_at_k(tampered, k)["X"]

    pd.testing.assert_frame_equal(before, after)


def test_only_sessions_longer_than_k_are_scored(df):
    out = features_at_k(df, 5)
    assert (out["n"] > 5).all()


def test_history_encoding_is_leave_one_out():
    cfg.RNG = np.random.default_rng(1)
    y = np.array([1.0, 2.0, 3.0, 10.0])
    g = np.array(["a", "a", "a", "b"])
    Xtr = pd.DataFrame({"x": range(4)})
    Xte = pd.DataFrame({"x": [0, 1]})
    A, B = add_history(Xtr, y, g, Xte, np.array(["a", "new"]), prior=1e-9)

    mu_ok = ~A["hist_mean"].isna()
    # row 0 of account "a" is encoded from the OTHER two rows (2 and 3), never from itself
    if mu_ok[0]:
        assert np.isclose(A["hist_mean"][0], (2.0 + 3.0) / 2, atol=1e-6)
    # account "b" has only one row, so leave-one-out leaves nothing
    assert np.isnan(A["hist_mean"][3])
    # test side: seen account gets its training mean, unseen account gets NaN and n = 0
    assert np.isclose(B["hist_mean"][0], 2.0, atol=1e-6)
    assert np.isnan(B["hist_mean"][1]) and B["hist_n"][1] == 0
