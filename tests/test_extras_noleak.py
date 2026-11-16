"""Every NEW feature path must also be blind to everything after request k."""
import numpy as np
import pandas as pd

from agent_session_forecast import config as cfg
from agent_session_forecast.evaluation.process import session_states, state_features
from agent_session_forecast.evaluation.sequence import seq_tensor
from agent_session_forecast.features import features_at_k

NUM = ["a_completion", "a_prompt", "a_cached", "a_ttft", "a_start", "a_end", "a_ntools", "a_nerr",
       "a_user", "a_mut", "a_lenfin", "a_restok", "a_lat", "a_nsub"]


def tamper(df, k):
    t = df.copy()
    for col in NUM:
        t[col] = [list(a[:k]) + [10 ** 9] * (len(a) - k) for a in t[col]]
    t["a_cats"] = [list(a[:k]) + ["ZZZ|YYY"] * (len(a) - k) for a in t["a_cats"]]
    return t


def test_rich_features_ignore_the_future(df):
    cfg.FEATURE_SET = "rich"
    k = 4
    pd.testing.assert_frame_equal(features_at_k(df, k)["X"], features_at_k(tamper(df, k), k)["X"])
    assert {"user_trigger_frac", "mutation_count", "n_tool_cats"} <= set(features_at_k(df, k)["X"].columns)


def test_base_feature_set_is_unchanged_by_the_new_code(df):
    cfg.FEATURE_SET = "base"
    X = features_at_k(df, 3)["X"]
    assert "mutation_count" not in X.columns


def test_state_features_and_sequence_tensor_ignore_the_future(df):
    k = 5
    d, t = features_at_k(df, k)["d"], features_at_k(tamper(df, k), k)["d"]
    pd.testing.assert_frame_equal(state_features(d, k), state_features(t, k))
    cfg.FEATURE_SET = "rich"
    assert np.array_equal(seq_tensor(d, k), seq_tensor(t, k))
    assert seq_tensor(d, k).shape[1] == k


def test_non_strict_includes_sessions_that_end_at_k(df):
    k = 4
    strict, loose = features_at_k(df, k), features_at_k(df, k, strict=False)
    assert len(loose["y"]) > len(strict["y"])
    assert (loose["n"] >= k).all() and (strict["n"] > k).all()


def test_session_states_are_a_known_alphabet(df):
    from agent_session_forecast.evaluation.process import STATES
    assert {s for ss in session_states(df, 6) for s in ss} <= set(STATES)
