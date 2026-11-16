"""Shared fixtures: a small synthetic dataset in the real schema, and config isolation."""
import copy

import pandas as pd
import pytest

from agent_session_forecast import config as cfg
from agent_session_forecast.data.loading import summarize_session
from agent_session_forecast.data.synthetic import make_synthetic
from agent_session_forecast.features import set_category_maps
from agent_session_forecast.profiling import flag_censoring


@pytest.fixture(autouse=True)
def _restore_config():
    """apply_fast_mode / FEATURE_SET / RNG are module globals: put them back after every test."""
    saved = {k: copy.deepcopy(v) for k, v in vars(cfg).items() if k.isupper()}
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)


@pytest.fixture(scope="session")
def df():
    recs = [summarize_session(s) for s in make_synthetic(n_users=30, n_sessions=900)]
    d = flag_censoring(pd.DataFrame([r for r in recs if r]), bounds=[(0, 1)])
    set_category_maps(d)
    return d


@pytest.fixture(scope="session")
def df_big():
    """Larger fake set for the few tests that run a whole analysis."""
    recs = [summarize_session(s) for s in make_synthetic(n_users=40, n_sessions=1500)]
    d = flag_censoring(pd.DataFrame([r for r in recs if r]), bounds=[(0, 1)])
    set_category_maps(d)
    return d
