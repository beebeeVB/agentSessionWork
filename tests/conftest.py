"""Shared fixture: a small synthetic dataset in the real schema."""
import pandas as pd
import pytest

from agent_session_forecast.data.loading import summarize_session
from agent_session_forecast.data.synthetic import make_synthetic
from agent_session_forecast.features import set_category_maps
from agent_session_forecast.profiling import flag_censoring


@pytest.fixture(scope="session")
def df():
    recs = [summarize_session(s) for s in make_synthetic(n_users=30, n_sessions=900)]
    d = flag_censoring(pd.DataFrame([r for r in recs if r]), bounds=[(0, 1)])
    set_category_maps(d)
    return d
