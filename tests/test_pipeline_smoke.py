"""End-to-end: the whole study runs on synthetic data and writes every output."""
import argparse
import os

import pytest

from agent_session_forecast import config as cfg
from agent_session_forecast.data import loading
from agent_session_forecast.data.synthetic import make_synthetic
from agent_session_forecast.pipeline import run


@pytest.mark.slow
def test_full_run_on_synthetic_data(tmp_path, monkeypatch):
    # smaller fake dataset than the CLI default so the test stays quick
    monkeypatch.setattr(loading, "make_synthetic", lambda: make_synthetic(n_users=40, n_sessions=1500))
    saved = (cfg.K_LIST, cfg.BOOT, cfg.PAIR_BOOT, cfg.CONF_REPS)
    try:
        args = argparse.Namespace(weeks=[], max_sessions=None, synthetic=True, fast=True,
                                  drop_censored=False, out=str(tmp_path))
        text = run(args)
    finally:
        cfg.K_LIST, cfg.BOOT, cfg.PAIR_BOOT, cfg.CONF_REPS = saved

    assert "SYNTHETIC DATA" in text
    for name in ["summary.txt", "dashboard.png", "dashboard2.png", "results_forecast.csv",
                 "results_paired.csv", "results_conformal.csv", "results_longsession.csv",
                 "results_importance.csv", "results_harness.csv"]:
        assert os.path.exists(tmp_path / name), f"missing output: {name}"
