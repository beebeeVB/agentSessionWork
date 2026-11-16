"""End-to-end: the whole study runs on synthetic data and writes every output."""
import argparse
import json
import os

import pytest

from agent_session_forecast import config as cfg
from agent_session_forecast.data import loading
from agent_session_forecast.data.synthetic import make_synthetic
from agent_session_forecast.pipeline import ALL_FLAGS, run

BASE_FILES = ["summary.txt", "dashboard.png", "dashboard2.png", "results_forecast.csv",
              "results_paired.csv", "results_conformal.csv", "results_longsession.csv",
              "results_importance.csv", "results_harness.csv"]


def _small(monkeypatch):
    # smaller fake dataset than the CLI default so the test stays quick
    monkeypatch.setattr(loading, "make_synthetic", lambda: make_synthetic(n_users=40, n_sessions=1500))
    orig = cfg.apply_fast_mode

    def fast():
        orig()
        cfg.EXTRA_KS = [3]
    monkeypatch.setattr(cfg, "apply_fast_mode", fast)


@pytest.mark.slow
def test_full_run_on_synthetic_data(tmp_path, monkeypatch):
    """The original outputs are still produced, unchanged, with no extras switched on."""
    _small(monkeypatch)
    args = argparse.Namespace(weeks=[], max_sessions=None, synthetic=True, fast=True,
                              drop_censored=False, out=str(tmp_path))
    text = run(args)
    assert "SYNTHETIC DATA" in text
    for name in BASE_FILES:
        assert os.path.exists(tmp_path / name), f"missing output: {name}"
    assert not os.listdir(tmp_path / "extras")          # nothing extra unless asked
    assert (tmp_path / "run_info" / "run_info.json").exists()


@pytest.mark.slow
def test_every_extra_analysis_and_the_output_layout(tmp_path, monkeypatch):
    _small(monkeypatch)
    args = argparse.Namespace(weeks=[], max_sessions=None, synthetic=True, fast=True,
                              drop_censored=False, out=str(tmp_path), all_extras=True, feature_set="rich")
    text = run(args)
    for name in BASE_FILES + ["dashboard3.png"]:
        assert os.path.exists(tmp_path / name), f"missing output: {name}"
    for sub in ("figures", "tables", "extras", "run_info"):
        assert (tmp_path / sub).is_dir()
    assert "ANALYSES THAT FAILED" not in text, text[text.index("ANALYSES THAT FAILED"):][:500]
    extras = set(os.listdir(tmp_path / "extras"))
    assert extras == {"hazard", "hierarchical", "distributional", "sequence", "process", "shift",
                      "decision", "mixture"}
    figs = {f for f in os.listdir(tmp_path / "figures") if f.endswith(".png")}
    assert {"fig01_error_vs_k.png", "fig05_conformal.png", "fig07_hazard.png"} <= figs
    assert all((tmp_path / "figures" / f.replace(".png", ".pdf")).exists() for f in figs)
    assert any(f.endswith(".tex") for f in os.listdir(tmp_path / "tables"))
    info = json.load(open(tmp_path / "run_info" / "run_info.json"))
    assert info["synthetic"] and set(info["extras_enabled"]) == set(ALL_FLAGS)
    assert "GROUP-AWARE CONFORMAL" in text and "DECISION-FOCUSED" in text
