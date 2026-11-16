"""Unit tests for the building blocks of the new analyses (fast; no full runs)."""
import numpy as np
import pandas as pd

from agent_session_forecast import config as cfg
from agent_session_forecast.evaluation import decision, distributional, hazard, hierarchical, process
from agent_session_forecast.evaluation.compare import evaluate_predictors, iter_folds
from agent_session_forecast.features import features_at_k
from agent_session_forecast.profiling import icc_oneway
from agent_session_forecast.uncertainty.conformal import group_intervals


# ---- hazard ------------------------------------------------------------------------------
def test_person_period_handles_censoring():
    rep, j, ev = hazard.person_period(np.array([1, 3, 3]), np.array([False, False, True]), J=5)
    # session 0: one row, event; session 1: 3 rows, event on the last; session 2 (censored): 2 rows, no event
    assert list(rep) == [0, 1, 1, 1, 2, 2]
    assert list(j) == [1, 1, 2, 3, 1, 2]
    assert list(ev) == [1, 0, 0, 1, 0, 0]


def test_person_period_truncates_at_horizon():
    rep, j, ev = hazard.person_period(np.array([10]), np.array([False]), J=4)
    assert list(j) == [1, 2, 3, 4] and ev.sum() == 0     # still alive at the horizon: no event


def test_median_remaining():
    S = np.array([[0.9, 0.4, 0.1], [0.9, 0.8, 0.7]])
    assert list(hazard.median_remaining(S)) == [2, 4]    # second never drops <= 0.5: J + 1


# ---- hierarchical ------------------------------------------------------------------------
def test_variance_components_recover_the_truth():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(200), 20)
    r = rng.normal(0, 0.5, 200)[g] + rng.normal(0, 1.0, len(g))
    su2, se2 = hierarchical.fit_variance_components(r, g)
    assert abs(su2 - 0.25) < 0.07 and abs(se2 - 1.0) < 0.1


def test_shrinkage_grows_with_fewer_sessions_and_unseen_gets_zero():
    r = np.r_[np.full(2, 1.0), np.full(200, 1.0)]
    g = np.r_[["small"] * 2, ["big"] * 200]
    eff = hierarchical.account_effects(r, g, su2=0.25, se2=1.0)
    assert eff["small"][1] < eff["big"][1] < 1.0
    assert eff["small"][0] < eff["big"][0]
    assert eff.get("never_seen", (0.0,))[0] == 0.0


def test_hier_prediction_for_unseen_accounts_is_the_harness_mean(df):
    out = features_at_k(df, 3)
    tr = np.arange(len(out["y"]) // 2)
    te = np.arange(len(out["y"]) // 2, len(out["y"]))
    out["groups"] = np.where(np.arange(len(out["y"])) < len(out["y"]) // 2, out["groups"], "STRANGER")
    p = hierarchical.pred_hier_eb(out, tr, te, {})
    mu = pd.Series(out["y"][tr]).groupby(out["X"]["harness"].values[tr]).mean()
    expect = np.array([mu.get(h, out["y"][tr].mean()) for h in out["X"]["harness"].values[te]])
    assert np.allclose(p, expect)


def test_icc_oneway_extremes():
    g = np.repeat(np.arange(30), 10)
    assert icc_oneway(np.repeat(np.arange(30.0), 10), g) > 0.99
    assert icc_oneway(np.random.default_rng(0).normal(size=300), g) < 0.15


# ---- distributional ----------------------------------------------------------------------
def test_pinball_is_asymmetric():
    y, q = np.array([1.0]), np.array([0.0])
    assert np.isclose(distributional.pinball(y, q, 0.9)[0], 0.9)    # under-prediction costs tau
    assert np.isclose(distributional.pinball(-y, q, 0.9)[0], 0.1)   # over-prediction costs 1 - tau


def test_predicted_quantiles_are_sorted(df):
    cfg.QUANTILES = [0.1, 0.5, 0.9]
    out = features_at_k(df, 3)
    tr, te = np.arange(0, len(out["y"]), 2), np.arange(1, len(out["y"]), 2)
    for m in ("lognormal", "resid_quantiles"):
        Q = distributional.predict_quantiles(m, out, tr, te)
        assert Q.shape == (len(te), 3) and (np.diff(Q, axis=1) >= 0).all()


# ---- group-aware conformal ---------------------------------------------------------------
def test_group_intervals_are_valid_and_ordered(df):
    out = features_at_k(df, 3)
    rng = np.random.default_rng(0)
    accts = np.array(sorted(set(out["groups"])))
    rng.shuffle(accts)
    third = len(accts) // 3
    part = {a: i for i, a in enumerate(accts)}
    idx = np.array([part[a] for a in out["groups"]])
    tr, ca, te = (np.where(idx < third)[0], np.where((idx >= third) & (idx < 2 * third))[0],
                  np.where(idx >= 2 * third)[0])
    iv = group_intervals(out["X"], out["y"], out["groups"], tr, ca, te)
    assert set(iv) == set(cfg.GROUP_CONF_METHODS)
    for m, (lo, hi) in iv.items():
        assert lo.shape == hi.shape == (len(te),) and (hi >= lo).all(), m
        assert np.isfinite(lo).all() and np.isfinite(hi).all(), m


# ---- process / shift / decision ----------------------------------------------------------
def test_state_priority():
    s = process.state_of
    assert s("read_file", 1, 0, 1) == "error"          # error beats everything
    assert s("read_file|edit_file", 0, 0, 2) == "edit"
    assert s("bash", 0, 0, 1) == "run" and s("grep", 0, 0, 1) == "explore"
    assert s("", 0, 0, 0) == "talk" and s("mystery", 0, 0, 1) == "other"


def test_process_transition_rows_sum_to_one(df):
    P, T, tab = process.estimate_process(df)
    rows = P.dropna(how="all").sum(axis=1)       # a state never followed by anything has no row
    assert len(rows) >= 3 and np.allclose(rows, 1.0)
    assert (tab.p_end_given_state.dropna().between(0, 1)).all()


def test_account_balanced_weights_equalise_accounts():
    g = np.array(["a"] * 90 + ["b"] * 10)
    n_g = pd.Series(g).map(pd.Series(g).value_counts()).values.astype(float)
    w = 1.0 / n_g
    assert np.isclose(w[g == "a"].sum(), w[g == "b"].sum())


def test_cost_matrix_matches_hand_calculation(monkeypatch):
    monkeypatch.setattr(cfg, "HOLD_RATE", 0.01)
    g, P = np.array([5.0, 500.0, np.inf]), np.array([100.0, 100.0, 100.0])
    C = decision._cost_matrix(g, P, [0, 60])
    # tau=0: continuing sessions always miss (cost P); ending session costs nothing
    assert np.allclose(C[:, 0], [100.0, 100.0, 0.0])
    # tau=60: gap 5 -> hit, hold 5s = 0.01*100*5 = 5 ; gap 500 -> hold 60s (60) + miss (100) ; ending -> hold 60 s
    assert np.allclose(C[:, 1], [5.0, 160.0, 60.0])


def test_decision_oracle_is_never_worse_than_any_policy(df_big):
    cfg.EXTRA_KS = [3]
    r = decision.evaluate_k(df_big, 3)
    assert r is not None
    oracle = r.set_index("policy").loc["oracle", "relative_to_fixed"]
    assert oracle <= r.relative_to_fixed.min() + 1e-9


# ---- shared harness ----------------------------------------------------------------------
def test_evaluate_predictors_pairs_on_identical_folds(df_big):
    cfg.apply_fast_mode()
    cfg.EXTRA_KS = [3]
    cfg.PAIR_BOOT = 100
    mean_pred = lambda out, tr, te, ctx: np.full(len(te), out["y"][tr].mean())
    res, paired, store = evaluate_predictors(df_big, {"mean": mean_pred}, designs=("grouped",), verbose=False)
    assert {"gbm", "baseline_harness", "mean"} <= set(res.model)
    assert set(paired.vs) == {"gbm", "baseline_harness"}
    out = features_at_k(df_big, 3)
    te_all = np.concatenate([te for _, _, te in iter_folds(out, "grouped")])
    assert len(set(te_all)) == len(out["y"])            # every session is tested exactly once


# ---- new in this version: shift skill column, decision sensitivity, robustness --------------
def test_shift_table_reports_new_account_skill(df_big):
    """The 'gap closed' number alone is misleading, so the table must carry the grouped skill too."""
    import tempfile
    from agent_session_forecast.evaluation import shift
    cfg.apply_fast_mode()
    cfg.EXTRA_KS = [3]
    R = shift.run(df_big, tempfile.mkdtemp())
    gap = R["payload"]["gap"]
    assert {"newacct_skill_vs_gbm", "newacct_skill_lo", "newacct_skill_hi"} <= set(gap.columns)
    r = gap[gap.method == "gbm_sqrt"].iloc[0]
    # skill = 1 - MAE(method, grouped) / MAE(gbm, grouped): recompute it from the table itself
    g0 = gap[gap.method == "gbm"].iloc[0]
    assert abs(r.newacct_skill_vs_gbm - (1 - r.mae_grouped / g0.mae_grouped)) < 1e-9
    assert any("new-account skill vs gbm" in line for line in R["lines"])


def test_decision_sensitivity_includes_the_main_setting_and_is_consistent(df_big):
    import tempfile
    cfg.apply_fast_mode()
    cfg.EXTRA_KS = [3]
    R = decision.run(df_big, tempfile.mkdtemp())
    sens, main = R["payload"]["sens"], R["payload"]["res"]
    assert set(decision.SWEEP_HOLD_RATES) | {cfg.HOLD_RATE} == set(np.round(sens.hold_rate.unique(), 6)) \
        or set(decision.SWEEP_HOLD_RATES) <= set(sens.hold_rate.unique())
    assert np.allclose(sens.break_even_s, 1.0 / sens.hold_rate)
    assert np.isclose(main.hold_rate, cfg.HOLD_RATE).all()
    # the main table is exactly the sensitivity table restricted to the configured rate
    again = sens[np.isclose(sens.hold_rate, cfg.HOLD_RATE)].reset_index(drop=True)
    pd.testing.assert_frame_equal(main, again)
    # the oracle can never lose to fixed_ttl, whatever the cost model
    orc = sens[sens.policy == "oracle"]
    assert (orc.saving_vs_fixed >= -1e-9).all()
    assert cfg.HOLD_RATE == 0.01                      # the sweep must restore the configured value


def test_robustness_scenarios_remove_the_heaviest_accounts(df_big):
    from agent_session_forecast.evaluation import robustness
    sc = robustness.scenarios(df_big)
    top = df_big.groupby("user_id").size().sort_values(ascending=False).index.tolist()
    assert "all" in sc and len(sc["all"]) == len(df_big)
    for n in (1, 3, 10):
        name = f"drop_top{n}"
        if name in sc:
            assert not sc[name].user_id.isin(top[:n]).any()
            assert len(sc[name]) < len(df_big)
    cap = sc.get(f"cap{robustness.CAP}")
    if cap is not None:
        assert cap.groupby("user_id").size().max() <= robustness.CAP
        assert cap.user_id.nunique() == df_big.user_id.nunique()      # nobody is dropped, only capped
    for d in sc.values():
        assert robustness._eff(d) >= 1.0
