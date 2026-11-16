"""The whole study, start to finish, in the order a reader would follow it.

load -> profile -> forecast (3 designs) -> paired tests -> long-session classifier
-> feature importance -> conformal intervals -> drift test -> dashboards + summary

Each optional analysis is wrapped in ``safe`` so one failure never throws away
the rest of the report.
"""
import datetime as dt
import importlib
import os
import time
import traceback

from . import config as cfg
from .data.loading import load_data
from .evaluation.classification import evaluate_long_sessions
from .evaluation.forecasting import evaluate_forecasts, paired_analysis
from .evaluation.importance import grouped_importance
from .features import set_category_maps
from .profiling import flag_censoring, profile
from .reporting import figures, outputs, summary
from .reporting.dashboards import make_dashboard, make_dashboard2
from .uncertainty.adaptive import run_aci
from .uncertainty.conformal import run_conformal


# (CLI flag, name, module) for the optional extra analyses; run in this order.
EXTRAS = [("hazard", "hazard", "evaluation.hazard"),
          ("hierarchical", "hierarchical", "evaluation.hierarchical"),
          ("distributional", "distributional", "evaluation.distributional"),
          ("sequence", "sequence", "evaluation.sequence"),
          ("process", "process", "evaluation.process"),
          ("shift", "shift", "evaluation.shift"),
          ("decision", "decision", "evaluation.decision"),
          ("mixture", "mixture", "evaluation.mixture")]
ALL_FLAGS = [e[0] for e in EXTRAS] + ["group_conformal"]

FAILURES = []   # (analysis name, short error) of every analysis that failed in this run


def safe(name, fn, *a, **kw):
    """Run an optional analysis; print the traceback but keep going if it fails."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  [warn] {name} failed:")
        traceback.print_exc()
        FAILURES.append((name, f"{type(e).__name__}: {e}"))
        return None


def run(args):
    """Run everything. ``args`` needs: weeks, max_sessions, synthetic, fast, drop_censored, out."""
    started = dt.datetime.now(dt.timezone.utc)
    dirs = outputs.make_dirs(args.out)
    FAILURES.clear()
    if getattr(args, "all_extras", False):
        for f in ALL_FLAGS:
            setattr(args, f, True)
    on = lambda flag: bool(getattr(args, flag, False))
    cfg.FEATURE_SET = getattr(args, "feature_set", "base") or "base"
    if args.fast:
        cfg.apply_fast_mode()

    # 1. load ---------------------------------------------------------------
    df, bounds = load_data(args)
    df = flag_censoring(df, bounds)
    if args.drop_censored:
        before = len(df)
        df = df[~(df.cens_right | df.cens_left)].reset_index(drop=True)
        print(f"Dropped {before - len(df)} censored sessions")
    set_category_maps(df)
    n_acc = df.user_id.nunique()
    share = df.groupby("user_id").size() / len(df)
    print(f"\nLoaded {len(df):,} sessions from {n_acc} accounts "
          f"(effective number of accounts {1 / (share ** 2).sum():.1f}); "
          f"weeks: {', '.join(sorted(df.week.astype(str).unique()))}; feature set: {cfg.FEATURE_SET}")

    # 2. profile ------------------------------------------------------------
    print("\nProfiling ...")
    lines, per_user = profile(df, args.out)

    # 3. forecast + paired tests ---------------------------------------------
    print("\nEvaluating early forecasts (random / account-grouped / temporal) ...")
    res, store = evaluate_forecasts(df)
    paired = None
    if len(res):
        res.to_csv(os.path.join(args.out, "results_forecast.csv"), index=False)
        lines += summary.forecast_tables(res)
        print("\nPaired cluster bootstrap ...")
        paired = safe("paired_analysis", paired_analysis, store)
        if paired is not None and len(paired):
            paired.to_csv(os.path.join(args.out, "results_paired.csv"), index=False)
            lines += summary.paired_tables(paired)

    # 4. will it be long? ----------------------------------------------------
    print("\nLong-session classifier ...")
    long_res = safe("long_sessions", evaluate_long_sessions, df)
    if long_res is not None and len(long_res):
        long_res.to_csv(os.path.join(args.out, "results_longsession.csv"), index=False)
        lines += summary.long_session_table(long_res)

    # 5. which features matter on new accounts --------------------------------
    k_imp = 5 if (df.n_requests > 5).sum() >= 200 else 2
    print(f"\nPermutation importance (held-out accounts, k={k_imp}) ...")
    imp = safe("importance", grouped_importance, df, k_imp)
    if imp is not None:
        imp.to_csv(os.path.join(args.out, "results_importance.csv"), header=["mae_increase"])
        lines += summary.importance_table(imp, k_imp)

    # 6. prediction intervals --------------------------------------------------
    k_conf = 5 if (df.n_requests > 5).sum() >= 200 else 2
    gc = on("group_conformal")
    print(f"\nConformal intervals at k={k_conf} ({5 + 3 * gc} methods) ...")
    cres = safe("conformal", run_conformal, df, k_conf, group_methods=gc)
    conf, conf_ph = cres if cres is not None else (None, None)
    if conf is not None:
        conf.to_csv(os.path.join(args.out, "results_conformal.csv"), index=False)
        lines += summary.conformal_table(conf, k_conf)
        if gc:
            lines += summary.group_conformal_table(conf, k_conf)
    else:
        lines.append("\nConformal skipped: too few accounts/sessions.")

    # 7. drift ----------------------------------------------------------------
    print("\nAdaptive conformal under drift ...")
    aci = safe("aci", run_aci, df, k_conf)
    if aci is not None:
        lines += summary.aci_lines(aci, k_conf)

    # 8. optional extra analyses -------------------------------------------------
    extras, timings = {}, {}
    for flag, name, mod in EXTRAS:
        if not on(flag):
            continue
        print(f"\nExtra analysis: {name} ...")
        t0 = time.time()
        R = safe(name, importlib.import_module(f".{mod}", __package__).run, df,
                 os.path.join(dirs["extras"], name))
        print(f"  [{name}] finished in {time.time() - t0:.0f}s")
        timings[name] = round(time.time() - t0, 1)
        if R is not None:
            extras[name] = R
            lines += R["lines"]

    # 9. report -----------------------------------------------------------------
    lines += summary.CAVEATS
    if FAILURES:
        lines += ["", "ANALYSES THAT FAILED IN THIS RUN (the rest of the report is unaffected):"]
        lines += [f"  - {n}: {e}" for n, e in FAILURES]
    if args.synthetic:
        lines.insert(0, "*** SYNTHETIC DATA: numbers below test the code, not the dataset ***")

    p1 = make_dashboard(df, per_user, res if len(res) else None, conf, conf_ph, k_conf, args.out)
    p2 = make_dashboard2(paired, long_res, imp, conf, aci, k_conf, args.out)

    # 10. figures, tables, run_info ----------------------------------------------
    ctx = dict(df=df, per_user=per_user, res=res if len(res) else None, paired=paired, long_res=long_res,
               imp=imp, conf=conf, conf_ph=conf_ph, aci=aci, k_conf=k_conf, k_imp=k_imp, extras=extras)
    saved = safe("figures", figures.make_all, ctx, dirs["figures"]) or []
    p3 = safe("dashboard3", figures.make_dashboard3, extras, args.out)
    base_tables = {"results_forecast": res, "results_paired": paired, "results_longsession": long_res,
                   "results_conformal": conf, "importance": imp.rename("mae_increase").reset_index()
                   if imp is not None else None, "per_account_sessions": per_user.rename("sessions").reset_index()}
    for n, t in base_tables.items():
        outputs.save_table(t, n, dirs["tables"])
    for ename, R in extras.items():
        for n, t in R["tables"].items():
            outputs.save_table(t, n, dirs["tables"])
    info = outputs.write_run_info(args, df, per_user, dirs["run_info"], [e for e in ALL_FLAGS if on(e)],
                                  [dict(analysis=n, error=e) for n, e in FAILURES], started, timings)
    lines += ["", f"Run info: {len(saved)} figure(s) in figures/, tables in tables/, extras in extras/, "
                  f"provenance in run_info/ (effective accounts {info['effective_n_accounts']:.1f})."]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text)
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)
    print(f"\nSaved: {p1}\n       {p2}" + (f"\n       {p3}" if p3 else "") +
          f"\nFigures, tables, extras, run_info and summary in: {args.out}/")
    return text
