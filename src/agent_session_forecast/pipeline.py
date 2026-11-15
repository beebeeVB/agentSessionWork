"""The whole study, start to finish, in the order a reader would follow it.

load -> profile -> forecast (3 designs) -> paired tests -> long-session classifier
-> feature importance -> conformal intervals -> drift test -> dashboards + summary

Each optional analysis is wrapped in ``safe`` so one failure never throws away
the rest of the report.
"""
import os
import traceback

from . import config as cfg
from .data.loading import load_data
from .evaluation.classification import evaluate_long_sessions
from .evaluation.forecasting import evaluate_forecasts, paired_analysis
from .evaluation.importance import grouped_importance
from .features import set_category_maps
from .profiling import flag_censoring, profile
from .reporting import summary
from .reporting.dashboards import make_dashboard, make_dashboard2
from .uncertainty.adaptive import run_aci
from .uncertainty.conformal import run_conformal


def safe(name, fn, *a, **kw):
    """Run an optional analysis; print the traceback but keep going if it fails."""
    try:
        return fn(*a, **kw)
    except Exception:
        print(f"  [warn] {name} failed:")
        traceback.print_exc()
        return None


def run(args):
    """Run everything. ``args`` needs: weeks, max_sessions, synthetic, fast, drop_censored, out."""
    os.makedirs(args.out, exist_ok=True)
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
    print(f"\nConformal intervals at k={k_conf} (5 methods) ...")
    cres = safe("conformal", run_conformal, df, k_conf)
    conf, conf_ph = cres if cres is not None else (None, None)
    if conf is not None:
        conf.to_csv(os.path.join(args.out, "results_conformal.csv"), index=False)
        lines += summary.conformal_table(conf, k_conf)
    else:
        lines.append("\nConformal skipped: too few accounts/sessions.")

    # 7. drift ----------------------------------------------------------------
    print("\nAdaptive conformal under drift ...")
    aci = safe("aci", run_aci, df, k_conf)
    if aci is not None:
        lines += summary.aci_lines(aci, k_conf)

    # 8. report -----------------------------------------------------------------
    lines += summary.CAVEATS
    if args.synthetic:
        lines.insert(0, "*** SYNTHETIC DATA: numbers below test the code, not the dataset ***")

    p1 = make_dashboard(df, per_user, res if len(res) else None, conf, conf_ph, k_conf, args.out)
    p2 = make_dashboard2(paired, long_res, imp, conf, aci, k_conf, args.out)
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text)
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)
    print(f"\nSaved: {p1}\n       {p2}\nSummary + CSVs in: {args.out}/")
    return text
