"""Everything that is about *where results go*: tables/ (CSV + LaTeX), run_info/, folders."""
import datetime as dt
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from .. import config as cfg


def make_dirs(out):
    paths = {n: os.path.join(out, n) for n in ("figures", "tables", "extras", "run_info")}
    for p in [out] + list(paths.values()):
        os.makedirs(p, exist_ok=True)
    return paths


def _tex_escape(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def to_latex_simple(df, float_digits=3):
    """Small booktabs-style LaTeX table with no jinja2 dependency."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{float_digits}g}"
        return _tex_escape(v)
    cols = list(df.columns)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
             " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def save_table(df, name, tables_dir, max_tex_rows=200):
    """Write tables/<name>.csv and tables/<name>.tex."""
    if df is None or not hasattr(df, "to_csv") or not len(df):
        return
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    df.to_csv(os.path.join(tables_dir, name + ".csv"), index=False)
    try:
        with open(os.path.join(tables_dir, name + ".tex"), "w") as f:
            f.write(to_latex_simple(df.head(max_tex_rows)))
    except Exception:
        pass


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _versions():
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "statsmodels", "torch", "pyarrow",
              "zstandard", "huggingface_hub"):
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", "?")
        except Exception:
            v[m] = None
    return v


def write_run_info(args, df, per_user, run_dir, extras_enabled, failures, started, timings=None):
    share = per_user / per_user.sum()
    info = {
        "command_line": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "synthetic": bool(args.synthetic),
        "fast_mode": bool(args.fast),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "random_seed": 0,
        "weeks": list(getattr(args, "weeks", [])) if not args.synthetic else ["synthetic"],
        "source": getattr(args, "source", "raw"),
        "drop_censored": bool(args.drop_censored),
        "n_sessions": int(len(df)),
        "n_accounts": int(per_user.size),
        "effective_n_accounts": float(1.0 / (share ** 2).sum()),
        "n_censored_right": int(df.cens_right.sum()), "n_censored_left": int(df.cens_left.sum()),
        "config": {k: getattr(cfg, k) for k in (
            "K_LIST", "EXTRA_KS", "KMAX", "ALPHA", "BOOT", "PAIR_BOOT", "CONF_REPS", "FEATURE_SET",
            "HAZARD_J", "QUANTILES", "SEQ_EPOCHS", "DRO_ROUNDS", "MIX_CLASSES", "HOLD_RATE", "TTL_GRID_S",
            "WEIGHT_MODE")},
        "extras_enabled": extras_enabled,
        "analyses_failed": failures,
        "extra_analysis_seconds": timings or {},
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    L = ["RUN INFO", "=" * 8]
    for k, v in info.items():
        if isinstance(v, dict):
            L.append(f"{k}:")
            L += [f"  {a}: {b}" for a, b in v.items()]
        else:
            L.append(f"{k}: {v}")
    with open(os.path.join(run_dir, "run_info.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    return info
