"""Process mining of the tool-call sequence (``--process``).

Each request is mapped to one coarse *state* from the tools it called (a keyword match on the
tool category, so unknown categories end up in ``other``):

    talk      no tool calls (the model answered in text)
    explore   read / search / list / fetch / grep ...
    edit      edit / write / patch / create ...
    run       bash / shell / exec / test / build ...
    subagent  a sub-agent was spawned
    error     at least one tool call failed       (takes priority)

Priority when several apply: error > subagent > edit > run > explore > other > talk.

From the first ``KMAX`` requests of every session we estimate

* the transition matrix between consecutive requests;
* P(session ends | current state) = (# requests in that state that were the session's last,
  excluding right-censored sessions) / (# requests in that state);
* the mean dwell (run length) of each state. Run lengths are truncated at the window, so they
  are slightly underestimated for sticky states.

This is a first-order Markov / semi-Markov summary, not a fitted generative model. It is also
turned into forecasting features (state counts, last state, number of state changes, longest
run), all computed from the first k requests only, and scored as ``gbm_states``.
"""
import os

import numpy as np
import pandas as pd

from ..models import new_gbm
from .compare import evaluate_predictors, fmt_paired

STATES = ["talk", "explore", "edit", "run", "subagent", "error", "other"]
_KW = {"explore": ("read", "grep", "glob", "search", "list", "find", "view", "fetch", "web", "ls", "cat", "open"),
       "edit": ("edit", "write", "patch", "replace", "create", "apply", "insert", "notebook", "save"),
       "run": ("bash", "shell", "exec", "run", "test", "command", "terminal", "build", "compile", "pytest")}


def state_of(cats, nerr, nsub, ntools):
    if nerr > 0:
        return "error"
    if nsub > 0:
        return "subagent"
    if ntools == 0:
        return "talk"
    toks = [c.lower() for c in cats.split("|") if c]
    for st in ("edit", "run", "explore"):
        if any(any(w in t for w in _KW[st]) for t in toks):
            return st
    return "other"


def session_states(d, upto=None):
    """List (per session) of the state of each of its first `upto` stored requests."""
    out = []
    for cats, ne, ns, nt in zip(d["a_cats"], d["a_nerr"], d["a_nsub"], d["a_ntools"]):
        m = len(cats) if upto is None else min(upto, len(cats))
        out.append([state_of(cats[i], ne[i], ns[i], nt[i]) for i in range(m)])
    return out


def state_features(d, k):
    """Features from the first k requests only (states, last state, switches, longest run)."""
    sts = session_states(d, k)
    rows = []
    for s in sts:
        cnt = {st: s.count(st) / len(s) for st in STATES}
        runs, cur = [], 1
        for a, b in zip(s[:-1], s[1:]):
            if a == b:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        row = {f"state_{st}": cnt[st] for st in STATES}
        row.update({f"last_{st}": float(s[-1] == st) for st in STATES})
        row["n_switches"] = len(runs) - 1
        row["longest_run"] = max(runs)
        rows.append(row)
    return pd.DataFrame(rows)


def pred_states(out, tr, te, ctx):
    X, y, d, k = out["X"], out["y"], out["d"], ctx["k"]
    F = state_features(d, k)
    Z = pd.concat([X.reset_index(drop=True), F], axis=1)
    return new_gbm().fit(Z.iloc[tr], y[tr]).predict(Z.iloc[te])


def estimate_process(df):
    """Transition matrix, end probability and dwell per state from the first KMAX requests."""
    sts = session_states(df)
    n, cens = df.n_requests.values, df.cens_right.values
    T = pd.DataFrame(0.0, index=STATES, columns=STATES)
    cnt, ends = dict.fromkeys(STATES, 0), dict.fromkeys(STATES, 0)
    runs = {st: [] for st in STATES}
    for s, nn, cr in zip(sts, n, cens):
        for t, st in enumerate(s):
            cnt[st] += 1
            if t + 1 == nn and not cr:
                ends[st] += 1
            if t + 1 < len(s):
                T.loc[st, s[t + 1]] += 1
        cur = 1
        for a, b in zip(s[:-1], s[1:]):
            if a == b:
                cur += 1
            else:
                runs[a].append(cur); cur = 1
        runs[s[-1]].append(cur)
    P = T.div(T.sum(axis=1).replace(0, np.nan), axis=0)
    tab = pd.DataFrame({"state": STATES, "n_requests": [cnt[s] for s in STATES],
                        "p_end_given_state": [ends[s] / cnt[s] if cnt[s] else np.nan for s in STATES],
                        "mean_dwell": [np.mean(runs[s]) if runs[s] else np.nan for s in STATES]})
    return P, T, tab


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    P, T, tab = estimate_process(df)
    res, paired, store = evaluate_predictors(df, {"gbm_states": pred_states})
    long = P.stack().rename("p").reset_index()
    long.columns = ["from", "to", "p"]
    long["count"] = T.stack().values
    tables = {"process_transitions": long, "process_states": tab,
              "process_forecast": res, "process_paired": paired}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", "PROCESS MINING (request states from tool calls; first-order transitions):",
         tab.round(3).to_string(index=False)]
    L += fmt_paired(paired, designs=("grouped", "random"), refs=("gbm",))
    return dict(tables=tables, lines=L, payload=dict(P=P, tab=tab, res=res, paired=paired))
