"""Fake data in the SAME schema as the real trace.

Only for testing the pipeline offline (and for the tests). Each account has its
own "habit" for session length, which is exactly the structure that makes random
splits look better than account-grouped ones, so the demo is still instructive.
Numbers computed on it say nothing about the real dataset.
"""
import datetime as dt

import numpy as np

from .. import config as cfg


_CATS = ["read_file", "search", "list_dir", "edit_file", "write_file", "bash", "run_tests", "web_fetch"]


def make_synthetic(n_users=60, n_sessions=4000):
    """Fake data in the SAME schema; each account has its own length 'habit'.

    The richer fields (step_trigger, mutation, finish_reason, tool category / result size /
    latency / subagent) come from a SEPARATE generator so the original synthetic numbers
    are unchanged.
    """
    rng2 = np.random.default_rng(12345)
    week_start = dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc)
    harnesses = ["claude-code", "cline", "kilo", "hermes", "opencode"]
    user_h = {u: harnesses[u % len(harnesses)] for u in range(n_users)}
    habit = cfg.RNG.normal(2.5, 0.8, n_users)
    weights = cfg.RNG.pareto(1.2, n_users) + 1
    weights /= weights.sum()
    sessions = []
    for i in range(n_sessions):
        u = int(cfg.RNG.choice(n_users, p=weights))
        n = int(max(1, min(400, np.exp(cfg.RNG.normal(habit[u], 0.9)))))
        t0 = int((week_start + dt.timedelta(seconds=float(cfg.RNG.uniform(0, 7 * 86400 - 3600)))).timestamp() * 1000)
        reqs, t = [], t0
        for j in range(n):
            dur = int(cfg.RNG.integers(1000, 20000))
            ntc = int(cfg.RNG.integers(0, 3))
            reqs.append({
                "step": j, "start_ms": t, "end_ms": t + dur,
                "ttft_ms": int(cfg.RNG.integers(300, 6000)), "model": "glm-5.1",
                "provider_tokens": {"prompt": int(5000 + 800 * j + cfg.RNG.integers(0, 2000)),
                                    "completion": int(cfg.RNG.integers(20, 600)),
                                    "cached": int(cfg.RNG.integers(0, 4000))},
                "role_tokens": {"system": 1800, "tool_definitions": 8000},
                "tool_definition_names": ["a"] * (10 + u % 5),
                "tool_calls": [{"outcome": "error" if cfg.RNG.random() < 0.15 else "success",
                                "category": _CATS[int(rng2.integers(0, len(_CATS)))],
                                "result_tokens": int(rng2.integers(0, 3000)),
                                "latency_ms": int(rng2.integers(20, 4000)),
                                "subagent": bool(rng2.random() < 0.02)}
                               for _ in range(ntc)],
                "step_trigger": "user" if (j == 0 or rng2.random() < 0.1) else "tool",
                "mutation": {"transition": "compact" if rng2.random() < 0.03 else "append"} if j else None,
                "finish_reason": "length" if rng2.random() < 0.02 else "stop",
            })
            t += dur + int(cfg.RNG.integers(500, 8000))
        sessions.append({"session_id": f"s{i}", "user_id": f"u{u}", "harness": user_h[u],
                         "day": "2026-08-30", "start_ms": t0, "end_ms": t, "requests": reqs})
    return sessions


def synthetic_to_parquet(sessions, outdir):
    """Write synthetic sessions as flat sessions/requests/tool_calls parquet tables.

    Only used to test the Parquet loader's mechanics (struct columns, joins). The column
    names are a GUESS at the real layout, so this proves the code path works, not that the
    real files match.
    """
    import os
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    S, R, T = [], [], []
    for s in sessions:
        S.append({k: s[k] for k in ["session_id", "user_id", "harness", "day", "start_ms", "end_ms"]})
        S[-1]["week"] = "2026-08-30_2026-09-05"
        for j, r in enumerate(s["requests"]):
            R.append({"session_id": s["session_id"], "step": j, "start_ms": r["start_ms"],
                      "end_ms": r["end_ms"], "ttft_ms": r["ttft_ms"], "model": r["model"],
                      "provider_tokens": r["provider_tokens"], "role_tokens": r["role_tokens"],
                      "tool_definition_names": r["tool_definition_names"],
                      "step_trigger": r["step_trigger"], "mutation": r["mutation"],
                      "finish_reason": r["finish_reason"]})
            for ci, c in enumerate(r["tool_calls"]):
                T.append({"session_id": s["session_id"], "step": j, "call_index": ci, **c})
    pd.DataFrame(S).to_parquet(os.path.join(outdir, "sessions.parquet"))
    pd.DataFrame(R).to_parquet(os.path.join(outdir, "requests.parquet"))
    pd.DataFrame(T).to_parquet(os.path.join(outdir, "tool_calls.parquet"))
