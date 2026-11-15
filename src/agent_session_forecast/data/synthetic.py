"""Fake data in the SAME schema as the real trace.

Only for testing the pipeline offline (and for the tests). Each account has its
own "habit" for session length, which is exactly the structure that makes random
splits look better than account-grouped ones, so the demo is still instructive.
Numbers computed on it say nothing about the real dataset.
"""
import datetime as dt

import numpy as np

from .. import config as cfg


def make_synthetic(n_users=60, n_sessions=4000):
    """Fake data in the SAME schema; each account has its own length 'habit'."""
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
                "tool_calls": [{"outcome": "error" if cfg.RNG.random() < 0.15 else "success"}
                               for _ in range(ntc)],
            })
            t += dur + int(cfg.RNG.integers(500, 8000))
        sessions.append({"session_id": f"s{i}", "user_id": f"u{u}", "harness": user_h[u],
                         "day": "2026-08-30", "start_ms": t0, "end_ms": t, "requests": reqs})
    return sessions
