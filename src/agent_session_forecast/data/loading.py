"""Reading the real trace from Hugging Face.

Streams the weekly ``.jsonl.zst`` files, keeps a light-weight summary of each
session (totals plus per-request arrays for the first KMAX requests) and drops
everything else, so a whole week fits comfortably in memory.
"""
import datetime as dt
import io
import json
import sys

import pandas as pd

from .. import config as cfg
from .synthetic import make_synthetic


def summarize_session(s):
    """Reduce one raw session dict to a flat record (first cfg.KMAX requests kept)."""
    reqs = s.get("requests") or []
    n = len(reqs)
    if n == 0:
        return None

    def pt(r, key):
        return (r.get("provider_tokens") or {}).get(key) or 0

    first = reqs[:cfg.KMAX]
    return {
        "session_id": s.get("session_id"),
        "user_id": s.get("user_id"),
        "harness": s.get("harness"),
        "day": s.get("day"),
        "start_ms": s.get("start_ms"),
        "end_ms": s.get("end_ms"),
        "n_requests": n,
        "total_completion": sum(pt(r, "completion") for r in reqs),
        "total_prompt": sum(pt(r, "prompt") for r in reqs),
        "n_tool_calls": sum(len(r.get("tool_calls") or []) for r in reqs),
        "model0": reqs[0].get("model"),
        "a_completion": [pt(r, "completion") for r in first],
        "a_prompt": [pt(r, "prompt") for r in first],
        "a_cached": [pt(r, "cached") for r in first],
        "a_ttft": [r.get("ttft_ms") or 0 for r in first],
        "a_start": [r.get("start_ms") or 0 for r in first],
        "a_ntools": [len(r.get("tool_calls") or []) for r in first],
        "a_nerr": [sum(1 for c in (r.get("tool_calls") or [])
                       if isinstance(c, dict) and c.get("outcome") == "error")
                   for r in first],
        "tooldefs0": (reqs[0].get("role_tokens") or {}).get("tool_definitions", 0),
        "system0": (reqs[0].get("role_tokens") or {}).get("system", 0),
        "n_tooldefs0": len(reqs[0].get("tool_definition_names") or []),
    }


def iter_week_sessions(week, max_sessions=None):
    """Yield raw session dicts for one week by streaming the .jsonl.zst files."""
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    start = dt.date.fromisoformat(week.split("_")[0])
    count = 0
    for i in range(7):
        day = (start + dt.timedelta(days=i)).isoformat()
        fn = f"release/{week}/traces/{day}.jsonl.zst"
        try:
            path = hf_hub_download(cfg.REPO, fn, repo_type="dataset")
        except Exception as e:
            print(f"  [skip] {fn}: {type(e).__name__}")
            continue
        print(f"  reading {fn}")
        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if max_sessions and count >= max_sessions:
                        return


def load_data(args):
    recs, bounds = [], []
    if args.synthetic:
        print("Using SYNTHETIC data (pipeline test only; results are meaningless).")
        for s in make_synthetic():
            r = summarize_session(s)
            if r:
                recs.append(r)
        ws = dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc)
        bounds.append((int(ws.timestamp() * 1000), int((ws + dt.timedelta(days=7)).timestamp() * 1000)))
    else:
        for week in args.weeks:
            print(f"Loading week {week} ...")
            ws = dt.datetime.fromisoformat(week.split("_")[0]).replace(tzinfo=dt.timezone.utc)
            bounds.append((int(ws.timestamp() * 1000), int((ws + dt.timedelta(days=7)).timestamp() * 1000)))
            for s in iter_week_sessions(week, args.max_sessions):
                r = summarize_session(s)
                if r:
                    r["week"] = week
                    recs.append(r)
    df = pd.DataFrame(recs)
    if df.empty:
        sys.exit("No sessions loaded. Check network access / week names.")
    if "week" not in df:
        df["week"] = "synthetic"
    return df, bounds
