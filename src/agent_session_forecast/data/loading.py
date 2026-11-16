"""Reading the real trace from Hugging Face.

Two sources are supported:

* ``raw``     - stream the weekly ``.jsonl.zst`` files (about 108 GB decompressed for all 16 weeks);
* ``parquet`` - read the flattened ``sessions`` / ``requests`` / ``tool_calls`` Parquet tables
                (about 9 GB), either from Hugging Face or from a local folder (``--parquet-dir``).

Both produce the same light-weight record per session: totals plus per-request arrays for the
first KMAX requests. Everything else is dropped so a whole week fits comfortably in memory.

NOTE ON THE PARQUET PATH: it was written from the dataset card, not from the files themselves.
Column names are therefore *discovered* (see ``_resolve``) and the mapping is printed at run
time. If a required column cannot be found the loader stops and lists the columns it saw, so a
schema mismatch is loud, never silent.
"""
import datetime as dt
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from .. import config as cfg
from .synthetic import make_synthetic


# --------------------------------------------------------------------------------------
# one raw session dict -> one flat record
# --------------------------------------------------------------------------------------
def _is_user_step(r):
    return 1 if str(r.get("step_trigger") or "").lower() == "user" else 0


def _is_mutation(r):
    t = (r.get("mutation") or {}).get("transition") if isinstance(r.get("mutation"), dict) else None
    return 0 if t in (None, "", "append") else 1


def _tc(r):
    return [c for c in (r.get("tool_calls") or []) if isinstance(c, dict)]


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
        "a_end": [r.get("end_ms") or 0 for r in first],
        "a_ntools": [len(r.get("tool_calls") or []) for r in first],
        "a_nerr": [sum(1 for c in (r.get("tool_calls") or [])
                       if isinstance(c, dict) and c.get("outcome") == "error")
                   for r in first],
        # ---- richer per-request fields (used by --feature-set rich and the extra analyses) ----
        "a_user": [_is_user_step(r) for r in first],
        "a_mut": [_is_mutation(r) for r in first],
        "a_lenfin": [1 if str(r.get("finish_reason") or "").lower() in ("length", "max_tokens") else 0
                     for r in first],
        "a_restok": [sum((c.get("result_tokens") or 0) for c in _tc(r)) for r in first],
        "a_lat": [sum((c.get("latency_ms") or 0) for c in _tc(r)) for r in first],
        "a_nsub": [sum(1 for c in _tc(r) if c.get("subagent")) for r in first],
        "a_cats": ["|".join(sorted(str(c.get("category") or c.get("name") or "unknown") for c in _tc(r)))
                   for r in first],
        "tooldefs0": (reqs[0].get("role_tokens") or {}).get("tool_definitions", 0),
        "system0": (reqs[0].get("role_tokens") or {}).get("system", 0),
        "n_tooldefs0": len(reqs[0].get("tool_definition_names") or []),
    }


# --------------------------------------------------------------------------------------
# raw .jsonl.zst source
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Parquet source (column names are discovered, never assumed)
# --------------------------------------------------------------------------------------
def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


_CANDIDATES = {
    "sessions": {
        "session_id": ["session_id", "session"],
        "user_id": ["user_id", "account_id", "user"],
        "harness": ["harness"],
        "day": ["day", "date"],
        "start_ms": ["start_ms", "session_start_ms"],
        "end_ms": ["end_ms", "session_end_ms"],
        "week": ["week"],
    },
    "requests": {
        "session_id": ["session_id", "session"],
        "step": ["step", "request_index", "req_index", "request_step", "index"],
        "start_ms": ["start_ms", "request_start_ms"],
        "end_ms": ["end_ms", "request_end_ms"],
        "ttft_ms": ["ttft_ms", "ttft"],
        "model": ["model"],
        "prompt": ["provider_tokens_prompt", "prompt_tokens", "provider_prompt", "prompt"],
        "completion": ["provider_tokens_completion", "completion_tokens", "provider_completion", "completion"],
        "cached": ["provider_tokens_cached", "cached_tokens", "provider_cached", "cached"],
        "system": ["role_tokens_system", "system_tokens"],
        "tooldefs": ["role_tokens_tool_definitions", "tool_definition_tokens", "tool_definitions_tokens"],
        "n_tooldefs": ["n_tool_definitions", "tool_definition_count", "n_tool_definition_names"],
        "trigger": ["step_trigger", "trigger"],
        "mutation": ["mutation_transition", "transition"],
        "finish": ["finish_reason"],
    },
    "tools": {
        "session_id": ["session_id", "session"],
        "step": ["step", "request_step", "request_index", "req_step", "req_index"],
        "name": ["name", "tool_name"],
        "category": ["category", "tool_category"],
        "outcome": ["outcome"],
        "result_tokens": ["result_tokens"],
        "latency_ms": ["latency_ms"],
        "subagent": ["subagent", "is_subagent", "subagent_session_id"],
    },
}
_REQUIRED = {"sessions": ["session_id"], "requests": ["session_id", "start_ms", "prompt", "completion"],
             "tools": []}


def _flat_schema_names(path):
    """Column names of a parquet file with struct columns expanded to 'parent.child'."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    names = []

    def walk(prefix, typ):
        if pa.types.is_struct(typ):
            for i in range(typ.num_fields):
                f = typ.field(i)
                walk(f"{prefix}.{f.name}" if prefix else f.name, f.type)
        else:
            names.append(prefix)

    sch = pq.read_schema(path)
    for f in sch:
        walk(f.name, f.type)
    return names


def _resolve(kind, names):
    """Map logical field -> actual column name. Exact normalized match only."""
    norm = {}
    for n in names:
        norm.setdefault(_norm(n), n)
    mapping = {}
    for logical, cands in _CANDIDATES[kind].items():
        for c in cands:
            if c in norm:
                mapping[logical] = norm[c]
                break
    missing = [f for f in _REQUIRED[kind] if f not in mapping]
    if missing:
        raise RuntimeError(f"[{kind} table] could not find required column(s) {missing}.\n"
                           f"Columns seen: {sorted(names)}\n"
                           f"Fix: add the right name to _CANDIDATES in data/loading.py.")
    return mapping


def _read_table(path, kind):
    """Read only the needed columns of one parquet file; returns (DataFrame, mapping)."""
    import pyarrow.parquet as pq
    names = _flat_schema_names(path)
    mapping = _resolve(kind, names)
    want = sorted(set(mapping.values()))
    # a list column holding tool-definition names: keep its length only
    extra = [n for n in names if _norm(n) == "tool_definition_names"] if kind == "requests" else []
    # pyarrow returns dotted nested selections under their CHILD name only, so read the whole
    # parent struct column (small: provider_tokens, role_tokens, mutation) and flatten it.
    top = sorted({w.split(".")[0] for w in want} | {e.split(".")[0] for e in extra})
    t = pq.read_table(path, columns=top).flatten()
    df = t.to_pandas()
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise RuntimeError(f"[{kind}] columns vanished while reading: {missing}; saw {list(df.columns)}")
    df = df.rename(columns={v: k for k, v in mapping.items()})
    if extra:
        col = extra[0].split(".")[-1]
        if col in df.columns and "n_tooldefs" not in df.columns:
            df["n_tooldefs"] = df[col].map(lambda v: 0 if v is None else len(v))
        df = df.drop(columns=[c for c in [col] if c in df.columns])
    print(f"  [{kind}] {os.path.basename(path)}: {len(df):,} rows; columns mapped: "
          f"{ {k: v for k, v in mapping.items()} }")
    miss = [k for k in _CANDIDATES[kind] if k not in mapping]
    if miss:
        print(f"  [{kind}] NOT FOUND (filled with defaults): {miss}")
    return df, mapping


def _classify_files(files):
    """Pick the sessions / requests / tool-call parquet files from a list of paths."""
    out = {"sessions": [], "requests": [], "tools": []}
    for f in files:
        b = os.path.basename(f).lower()
        if not b.endswith(".parquet"):
            continue
        if "tool" in b:
            out["tools"].append(f)
        elif "request" in b:
            out["requests"].append(f)
        elif "session" in b:
            out["sessions"].append(f)
    return out


def _records_from_tables(sess, req, tool):
    """Build summarize_session-style records from flat tables (one week or everything)."""
    req = req.copy()
    if "step" not in req.columns:
        req["step"] = req.groupby("session_id").cumcount()
    req = req.sort_values(["session_id", "step"]).reset_index(drop=True)
    for c in ["start_ms", "end_ms", "ttft_ms", "prompt", "completion", "cached", "system",
              "tooldefs", "n_tooldefs"]:
        if c not in req.columns:
            req[c] = 0
        req[c] = pd.to_numeric(req[c], errors="coerce").fillna(0)
    for c in ["trigger", "mutation", "finish", "model"]:
        if c not in req.columns:
            req[c] = None

    g = req.groupby("session_id", sort=False)
    n_req = g.size()
    tot_c, tot_p = g["completion"].sum(), g["prompt"].sum()
    req["_pos"] = g.cumcount()
    first = req[req["_pos"] < cfg.KMAX]

    # tool calls attach to (session_id, step)
    tool_by_req, n_tool_total = {}, {}
    if tool is not None and len(tool):
        t = tool.copy()
        if "step" not in t.columns:
            print("  [tools] no step column: per-request tool features will be zero")
        else:
            n_tool_total = t.groupby("session_id").size().to_dict()
            for c in ["result_tokens", "latency_ms"]:
                t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0) if c in t.columns else 0
            for c in ["category", "name", "outcome", "subagent"]:
                if c not in t.columns:
                    t[c] = None
            t["_sub"] = t["subagent"].map(lambda v: 0 if v in (None, False, 0, "", np.nan) or
                                          (isinstance(v, float) and np.isnan(v)) else 1)
            t["_cat"] = t["category"].fillna(t["name"]).fillna("unknown").astype(str)
            t["_err"] = (t["outcome"] == "error").astype(int)
            fs = first[["session_id", "step"]].drop_duplicates()
            t = t.merge(fs, on=["session_id", "step"], how="inner")
            for (sid, stp), grp in t.groupby(["session_id", "step"], sort=False):
                tool_by_req[(sid, stp)] = (len(grp), int(grp["_err"].sum()), float(grp["result_tokens"].sum()),
                                           float(grp["latency_ms"].sum()), int(grp["_sub"].sum()),
                                           "|".join(sorted(grp["_cat"])))

    sess = sess.set_index("session_id") if sess is not None and len(sess) else None
    recs = []
    for sid, f in first.groupby("session_id", sort=False):
        tb = [tool_by_req.get((sid, st), (0, 0, 0.0, 0.0, 0, "")) for st in f["step"].values]
        meta = sess.loc[sid] if sess is not None and sid in sess.index else None
        start = meta["start_ms"] if meta is not None and "start_ms" in meta and pd.notna(meta["start_ms"]) \
            else float(f["start_ms"].iloc[0])
        end = meta["end_ms"] if meta is not None and "end_ms" in meta and pd.notna(meta["end_ms"]) \
            else float(f["end_ms"].max())
        r0 = f.iloc[0]
        recs.append({
            "session_id": sid,
            "user_id": (meta["user_id"] if meta is not None and "user_id" in meta else None),
            "harness": (meta["harness"] if meta is not None and "harness" in meta else None),
            "day": (meta["day"] if meta is not None and "day" in meta else None),
            "start_ms": start, "end_ms": end,
            "n_requests": int(n_req[sid]),
            "total_completion": float(tot_c[sid]), "total_prompt": float(tot_p[sid]),
            "n_tool_calls": int(n_tool_total.get(sid, sum(x[0] for x in tb))),
            "model0": r0["model"],
            "a_completion": f["completion"].tolist(), "a_prompt": f["prompt"].tolist(),
            "a_cached": f["cached"].tolist(), "a_ttft": f["ttft_ms"].tolist(),
            "a_start": f["start_ms"].tolist(), "a_end": f["end_ms"].tolist(),
            "a_ntools": [x[0] for x in tb], "a_nerr": [x[1] for x in tb],
            "a_user": [1 if str(v).lower() == "user" else 0 for v in f["trigger"]],
            "a_mut": [0 if (v is None or (isinstance(v, float) and np.isnan(v)) or v in ("", "append")) else 1
                      for v in f["mutation"]],
            "a_lenfin": [1 if str(v).lower() in ("length", "max_tokens") else 0 for v in f["finish"]],
            "a_restok": [x[2] for x in tb], "a_lat": [x[3] for x in tb], "a_nsub": [x[4] for x in tb],
            "a_cats": [x[5] for x in tb],
            "tooldefs0": float(r0["tooldefs"]), "system0": float(r0["system"]),
            "n_tooldefs0": int(r0["n_tooldefs"]),
        })
    return recs


def _load_parquet(args, bounds):
    """Load via Parquet tables (local folder or Hugging Face). Returns list of records."""
    weeks = args.weeks
    local = getattr(args, "parquet_dir", None)
    if local:
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(local) for f in fs]
        fetch = lambda p: p
    else:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(cfg.REPO, repo_type="dataset")
        fetch = lambda p: hf_hub_download(cfg.REPO, p, repo_type="dataset")
    groups = _classify_files(files)
    if not groups["requests"] or not groups["sessions"]:
        sys.exit("Parquet source: could not find sessions/requests tables. Files seen (first 30):\n  " +
                 "\n  ".join(sorted(files)[:30]))
    recs = []
    for week in weeks:
        pick = {k: [f for f in v if week in f] or v for k, v in groups.items()}
        print(f"Loading week {week} from parquet "
              f"({len(pick['sessions'])} session / {len(pick['requests'])} request / "
              f"{len(pick['tools'])} tool-call file(s)) ...")
        sess = pd.concat([_read_table(fetch(p), "sessions")[0] for p in pick["sessions"]], ignore_index=True)
        ws, we = bounds[weeks.index(week)]
        if "week" in sess.columns and (sess["week"].astype(str) == week).any():
            sess = sess[sess["week"].astype(str) == week]
        elif "start_ms" in sess.columns:
            sess = sess[(sess["start_ms"] >= ws) & (sess["start_ms"] < we)]
        if getattr(args, "max_sessions", None):
            sess = sess.head(args.max_sessions)
        keep = set(sess["session_id"])
        req = pd.concat([_read_table(fetch(p), "requests")[0] for p in pick["requests"]], ignore_index=True)
        req = req[req["session_id"].isin(keep)]
        tool = None
        if pick["tools"]:
            tool = pd.concat([_read_table(fetch(p), "tools")[0] for p in pick["tools"]], ignore_index=True)
            tool = tool[tool["session_id"].isin(keep)]
        part = _records_from_tables(sess, req, tool)
        for r in part:
            r["week"] = week
        print(f"  -> {len(part):,} sessions")
        recs.extend(part)
    return recs


# --------------------------------------------------------------------------------------
def _week_bounds(week):
    ws = dt.datetime.fromisoformat(week.split("_")[0]).replace(tzinfo=dt.timezone.utc)
    return int(ws.timestamp() * 1000), int((ws + dt.timedelta(days=7)).timestamp() * 1000)


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
        weeks = list(args.weeks)
        if len(weeks) == 1 and weeks[0].lower() == "all":
            weeks = cfg.all_weeks()
        args.weeks = weeks
        bounds = [_week_bounds(w) for w in weeks]
        source = getattr(args, "source", "raw")
        if source == "parquet" or getattr(args, "parquet_dir", None):
            recs = _load_parquet(args, bounds)
        else:
            for week in weeks:
                print(f"Loading week {week} ...")
                n0 = len(recs)
                try:
                    for s in iter_week_sessions(week, args.max_sessions):
                        r = summarize_session(s)
                        if r:
                            r["week"] = week
                            recs.append(r)
                except Exception as e:     # one bad week must not lose the others
                    print(f"  [warn] week {week} failed ({type(e).__name__}: {e}); continuing")
                print(f"  -> {len(recs) - n0:,} sessions")
    df = pd.DataFrame(recs)
    if df.empty:
        sys.exit("No sessions loaded. Check network access / week names.")
    if "week" not in df:
        df["week"] = "synthetic"
    return df, bounds
