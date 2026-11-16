"""Command-line entry point: ``session-forecast`` or ``python -m agent_session_forecast``."""
import argparse

from .pipeline import run


def build_parser():
    ap = argparse.ArgumentParser(
        prog="session-forecast",
        description="How well can we predict an agent session's length from its first k requests, "
                    "and can we trust the answer on brand-new accounts?")
    ap.add_argument("--weeks", nargs="+", default=["2026-08-30_2026-09-05"],
                    help="week folder name(s) in the Hugging Face dataset, e.g. 2026-08-30_2026-09-05; "
                         "or the single word 'all' for all 16 weeks")
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="cap sessions per week (handy for quick tests)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use fake data in the same schema (no network needed)")
    ap.add_argument("--fast", action="store_true",
                    help="fewer k values and bootstrap reps: a quick pipeline check, not for results")
    ap.add_argument("--drop-censored", action="store_true",
                    help="drop sessions that look cut off at a week boundary (sensitivity run)")
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    ap.add_argument("--source", choices=["raw", "parquet"], default="raw",
                    help="read the raw .jsonl.zst traces (default) or the flattened Parquet tables")
    ap.add_argument("--parquet-dir", default=None,
                    help="folder with already downloaded sessions/requests/tool_calls parquet files")
    ap.add_argument("--feature-set", choices=["base", "rich"], default="base",
                    help="'rich' adds trigger/mutation/tool-result/latency/sub-agent features (default: base)")
    g = ap.add_argument_group("optional extra analyses (each writes to <out>/extras/<name>/)")
    g.add_argument("--hazard", action="store_true", help="landmark discrete-time hazard / survival model")
    g.add_argument("--hierarchical", action="store_true", help="mixed-effects model with partial pooling")
    g.add_argument("--distributional", action="store_true", help="quantile / distributional forecasts")
    g.add_argument("--group-conformal", dest="group_conformal", action="store_true",
                   help="group-aware conformal methods (LOAO, CV+, weighted)")
    g.add_argument("--sequence", action="store_true", help="GRU / n-gram models on the request stream")
    g.add_argument("--process", action="store_true", help="process mining of the tool-call states")
    g.add_argument("--shift", action="store_true", help="domain-shift methods (weighting, group DRO, ...)")
    g.add_argument("--decision", action="store_true", help="KV-cache retention cost evaluation")
    g.add_argument("--mixture", action="store_true", help="latent-class mixture-of-experts")
    g.add_argument("--all-extras", dest="all_extras", action="store_true", help="turn on every extra above")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
