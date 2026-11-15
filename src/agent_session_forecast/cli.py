"""Command-line entry point: ``session-forecast`` or ``python -m agent_session_forecast``."""
import argparse

from .pipeline import run


def build_parser():
    ap = argparse.ArgumentParser(
        prog="session-forecast",
        description="How well can we predict an agent session's length from its first k requests, "
                    "and can we trust the answer on brand-new accounts?")
    ap.add_argument("--weeks", nargs="+", default=["2026-08-30_2026-09-05"],
                    help="week folder name(s) in the Hugging Face dataset, e.g. 2026-08-30_2026-09-05")
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="cap sessions per week (handy for quick tests)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use fake data in the same schema (no network needed)")
    ap.add_argument("--fast", action="store_true",
                    help="fewer k values and bootstrap reps: a quick pipeline check, not for results")
    ap.add_argument("--drop-censored", action="store_true",
                    help="drop sessions that look cut off at a week boundary (sensitivity run)")
    ap.add_argument("--out", default="results", help="output directory (default: results)")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
