"""The Parquet loader must agree with the raw loader and fail loudly on a schema mismatch."""
import argparse

import numpy as np
import pandas as pd
import pytest

from agent_session_forecast import config as cfg
from agent_session_forecast.data import loading
from agent_session_forecast.data.synthetic import make_synthetic, synthetic_to_parquet


def test_all_weeks_are_the_16_release_weeks():
    w = cfg.all_weeks()
    assert len(w) == 16 and w[0] == "2026-05-17_2026-05-23" and w[-1] == "2026-08-30_2026-09-05"


def test_parquet_records_equal_raw_records(tmp_path):
    pytest.importorskip("pyarrow")
    S = make_synthetic(n_users=15, n_sessions=200)
    synthetic_to_parquet(S, str(tmp_path))
    raw = pd.DataFrame([loading.summarize_session(s) for s in S])
    args = argparse.Namespace(weeks=["2026-08-30_2026-09-05"], synthetic=False, source="parquet",
                              parquet_dir=str(tmp_path), max_sessions=None)
    pq, _ = loading.load_data(args)
    pq = pq.set_index("session_id").loc[raw.session_id].reset_index()
    for c in raw.columns:
        assert all(np.array_equal(np.array(a, dtype=object), np.array(b, dtype=object))
                   for a, b in zip(raw[c], pq[c])), f"column {c} differs between raw and parquet loaders"


def test_missing_required_column_is_a_loud_error():
    with pytest.raises(RuntimeError, match="could not find required column"):
        loading._resolve("requests", ["session_id", "start_ms", "something_else"])


def test_resolve_accepts_flat_and_dotted_names():
    m = loading._resolve("requests", ["session_id", "start_ms", "provider_tokens.prompt",
                                      "provider_tokens_completion"])
    assert m["prompt"] == "provider_tokens.prompt" and m["completion"] == "provider_tokens_completion"
