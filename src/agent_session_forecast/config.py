"""Shared settings in one place.

Other modules read these as ``cfg.NAME`` at call time (not ``from config import
NAME``), so ``apply_fast_mode()`` can change them after import.
"""
import numpy as np

REPO = "harvardMadsys/freeinference_agentic_trace"   # Hugging Face dataset

K_LIST = [1, 2, 3, 5, 8, 12, 20]   # "how many requests have we seen so far?"
KMAX = 20                           # requests per session kept from the raw trace (must be >= max(K_LIST))
ALPHA = 0.10                        # conformal miscoverage -> 90% intervals

BOOT = 300          # bootstrap reps for per-model MAE CIs
PAIR_BOOT = 1000    # bootstrap reps for paired comparisons
CONF_REPS = 20      # random account re-splits for the conformal comparison

RNG = np.random.default_rng(0)      # one seeded generator => reproducible runs

MODELS = ["baseline_global", "baseline_harness", "gbm", "gbm_hist"]
DESIGNS = ["random", "grouped", "temporal"]
CONF_METHODS = ["split", "mondrian", "normalized", "cqr", "random_calib"]

# Category -> integer code maps, filled by features.set_category_maps(df) so the
# codes stay identical across every k.
HARN, MOD = {}, {}


def apply_fast_mode():
    """Fewer k values and bootstrap reps: a quick pipeline check, not for results."""
    global K_LIST, BOOT, PAIR_BOOT, CONF_REPS
    K_LIST = [1, 3, 5, 12]
    BOOT, PAIR_BOOT, CONF_REPS = 100, 300, 6
