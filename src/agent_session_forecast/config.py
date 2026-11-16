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

# ---- settings for the optional extra analyses (see docs/METHODS.md sections 11-19) ----
EXTRA_KS = [1, 3, 5, 12]    # k values used by the extra analyses (kept small: they are heavier)
HAZARD_J = 40               # horizon (requests ahead) of the landmark hazard model
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]   # quantile grid for distributional scores
SEQ_EPOCHS = 25             # training epochs for the GRU (ignored by the MLP fallback)
WEIGHT_MODE = "exact"       # "exact": GBM sample_weight (slow with absolute-error loss); "resample": weighted bootstrap (fast, approximate; used by --fast)
DRO_ROUNDS = 3              # group-DRO reweighting rounds
MIX_CLASSES = [2, 3, 4]     # candidate numbers of latent classes (chosen by BIC)
FEATURE_SET = "base"        # "base" = original features; "rich" = + trigger/mutation/tool features

# Decision-focused (KV-cache retention) cost model. Everything is in units of "one
# prefill token recomputed". Holding a cached prefix costs HOLD_RATE per token per second;
# a miss costs 1 per prompt token. Break-even gap is therefore 1 / HOLD_RATE seconds.
HOLD_RATE = 0.01
TTL_GRID_S = [0, 10, 30, 60, 120, 300, 600, 1800]

# The 16 weekly folders of the public release (Sunday-Saturday).
FIRST_WEEK_START = "2026-05-17"
N_WEEKS = 16

MODELS = ["baseline_global", "baseline_harness", "gbm", "gbm_hist"]
DESIGNS = ["random", "grouped", "temporal"]
CONF_METHODS = ["split", "mondrian", "normalized", "cqr", "random_calib"]
GROUP_CONF_METHODS = ["loao", "cv_plus_group", "weighted"]   # group-aware methods (--group-conformal)

# Category -> integer code maps, filled by features.set_category_maps(df) so the
# codes stay identical across every k.
HARN, MOD = {}, {}


def all_weeks():
    """The 16 weekly folder names of the release, e.g. '2026-05-17_2026-05-23'."""
    import datetime as dt
    d0 = dt.date.fromisoformat(FIRST_WEEK_START)
    out = []
    for i in range(N_WEEKS):
        a = d0 + dt.timedelta(days=7 * i)
        out.append(f"{a.isoformat()}_{(a + dt.timedelta(days=6)).isoformat()}")
    return out


def apply_fast_mode():
    """Fewer k values and bootstrap reps: a quick pipeline check, not for results."""
    global K_LIST, BOOT, PAIR_BOOT, CONF_REPS, EXTRA_KS, HAZARD_J, QUANTILES, SEQ_EPOCHS
    global DRO_ROUNDS, MIX_CLASSES, WEIGHT_MODE
    K_LIST = [1, 3, 5, 12]
    BOOT, PAIR_BOOT, CONF_REPS = 100, 300, 6
    EXTRA_KS = [3, 12]
    HAZARD_J = 25
    QUANTILES = [0.05, 0.5, 0.95]
    SEQ_EPOCHS = 8
    DRO_ROUNDS = 2
    WEIGHT_MODE = "resample"
    MIX_CLASSES = [2, 3]
