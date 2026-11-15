# Methods

What every analysis does and, more importantly, **why it's there**. Code locations are in brackets.

## 1. The target and what gets scored

We predict `log(1 + n_requests)`, where `n_requests` is the number of top-level LLM requests in a session.
Lengths are heavy-tailed (a few sessions run to hundreds of requests), so on the raw scale a handful of giants
would decide every metric. On the log scale an error of 0.69 means "off by a factor of about 2".

To predict from the first *k* requests, a session needs at least *k*+1 requests. So for each k we score only sessions
with `n_requests > k`. This has a consequence worth repeating: **as k grows, short sessions drop out and the target
becomes narrower and easier**. A falling error curve is therefore *not* proof that "more observed requests = better model".
The fair comparison is always between models at the *same* k. [`features.py`, `evaluation/forecasting.py`]

## 2. Features: only the first k requests

For each session we build features from the first k requests only: token counts (cumulative, mean, last, prompt
growth), cached-token fraction, time-to-first-token, elapsed time and mean gap between requests, tool-call counts and
error rates, how many tools and how large a system prompt the harness shipped, plus which harness and model it is.

Nothing after request k is ever touched, and there is a test for it (`tests/test_features.py`) that scrambles the
future and checks the features don't move. [`features.py`]

## 3. Three ways to hold out data

| Design | Splits | Why it exists |
|---|---|---|
| Random | 5-fold over sessions | The usual approach; the optimistic one |
| Account-grouped | 5-fold, whole accounts held out | New-user performance |
| Temporal | first 70% of time -> last 30% | "Tomorrow" performance; a single split, so noisier |

Sessions of one account are strongly alike (the **ICC** printed by the profiler is the share of variance explained by
*who the account is*). Random splits let the model learn an account's habits from some sessions and be graded on its
others. Grouped splits don't. The difference between the two is the price of meeting a stranger. [`models.py`, `evaluation/forecasting.py`]

## 4. Models

- **baseline_global**: the median log-length among sessions that survived past k. No features at all.
- **baseline_harness**: the same, per harness (Claude Code, Cline, ...). Both baselines are trained only on sessions with more than k requests, so they already account for survival.
- **gbm**: scikit-learn `HistGradientBoostingRegressor` on all the first-k features.
- **gbm_hist**: the same plus **account history** (below).

If the GBM can't beat the per-harness median, the first-k request details aren't adding much beyond "which tool is this?".
[`models.py`]

### Account history and "history dropout"

`gbm_hist` adds two features: a smoothed average of the account's *other* session lengths (leave-one-out, so a row never
sees its own label) and how many such sessions exist. For a brand-new account both are missing.

If the model only ever trained with history present, it would lean on it and then fall apart for new accounts
(this really happened in an early version, on the synthetic test data: error jumped from 0.76 to 1.03 under grouped evaluation).
So 20% of training rows have their history hidden at random, which teaches the model what to do without it.
Read `gbm_hist` as "returning user" performance. For strangers it should match `gbm`. [`features.add_history`]

## 5. Honest error bars: the cluster bootstrap

Ordinary bootstrap resamples sessions, which pretends they're independent and gives intervals that are too narrow.
Here we resample **accounts** (with all their sessions) and recompute the metric. [`evaluation/bootstrap.py`]

### Paired comparisons

To ask "is model A better than model B?", both models are scored on exactly the same sessions, then accounts are
resampled and the *difference* is recomputed. Because noise from hard sessions cancels, this is far tighter than comparing
two separate intervals. Reported as:

- `diff`: mean absolute error of A minus B, with a 95% interval and a two-sided bootstrap p-value.
- `skill`: `1 - MAE_A / MAE_B`. Positive means A is better; +10% means 10% lower error.

The comparisons run: grouped-minus-random (the price of new accounts), GBM vs per-harness baseline, harness baseline
vs global baseline (does knowing the tool help at all?), and history vs plain GBM. [`evaluation/forecasting.paired_analysis`]

## 6. Will the session be long?

A friendlier question than "how many requests": predict `P(n_requests > T)` for T = 20 and 50. Scored with AUC and
Brier skill against the per-harness base rate. AUC is computed **within each fold and averaged**. Pooling out-of-fold
scores across folds with different base rates can push AUC below 0.5 under grouped CV even when there is no signal
at all (the training mean moves opposite to the test mean). [`evaluation/classification.py`]

## 7. Feature importance on new accounts

Permutation importance: shuffle one feature on held-out *accounts* and see how much the error rises. Averaged over three
random account splits. Features that matter only when the account is known will show up near zero here. [`evaluation/importance.py`]

## 8. Prediction intervals (conformal)

Conformal prediction wraps any model to produce intervals with a coverage guarantee: with a 90% target, about 90% of true
values should land inside. The guarantee needs calibration data that looks like the test data, and that's where
account shift bites. Every method here is tested on **accounts the model never saw**, repeated over many random account
splits. [`uncertainty/conformal.py`]

| Method | Idea | Strength | Weakness |
|---|---|---|---|
| `split` | One width for everybody, from calibration residuals | Simple, guaranteed on average | Same width for easy and hard cases |
| `mondrian` | A separate width per harness | Fixes harnesses that were over/under-covered | Needs enough calibration data per harness |
| `normalized` | Width scaled by a second model's guess of the error size | Adapts to the case | Extra model; can be unstable |
| `cqr` | Learn lower/upper quantiles, then correct them conformally | Adapts to the case, usually tighter | Quantile models can be jumpy |
| `random_calib` | Calibrate on random sessions of *seen* accounts | (the trap) | Looks fine in-house, under-covers on new accounts |

Besides overall coverage we report:

- **worst-harness coverage**: the lowest coverage among harnesses with at least 30 test sessions;
- **10th-percentile account coverage**: coverage for the unluckiest tenth of accounts (with at least 10 test sessions);
- **width factor**: the median multiplicative span of the interval in requests, `exp(upper - lower)`. A 90% interval that is "x20 wide" is correct but may not be useful.

Marginal coverage (the headline number) can look perfect while particular harnesses or accounts are badly under-covered.
Those two columns are there to catch that.

## 9. Adaptive Conformal Inference (ACI)

Everything above assumes the future looks like the calibration set. To test drift: train on the first 40% of the week,
calibrate on the next 20%, then stream the final 40% in time order. *Static* intervals keep the calibration width forever.
ACI nudges its miscoverage level after every outcome: a miss widens future intervals, a hit tightens them, so long-run coverage
tracks the target even when the distribution moves. The model itself is not retrained during the stream, and the
true length is assumed to be known right after each session (in practice it is only known when the session ends). [`uncertainty/adaptive.py`]

## 10. Profiling and censoring

- **Gini / Lorenz curve / effective number of accounts (1/HHI)**: how concentrated the data is. The effective number is
  what limits grouped analysis; 60 accounts where three dominate behave like far fewer.
- **ICC**: share of length variance explained by account identity.
- **Censoring flags**: sessions are split at week boundaries, so a session ending within two hours of the week's end may have
  been cut short, and one starting within two hours of the start may be a continuation. This is a heuristic
  (assumes week folders are UTC midnight). `--drop-censored` removes them for a sensitivity check. [`profiling.py`]

## Known simplifications

- One seed, one family of model (gradient boosting). The conclusions are about this setup, not about every possible model.
- Hyper-parameters are fixed, not tuned per k, so every model gets the same modest budget.
- The temporal design is a single split, so read it as supporting evidence.
- Sub-agent requests nested in tool calls are not counted toward the parent session's length.
