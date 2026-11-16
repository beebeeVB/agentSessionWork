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

## 11. Optional extra analyses: overview

Sections 11-19 are **optional** (`--hazard`, `--hierarchical`, `--distributional`, `--group-conformal`,
`--sequence`, `--process`, `--shift`, `--decision`, `--mixture`, or `--all-extras`). None of them changes the
results of sections 1-10. Each runs on the same folds as the main experiment, is compared **at the same k** with the
same paired account-level bootstrap, and writes to `<out>/extras/<name>/`. They use `cfg.EXTRA_KS` (default
`[1, 3, 5, 12]`) because they are heavier than the main experiment. Hyper-parameters are fixed, not tuned to win: a
tie with the plain GBM is an acceptable, publishable result. [`evaluation/compare.py`]

`--feature-set rich` adds seven features computed from the first k requests only: share of steps triggered by the
user, number of prompt mutations other than `append` (e.g. context compaction), number of length-truncated
outputs, total tool result tokens, mean tool latency, number of sub-agent spawns, and number of distinct tool
categories. The default (`base`) reproduces the original features. [`features.py`]

## 12. Hazard / survival model (`--hazard`)

Instead of regressing the length, model *how long the session keeps going*. At landmark k (session still alive) the
hazard h(j) is the probability it ends exactly j requests later given it lasted j-1 more. Person-period rows turn this
into classification; gradient boosting and logistic regression are both fitted. Survival S(j) = prod(1 - h(i));
the **median remaining length** (smallest j with S(j) <= 0.5) is the point forecast compared with the other models.
Right-censored sessions contribute hazard terms for the steps seen and no terminal event.
Outputs: forecast and paired skill vs baselines, survival-curve calibration, AUC / Brier skill for "lasts more than h more
requests" at h = 5, 10, 20, and a calibration table. *Limitation*: the horizon is capped at `HAZARD_J` (40); a median
beyond it is reported as J + 1. [`evaluation/hazard.py`]

## 13. Hierarchical model (`--hierarchical`)

`y = harness mean + account random intercept + noise`, with variance components estimated by one-way ANOVA moments
(deterministic) and, if `statsmodels` is installed, cross-checked with a REML `MixedLM`. Seen accounts are shrunk toward
the harness mean by `B = n s_u^2 / (n s_u^2 + s_e^2)`; **unseen accounts get exactly the population prediction**. Two
predictors are scored: `hier_eb` (pure mixed model) and `gbm_re` (GBM plus a shrunken account offset from out-of-fold
residuals), next to the older `gbm_hist`. Outputs include the variance components, the model ICC vs the raw ICC from
profiling, and a per-account table of raw vs shrunken effects. *Not included*: a random slope on k (each landmark is its own
model, so k is constant within it). [`evaluation/hierarchical.py`]

## 14. Distributional forecasts (`--distributional`)

Predicts a quantile for each level in `cfg.QUANTILES` and scores the whole distribution: pinball loss, grid CRPS
(2 x mean pinball over the levels), interval score of the central interval, central coverage, and quantile calibration.
Methods: GBM mean + empirical OOF-residual quantiles (the baseline), per-level quantile GBMs, lognormal, and
heteroscedastic lognormal. Paired skill is relative to the residual-quantile baseline. *Limitation*: the default grid is
coarse (5 levels), so CRPS is an approximation useful for comparing methods, not as an absolute number.
[`evaluation/distributional.py`]

## 15. Group-aware conformal (`--group-conformal`)

Adds three methods to the conformal comparison of section 8, evaluated **on the same account re-splits**:
`loao` (leave-accounts-out calibration residuals), `cv_plus_group` (CV+ with account folds) and `weighted`
(split-conformal reweighted by a classifier-estimated calibration-vs-test density ratio, weights clipped to
[0.05, 20]). `loao` and `cv_plus_group` use train + calibration accounts together, so they see more data than `split`;
read the comparison with that in mind. `random_calib` stays as the known-bad control. [`uncertainty/conformal.py`]

## 16. Sequence models (`--sequence`)

A small GRU over the per-request feature sequence (plus the static features), or an MLP on the flattened sequence if
PyTorch is not installed (the output then says `seq_mlp`), and a bag of tool-signature n-grams fed to the GBM
(`tool_ngram_gbm`). Fixed hyper-parameters, seeded, not tuned. At a few thousand sessions a tie with the GBM is the expected
outcome. [`evaluation/sequence.py`]

## 17. Process mining (`--process`)

Each request is mapped to one state (`talk, explore, edit, run, subagent, error, other`) by keyword matching on the tool
category (unknown categories go to `other`; priority error > subagent > edit > run > explore > other > talk). From the first
20 requests we estimate the transition matrix, P(session ends | state) (excluding right-censored sessions) and the mean
dwell per state (truncated at the window), and test state features (`gbm_states`). It is a first-order summary, not a fitted
generative model. *Check the keyword lists against the real tool categories* before interpreting the states.
[`evaluation/process.py`]

## 18. Domain-shift methods (`--shift`)

> A smaller grouped-minus-random gap is not an improvement: it also appears when a method makes random-split accuracy worse. `shift_gap.csv` therefore includes `newacct_skill_vs_gbm` (paired, account-grouped); judge methods by that column.


Does the account-grouped penalty shrink if the GBM is trained to generalise? Methods: account-balanced weights (1/n),
square-root weights, group DRO (exponentiated-gradient reweighting of accounts with out-of-fold losses), and dropping the
quarter of features that most separate accounts (ANOVA F). The headline is the **change in the grouped-minus-random gap vs the
plain GBM**, with a paired account-level bootstrap. `--fast` uses weighted resampling instead of exact sample weights (the
GBM's absolute-error loss makes exact weights several times slower); real runs use exact weights.
[`evaluation/shift.py`]

## 19. Decision-focused evaluation (`--decision`) and latent classes (`--mixture`)

**Decision.** After request k, how long should the session's prompt prefix stay cached? Cost (in prefill-token units) is
`HOLD_RATE * P * seconds_held + P * [miss]`, with P the prompt tokens of request k, so large sessions weigh more. Policies:
always evict, best fixed TTL, harness continuation rate, a GBM continuation classifier (the discrete-time hazard at step k),
and an oracle (lower bound). Account-grouped CV, paired bootstrap of the saving vs the best fixed TTL. **The cost constants
(`HOLD_RATE = 0.01`, break-even gap 100 s; `TTL_GRID_S`) are assumptions, not measurements**; vary them. Only the first 20
requests are stored, so k must be below 20. [`evaluation/decision.py`]

**Mixture.** Inside each training fold: standardise, reduce to 8 principal components, pick the number of Gaussian-mixture
classes by BIC, train one GBM expert per class (hard assignment, classes with < 60 sessions fall back to the global model),
predict with the posterior-weighted average. Class profiles (size, length quantiles, harness mix) are written for
interpretation. [`evaluation/mixture.py`]

## 19b. Cost-model sensitivity (inside `--decision`)

The retention cost model has two assumed numbers: the price of holding a prefix per second
(`HOLD_RATE`) and, implicitly, the price of a miss (one prefix recompute). Their ratio is the
break-even gap 1/HOLD_RATE. `decision_sensitivity.csv` re-scores every policy for break-even gaps of
20, 50, 100, 200 and 500 s; the forecasting models are fitted once per fold (they do not depend on the
cost model), only the decision step is repeated. The summary prints, for the hazard policy, in how many
(k, cost-model) cells it is significantly cheaper / costlier than the best fixed TTL. Report the whole
sweep, not the single default.

## 19c. Robustness to heavy accounts (`--robustness`)

Grouped analyses rest on the effective number of accounts (1/HHI), not the raw count. This analysis
re-runs the whole cross-validation (random and account-grouped) after dropping the 1, 3 and 10 heaviest
accounts, and after capping every account at 50 sessions (fixed random subsample), and reports the
new-account gap and the GBM's new-account skill with paired cluster-bootstrap CIs, plus
`robustness_stability.csv` (largest change from the full data, and whether the CIs still exclude 0).

## 20. Data sources

`--source raw` (default) streams the weekly `.jsonl.zst` files (about 108 GB decompressed for all 16 weeks). `--source parquet`
reads the flattened `sessions` / `requests` / `tool_calls` Parquet tables (about 9 GB), from Hugging Face or from a local folder
(`--parquet-dir`). Both produce identical per-session records (a test checks this on synthetic data). **The Parquet column
names are discovered at run time and the mapping is printed; if a required column is missing the loader stops and lists the
columns it saw.** It has not been run against the real files, so check the printed mapping on first use. `--weeks all` expands
to the 16 weekly folders `2026-05-17_2026-05-23 ... 2026-08-30_2026-09-05`; a week that fails to download is skipped with a warning.
[`data/loading.py`]

## Known simplifications

- One seed, one family of model (gradient boosting). The conclusions are about this setup, not about every possible model.
- Hyper-parameters are fixed, not tuned per k, so every model gets the same modest budget.
- The temporal design is a single split, so read it as supporting evidence.
- Sub-agent requests nested in tool calls are not counted toward the parent session's length.
- The extra analyses (11-19) use fixed, modest hyper-parameters and a small set of k values; none was tuned to produce a win.
