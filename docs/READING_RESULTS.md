# Reading the results

Everything a run writes to `--out/`, and how to read it.

## Files

| File | What's in it |
|---|---|
| `summary.txt` | The whole story in text. **Start here.** |
| `dashboard.png` | Data profile, forecast error, interval coverage |
| `dashboard2.png` | Paired tests, skill scores, long-session AUC, feature importance, drift |
| `results_forecast.csv` | One row per (k, design, model): MAE with 95% CI, account-macro MAE, median error in requests, R² |
| `results_paired.csv` | Paired comparisons: diff, skill, 95% CIs, bootstrap p-value |
| `results_longsession.csv` | AUC and Brier for "will it exceed T requests?" |
| `results_importance.csv` | Permutation importance on held-out accounts |
| `results_conformal.csv` | Per account-split results for the 5 interval methods |
| `results_harness.csv` | Count, median and mean length per harness |
| `figures/` | One PNG (200 dpi) and one PDF per figure (see "Figures" below); a figure is skipped if its analysis did not run |
| `tables/` | Every table as CSV **and** LaTeX (`.tex`, booktabs style) ready for a paper |
| `extras/<name>/` | CSVs of each optional analysis (`--hazard`, `--hierarchical`, ...) |
| `dashboard3.png` | One page collecting the panels of every extra analysis that ran |
| `run_info/run_info.{json,txt}` | Provenance: command line, git commit, library versions, seed, config values, weeks, session / account / effective-account counts, timings, and any analysis that failed |

## `summary.txt`, block by block

1. **Header lines**: sessions, accounts, length quantiles, Gini, *effective number of accounts*, censoring share, ICC.
   Check the observed start/end range here to confirm the UTC assumption.
2. **Forecast MAE table**: lower is better. Compare columns within one k.
3. **Account-macro MAE**: the same but every account counts equally. If this differs a lot from the main table, a few big accounts are driving the results.
4. **Median error in requests**: the number to quote to a human ("typically off by about 12 requests").
5. **PAIRED blocks**: each line is `k: estimate [95% CI] p=...`.
   - `grouped - random` **positive and CI above 0**: new accounts really are harder.
   - `skill` **CI above 0**: the first model genuinely beats its reference. CI straddling 0: no evidence either way.
6. **Long-session table**: AUC 0.5 is a coin flip; compare `auc_gbm` against `auc_baseline`.
7. **Permutation importance**: features near zero don't help on new accounts.
8. **Conformal table**: target is 90%.
9. **Drift test**: static vs ACI coverage over the streamed sessions.

## `dashboard.png`

| Panel | Look for |
|---|---|
| Session length | Heavy right tail; red sliver = sessions cut off at the week end (small is good) |
| Length by harness | Do harnesses differ? If yes, "harness" is a strong baseline |
| Account concentration | Bowed far below the diagonal = a few accounts dominate |
| Early forecasting error | Solid = GBM, dotted = baseline. Dotted at or below solid means the GBM isn't earning its keep. Red above blue = new accounts are harder. The band is the GBM-grouped 95% CI |
| Conformal methods | Bars should sit near the dashed 90% line. Labels show how wide the interval is (a factor in requests). `random_calib` is the trap |
| Coverage by harness | Bars should hug the dashed 90% line. Mondrian should pull stray harnesses toward it |

## `dashboard2.png`

| Panel | Look for |
|---|---|
| Cost of holding out accounts | Points above 0 with bars not touching 0 = a real penalty for new accounts. Bars crossing 0 = could be noise |
| Skill vs reference | Above 0 = better than the reference. Green (history) high but red/blue (GBM vs harness median) near 0 means almost all the signal is "I recognise this account" |
| Conditional coverage | Bars well under the 90% line mean the *average* looked fine but some harnesses or accounts are badly served |
| Will the session be long? | AUC above 0.5 with CI above 0.5 = real signal. Grouped below random is the price of strangers |
| Which features matter | Longer bar = shuffling it hurts more. Bars at 0 are decoration |
| Drift | Red (static) wandering away from 0.9 while green (ACI) stays close = drift matters and ACI helps |

## Red flags

- **GBM never beats the per-harness median**: the request-level features add little; look at importance and the history skill.
- **Huge gap between `gbm_hist` (random) and `gbm`, none under grouped**: the signal is mostly account identity.
- **`random_calib` covers ~90% on seen accounts but less on new ones**: exactly the in-house illusion this repo exists to expose.
- **Worst-harness or 10th-percentile coverage far below 90%**: the intervals are unfair even if the average is fine.
- **Conformal width factor in the tens**: a valid interval that may be too loose to use.
- **Few effective accounts**: every grouped conclusion needs wide error bars; believe the CIs.

## Figures (`figures/`)

| File | Shows | Read it as |
|---|---|---|
| `fig01_error_vs_k` | GBM and harness baseline error vs k for the 3 designs | Band = cluster-bootstrap CI. Red above blue = new accounts are harder |
| `fig02_gap_vs_k` | MAE(grouped) - MAE(random), paired CI | Above 0 with the bar clear of 0 = random CV was too optimistic |
| `fig03_skill_vs_harness` | Skill of the GBM (left) and of the new models (right) vs the per-harness median | Above 0 = better than that baseline |
| `fig04_account_concentration` | Lorenz curve, Gini, effective number of accounts | The effective number limits every grouped conclusion |
| `fig05_conformal` | Coverage vs width; worst-harness and 10th-percentile-account coverage, including group-aware methods | Want coverage near 90% **and** narrow. `random_calib` is the control |
| `fig06_aci_drift` | Rolling coverage of static vs adaptive intervals | Static drifting from 0.9 while ACI holds = drift matters |
| `fig07_hazard` | Predicted vs observed survival; predicted vs actual remaining length | Curves on top of each other = calibrated |
| `fig08_hierarchical` | Raw vs shrunken account effects; model ICC vs raw ICC | Small accounts pulled toward 0 = partial pooling working |
| `fig09_distributional` | Quantile calibration; grid CRPS by method | Points on the diagonal = calibrated; lower CRPS = better |
| `fig10_shift` | Change in the new-account gap vs plain GBM | Left of 0 with CI clear of 0 = the method closes part of the gap |
| `fig11_process` | Transition matrix between request states; P(end | state) | Rows are "current state"; look for states that precede the end |
| `fig12_decision` | Cache-retention saving vs the best fixed TTL (symlog axis) | Above 0 = cheaper. `oracle` is a lower bound, not a policy |
| `fig13_mixture` | Latent class sizes, length and harness mix | Classes should differ in length to be worth having |
| `fig14_weekly_drift` | Session length and harness mix by week (only with 2+ weeks) | Shifts here mean pooling weeks needs care |
| `fig15_extras_vs_gbm` | Skill of every new model vs the plain GBM, grouped and random | Most bars straddling 0 = no evidence of gain |

## Extra sections of `summary.txt`

Each enabled analysis adds a block (HAZARD MODEL, HIERARCHICAL MODEL, DISTRIBUTIONAL FORECASTS, GROUP-AWARE CONFORMAL,
SEQUENCE MODELS, PROCESS MINING, DOMAIN-SHIFT METHODS, DECISION-FOCUSED EVALUATION, LATENT-CLASS MIXTURE). Lines marked
`<- better` / `<- worse` are those whose 95% CI excludes 0; everything else is "not distinguishable from no effect".
If an analysis crashed, a final block `ANALYSES THAT FAILED IN THIS RUN` names it and the error; the rest is unaffected.
**With `--synthetic` every number is meaningless.** A tie with the GBM is a normal result for the neural and mixture models.
