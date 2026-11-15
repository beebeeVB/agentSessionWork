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
