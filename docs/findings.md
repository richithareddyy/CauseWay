# Causeway: findings

> Copied from `reports/results/findings/findings_report.md` by `make docs`. Figures marked as paths and every result table are produced locally by `make analysis`.

Does assigning a customer to a marketing e-mail make them spend more, or are the customers who get promotions simply the ones who would have bought anyway? This report gathers the answers from every stage of the analysis. Every number is read from the saved result tables.

Estimand throughout Part I: the **intention-to-treat effect of assignment to an e-mail campaign**, over the two weeks after the campaign. It is the effect of being assigned an e-mail, not of receiving or opening one.

## 1. What randomization establishes

The Hillstrom (2008) experiment randomly assigned 64,000 customers to a Mens e-mail, a Womens e-mail or no e-mail. Because assignment was random, the difference in means is an unbiased estimate of the effect of assignment, with no modelling assumptions.

| Contrast | Role | Outcome | Estimand | Estimate | 95% CI | Relative lift | Holm p | Randomization p |
|---|---|---|---|---|---|---|---|---|
| Mens vs No E-Mail | co-primary | spend | ITT effect of assignment | $0.77 | [$0.49, $1.05] | 118% | < 0.0001 | ≤ 0.0001 |
| Mens vs No E-Mail | co-primary | visit | ITT effect of assignment | 7.66 pp | [7.00, 8.32] | 72% | < 0.0001 | ≤ 0.0001 |
| Mens vs No E-Mail | co-primary | conversion | ITT effect of assignment | 0.68 pp | [0.50, 0.86] | 119% | < 0.0001 | ≤ 0.0001 |
| Womens vs No E-Mail | co-primary | spend | ITT effect of assignment | $0.42 | [$0.17, $0.68] | 65% | 0.0011 | 0.0010 |
| Womens vs No E-Mail | co-primary | visit | ITT effect of assignment | 4.52 pp | [3.89, 5.16] | 43% | < 0.0001 | ≤ 0.0001 |
| Womens vs No E-Mail | co-primary | conversion | ITT effect of assignment | 0.31 pp | [0.15, 0.47] | 54% | 0.0002 | 0.0003 |
| Mens vs Womens | secondary | spend | ITT effect of assignment | $0.35 | [$0.03, $0.66] | 32% | — | 0.0294 |
| Mens vs Womens | secondary | visit | ITT effect of assignment | 3.14 pp | [2.43, 3.84] | 21% | — | ≤ 0.0001 |
| Mens vs Womens | secondary | conversion | ITT effect of assignment | 0.37 pp | [0.17, 0.56] | 42% | — | 0.0003 |

Holm p-values adjust the two co-primary contrasts for multiplicity; randomization p-values come from 10,000 re-randomizations. Visit and conversion effects are in percentage points.

- Assignment to the **Mens e-mail** raised mean two-week spend by $0.77 (95% CI [$0.49, $1.05]), a 118% lift over the No E-Mail mean of $0.65.
- Assignment to the **Womens e-mail** raised it by $0.42 (95% CI [$0.17, $0.68]).
- The Mens e-mail out-earned the Womens e-mail by $0.35 (95% CI [$0.03, $0.66]) (secondary contrast, randomization p = 0.0294).
- Visit rates rose by 7.66 pp (95% CI [7.00, 8.32]) (Mens) and 4.52 pp (95% CI [3.89, 5.16]) (Womens).

**Is the randomization credible?** A classifier cannot tell the arms apart from pre-treatment covariates (out-of-fold AUC 0.498 and 0.494, likelihood-ratio p = 0.82 and 0.92, Mens and Womens), and the largest absolute standardized mean difference is 0.014.

**Is the spend effect robust?** Across the pre-specified spend versions (recoding the $499 values, winsorizing) the Mens estimate ranges from $0.74 to $0.77 and the Womens estimate from $0.41 to $0.42. Keeping one copy of each exact duplicate row gives $0.85 and $0.47. Under five further bootstrap seeds the lower bound of the Mens BCa interval stays between $0.484 and $0.502.

![ITT effects with 95% intervals](figures/itt_effects.png)

## 2. What happens without randomization

Part II removes the protection of randomization on purpose. Customers are subsampled from each randomized pair with a known selection rule (RCT rejection sampling, Keith et al. 2023), so that high-spending, recent, multichannel and established customers are more likely to appear e-mailed. The full randomized pair still supplies the true answer. Three scenarios, 200 draws each:

- **A**: moderate selection on covariates the analyst observes.
- **B**: twice as strong, which stresses overlap and produces extreme weights.
- **C**: the strength of A, but recency is hidden from the analyst, so unconfoundedness fails.

In Scenario A the naive comparison overstates the spend effect by 48% (Mens vs No E-Mail) and 56% (Womens vs No E-Mail). The error splits exactly into a selection term (the e-mailed customers would have spent more anyway) and an effect-heterogeneity term (ATT minus ATE, which can take either sign): $0.26 and $0.12 per customer for Mens vs No E-Mail; $0.26 and -$0.02 per customer for Womens vs No E-Mail.

## 3. Which estimators recover the experimental answer

![Mean bias of each estimator by comparison and scenario](figures/estimator_recovery.png)

Feasible estimators whose mean bias is within Monte Carlo error for both comparisons:

- **Scenario A** (moderate selection): spend: IPW (ATE), AIPW (linear, cross-fitted) and IPW (ATT); visit: IPW (ATE), AIPW (linear, cross-fitted) and IPW (ATT).
- **Scenario B** (strong selection): spend: IPW (ATE), AIPW (linear, cross-fitted), DML (gradient boosting) and IPW (ATT); visit: IPW (ATE), AIPW (linear, cross-fitted), DML (gradient boosting) and IPW (ATT).
- **Scenario C** (recency hidden): spend: none; visit: none.

Each flag is a separate test at roughly the 5% level, so among the 48 feasible cells in Scenarios A and B about 2 would be flagged by chance even if every estimator were unbiased. An isolated flag, such as one comparison in one scenario, is weak evidence on its own; a pattern across cells is not.

1:1 matching with a 0.2-SD caliper is biased in every cell: the caliper discards treated customers without a close control, which changes who the estimate describes.

With recency hidden (C), no feasible estimator recovers the benchmark: adjustment cannot remove bias from a variable the analyst does not observe.

The tables below are the estimator comparison for spend in Scenario A. The estimate and interval come from one constructed dataset (draw 0, fixed before any result was seen); bias and coverage summarize all 200 draws. `*` marks a bias larger than twice its Monte Carlo standard error. Every scenario and outcome is in `estimator_table.csv`.

**Mens vs No E-Mail, Scenario A, spend**

| Estimator | Estimand | Estimated effect (draw 0) | 95% CI (draw 0) | Mean bias vs benchmark (200 draws) | CI coverage | Key assumptions |
|---|---|---|---|---|---|---|
| Randomized experiment | ATE (randomized, full pair) | $0.77 | [$0.49, $1.05] | benchmark | — | Randomized assignment; SUTVA (no interference, one version of each e-mail). |
| Randomized experiment | ATT (randomized pair, e*-weighted) | $0.89 | [$0.52, $1.25] | benchmark | — | Randomized assignment; SUTVA (no interference, one version of each e-mail). |
| Naive difference | unadjusted association | $0.97 | [$0.57, $1.36] | +48.1% * | 64% | Assignment independent of potential outcomes (violated by construction in Part II). |
| Oracle IPW (true e*) | ATE | $0.54 | [$0.18, $0.90] | +0.7% | 100% | Known propensity; positivity. Checks the construction; not a feasible estimator. |
| Outcome regression (OLS) | ATE | $0.56 | [$0.17, $0.94] | -4.0% * | 100% | Unconfoundedness given the analyst covariates; correctly specified outcome model. |
| IPW (ATE) | ATE | $0.54 | [$0.18, $0.90] | +0.4% | 100% | Unconfoundedness; positivity for both arms; correctly specified propensity model. |
| AIPW (linear, cross-fitted) | ATE | $0.54 | [$0.19, $0.90] | +1.1% | 100% | Unconfoundedness; positivity; at least one of the propensity and outcome models correct. |
| DML (gradient boosting) | ATE | $0.52 | [$0.15, $0.89] | +0.1% | 100% | Unconfoundedness; positivity; nuisance learners converge fast enough (o(n^-1/4)) under cross-fitting. |
| IPW (ATT) | ATT | $0.82 | [$0.35, $1.29] | +0.6% | 98% | Unconfoundedness for controls; controls overlap the treated; correct propensity model. |
| 1:1 NN matching (ATT) | ATT | $0.58 | [$0.15, $1.01] | -12.0% * | 97% | Unconfoundedness; treated units have close controls; unmatched treated change the estimand to the ATT among matched treated. |

**Womens vs No E-Mail, Scenario A, spend**

| Estimator | Estimand | Estimated effect (draw 0) | 95% CI (draw 0) | Mean bias vs benchmark (200 draws) | CI coverage | Key assumptions |
|---|---|---|---|---|---|---|
| Randomized experiment | ATE (randomized, full pair) | $0.42 | [$0.17, $0.68] | benchmark | — | Randomized assignment; SUTVA (no interference, one version of each e-mail). |
| Randomized experiment | ATT (randomized pair, e*-weighted) | $0.40 | [$0.09, $0.72] | benchmark | — | Randomized assignment; SUTVA (no interference, one version of each e-mail). |
| Naive difference | unadjusted association | $0.76 | [$0.39, $1.14] | +56.9% * | 86% | Assignment independent of potential outcomes (violated by construction in Part II). |
| Oracle IPW (true e*) | ATE | $0.60 | [$0.12, $1.09] | +0.1% | 100% | Known propensity; positivity. Checks the construction; not a feasible estimator. |
| Outcome regression (OLS) | ATE | $0.64 | [$0.18, $1.10] | +9.8% * | 98% | Unconfoundedness given the analyst covariates; correctly specified outcome model. |
| IPW (ATE) | ATE | $0.61 | [$0.12, $1.09] | +0.4% | 100% | Unconfoundedness; positivity for both arms; correctly specified propensity model. |
| AIPW (linear, cross-fitted) | ATE | $0.60 | [$0.12, $1.09] | +0.9% | 99% | Unconfoundedness; positivity; at least one of the propensity and outcome models correct. |
| DML (gradient boosting) | ATE | $0.62 | [$0.11, $1.12] | +6.7% * | 98% | Unconfoundedness; positivity; nuisance learners converge fast enough (o(n^-1/4)) under cross-fitting. |
| IPW (ATT) | ATT | $0.48 | [-$0.01, $0.97] | -2.9% | 98% | Unconfoundedness for controls; controls overlap the treated; correct propensity model. |
| 1:1 NN matching (ATT) | ATT | $0.69 | [$0.20, $1.18] | +20.3% * | 98% | Unconfoundedness; treated units have close controls; unmatched treated change the estimand to the ATT among matched treated. |

ATE and ATT estimators are compared only with the benchmark for their own estimand. The naive difference targets no causal estimand; it is set against the ATE only to show its error.

![Estimates from one constructed dataset](figures/forest_display_draw_spend.png)

## 4. Does the effect differ between customers?

Heterogeneity was claimed only if the pre-specified tests (the BLP slope b2, or the spread of sorted group effects, over 100 sample splits) supported it.

| Contrast | Outcome | BLP slope b2 | p (split-adjusted) | Heterogeneity detected |
|---|---|---|---|---|
| Mens vs No E-Mail | spend | -0.07 | 1.0000 | no |
| Mens vs No E-Mail | visit | 0.27 | 0.9284 | no |
| Mens vs No E-Mail | conversion | 0.11 | 1.0000 | no |
| Womens vs No E-Mail | spend | 0.41 | 0.4399 | no |
| Womens vs No E-Mail | visit | 0.90 | < 0.0001 | yes |
| Womens vs No E-Mail | conversion | 0.54 | 0.2320 | no |

Heterogeneity was detected only for Womens vs No E-Mail on visit. Elsewhere the predicted differences did not replicate on held-out data.

**Targeting.** Sending each customer the e-mail the model predicts is best changes expected spend by -$0.08 per customer, compared with sending everyone the Mens e-mail (95% CI [-$0.25, $0.10]). The experiment therefore supports the simple policy over any targeting these covariates allow. The interval holds the fitted predictions fixed, so it leaves out model-fitting variability; a wider interval would only strengthen this conclusion. These are estimated values of policies that were never run, not realized outcomes.

![Sorted group average effects](figures/gates_by_predicted_effect.png)

## 5. Robustness and sensitivity

**Refutation tests do not detect hidden confounding.** A placebo treatment is rejected in 4% (A) and 4% (C) of draws on average, and adding a random covariate or refitting on 80% subsets barely moves the estimates, in Scenario C just as in A. Yet in C the spend estimates are biased by 18% to 44%. Passing a refutation test is not evidence that no confounder is missing.

**Sensitivity analysis bounds; it does not detect.** The robustness value for visit is 0.091 (A) vs 0.116 (C) for Mens vs No E-Mail and 0.057 (A) vs 0.079 (C) for Womens vs No E-Mail: the estimates look more robust when a confounder is hidden, not less. What sensitivity analysis does do is turn a hypothesis about confounder strength into a bound. Bounds for a confounder one, two or three times as strong as the observed covariate giving the largest bound cover the true visit error in 18%, 90% and 100% of Mens draws in Scenario C.

![Sensitivity summaries in Scenarios A and C](figures/sensitivity_summary.png)

## 6. Statistical uncertainty and identification uncertainty

Confidence intervals describe sampling variability under the stated assumptions. They say nothing about whether the assumptions hold.

- **Part I** intervals are the whole story: randomization secures identification, so the remaining uncertainty is statistical.
- **Part II, Scenario C**: with recency hidden, the cross-fitted AIPW interval for the visit effect covers the truth in 8% and 12% of draws (Mens, Womens), against 97% and 98% for the oracle. The intervals are centred in the wrong place, and no standard error can show that. For spend the same intervals still cover in 97% and 95% of draws, only because spend is noisy enough to make them wide: high coverage there is not evidence of small bias.

## 7. Assumptions

| Assumption | Status | Part I | Part II | How it is examined |
|---|---|---|---|---|
| Randomized assignment | supported by design | Holds: assignment was randomized by the data provider. | Removed on purpose by the constructed selection. | Covariate SMDs, a likelihood-ratio test and an out-of-fold classifier AUC for each arm. |
| Unconfoundedness given the analyst's covariates | fundamentally untestable | Not needed. | True by construction in A and B, false in C (recency hidden). | Cannot be checked from observed data; omitted-variable bounds, E-values and Rosenbaum bounds show how strong a hidden confounder would have to be. |
| Positivity (overlap) | empirically diagnosable | Holds: each arm received about a third of customers. | Weakened in A, stressed in B. | Propensity overlap plots, extreme-weight shares, effective sample size, trimming and calipers. |
| Consistency (one version of each e-mail) | partially testable | Assumed: each arm is one campaign e-mail. | Inherited from Part I. | Only through the data documentation; delivery and opening were not recorded. |
| No interference (SUTVA) | fundamentally untestable | Assumed: one customer's e-mail does not change another's spending. | Inherited from Part I. | No household or network identifiers exist to examine it. |
| Temporal ordering | supported by design | Covariates describe the prior twelve months; outcomes the two weeks after the campaign. | Same. | Documented by the data source. |
| Outcome and propensity model specification | partially testable | Not needed for unbiasedness; regression adjustment only sharpens precision. | Matters for every adjusted estimator. | Alternative specifications, doubly robust estimators and the comparison with the benchmark. |

## 8. What the evidence supports, and what it does not

**Supported by the randomized design alone:**

- Assignment to either e-mail increased two-week spend, visits and conversions, and the Mens e-mail outperformed the Womens e-mail on all three.
- The size of those effects, with the stated statistical uncertainty.
- Little evidence that the effect varies with these covariates, except visit under the Womens vs No E-Mail contrast.

**Requires further assumptions:**

- Any effect of *receiving* or *opening* an e-mail. Neither was recorded; estimating it would need an instrumental-variable analysis and its exclusion restriction.
- Transport to other retailers, years or campaigns. The data are from 2008 and the outcome window is two weeks.
- Any conclusion from real observational data. Part II shows what estimators recover when the assumptions hold or fail by construction; it cannot certify those assumptions where the selection rule is unknown.

**Where the conclusions stop.** Subgroup estimates are imprecise and exploratory. The observational scenarios are constructed, with a selection rule chosen in advance; real selection may be stronger, nonlinear or driven by unrecorded variables. The data hold an observed maximum of $499; whether it reflects a true maximum or a data-generation constraint is unknown, and the spend effect barely moves when those values are recoded or winsorized.

## 9. Figure guide

Each figure answers one question, taken from the docstring of the function that draws it.

| Required visualization | Figure | Question it answers |
|---|---|---|
| Outcome distributions by arm | `reports/figures/experiment/outcome_levels_by_arm.png` | How do mean spend, visit rate and conversion rate differ across the randomized arms? |
| Outcome distributions by arm | `reports/figures/experiment/spend_among_purchasers.png` | What does spend look like among purchasers, and is there structure at the $499 maximum? |
| Covariate balance in the experiment | `reports/figures/experiment/balance_randomized.png` | Are pre-treatment covariates balanced across arms? (standardized mean differences) |
| Experimental treatment effects | [experiment/itt_effects.png](figures/itt_effects.png) | What is the ITT effect of assignment, and do the pre-specified estimators agree? |
| Imbalance after constructed selection | [benchmark/selection_imbalance.png](figures/selection_imbalance.png) | How much covariate imbalance does the constructed selection create? (fixed display draw) |
| Propensity overlap | `reports/figures/benchmark/design_propensity_overlap.png` | Do treated and control customers overlap in the true design propensity e*(x)? |
| Propensity overlap | [propensity/estimated_propensity_overlap.png](figures/estimated_propensity_overlap.png) | Where do estimated propensities of treated and control customers overlap, by scenario? |
| Love plot before and after adjustment | `reports/figures/propensity/balance_before_after_adjustment.png` | Does weighting or matching remove the imbalance, including on a hidden covariate? |
| Observational estimator comparison | [comparison/forest_display_draw_spend.png](figures/forest_display_draw_spend.png) | On one realized observational dataset, which estimates land near their own experimental benchmark? |
| Bias relative to the experimental benchmark | [findings/estimator_recovery.png](figures/estimator_recovery.png) | Which estimators recover their experimental benchmark, and under which kind of selection? |
| Bias relative to the experimental benchmark | `reports/figures/comparison/bias_rmse_spend.png` | Across 200 draws, how large is each estimator's bias relative to its total error (RMSE)? |
| Bootstrap and sampling distributions | `reports/figures/experiment/bootstrap_spend_effects.png` | Is the normal approximation adequate for the skewed spend outcome? |
| Bootstrap and sampling distributions | `reports/figures/comparison/sampling_distributions_spend.png` | How close does each estimator get to its estimand-matched experimental benchmark? |
| Distribution of heterogeneous effects | `reports/figures/heterogeneity/forest_cate_distribution.png` | How widely do the forest's predicted individual effects spread, relative to their uncertainty? |
| Effects by customer segment | [heterogeneity/gates_by_predicted_effect.png](figures/gates_by_predicted_effect.png) | Do customers with larger predicted effects actually show larger effects on held-out data? |
| Effects by customer segment | `reports/figures/heterogeneity/segment_effects_spend.png` | Within each pre-specified segment, how large is the effect and how uncertain is it? |
| Uplift curves | [heterogeneity/uplift_curves_spend.png](figures/uplift_curves_spend.png) | Does ranking customers by predicted effect beat random targeting of the same number of customers? |
| Sensitivity analysis | [robustness/sensitivity_summary.png](figures/sensitivity_summary.png) | Do standard sensitivity summaries look any different when a confounder really is hidden? |
| Sensitivity analysis | `reports/figures/robustness/specification_bias_spend.png` | Do the Part II conclusions survive other propensity and outcome models? |
| Sensitivity analysis | `reports/figures/robustness/trimming_and_calipers.png` | What do trimming and tighter calipers buy, and what do they cost in discarded customers? |

## 10. Provenance

- Processed data SHA-256 `5e722de97e02d52ad55a9db56cbbe37654c1d1390bdb3cfd0b2ef9583f8bb05b` (64,000 rows, none removed); raw data SHA-256 `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
- Benchmark specification v1.0.0 (SHA-256 `d88ce1a03de54eed5c8a38e095a49d01c0e07f0d12f95e66d3554f23b6f6697e`), base seed 20080320, 200 draws.
- Code `0fe84ed`.
- Python 3.12.15; numpy 2.4.6, pandas 3.0.6, pyarrow 26.0.0, scipy 1.15.3, statsmodels 0.15.0, scikit-learn 1.9.1, econml 0.17.0, doubleml 0.11.4.
