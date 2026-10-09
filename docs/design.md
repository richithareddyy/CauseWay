# Causeway — Milestone 1: Dataset Verification and Causal Design

**Status:** design for review. No estimators have been implemented. Every number in Section 1 is printed by `scripts/audit_raw_data.py`, run on the original source file.

**Disclosure on pre-specification.** The integrity audit computed outcome rates by arm (Section 1.11) to confirm the file is internally coherent, and the power calculations in Section 2.13 use outcome variances. This design is therefore *not* blind to the arm-level means. The choices below are made so that they do not exploit that knowledge: both e-mail arms are analysed as co-primary contrasts, the benchmark selection mechanism uses covariates only, and every robustness specification is listed here before any of them is run.

---

## 1. Dataset verification

### 1.1 Original source and provenance

| Item | Verified value |
|---|---|
| Name | MineThatData E-Mail Analytics and Data Mining Challenge |
| Author / publisher | Kevin Hillstrom, MineThatData |
| Original announcement | Blog post, 20 March 2008: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html> |
| Original file | <http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv> (still served, HTTP 200, checked 2026-10-08) |
| SHA-256 of original file | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` (3,964,977 bytes, CRLF line endings) |

The blog post describes the data as an e-mail test on 64,000 customers who last purchased within twelve months. Customers were randomly split into thirds: Mens e-mail, Womens e-mail, or no e-mail. Results were tracked for two weeks after the campaign. The post posed eight analytical questions, including which version performed best, how much incremental spend each drove, and whether the versions performed differently across customer segments.

What the post does **not** document: the calendar date of the campaign, the retailer, the e-mail content, whether e-mails were delivered or opened, how the randomization was carried out (simple or blocked), and which purchase channels the outcomes cover.

### 1.2 Dataset size

64,000 rows × 12 columns. Every row has exactly 12 fields.

### 1.3 Unit of analysis

One row per customer. There is no customer identifier, so repeat appearances of the same person cannot be detected (see 1.9).

### 1.4 Treatment arms (`segment`)

| Arm | n |
|---|---|
| Mens E-Mail | 21,307 |
| Womens E-Mail | 21,387 |
| No E-Mail | 21,306 |

The counts are consistent with equal-probability assignment: χ²(2) = 0.20, p = 0.90. `segment` records **assignment** to a campaign. The data contain no delivery or open indicators.

### 1.5 Outcomes (two-week window after the campaign)

| Field | Definition (source post) | Observed |
|---|---|---|
| `visit` | 1 if the customer visited the website in the following two weeks | 9,394 ones (14.7%) |
| `conversion` | 1 if the customer purchased merchandise in the following two weeks | 578 ones (0.90%) |
| `spend` | Actual dollars spent in the following two weeks | 0 for 63,422 customers; 29.99–499.00 for the 578 converters |

### 1.6 Pre-treatment covariates (measured over the year before the campaign)

| Field | Definition (source post) | Observed domain |
|---|---|---|
| `recency` | Months since last purchase | integers 1–12 |
| `history` | Actual dollars spent in the past year | 29.99–3,345.93 |
| `history_segment` | Binned version of `history` (7 bins) | deterministic function of `history`; 0 violations of bin bounds |
| `mens` | 1 if bought Mens merchandise in the past year | 35,266 ones |
| `womens` | 1 if bought Womens merchandise in the past year | 35,182 ones; every customer has `mens` or `womens` = 1 |
| `zip_code` | Urban / Suburban / Rural classification of the ZIP code | spelled **`Surburban`** in the source (28,776 rows) |
| `newbie` | 1 if a new customer in the past twelve months | 32,144 ones |
| `channel` | Channels purchased from in the past year | Phone / Web / Multichannel |

The data contain no tenure beyond `newbie`, no purchase frequency, and no demographics (`zip_code` is a geographic urbanicity class, not a personal attribute). Causeway will not construct or impute any of these.

### 1.7 Treatment assignment mechanism

The assignment is randomized, per the source. The data are consistent with that: arm sizes match equal thirds, and the largest absolute standardized mean difference against control is 0.014 (Mens) and 0.009 (Womens) across nine covariate contrasts. The randomization procedure itself (simple, stratified, or blocked) is undocumented. The near-identical Mens and control counts (21,307 vs 21,306) are compatible with either.

### 1.8 Observation window

The covariates describe the twelve months before the campaign; the outcomes cover the two weeks after it. The calendar dates are not published. The file name and post are dated 20 March 2008, so the campaign took place on or before that date.

### 1.9 Missing data, duplicates, and anomalies

- **Missing values:** none in any column (empty, `NA`, `NaN` and `NULL` were all checked).
- **Exact duplicate rows:** 6,562 rows beyond a first copy (7,634 rows in 1,072 duplicate groups). 7,570 of those 7,634 rows have `history` = 29.99, and no duplicate group contains a conversion. These are customers with coarse, identical covariate profiles and zero outcomes, which is expected when one-eighth of the file sits at a single `history` value. They are **not** evidence of duplicated records, and they will not be removed. The validation log will report them.
- **`history` = 29.99:** 7,947 customers (12.4%). The file gives no lower values, so this is plausibly a single-item minimum price or a floor in how history was recorded. That cannot be determined from the source.
- **`spend` among converters:** the minimum is 29.99 (136 of 578 converters, 23.5%) and the maximum is 499.00 (12 converters, none above). A ceiling at exactly 499.00 is consistent with top-coding, but the source does not say so. Section 2.12 lists a sensitivity analysis for this.
- **Logical consistency:** every converter also visited (0 cases of `conversion`=1 with `visit`=0). `spend` > 0 if and only if `conversion` = 1. This pattern suggests that conversion was measured on website-linked purchases, but the source does not say whether phone or other-channel purchases are counted.
- **Category spelling:** `Surburban` will be recoded to `Suburban` as a logged transformation.

### 1.10 Access and licensing

Neither the blog post nor the file states a license or terms of use. The dataset is redistributed by several open-source libraries (scikit-uplift, TensorFlow Datasets, causeinfer, pyuplift), whose licenses cover their code, not the data. Causeway therefore:

- does **not** commit the data to the repository,
- downloads it at build time (`make data`) from the original MineThatData URL and verifies the SHA-256 above, falling back to the scikit-uplift mirror (with its MD5 verified) only if the original is unavailable,
- credits Kevin Hillstrom and the original post as the data source.

### 1.11 Discrepancies between copies

| Copy | Finding |
|---|---|
| Original MineThatData CSV | Reference copy (CRLF line endings, integers written as `0`). |
| scikit-uplift mirror (`hillstorm_no_indices.csv.gz`, MD5 `a68a81291f53a14f4e29002629803ba3`, matches the checksum in the scikit-uplift source) | Same header and the same 64,000 rows in the same order. **Numerically identical on all 64,000 rows.** Text differs only in formatting (LF endings, `spend` written as `0.0`), which explains the different byte counts (4,028,958 vs 3,964,977). It preserves the `Surburban` spelling. |
| TensorFlow Datasets `hillstrom` | 64,000 examples with the same 12 features. Its catalog citation (`author={ENTRY, WINNING}`) is malformed, so Causeway cites the original blog post instead. |
| Common secondary descriptions | Some write-ups name the treatment column "treatment"; the original column is `segment`. Some uplift tutorials collapse the two e-mail arms into one binary treatment; Causeway does not (Section 2.2). |

**Integrity-check outcome rates** (descriptive only; the formal analysis is Milestone 3):

| Arm | Visit | Conversion | Mean spend ($) |
|---|---|---|---|
| Mens E-Mail | 0.1828 | 0.01253 | 1.4226 |
| Womens E-Mail | 0.1514 | 0.00884 | 1.0772 |
| No E-Mail | 0.1062 | 0.00573 | 0.6528 |

---

## 2. Causal design

### 2.1 Primary research question

> Did assigning a customer to receive a promotional e-mail causally increase the customer's spending over the following two weeks, and by how much?

### 2.2 Primary treatment comparison

**Two co-primary pairwise contrasts against the shared control:** Mens E-Mail vs No E-Mail, and Womens E-Mail vs No E-Mail. Both are estimated jointly from one three-arm model, and the family-wise error is controlled with a Holm correction across the two.

Rejected alternatives:

- *Collapsing both arms into "any e-mail".* The estimand would be the effect of a 50/50 lottery between two different creatives. That corresponds to no campaign anyone would run.
- *Choosing a single arm as primary.* The arm-level means have already been seen (see the disclosure above), so picking one now would be a post-hoc choice.
- *A full multi-treatment model.* With a single control and no factorial structure, the three-arm regression with two indicators *is* the valid multi-treatment analysis. It is numerically equivalent to the two pairwise differences in means and supplies their covariance.

Mens vs Womens is a **secondary** head-to-head contrast.

### 2.3 Primary outcome

`spend`: dollars spent in the two weeks after the campaign, with non-purchasers coded 0. This outcome answers the business question directly. It is zero-inflated and right-skewed (99.1% zeros), which drives several design choices below.

### 2.4 Primary estimand

For each e-mail arm *a* ∈ {Mens, Womens}, the **intention-to-treat average treatment effect**:

τ_a = E[ Y(a) − Y(none) ]

where *Y(a)* is two-week spend under assignment to arm *a*, and the expectation is over the population of customers who purchased in the prior twelve months. It is ITT because only assignment is observed, not receipt or opening. Under randomization, ATT = ATE, since the assigned group is a random sample of the population.

### 2.5 Secondary outcomes

- `visit` (binary): the most frequent outcome, so the best-powered one.
- `conversion` (binary).

### 2.6 Secondary estimands

- ITT risk differences and risk ratios for `visit` and `conversion`.
- The Mens vs Womens contrast.
- **Conditional average treatment effects (CATE)**, τ_a(x) = E[Y(a) − Y(none) | X = x], and group average treatment effects (GATEs) over pre-specified segments: `history_segment`, recency bands (1–3, 4–6, 7–9, 10–12 months), `channel`, `newbie`, `zip_code`, and `mens`/`womens` purchase history. These are estimated **on the randomized data**, where the propensity is known by design.
- **Policy value** of targeting rules (treat everyone / random / model-ranked / best-arm-per-customer), evaluated on held-out folds.
- *Not a causal estimand:* mean spend among converters. Conditioning on conversion conditions on a post-treatment variable. If this quantity is reported, it will be through Lee (2009) bounds for always-converters, stated with the monotonicity assumption those bounds require.

### 2.7 Experimental estimators (Part I)

| Estimator | Purpose |
|---|---|
| Difference in means, Neyman/HC2 standard error, 95% CI | Primary. Design-based, no model. |
| Fisher randomization test (10,000 permutations within the pair) | Exact test of the sharp null of no effect for any customer. |
| Lin (2013) fully interacted regression adjustment, HC2 SE | Precision only. It cannot add identification that randomization has not already supplied. |
| Nonparametric bootstrap (B = 5,000, seeded), percentile and BCa intervals | Checks on the stability of the normal-approximation interval for a skewed outcome. |
| Risk difference and risk ratio (Newcombe and log-scale CIs) | Binary outcomes. |
| Causal forest (EconML `CausalForestDML`) with the known design propensity; best-linear-predictor heterogeneity test and GATEs (Chernozhukov, Demirer, Duflo and Fernández-Val, 2018) | Heterogeneity, with a formal test of whether it exists at all. |

Propensity models, matching and weighting are **not** used to identify the Part I effect. In Part I the propensity score appears only as a randomization diagnostic: the AUC of a treatment classifier should be ≈ 0.5.

### 2.8 Proposed observational benchmark construction (Part II)

**Purpose:** measure how naive and adjusted observational estimators behave when randomization is removed in a known, documented way. The benchmark is a *constructed* dataset. It is never described as Hillstrom's data-generating behaviour.

**Population:** each co-primary pair separately (Mens + control, n = 42,613; Womens + control, n = 42,693).

**Mechanism: RCT rejection sampling** (Keith, Feldman, Jurgens, Bragg and Bhattacharya, 2023).

1. Define a design propensity e*(x) that mimics a marketer who targets recent, high-value, multichannel, established customers:

   logit e*(x) = α + k · [ 0.75·z(log history) − 0.75·z(recency) + 0.50·1{channel = Multichannel} − 0.50·newbie ]

   *z(·)* is a z-score computed on the pair population. α is solved numerically so that the mean of e*(X) is 0.5. These coefficients are fixed in this document.
2. For unit *i* with trial arm *t_i*, compute the acceptance probability

   a_i = P*(T = t_i | x_i) / ( P̂(T = t_i) · M ),  with M = max_i P*(T = t_i | x_i) / P̂(T = t_i),

   where P̂(T = t) is the arm share in the pair.
3. Draw U_i ~ Uniform(0, 1) and keep unit *i* if U_i ≤ a_i.

In the retained sample, P(T = 1 | X) = e*(X), and the covariate distribution P(X) is unchanged in expectation. Keith et al. prove that this preserves identification. They also show that the earlier OSRCT sampler (Gentzel, Pruthi and Jensen, 2021, Algorithm 2) changes P(X) unless the trial's treatment share is exactly 0.5, and that oracle estimators are then biased. Each Hillstrom pair has a share of ~0.50, so the two samplers would nearly coincide here. Causeway uses rejection sampling because its validity does not depend on that coincidence.

**Pre-specified scenarios** (all defined before any observational estimate is computed):

| Scenario | Selection strength | Analyst's adjustment set | Purpose |
|---|---|---|---|
| A — moderate selection (primary) | k = 1 | all pre-treatment covariates | Main estimator comparison |
| B — strong selection | k = 2 | all pre-treatment covariates | Overlap, positivity and extreme-weight stress test |
| C — hidden confounder | k = 1 | omits `recency` | A known violation of unconfoundedness: tests whether sensitivity analysis flags it |

If the e*(X) distribution in a scenario (computed from covariates only) produces degenerate overlap, k may be reduced. That change would be made and documented **before** any outcome-based estimate is computed, and never after.

**Seeds and replication:**

- Master seed `20080320`.
- A single fixed draw (replicate 0) is used for figures and walkthroughs.
- R = 200 independent selection draws per scenario measure the estimators' bias, RMSE and CI coverage. One draw alone would confound estimator bias with subsampling noise. If runtime forces a smaller R for DML or forest estimators, that is decided from timings, not from results.

**Benchmarks (matched by estimand):** conditional on the realized trial, each benchmark is computed from the *full* randomized pair.

| Estimand in the observational sample | Experimental benchmark |
|---|---|
| ATE | Difference in means on the full pair. Valid because P(X) is preserved. |
| ATT = E[τ(X) \| T = 1, S = 1] | Difference in Hájek means on the full pair, with each unit weighted by e*(x_i). The weight is known by design, not estimated. |
| ATC | As for ATT, with weight 1 − e*(x_i). |

Matching estimates are compared with the ATT benchmark, and IPW, AIPW, DML and regression ATE estimates with the ATE benchmark. An oracle IPW estimator that uses the *true* e*(x) is included as a check on the construction itself: it should be unbiased within Monte Carlo error.

**Relation to prior work:**

- LaLonde (1986) compared experimental estimates from the National Supported Work Demonstration with nonexperimental estimates built from comparison groups drawn from survey data. There, selection was real and unknown.
- In Causeway, selection is on observed covariates **by construction**. Part II therefore tests estimator mechanics (finite-sample behaviour, model misspecification, overlap) under a *true* unconfoundedness assumption. It does not test whether real observational marketing data satisfy that assumption. Scenario C is the only place where the assumption is deliberately broken.
- Causeway cites LaLonde as the origin of the within-study comparison idea, and Gentzel et al. (2021) and Keith et al. (2023) as the direct methodological basis. It does **not** claim to replicate LaLonde's design.

### 2.9 Confounders intentionally used in benchmark selection

`history` (log), `recency`, `channel = Multichannel` and `newbie`. All four are pre-treatment and plausibly predictive of spend under standard recency-frequency-monetary reasoning; this choice was made from domain reasoning, not from fitted outcome models. `mens`, `womens` and `zip_code` are deliberately left out of selection. They remain available to the analyst, so a correct adjustment set and a superset of it can be compared.

### 2.10 DAG for the observational benchmark

```
              U (unobserved customer traits)
             /                     \
            v                       v
   X_sel = {history, recency,     X_other = {mens, womens, zip_code}
            channel, newbie}            |
       |        \                       |
       |         \                      v
       |          +-------------------> Y (spend)
       v                                ^
       S  <----------- R (randomized    |
  (retained in          trial arm) ---> visit ---> conversion ---> spend
   benchmark)                 |                                    ^
                              +------------------------------------+
   T = R among units with S = 1
```

- **Treatment:** T, the trial arm among retained units.
- **Outcome:** Y (spend). `visit` and `conversion` are secondary outcomes in their own right.
- **Confounding:** created by **S**, which is a *collider* of R and X_sel. Conditioning on S = 1 (keeping only retained units) makes T and X_sel dependent. That opens the back-door path T – [S] – X_sel → Y, and also T – [S] – X_sel ← U → Y.
- **Adjustment set:** {history, recency, channel, newbie} is the minimal sufficient set. It blocks both paths, because X_sel is a non-collider on each of them. Adding X_other is valid (it is pre-treatment, not a collider, and not a descendant of T) and may improve precision. `history_segment` is a deterministic function of `history` and is dropped to avoid redundancy.
- **Mediators:** visit → conversion lie on the path from T to spend. They are **never** adjusted for when estimating the total effect on spend. Doing so would remove part of the effect and, because U → visit is plausible, open a collider path.
- **Part I DAG:** the same graph without S. R has no parents, so no adjustment is needed.

### 2.11 Identification assumptions

| Assumption | Part I (randomized) | Part II (constructed) |
|---|---|---|
| Exchangeability / unconfoundedness | Supported by design (randomization) | True by construction for A and B, but *assumed* by every estimator; violated by construction in C. Untestable from the observed data in a real study. |
| Positivity | Supported by design (P(T) ≈ 1/2) | Empirically diagnosable: propensity overlap, extreme weights, effective sample size |
| Consistency (well-defined treatment) | Partially testable. "Assignment to e-mail *a*" is well defined; e-mail content and delivery are unknown. | Same |
| SUTVA / no interference | Fundamentally untestable here. Household or social spillover is plausible but unobserved. | Same |
| Temporal ordering | Supported by design. Covariates cover the prior year and outcomes the following two weeks. | Same |
| Correct model specification | Not required for difference in means; Lin adjustment is robust to misspecification | Required for regression; relaxed by AIPW and DML (double robustness, flexible nuisance models); diagnosable in part via balance and agreement between estimators |

### 2.12 Main threats to validity

1. **External validity and age:** a single retailer, a single campaign, 2008 e-mail behaviour and an unknown season.
2. **Treatment definition:** ITT on assignment only, with no delivery or open data. A per-protocol effect is not identifiable.
3. **Outcome measurement:** conversion may cover only website purchases (every converter visited). Spend may be top-coded at 499.00. There is no margin or cost data, so policy values are in incremental *revenue*, not profit.
4. **Heavy-tailed, sparse outcome:** only 578 converters. The planned robustness checks are winsorized spend, a fit excluding the twelve 499.00 values, conversion as an alternative outcome, and bootstrap BCa intervals.
5. **Subgroup power:** the minimum detectable spend effect (80% power) for the overall Mens contrast is $0.41, but within pre-specified subgroups it ranges from $0.52 (`newbie` = 1) to $5.41 (`history_segment` = $1,000+). Most subgroup spend effects cannot be distinguished from zero or from each other. For `visit` the overall MDE is ≈ 0.009, far better powered. This is why the heterogeneity analysis leads with a formal test, and why subgroup claims will be labelled exploratory unless they survive multiplicity control.
6. **Multiplicity:** Holm correction across the two co-primary contrasts. Segment-level results carry Benjamini–Hochberg-adjusted q-values and are labelled exploratory.
7. **Part II construction:** selection is on observed covariates by design, so good recovery in Scenarios A and B shows that the estimators work *when their assumptions hold*. It is not evidence that real observational marketing data satisfy those assumptions.
8. **Undocumented randomization procedure:** blocked or stratified randomization would make Neyman standard errors conservative, which is acceptable.

### 2.13 Planned statistical tests and intervals

- **Primary:** difference in means with HC2 standard errors and 95% Wald CIs; Holm-adjusted p-values for the two co-primary contrasts; Fisher randomization test p-values; bootstrap percentile and BCa intervals (B = 5,000).
- **Precision:** Lin-adjusted estimates with HC2 standard errors.
- **Binary outcomes:** Newcombe hybrid-score CIs for risk differences; log-scale CIs for risk ratios.
- **Balance:** standardized mean differences (|SMD| < 0.1 as the conventional reference), a randomization χ² test on arm counts, and the treatment-classifier AUC.
- **Part II:** for each estimator, the point estimate with a 95% CI (analytic where one exists, otherwise bootstrap), plus bias, RMSE and CI coverage against the estimand-matched benchmark over the R draws, with Monte Carlo standard errors.
- **Heterogeneity:** a best-linear-predictor test of heterogeneity, GATE CIs by quantile of the predicted CATE, and segment GATEs with BH q-values.
- **Targeting:** cross-fitted policy value (IPW/AIPW on held-out folds) with bootstrap CIs; uplift and Qini curves on held-out folds only.
- **Sensitivity:** E-values (VanderWeele and Ding, 2017) for risk-ratio estimates; Cinelli–Hazlett (2020) omitted-variable-bias bounds in partial-R² terms for the spend outcome model; Rosenbaum-style Γ bounds for the matched design. All three are applied to Part II, and Scenario C checks whether each one flags the known hidden confounder.
- **Refutation (Part II only):** DoWhy placebo treatment, random common cause and data-subset refuters, each described by what it does and does not test.

### 2.14 Definition of success

The project succeeds if it **measures and communicates causal evidence correctly**. It does not need any particular estimator to look good. Concretely:

1. `make data` rebuilds the analysis dataset from the original URL, verifies the checksum, and writes a validation log of every check and transformation. No record is silently dropped.
2. Part I reports both co-primary ITT effects on spend as point estimate, SE, 95% CI, Holm-adjusted p-value and randomization p-value, with randomization diagnostics, alongside the secondary outcomes. The pre-specified robustness specifications are all reported, including any that disagree.
3. The Part II construction is shown to be valid before it is used. Covariate SMDs between the retained sample and the full pair are < 0.1, and the oracle IPW is unbiased within Monte Carlo error. Each estimator is then reported with bias, RMSE and coverage against its estimand-matched benchmark.
4. Heterogeneity is claimed only where the formal test or the multiplicity-adjusted GATEs support it. Otherwise the project reports that no heterogeneity was detected, with the MDE that bounds what could have been detected.
5. Targeting results come from held-out data and are labelled as estimated policy values in incremental revenue, not realized outcomes.
6. Tests pass, CI is green, every reported number is produced by code in the repository, and the README separates what randomization establishes from what requires observational assumptions.

---

## 3. Points where the original specification needs adjustment

| # | Specification item | Issue | Resolution in this design |
|---|---|---|---|
| 1 | "T = whether a customer *received* the promotion" | Only assignment is observed | The estimand is ITT on assignment |
| 2 | Propensity scores, matching and IPW applied to the experiment | Unnecessary for identification under randomization | Part I uses design-based estimators; the propensity score appears only as a diagnostic |
| 3 | Heterogeneity by "customer tenure", "purchase frequency" and "demographic variables" | None exists in the data | Use `newbie`, `recency`, `history`, `channel` and `zip_code` (urbanicity); nothing is invented |
| 4 | Spend-based CATE and "potentially harmful" segments | Subgroup spend MDEs ($0.52–$5.41) exceed most plausible effects | Lead with a formal heterogeneity test; give `visit` heterogeneity equal billing; label segment spend effects exploratory |
| 5 | A single constructed observational dataset | One draw confounds estimator bias with subsampling noise | R = 200 seeded draws, plus one fixed draw for display |
| 6 | Generic "introduce confounding" | Naive subsampling (OSRCT) can shift P(X) and bias even oracle estimators | Use RCT rejection sampling, with P(X) preservation verified |
| 7 | Comparing every estimator against "the experimental benchmark" | ATT and ATE differ under heterogeneity | Estimand-matched benchmarks (ATE, ATT and ATC) |
| 8 | E-values as general sensitivity analysis | E-values are defined for risk ratios, not mean differences in dollars | E-values for binary outcomes; partial-R² OVB analysis for spend |
| 9 | Uplift and gain curves | In-sample uplift curves overfit; there is no cost data | Cross-fitted, held-out evaluation; values reported in incremental revenue |
| 10 | Python environment | DoWhy 0.14 requires Python < 3.14; this machine has only Python 3.14 | Use Python 3.12 locally and in CI |

---

## 4. Leakage review for this milestone

- All eight covariates are measured over the year **before** the campaign. None of `visit`, `conversion` or `spend` will ever be a model feature.
- `history_segment` is derived from `history`. It is redundant, not leaked.
- The Part II selection mechanism uses only pre-treatment covariates and the randomized arm, never outcomes.
- Held-out evaluation is required for every uplift or policy result.

---

## 5. References

These were verified against the publisher or author pages during this milestone:

- Hillstrom, K. (2008, March 20). *The MineThatData E-Mail Analytics and Data Mining Challenge.* MineThatData blog.
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. *American Economic Review*, 76(4), 604–620.
- Gentzel, A. M., Pruthi, P., & Jensen, D. (2021). How and why to use experimental data to evaluate methods for observational causal inference. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 3660–3671.
- Keith, K. A., Feldman, S., Jurgens, D., Bragg, J., & Bhattacharya, R. (2023). RCT rejection sampling for causal estimation evaluation. *Transactions on Machine Learning Research.* arXiv:2307.15176.

The methods references below will be checked against publisher records before they appear in the README: Lin (2013), *Annals of Applied Statistics*; Lee (2009), *Review of Economic Studies*; Chernozhukov et al. (2018), *Econometrics Journal* (double machine learning); Chernozhukov, Demirer, Duflo and Fernández-Val (2018), NBER Working Paper 24678; Wager and Athey (2018), *JASA*; Austin (2011), *Pharmaceutical Statistics* (caliper width); VanderWeele and Ding (2017), *Annals of Internal Medicine*; Cinelli and Hazlett (2020), *JRSS-B*; Rosenbaum (2002), *Observational Studies*.
