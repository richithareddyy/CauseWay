# CauseWay

A causal-inference study asking whether promotional emails increased customer spending—and what happens when observational methods try to recover that experimental answer.

**Python 3.12 · pandas · scikit-learn · EconML · DoubleML · Matplotlib**

The starting point is the Hillstrom randomized email experiment, covering 64,000 customers. Part I estimates the effect of assignment to each email campaign. Part II deliberately introduces selection bias, then compares weighting, matching, regression, AIPW, and DML against the appropriate experimental benchmark.

## Main findings

- Assignment to the Mens email increased two-week spending by about $0.77 per customer; the Womens email increased it by about $0.42.
- With observed confounders, weighting and cross-fitted AIPW recover the benchmark within Monte Carlo error.
- Hiding a confounder leaves every tested estimator biased, even when refutation checks pass.

![Experimental effects with 95% intervals](docs/figures/itt_effects.png)

The [findings report](docs/findings.md) contains intervals, estimator comparisons, sensitivity checks, and qualifications. These findings concern this experiment and the constructed selection scenarios, not a general ranking of causal methods.

## Reproduce

From the repository root, with Python 3.12 installed:

```bash
make setup
make data
make analysis
make docs
make test
```

The full analysis took about 75 minutes on eight cores in the documented run. Data downloads are checksum-verified; configurations fix the benchmark design and random seeds. Generated reports record the data, configuration, and environment used.

The engineering work includes handwritten estimator calculations checked against library implementations, validation of the source data, and tests for reporting and published figures. Policy comparisons are estimates of policies that were not deployed.

[Research design](docs/design.md) · [Results](docs/findings.md) · [Setup and technical details](GUIDE.md)
