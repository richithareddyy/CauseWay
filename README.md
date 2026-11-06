# CauseWay

A company emails its best customers and sees them spend more. Did the email cause that, or would they have bought anyway? That comparison alone can’t tell you. CauseWay starts with a randomized email experiment, then deliberately creates biased samples to test how well causal methods recover the experimental estimate.

The study uses the Hillstrom experiment, with 64,000 customers. It first estimates the effect of assigning customers to each email campaign. It then selects biased subsets using a known sampling rule and checks how closely different methods recover the experimental estimate.

Python 3.12, pandas, scikit-learn, EconML, DoubleML.

## Main findings

- Assignment to the Mens email increased two-week spending by about $0.77 per customer; the Womens email increased it by about $0.42.
- When the variables responsible for selection are available, weighting and AIPW—a method combining outcome and treatment models—recover the benchmark within the uncertainty from repeated sampling.
- When one of those variables is hidden, every tested method remains biased, even when diagnostic checks pass.

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

The comparison includes matching, weighting, regression, AIPW, and double machine learning (DML). The AIPW and DML calculations are implemented here and checked against EconML and DoubleML. Analysis plans were committed before results were available. The targeting policies are evaluated on data, not deployed campaigns.

[Research design](docs/design.md) · [Results](docs/findings.md) · [Setup and technical details](GUIDE.md)
