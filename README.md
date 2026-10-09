# Causeway: From Correlation to Causal Evidence

Causeway asks whether a promotional e-mail actually *caused* customers to spend more, using the
randomized MineThatData e-mail experiment (Hillstrom, 2008). Part I analyses the randomized
experiment directly. Part II deliberately breaks randomization in a documented, reproducible way to
measure how well observational causal estimators recover the experimental benchmark.

The project is under active development. The verified dataset description and the pre-specified
causal design are in [`docs/design.md`](docs/design.md).

The raw data are not stored in this repository. `scripts/audit_raw_data.py` audits the original
file:

```bash
curl -L -o hillstrom.csv "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
python scripts/audit_raw_data.py hillstrom.csv
```
