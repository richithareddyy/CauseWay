# Causeway: From Correlation to Causal Evidence

Causeway asks whether assigning customers to a promotional e-mail *caused* them to spend more. It uses
the randomized MineThatData e-mail experiment (Hillstrom, 2008).

- **Part I** analyses the randomized experiment. Its estimand is the intention-to-treat effect of
  assignment to each e-mail campaign: Mens vs No E-Mail and Womens vs No E-Mail, as co-primary contrasts.
- **Part II** deliberately breaks randomization in a documented, reproducible way. It then measures how
  well common observational estimators recover the matching experimental benchmark, first under observed
  confounding and then with a hidden confounder.

The project is under active development, milestone by milestone. The verified dataset description, the
pre-specified causal design, and the review decisions are in [`docs/design.md`](docs/design.md). The
observational benchmark was frozen in [`configs/benchmark/v1.toml`](configs/benchmark/v1.toml) before any
benchmark estimate was computed.

## Reproducing the data pipeline

Requires **Python 3.12** (`requires-python = ">=3.12,<3.13"`), which is used everywhere: locally, in the
lock file, in tests and in CI.

```bash
make setup   # .venv with Python 3.12 and the pinned environment in requirements.txt
make data    # download, verify checksum, validate, and build the processed dataset
make analysis  # every analysis and the findings report (about 75 min on 8 cores)
make test    # run the test suite
```

`make data` does the following:

1. Downloads the original file from MineThatData. If it is unavailable, it falls back to the scikit-uplift
   mirror, which is numerically identical.
2. Accepts the file only if its SHA-256 matches [`configs/data.toml`](configs/data.toml), and stores it
   read-only in `data/raw/`. The raw file is never modified. If it has changed, the pipeline stops
   instead of overwriting it.
3. Runs explicit validation checks: schema, row count, missing values, types, category domains, binary
   encodings, impossible values, outcome consistency, treatment arms, allocation ratio, covariate balance,
   duplicates and the spend boundary. Any failure stops the pipeline.
4. Writes `data/interim/` (typed, unchanged values) and `data/processed/hillstrom.parquet` (canonical
   categories plus documented derived columns). No row is removed.
5. Writes `reports/validation/validation_report.md`, which records every check and every transformation,
   and `data/processed/manifest.json`, which records checksums, row counts and package versions.

The data are not stored in this repository. The source states no license, so the pipeline downloads
them from the original publisher.

## Data source

Hillstrom, K. (2008, March 20). *The MineThatData E-Mail Analytics and Data Mining Challenge.*
MineThatData blog. <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html>

The file covers 64,000 customers who purchased within the prior twelve months, randomly assigned in thirds
to a Mens e-mail, a Womens e-mail, or no e-mail. Visits, conversions and spend were tracked for two weeks
after the campaign.

## Repository structure

```
configs/          data sources and the frozen observational benchmark specification
docs/design.md    dataset verification, causal design, review decisions
scripts/          standalone raw-file audit (standard library only)
src/causeway/     pipeline package: acquisition, validation, processing, balance diagnostics
tests/            pytest suite (synthetic fixtures for pipeline mechanics)
data/             raw / interim / processed (generated, not versioned)
reports/          validation reports, figures, results (generated, not versioned)
```
