"""Validation checks for the raw and processed data. They report problems and never change the data."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from causeway import schema
from causeway.balance import standardized_mean_difference
from causeway.features import covariate_matrix

Status = Literal["pass", "warn", "fail", "info"]
MISSING_TOKENS = {"", "NA", "NAN", "NULL", "NONE"}
INTEGER_PATTERN = re.compile(r"^-?\d+$")
SMD_WARNING_THRESHOLD = 0.1
ALLOCATION_WARNING_P = 0.001


@dataclass
class CheckResult:
    name: str
    category: str
    status: Status
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    stage: str
    n_rows: int
    n_columns: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(check.status == "fail" for check in self.checks)

    def add(self, name: str, category: str, status: Status, message: str, **details) -> None:
        self.checks.append(CheckResult(name, category, status, message, details))

    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "fail"]

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_json_default) + "\n"

    def to_markdown(self) -> str:
        lines = [
            f"## Validation: {self.stage}",
            "",
            f"Rows: {self.n_rows:,} | Columns: {self.n_columns} | "
            f"Result: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "| Check | Category | Status | Result |",
            "|---|---|---|---|",
        ]
        for check in self.checks:
            message = check.message.replace("|", "\\|")
            lines.append(f"| `{check.name}` | {check.category} | {check.status.upper()} | {message} |")
        return "\n".join(lines) + "\n"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


class ValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        names = ", ".join(check.name for check in report.failures())
        super().__init__(f"{report.stage} validation failed: {names}")
        self.report = report


def parse_raw_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert validated raw text to numeric types without changing any value or label."""
    typed = raw.copy()
    for column in schema.INTEGER_COLUMNS:
        typed[column] = typed[column].astype("int64")
    for column in schema.FLOAT_COLUMNS:
        typed[column] = typed[column].astype("float64")
    return typed


def covariate_design(typed: pd.DataFrame) -> pd.DataFrame:
    """Shared covariate encoding, applied after normalising raw zip_code spellings."""
    return covariate_matrix(typed.assign(zip_code=typed["zip_code"].map(schema.ZIP_CODE_NORMALIZATION)))


def validate_raw(raw: pd.DataFrame, expected_rows: int | None = None) -> ValidationReport:
    """Run every raw-data check. Content checks run only once structure and types are sound."""
    report = ValidationReport("raw", n_rows=len(raw), n_columns=raw.shape[1])
    if not _check_structure(raw, report, expected_rows):
        return report
    if not _check_types(raw, report):
        return report
    typed = parse_raw_types(raw)
    _check_domains(raw, report)
    _check_ranges(typed, report)
    _check_duplicates(raw, report)
    if not report.passed:
        return _skip(
            report, "consistency, treatment, balance and boundary checks need valid labels and ranges"
        )
    _check_consistency(typed, report)
    _check_treatment(typed, report)
    if not report.passed:
        return _skip(report, "balance and boundary checks need consistent outcomes and all three arms")
    _check_balance(typed, report)
    _check_spend_boundary(typed, report)
    return report


def _skip(report: ValidationReport, reason: str) -> ValidationReport:
    report.add("remaining_checks", "control", "info", f"skipped after earlier failures: {reason}")
    return report


def _check_structure(raw: pd.DataFrame, report: ValidationReport, expected_rows: int | None) -> bool:
    columns = list(raw.columns)
    if columns == schema.COLUMNS:
        report.add("schema_columns", "structure", "pass", "12 expected columns, in the documented order")
    else:
        missing = [c for c in schema.COLUMNS if c not in columns]
        unexpected = [c for c in columns if c not in schema.COLUMNS]
        report.add(
            "schema_columns",
            "structure",
            "fail",
            f"missing={missing}, unexpected={unexpected}"
            if missing or unexpected
            else "all expected columns present but in a different order",
            missing=missing,
            unexpected=unexpected,
            observed=columns,
        )
        return False

    if expected_rows is None:
        report.add("row_count", "structure", "info", f"{len(raw):,} rows (no expected count configured)")
    elif len(raw) == expected_rows:
        report.add("row_count", "structure", "pass", f"{len(raw):,} rows, as documented by the source")
    else:
        report.add("row_count", "structure", "fail", f"{len(raw):,} rows; source documents {expected_rows:,}")

    missing_counts = {
        column: int(raw[column].str.strip().str.upper().isin(MISSING_TOKENS).sum()) for column in columns
    }
    total_missing = sum(missing_counts.values())
    if total_missing == 0:
        report.add("missing_values", "completeness", "pass", "no empty or NA-like values in any column")
    else:
        report.add(
            "missing_values",
            "completeness",
            "fail",
            f"{total_missing} missing values; the pipeline does not impute",
            by_column={k: v for k, v in missing_counts.items() if v},
        )
    return total_missing == 0


def _check_types(raw: pd.DataFrame, report: ValidationReport) -> bool:
    problems = {}
    for column in schema.INTEGER_COLUMNS:
        bad = ~raw[column].str.match(INTEGER_PATTERN)
        if bad.any():
            problems[column] = raw.loc[bad, column].unique()[:5].tolist()
    for column in schema.FLOAT_COLUMNS:
        parsed = pd.to_numeric(raw[column], errors="coerce")
        bad = parsed.isna() | ~np.isfinite(parsed)
        if bad.any():
            problems[column] = raw.loc[bad, column].unique()[:5].tolist()
    if problems:
        report.add(
            "data_types", "types", "fail", f"unparseable values in {sorted(problems)}", examples=problems
        )
        return False
    report.add("data_types", "types", "pass", "integer and decimal columns parse without loss")
    return True


def _check_domains(raw: pd.DataFrame, report: ValidationReport) -> None:
    for column, allowed in schema.RAW_DOMAINS.items():
        observed = raw[column].value_counts().to_dict()
        unknown = sorted(set(observed) - allowed)
        if unknown:
            report.add(
                f"domain_{column}", "categories", "fail", f"unrecognised labels {unknown}", counts=observed
            )
        else:
            report.add(
                f"domain_{column}",
                "categories",
                "pass",
                f"{len(observed)} recognised labels",
                counts=observed,
            )

    variants = {
        label: count
        for label, count in raw["zip_code"].value_counts().items()
        if schema.ZIP_CODE_NORMALIZATION.get(label, label) != label
    }
    if variants:
        message = ", ".join(
            f"'{label}' x{count:,} -> '{schema.ZIP_CODE_NORMALIZATION[label]}'"
            for label, count in variants.items()
        )
        report.add(
            "category_spelling",
            "categories",
            "info",
            f"source spelling normalised in processing: {message}",
            variants=variants,
        )

    binary_problems = {c: sorted(set(raw[c]) - {"0", "1"}) for c in schema.BINARY_COLUMNS}
    binary_problems = {c: v for c, v in binary_problems.items() if v}
    if binary_problems:
        report.add("binary_encoding", "encoding", "fail", f"non-0/1 values: {binary_problems}")
    else:
        report.add("binary_encoding", "encoding", "pass", f"{', '.join(schema.BINARY_COLUMNS)} are all 0/1")


def _check_ranges(typed: pd.DataFrame, report: ValidationReport) -> None:
    low, high = schema.RECENCY_RANGE
    out = typed[(typed["recency"] < low) | (typed["recency"] > high)]
    report.add(
        "recency_range",
        "impossible_values",
        "fail" if len(out) else "pass",
        f"{len(out)} rows outside {low}-{high} months" if len(out) else f"all within {low}-{high} months",
    )
    bad_history = typed[typed["history"] <= 0]
    report.add(
        "history_positive",
        "impossible_values",
        "fail" if len(bad_history) else "pass",
        f"{len(bad_history)} rows with history <= 0"
        if len(bad_history)
        else f"all positive (min {typed['history'].min():.2f}, max {typed['history'].max():.2f})",
    )
    bad_spend = typed[typed["spend"] < 0]
    report.add(
        "spend_nonnegative",
        "impossible_values",
        "fail" if len(bad_spend) else "pass",
        f"{len(bad_spend)} rows with negative spend" if len(bad_spend) else "no negative spend",
    )


def _check_consistency(typed: pd.DataFrame, report: ValidationReport) -> None:
    bounds = typed["history_segment"].map(schema.HISTORY_SEGMENTS)
    lower = bounds.map(lambda b: b[0])
    upper = bounds.map(lambda b: b[1])
    outside = int(((typed["history"] < lower) | (typed["history"] > upper)).sum())
    report.add(
        "history_segment_consistency",
        "consistency",
        "fail" if outside else "pass",
        f"{outside} rows whose history falls outside its history_segment bin"
        if outside
        else "every history value lies inside its history_segment bin",
    )

    converted = typed["conversion"] == 1
    no_visit = int((converted & (typed["visit"] == 0)).sum())
    report.add(
        "conversion_implies_visit",
        "outcome_encoding",
        "warn" if no_visit else "pass",
        f"{no_visit} converters without a recorded visit" if no_visit else "every converter also visited",
    )
    spend_without_conversion = int((~converted & (typed["spend"] > 0)).sum())
    conversion_without_spend = int((converted & (typed["spend"] == 0)).sum())
    inconsistent = spend_without_conversion + conversion_without_spend
    report.add(
        "spend_conversion_consistency",
        "outcome_encoding",
        "fail" if inconsistent else "pass",
        f"{spend_without_conversion} spend>0 without conversion; "
        f"{conversion_without_spend} conversions with zero spend"
        if inconsistent
        else "spend > 0 exactly when conversion = 1",
    )

    neither = int(((typed["mens"] == 0) & (typed["womens"] == 0)).sum())
    report.add(
        "merchandise_history",
        "consistency",
        "warn" if neither else "pass",
        f"{neither} customers bought neither Mens nor Womens merchandise"
        if neither
        else "every customer bought Mens and/or Womens merchandise in the past year",
    )


def _check_treatment(typed: pd.DataFrame, report: ValidationReport) -> None:
    counts = typed["segment"].value_counts().reindex(schema.ARMS, fill_value=0)
    missing_arms = [arm for arm, n in counts.items() if n == 0]
    if missing_arms:
        report.add("treatment_arms", "treatment_encoding", "fail", f"arms with no customers: {missing_arms}")
        return
    report.add(
        "treatment_arms",
        "treatment_encoding",
        "pass",
        ", ".join(f"{arm}: {n:,}" for arm, n in counts.items()),
        counts=counts.to_dict(),
    )
    chi2, p_value = stats.chisquare(counts.to_numpy())
    report.add(
        "allocation_ratio",
        "sample_counts",
        "warn" if p_value < ALLOCATION_WARNING_P else "pass",
        f"chi-square test of equal thirds: chi2(2) = {chi2:.3f}, p = {p_value:.3f}",
        chi2=float(chi2),
        p_value=float(p_value),
    )


def _check_balance(typed: pd.DataFrame, report: ValidationReport) -> None:
    design = covariate_design(typed)
    control = typed["segment"] == schema.CONTROL_ARM
    for arm in schema.ARMS[1:]:
        treated = typed["segment"] == arm
        smds = {
            column: standardized_mean_difference(design.loc[treated, column], design.loc[control, column])
            for column in design.columns
        }
        worst = max(smds, key=lambda c: abs(smds[c]))
        imbalanced = {c: v for c, v in smds.items() if abs(v) > SMD_WARNING_THRESHOLD}
        report.add(
            f"balance_{schema.ARM_CODES[arm]}_vs_control",
            "treatment_balance",
            "warn" if imbalanced else "pass",
            f"max |SMD| = {abs(smds[worst]):.4f} ({worst}); threshold {SMD_WARNING_THRESHOLD}",
            smd=smds,
        )


def _check_duplicates(raw: pd.DataFrame, report: ValidationReport) -> None:
    duplicated = raw.duplicated(keep="first")
    n_extra = int(duplicated.sum())
    in_groups = raw.duplicated(keep=False)
    floor_share = float((raw.loc[in_groups, "history"] == "29.99").mean()) if in_groups.any() else 0.0
    groups_with_conversion = int(raw.loc[in_groups & (raw["conversion"] == "1")].drop_duplicates().shape[0])
    report.add(
        "exact_duplicate_rows",
        "duplicates",
        "info",
        f"{n_extra:,} rows repeat an earlier row exactly ({int(in_groups.sum()):,} rows in "
        f"{int(raw[in_groups].drop_duplicates().shape[0]):,} groups; "
        f"{floor_share:.1%} of them have history = 29.99). "
        "Retained: with no customer identifier, identical features do not establish duplicate customers.",
        extra_copies=n_extra,
        rows_in_groups=int(in_groups.sum()),
        share_history_29_99=floor_share,
        groups_with_conversion=groups_with_conversion,
    )


def _check_spend_boundary(typed: pd.DataFrame, report: ValidationReport) -> None:
    maximum = float(typed["spend"].max())
    at_max = int((typed["spend"] == maximum).sum())
    positive = typed.loc[typed["spend"] > 0, "spend"]
    report.add(
        "spend_upper_boundary",
        "anomalies",
        "info",
        f"observed maximum spend {maximum:.2f} reached by {at_max} customers; whether this is a true maximum "
        "or a data-generation constraint is unknown",
        maximum=maximum,
        n_at_maximum=at_max,
        n_positive=int(len(positive)),
        minimum_positive=float(positive.min()) if len(positive) else None,
    )


def validate_processed(processed: pd.DataFrame, n_raw_rows: int) -> ValidationReport:
    """Checks that processing preserved every record and produced the canonical encodings."""
    report = ValidationReport("processed", n_rows=len(processed), n_columns=processed.shape[1])
    report.add(
        "row_count_preserved",
        "structure",
        "pass" if len(processed) == n_raw_rows else "fail",
        f"{len(processed):,} processed rows vs {n_raw_rows:,} raw rows",
    )
    row_ids = processed["row_id"].to_numpy()
    report.add(
        "row_id_traceability",
        "structure",
        "pass" if np.array_equal(row_ids, np.arange(n_raw_rows)) else "fail",
        "row_id is 0..n-1 in raw-file order",
    )
    n_missing = int(processed.isna().sum().sum())
    report.add(
        "no_missing_after_processing",
        "completeness",
        "fail" if n_missing else "pass",
        f"{n_missing} missing values" if n_missing else "no missing values",
    )
    canonical = {"zip_code": schema.ZIP_CODES, "channel": schema.CHANNELS, "segment": schema.ARMS}
    for column, categories in canonical.items():
        observed = set(processed[column].astype(str).unique())
        report.add(
            f"canonical_{column}",
            "categories",
            "pass" if observed <= set(categories) else "fail",
            f"labels {sorted(observed)}",
        )
    arm_ok = (processed["segment"].astype(str).map(schema.ARM_CODES) == processed["arm"].astype(str)).all()
    report.add(
        "arm_code_mapping",
        "treatment_encoding",
        "pass" if arm_ok else "fail",
        "arm code agrees with segment for every row",
    )
    return report
