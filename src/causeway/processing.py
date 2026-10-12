"""Raw -> interim -> processed transformations, each one recorded in a log."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from causeway import schema
from causeway.validation import parse_raw_types


@dataclass
class Transformation:
    step: int
    stage: str
    columns: list[str]
    action: str
    rows_affected: int
    reason: str


class TransformationLog:
    def __init__(self) -> None:
        self.entries: list[Transformation] = []

    def record(self, stage: str, columns: list[str], action: str, rows_affected: int, reason: str) -> None:
        self.entries.append(
            Transformation(len(self.entries) + 1, stage, columns, action, int(rows_affected), reason)
        )

    def to_records(self) -> list[dict]:
        return [asdict(entry) for entry in self.entries]

    def to_markdown(self) -> str:
        lines = ["| # | Stage | Columns | Action | Rows affected | Reason |", "|---|---|---|---|---|---|"]
        for e in self.entries:
            lines.append(
                f"| {e.step} | {e.stage} | {', '.join(e.columns)} | {e.action} "
                f"| {e.rows_affected:,} | {e.reason} |"
            )
        return "\n".join(lines) + "\n"


def build_interim(raw: pd.DataFrame, log: TransformationLog) -> pd.DataFrame:
    """Typed copy of the raw data. Values and labels are unchanged; only storage types change."""
    interim = parse_raw_types(raw)
    log.record(
        "interim",
        schema.INTEGER_COLUMNS,
        "parse text to int64",
        len(raw),
        "typed storage; every value validated as an integer beforehand",
    )
    log.record(
        "interim",
        schema.FLOAT_COLUMNS,
        "parse text to float64",
        len(raw),
        "typed storage; every value validated as a finite number beforehand",
    )
    return interim


def normalize_zip_code(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Map raw zip_code labels to canonical labels; unknown labels raise instead of passing through."""
    unknown = sorted(set(values.unique()) - set(schema.ZIP_CODE_NORMALIZATION))
    if unknown:
        raise ValueError(f"unrecognised zip_code labels {unknown}; add them explicitly to the schema")
    changed = {
        label: int(count)
        for label, count in values.value_counts().items()
        if schema.ZIP_CODE_NORMALIZATION[label] != label
    }
    return values.map(schema.ZIP_CODE_NORMALIZATION), changed


def recency_band(recency: pd.Series) -> pd.Series:
    labels = list(schema.RECENCY_BANDS)
    edges = [schema.RECENCY_BANDS[labels[0]][0] - 0.5] + [
        high + 0.5 for _, high in schema.RECENCY_BANDS.values()
    ]
    return pd.cut(recency, bins=edges, labels=labels, ordered=True)


def build_processed(interim: pd.DataFrame, log: TransformationLog) -> pd.DataFrame:
    """Analysis dataset: canonical categories, explicit categorical types, and documented derived columns."""
    processed = interim.copy()
    n = len(processed)

    processed.insert(0, "row_id", np.arange(n, dtype="int64"))
    log.record(
        "processed",
        ["row_id"],
        "add 0-based position in the raw file",
        n,
        "traceability to the raw file; it is not a customer identifier",
    )

    processed["zip_code"], changed = normalize_zip_code(processed["zip_code"])
    for label, count in changed.items():
        canonical = schema.ZIP_CODE_NORMALIZATION[label]
        log.record(
            "processed",
            ["zip_code"],
            f"recode '{label}' -> '{canonical}'",
            count,
            "spelling error in the source file; same category",
        )

    categorical = {
        "segment": (schema.ARMS, False),
        "history_segment": (list(schema.HISTORY_SEGMENTS), True),
        "zip_code": (schema.ZIP_CODES, False),
        "channel": (schema.CHANNELS, False),
    }
    for column, (categories, ordered) in categorical.items():
        processed[column] = pd.Categorical(processed[column], categories=categories, ordered=ordered)
        if processed[column].isna().any():
            raise ValueError(f"{column} has labels outside {categories}")
    log.record(
        "processed",
        list(categorical),
        "store as categorical with explicit category order",
        n,
        "fixed reference levels; history_segment ordered by spend bin",
    )

    processed["arm"] = pd.Categorical(
        processed["segment"].map(schema.ARM_CODES).astype(str), categories=list(schema.ARM_CODES.values())
    )
    log.record(
        "processed",
        ["arm"],
        "derive arm code (control / mens / womens) from segment",
        n,
        "short, stable treatment labels; segment is kept unchanged",
    )

    processed["ln_history"] = np.log(processed["history"])
    log.record(
        "processed",
        ["ln_history"],
        "derive natural log of history",
        n,
        "history is right-skewed; ln(history) is the encoding fixed in configs/benchmark/v1.toml",
    )

    processed["recency_band"] = recency_band(processed["recency"])
    log.record(
        "processed",
        ["recency_band"],
        "derive recency bands 1-3 / 4-6 / 7-9 / 10-12 months",
        n,
        "pre-specified segments for heterogeneity analysis (docs/design.md, Section 2.6)",
    )

    return processed
