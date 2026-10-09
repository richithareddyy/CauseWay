"""Milestone 1 audit of the raw Hillstrom e-mail experiment file.

Standard library only, so it can run before the project environment exists.
Every number quoted in docs/design.md (Sections 1-2) is printed by this script.

Usage:
    python scripts/audit_raw_data.py PATH_TO_ORIGINAL_CSV [PATH_TO_MIRROR_CSV_GZ]
"""

from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import io
import math
import statistics
import sys
from pathlib import Path

COVARIATES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
CATEGORICAL = [
    "recency",
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
]
SEGMENT_BOUNDS = {
    "1) $0 - $100": (0, 100),
    "2) $100 - $200": (100, 200),
    "3) $200 - $350": (200, 350),
    "4) $350 - $500": (350, 500),
    "5) $500 - $750": (500, 750),
    "6) $750 - $1,000": (750, 1000),
    "7) $1,000 +": (1000, math.inf),
}
CONTROL = "No E-Mail"
EMAIL_ARMS = ["Mens E-Mail", "Womens E-Mail"]
Z_POWER = 1.959964 + 0.841621  # two-sided alpha = 0.05, power = 0.80


def parse(raw: bytes) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    return rows[0], rows[1:]


def numerically_equal(a: list[str], b: list[str]) -> bool:
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) > 1e-9:
                return False
        except ValueError:
            return False
    return True


def smd(treated: list[float], control: list[float]) -> float:
    pooled = math.sqrt((statistics.variance(treated) + statistics.variance(control)) / 2)
    return (statistics.mean(treated) - statistics.mean(control)) / pooled


def diff_se(treated: list[float], control: list[float]) -> float:
    return math.sqrt(
        statistics.variance(treated) / len(treated) + statistics.variance(control) / len(control)
    )


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main(orig_path: str, mirror_path: str | None) -> None:
    raw = Path(orig_path).read_bytes()
    header, rows = parse(raw)
    col = {c: i for i, c in enumerate(header)}
    get = lambda r, c: r[col[c]]  # noqa: E731

    section("File")
    print("sha256:", hashlib.sha256(raw).hexdigest())
    print("bytes:", len(raw), "| line endings:", "CRLF" if b"\r\n" in raw else "LF")
    print("header:", header)
    print("rows:", len(rows), "| field counts:", dict(collections.Counter(len(r) for r in rows)))

    if mirror_path:
        section("Mirror comparison")
        packed = Path(mirror_path).read_bytes()
        print("mirror md5 (compressed):", hashlib.md5(packed).hexdigest())
        m_header, m_rows = parse(gzip.decompress(packed))
        print("headers equal:", m_header == header, "| mirror rows:", len(m_rows))
        print("rows identical as text:", sum(a == b for a, b in zip(rows, m_rows, strict=True)))
        print(
            "rows numerically identical:",
            sum(numerically_equal(a, b) for a, b in zip(rows, m_rows, strict=True)),
        )

    section("Missingness")
    for c in header:
        missing = sum(1 for r in rows if get(r, c).strip().upper() in ("", "NA", "NAN", "NULL"))
        print(f"{c}: {missing}")

    section("Categorical domains")
    for c in CATEGORICAL:
        counts = collections.Counter(get(r, c) for r in rows)
        print(c, sorted(counts.items(), key=lambda kv: (len(kv[0]), kv[0])))

    section("Continuous ranges")
    for c in ["history", "spend"]:
        v = [float(get(r, c)) for r in rows]
        print(f"{c}: min={min(v)} max={max(v)} mean={statistics.mean(v):.4f} n>0={sum(x > 0 for x in v)}")
    history_counts = collections.Counter(get(r, "history") for r in rows)
    print("distinct history values:", len(history_counts), "| most common:", history_counts.most_common(3))
    spend_conv = [float(get(r, "spend")) for r in rows if get(r, "conversion") == "1"]
    print(
        "spend | conversion: min",
        min(spend_conv),
        "| n == 29.99:",
        sum(s == 29.99 for s in spend_conv),
        "| n == 499.0:",
        sum(s == 499.0 for s in spend_conv),
        "| n > 499:",
        sum(s > 499 for s in spend_conv),
    )

    section("Duplicates")
    full = collections.Counter(tuple(r) for r in rows)
    dup_groups = {k: c for k, c in full.items() if c > 1}
    print("exact-duplicate rows beyond first copy:", sum(c - 1 for c in dup_groups.values()))
    print(
        "rows in duplicate groups:",
        sum(dup_groups.values()),
        "| of which history == 29.99:",
        sum(c for k, c in dup_groups.items() if k[col["history"]] == "29.99"),
    )
    print(
        "duplicate groups containing a conversion:", sum(1 for k in dup_groups if k[col["conversion"]] == "1")
    )

    section("Logical consistency")
    print(
        "conversion=1 & visit=0:",
        sum(1 for r in rows if get(r, "conversion") == "1" and get(r, "visit") == "0"),
    )
    print(
        "conversion=1 & spend=0:",
        sum(1 for r in rows if get(r, "conversion") == "1" and float(get(r, "spend")) == 0),
    )
    print(
        "conversion=0 & spend>0:",
        sum(1 for r in rows if get(r, "conversion") == "0" and float(get(r, "spend")) > 0),
    )
    print("mens=0 & womens=0:", sum(1 for r in rows if get(r, "mens") == "0" and get(r, "womens") == "0"))
    outside = sum(
        1
        for r in rows
        if not (
            SEGMENT_BOUNDS[get(r, "history_segment")][0]
            <= float(get(r, "history"))
            <= SEGMENT_BOUNDS[get(r, "history_segment")][1]
        )
    )
    print("history outside its history_segment bounds:", outside)

    section("Randomization")
    arms = collections.defaultdict(list)
    for r in rows:
        arms[get(r, "segment")].append(r)
    expected = len(rows) / 3
    chi2 = sum((len(v) - expected) ** 2 / expected for v in arms.values())
    print({k: len(v) for k, v in arms.items()}, f"| chi2(2)={chi2:.4f} p={math.exp(-chi2 / 2):.4f}")
    features = {
        "recency": lambda r: float(get(r, "recency")),
        "log_history": lambda r: math.log(float(get(r, "history"))),
        "mens": lambda r: float(get(r, "mens")),
        "womens": lambda r: float(get(r, "womens")),
        "newbie": lambda r: float(get(r, "newbie")),
        "zip_urban": lambda r: float(get(r, "zip_code") == "Urban"),
        "zip_rural": lambda r: float(get(r, "zip_code") == "Rural"),
        "channel_web": lambda r: float(get(r, "channel") == "Web"),
        "channel_multi": lambda r: float(get(r, "channel") == "Multichannel"),
    }
    control = arms[CONTROL]
    for arm in EMAIL_ARMS:
        smds = {k: smd([f(r) for r in arms[arm]], [f(r) for r in control]) for k, f in features.items()}
        print(f"{arm} vs {CONTROL}: max |SMD| = {max(abs(v) for v in smds.values()):.4f}")

    section("Outcome rates by arm (integrity check)")
    for arm, members in sorted(arms.items()):
        visit = statistics.mean(float(get(r, "visit")) for r in members)
        conv = statistics.mean(float(get(r, "conversion")) for r in members)
        spend = statistics.mean(float(get(r, "spend")) for r in members)
        print(f"{arm:14s} n={len(members)} visit={visit:.4f} conversion={conv:.5f} spend={spend:.4f}")

    section("Minimum detectable effects (80% power, alpha=0.05, two-sided)")
    for outcome in ["spend", "visit", "conversion"]:
        for arm in EMAIL_ARMS:
            se = diff_se(
                [float(get(r, outcome)) for r in arms[arm]], [float(get(r, outcome)) for r in control]
            )
            print(f"{outcome:10s} {arm:13s} SE={se:.5f} MDE={Z_POWER * se:.5f}")
    print("spend MDE by subgroup, Mens E-Mail vs No E-Mail:")
    for key in ["history_segment", "channel", "zip_code", "newbie"]:
        for level in sorted({get(r, key) for r in rows}):
            t = [r for r in arms["Mens E-Mail"] if get(r, key) == level]
            c = [r for r in control if get(r, key) == level]
            se = diff_se([float(get(r, "spend")) for r in t], [float(get(r, "spend")) for r in c])
            conversions = sum(int(get(r, "conversion")) for r in t + c)
            print(
                f"  {key}={level:18s} n={len(t) + len(c):6d} "
                f"conversions={conversions:4d} MDE={Z_POWER * se:.3f}"
            )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
