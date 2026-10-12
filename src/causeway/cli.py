"""Command-line entry point (`python -m causeway.cli <command>`)."""

from __future__ import annotations

import argparse
import json
import sys

from causeway.config import DataPaths, load_data_config
from causeway.pipeline import environment, run_data_pipeline
from causeway.validation import ValidationError

# command -> (report function in causeway.reporting, help text). Imported lazily so `causeway data`
# doesn't pay for loading EconML and DoubleML.
REPORTS = {
    "experiment": ("run_experiment_report", "Part I: analyse the randomized experiment"),
    "benchmark": ("run_benchmark_construction_report", "Part II: construct and check the benchmark"),
    "propensity": ("run_propensity_report", "Part II: propensity scores, weighting and matching"),
    "estimation": ("run_estimation_report", "Part II: outcome regression, AIPW and DML"),
    "compare": ("run_comparison_report", "Part II: compare every estimator with the benchmark"),
    "heterogeneity": ("run_heterogeneity_report", "heterogeneous effects and targeting"),
    "robustness": ("run_robustness_report", "robustness and sensitivity checks"),
}


def _data(_: argparse.Namespace) -> int:
    paths = DataPaths()
    try:
        result = run_data_pipeline(load_data_config(), paths)
    except ValidationError as error:
        print(error, file=sys.stderr)
        print(f"See {paths.reports / 'validation_report.md'}", file=sys.stderr)
        return 1
    raw, processed = result.manifest["raw"], result.manifest["processed"]
    print(f"raw        {raw['path']}  ({raw['rows']:,} x {raw['columns']}, source={raw['source']})")
    print(f"           sha256 {raw['sha256']}")
    for report in (result.raw_report, result.processed_report):
        counts = {s: sum(c.status == s for c in report.checks) for s in ("pass", "warn", "info", "fail")}
        print(f"validation {report.stage:9s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(
        f"processed  {processed['path']}  ({processed['rows']:,} x {processed['columns']}, "
        f"rows removed: {result.manifest['rows_removed']})"
    )
    print(f"report     {paths.reports / 'validation_report.md'}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from causeway import reporting

    if getattr(args, "render_only", False):
        outputs = getattr(reporting, f"render_{args.command}_report")()
    else:
        outputs = getattr(reporting, REPORTS[args.command][0])()
    print(f"report   {outputs.report_path}")
    for figure in outputs.figures:
        print(f"figure   {figure}")
    return 0


def _env(_: argparse.Namespace) -> int:
    print(json.dumps(environment(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causeway")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("data", help="download, verify, validate and process the dataset").set_defaults(
        func=_data
    )
    for name, (_, help_text) in REPORTS.items():
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(func=_report)
    for name in ("heterogeneity", "robustness"):
        commands.choices[name].add_argument(
            "--render-only", action="store_true", help="redraw figures and the report from saved tables"
        )
    commands.add_parser("env", help="print Python and dependency versions").set_defaults(func=_env)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
