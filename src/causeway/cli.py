"""Command-line entry point: `causeway data` and `causeway env`."""

from __future__ import annotations

import argparse
import json
import sys

from causeway.config import DataPaths, load_data_config
from causeway.pipeline import environment, run_data_pipeline
from causeway.validation import ValidationError


def _data(_: argparse.Namespace) -> int:
    paths = DataPaths()
    try:
        result = run_data_pipeline(load_data_config(), paths)
    except ValidationError as error:
        print(error, file=sys.stderr)
        print(f"See {paths.reports / 'validation_report.md'}", file=sys.stderr)
        return 1
    manifest = result.manifest
    print(
        f"raw        {manifest['raw']['path']}  ({manifest['raw']['rows']:,} x {manifest['raw']['columns']}, "
        f"source={manifest['raw']['source']})"
    )
    print(f"           sha256 {manifest['raw']['sha256']}")
    for report in (result.raw_report, result.processed_report):
        counts = {s: sum(c.status == s for c in report.checks) for s in ("pass", "warn", "info", "fail")}
        print(f"validation {report.stage:9s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(
        f"processed  {manifest['processed']['path']}  ({manifest['processed']['rows']:,} x "
        f"{manifest['processed']['columns']}, rows removed: {manifest['rows_removed']})"
    )
    print(f"report     {paths.reports / 'validation_report.md'}")
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
    commands.add_parser("env", help="print Python and dependency versions").set_defaults(func=_env)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
