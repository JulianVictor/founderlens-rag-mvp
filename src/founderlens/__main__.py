"""Entry point: ``python -m founderlens``.

Prints package and path diagnostics. Requires no API key, network access or
external service, and creates no files or directories.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from founderlens import __version__
from founderlens.config import ConfigError, load_settings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens",
        description="FounderLens AI RAG MVP: package status check.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Absolute path to the project root (overrides discovery).",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)
    except ConfigError as exc:
        print(f"founderlens: configuration error: {exc}", file=sys.stderr)
        return 1

    print(f"FounderLens AI RAG MVP {__version__}")
    print(f"Python {platform.python_version()}")
    print(f"Project root:   {settings.project_root}")
    for label, path in (
        ("Raw data:", settings.raw_data_dir),
        ("Processed data:", settings.processed_data_dir),
        ("Artifacts:", settings.artifacts_dir),
        ("Reports:", settings.reports_dir),
    ):
        status = "present" if path.is_dir() else "not created"
        print(f"{label:<16}{path} ({status})")
    print(
        "Status: OK (extraction, chunking, BM25/dense/hybrid retrieval, OpenAI provider and "
        "structured grounded generation implemented; citation resolution not implemented)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
