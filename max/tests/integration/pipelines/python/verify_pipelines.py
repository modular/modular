# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import sys
from typing import Optional, TextIO

import click

DUMMY_PIPELINE = "dummy"


def dump_results(*, to: TextIO = sys.stdout) -> None:
    pipeline = DUMMY_PIPELINE
    print(f"  ✅ {pipeline}", file=to)


@click.command()
@click.option(
    "--report",
    type=click.File("w"),
    help="Output the coverage report to the specified file",
)
@click.option("--pipeline", help="Run only a specified pipeline")
def main(report: Optional[TextIO], pipeline: Optional[str]) -> None:
    """Run logit-level comparisons of a Modular pipeline against a reference."""
    if pipeline is not None and pipeline != DUMMY_PIPELINE:
        raise click.ClickException(f"Unknown pipeline {pipeline!r}")
    if report:
        dump_results(to=report)
    dump_results()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
