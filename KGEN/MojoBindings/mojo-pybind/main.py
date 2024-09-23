# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import glob
import sys

import click

from src import generate_mojo_extension_module
from src.utils import eprint


@click.command(name="mojo-pybind")
@click.argument("mojo_file")
@click.option(
    "--verbose",
    is_flag=True,
    help=(
        "Whether to print additional verbose information about the binding"
        " generation."
    ),
)
def main(
    mojo_file: click.Path,
    verbose: bool,
):
    # Set working directory if this script was from by Bazel
    if directory := os.getenv("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)

    absolute_path = os.path.abspath(str(mojo_file))

    generate_mojo_extension_module(absolute_path, verbose=verbose)


# ==========================================================

if __name__ == "__main__":
    main()
