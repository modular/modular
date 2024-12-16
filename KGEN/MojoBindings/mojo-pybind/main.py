# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import click
from src import generate_mojo_extension_module


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
@click.option(
    "--raw-bindings",
    is_flag=True,
    help=(
        'Whether the specified Mojo source file contains "raw" bindings, and'
        " that automatic binding generation should be skipped.\n"
        "\n"
        "When this flag is specified only linking is performed."
    ),
)
def main(
    mojo_file: click.Path,
    verbose: bool,
    raw_bindings: bool,
):
    # Set working directory if this script was from by Bazel
    if directory := os.getenv("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)

    absolute_path = os.path.abspath(str(mojo_file))

    generate_mojo_extension_module(
        absolute_path,
        raw_bindings=raw_bindings,
        verbose=verbose,
    )


# ==========================================================

if __name__ == "__main__":
    main()
