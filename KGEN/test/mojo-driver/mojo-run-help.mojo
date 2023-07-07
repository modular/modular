# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Check that passing the `--help` option without an input argument, or before
# the input argument, prints the `run` command's help text.
# RUN: mojo-driver run --help | FileCheck %s --check-prefix CHECK-HELP
# RUN: mojo-driver run --help %s | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: Build and execute a Mojo file

# Check that passing the `--help` option after the input argument passes it
# along to the underlying Mojo program.
# RUN: mojo-driver run %s --help | FileCheck %s --check-prefix CHECK-NOT-HELP
# RUN: mojo-driver %s --help | FileCheck %s --check-prefix CHECK-NOT-HELP

from IO import print
from Sys import argv


fn main() -> None:
    # CHECK-NOT-HELP: mojo-run-help.mojo
    # CHECK-NOT-HELP: --help
    print(argv()[0])
    print(argv()[1])
