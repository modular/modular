# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Check that passing the `--help-text` option without an input argument, or before
# the input argument, prints the `run` command's help text.
# RUN: mojo run --help-text | FileCheck %s --check-prefix CHECK-HELP
# RUN: mojo run --help-text %s | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: Build and execute a Mojo file

# Check that passing the `--help-text` option after the input argument passes it
# along to the underlying Mojo program.
# RUN: mojo run %s --help-text | FileCheck %s --check-prefix CHECK-NOT-HELP
# RUN: mojo %s --help-text | FileCheck %s --check-prefix CHECK-NOT-HELP

from IO import print
from Sys import argv


fn main() -> None:
    # CHECK-NOT-HELP: mojo-run-help-text.mojo
    # CHECK-NOT-HELP: --help-text
    print(argv()[0])
    print(argv()[1])
