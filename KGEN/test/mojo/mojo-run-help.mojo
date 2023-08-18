# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` displays its help text.
# RUN: mojo run -invalid-option --help | FileCheck %s

# Check that passing the `--help-text` option without an input argument, or
# before the input argument, prints the `run` command's help text.
# RUN: mojo run --help-text | FileCheck %s
# RUN: mojo run %mojo_cpu_build_arch --help-text %s | FileCheck %s

# CHECK: Builds and executes a Mojo file

# Check that passing the `--help-text` option after the input argument passes it
# along to the underlying Mojo program.
# RUN: mojo run %mojo_cpu_build_arch %s --help-text | FileCheck %s --check-prefix CHECK-NOT-HELP
# RUN: %mojo %s --help-text | FileCheck %s --check-prefix CHECK-NOT-HELP

from sys import argv


fn main() -> None:
    # CHECK-NOT-HELP: mojo-run-help.mojo
    # CHECK-NOT-HELP: --help-text
    print(argv()[0])
    print(argv()[1])
