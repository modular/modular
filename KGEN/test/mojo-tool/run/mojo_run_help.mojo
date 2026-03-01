# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` displays its help text.
# RUN: mojo run -invalid-option --help | FileCheck %s

# Check that passing the `--help` option without an input argument, or
# before the input argument, prints the `run` command's help text.
# RUN: mojo run --help | FileCheck %s
# RUN: mojo run --help %s | FileCheck %s

# CHECK: mojo-run

# Check that passing the `--help` option after the input argument passes it
# along to the underlying Mojo program.
# RUN: mojo run %s --help | FileCheck %s --check-prefix CHECK-ARGV
# RUN: %mojo %s --help | FileCheck %s --check-prefix CHECK-ARGV

from std.sys import argv


fn main() -> None:
    # CHECK-ARGV: mojo_run_help.mojo
    # CHECK-ARGV: --help
    print(argv()[0])
    print(argv()[1])
