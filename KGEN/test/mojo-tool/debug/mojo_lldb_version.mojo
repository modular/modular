# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo debug -X --version | FileCheck %s

# CHECK: lldb version
