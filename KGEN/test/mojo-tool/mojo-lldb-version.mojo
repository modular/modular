# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo debug --version | FileCheck %s

# CHECK: lldb version
