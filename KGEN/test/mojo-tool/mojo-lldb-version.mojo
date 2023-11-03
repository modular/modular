# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# LLDB fails with asan, see https://github.com/modularml/modular/actions/runs/6748079891/job/18345656726
# UNSUPPORTED: asan

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo debug --version | FileCheck %s

# CHECK: lldb version
