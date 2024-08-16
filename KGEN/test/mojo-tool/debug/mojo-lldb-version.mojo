# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# LLDB fails with asan because it's built by default with python support in the
# CI, and python fails asan.
# UNSUPPORTED: asan

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo debug -X --version | FileCheck %s

# CHECK: lldb version
