# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ASAN fails due to ODR violation in liblldb and lldb
# UNSUPPORTED: asan

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo debug -X --version | FileCheck %s

# CHECK: lldb version
