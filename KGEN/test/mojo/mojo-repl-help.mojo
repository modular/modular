# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` prints its help text.
# RUN: %repl -invalid-option --help | FileCheck %s

# CHECK: Launches the REPL
