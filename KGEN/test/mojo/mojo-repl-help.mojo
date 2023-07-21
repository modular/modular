# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` displays its help text.
# RUN: %repl -invalid-option --help | FileCheck %s

# Invoking the subcommand with `--help-text` prints its help text.
# RUN: %repl -invalid-option --help-text | FileCheck %s

# CHECK: Launches the REPL.
