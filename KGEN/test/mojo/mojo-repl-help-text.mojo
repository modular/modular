# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help-text` prints its help text.
# RUN: %repl -invalid-option --help-text | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: Launch the REPL
