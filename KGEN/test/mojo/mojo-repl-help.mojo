# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: man-page

# Invoking the subcommand with `--help` displays its help text.
# RUN: %repl -invalid-option --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: MOJO-REPL(1)
