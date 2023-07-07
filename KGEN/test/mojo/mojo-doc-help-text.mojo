# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help-text` prints its help text.
# RUN: mojo doc -bad-option --doesnt-matter --help-text | FileCheck %s
# CHECK: Compile doc strings
