# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` displays its manual page.
# RUN: mojo doc -bad-option --doesnt-matter --help | FileCheck %s

# Invoking the subcommand with `--help-text` prints its help text.
# RUN: mojo doc -bad-option --doesnt-matter --help-text | FileCheck %s

# CHECK: Compiles doc strings.
