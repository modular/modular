# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Invoking the subcommand with `--help` prints its help text.
# RUN: mojo doc -bad-option --doesnt-matter --help | FileCheck %s

# CHECK: mojo-doc
