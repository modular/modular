# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: man-page

# Invoking the subcommand with `--help` displays its manual page.
# RUN: mojo doc -bad-option --doesnt-matter --help | FileCheck %s
# CHECK: MOJO-DOC(1)
