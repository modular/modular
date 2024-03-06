# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# The user must provide an input.
# RUN: not mojo doc 2>&1 | FileCheck %s --check-prefix CHECK-NO-INPUT
# CHECK-NO-INPUT: mojo{{.*}}: error: no input file provided
