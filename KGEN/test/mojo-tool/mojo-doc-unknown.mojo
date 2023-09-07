# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Reject unknown options.
# RUN: not mojo doc -one --two 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo{{.*}}: error: unrecognized argument '--two'
