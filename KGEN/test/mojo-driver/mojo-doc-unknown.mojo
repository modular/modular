# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Reject unknown options.
# RUN: not mojo-driver doc -one --two 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo-driver{{.*}}: error: unrecognized argument '--two'
