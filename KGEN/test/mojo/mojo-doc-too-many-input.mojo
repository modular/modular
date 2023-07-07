# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# More than one input is not allowed.
# RUN: not mojo doc %t.1.mojo %t.2.mojo 2>&1 | FileCheck %s --check-prefix CHECK-TOO-MANY-INPUT
# CHECK-TOO-MANY-INPUT: mojo{{.*}}: error: too many input files, cannot process both '{{.*}}.1.mojo' and '{{.*}}.2.mojo'
