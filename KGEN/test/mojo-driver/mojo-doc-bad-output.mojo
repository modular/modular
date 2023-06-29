# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# XFAIL: windows
# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo-driver doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-OUTPUT
# CHECK-BAD-OUTPUT: mojo-driver{{.*}}: error: cannot open output file 'no/such/directory.mojo': No such file or directory
