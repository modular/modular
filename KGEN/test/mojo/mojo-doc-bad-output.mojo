# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-BAD-OUTPUT
# CHECK-BAD-OUTPUT: mojo{{.*}}: error: cannot open output file 'no/such/directory.mojo': {{N|n}}o such file or directory
