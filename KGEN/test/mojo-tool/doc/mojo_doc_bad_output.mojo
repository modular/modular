# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s
# CHECK: mojo{{.*}}: error: cannot open output file 'no/such/directory.mojo': {{N|n}}o such file or directory

# RUN: not mojo doc --diagnostic-format json %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-DIAG
# CHECK-DIAG: {"kind":"error","message":"cannot open output file{{.*}}"}
