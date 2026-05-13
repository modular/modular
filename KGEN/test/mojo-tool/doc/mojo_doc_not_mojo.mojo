# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We don't allow input files that don't end in '.mojo'.
# RUN: touch %t.foo
# RUN: not mojo doc %t.foo 2>&1 | FileCheck %s --check-prefix CHECK-NOT-MOJO
# CHECK-NOT-MOJO: mojo{{.*}}: error: cannot open '{{.*}}.foo', since it does not appear to be a Mojo file (it does not end in '.mojo', '.mojoc' or '.mojopkg') or a Mojo source package
