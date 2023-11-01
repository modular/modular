# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package %S/inputs/test_package --debug-level full -o %T/test_package_debug.mojopkg
# RUN: kgen -elaborate --debug-level none -I %T -S %s | FileCheck %s

# Generate a package with debuginfo. Then use it to build an executable without debuginfo.

# This `test_package_debug` package is only built during testing.
from test_package_debug.module import identity


# CHECK-NOT: debuginfo
fn main():
    print(identity(2))
