# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test using a child trait from a package without using the parent trait.

# RUN: mojo package %S/inputs/test_package -o %T/test_package_trait.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %T %s --kgen-print-inline-type-values | FileCheck %s

from test_package_trait.module2 import *


# CHECK: lit.struct.decl @MyType({{.*}}PackageChildTrait
struct MyType(PackageChildTrait):
    fn method(self):
        pass

    fn method2(self):
        pass
