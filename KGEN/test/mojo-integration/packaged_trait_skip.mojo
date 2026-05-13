# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test using a child trait from a package without using the parent trait.

# RUN: mkdir -p %t.packaged-trait-skip
# RUN: mojo package %S/inputs/test_package -o %t.packaged-trait-skip/test_package_trait.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.packaged-trait-skip %s --kgen-print-inline-type-values | FileCheck %s

from test_package_trait.module2 import *


# CHECK: lit.struct.decl @MyType({{.*}}PackageChildTrait
struct MyType(PackageChildTrait):
    def method(self):
        pass

    def method2(self):
        pass
