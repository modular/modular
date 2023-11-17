# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo package %S/inputs/test_package -o %T/test_package_trait.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %T %s | FileCheck %s

from test_package_trait.module import PackageTrait, UseTrait


struct MyType(PackageTrait):
    fn method(self):
        pass


fn use_trait[T: PackageTrait](x: UseTrait, y: T):
    y.method()


# CHECK: lit.trait.decl @PackageTrait
# CHECK: lit.trait.decl @UsedInPackageTrait
# CHECK: lit.struct.decl @UseTrait
# CHECK-SAME: {{.*}}@UsedInPackageTrait
