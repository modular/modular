# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.extension-in-trait-module
# RUN: mojo package %S/inputs/simple_struct_package -o %t.extension-in-trait-module/simple_struct_package.mojoc
# RUN: mojo package %S/inputs/trait_and_extension_package -I %t.extension-in-trait-module -o %t.extension-in-trait-module/trait_and_extension_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.extension-in-trait-module %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from trait_and_extension_package import MyTrait
from simple_struct_package import MyStruct


def use_trait[T: MyTrait & Copyable](value: T):
    pass


# CHECK-LABEL: kgen.generator @"extension_composite::test
def test(var s: MyStruct):
    # CHECK: kgen.call tail @"extension_composite::use_trait
    use_trait(s^)
