# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.cross-package-extension
# RUN: mojo package %S/inputs/struct_only_package -o %t.cross-package-extension/struct_only_package.mojopkg
# RUN: mojo package -I %t.cross-package-extension %S/inputs/extension_only_package -o %t.cross-package-extension/extension_only_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.cross-package-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from struct_only_package import MyStruct
from extension_only_package import MyStruct


# CHECK-LABEL: kgen.generator @"struct_extensions_cross_package::test
fn test():
    var car = MyStruct()

    # Call method from the original struct
    car.accelerate()
    # CHECK: kgen.call @"struct_only_package::my_struct::MyStruct::accelerate

    # Call method from the extension
    var current_speed = car.get_speed()
    # CHECK: kgen.call @"extension_only_package::my_extension::extension:MyStruct::get_speed
