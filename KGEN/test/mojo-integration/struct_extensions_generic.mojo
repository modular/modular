# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.generic-struct
# RUN: mojo package %S/inputs/generic_struct_package -o %t.generic-struct/generic_struct_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.generic-struct %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from generic_struct_package import Container


__extension Container:
    def double(self) -> T:
        return self.value


# CHECK-LABEL: kgen.generator @"struct_extensions_generic::my_test
def my_test():
    var container = Container(42)
    # CHECK: kgen.call @"struct_extensions_generic::extension:Container::double
    var result = container.double()
