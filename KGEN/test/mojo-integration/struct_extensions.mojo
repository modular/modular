# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.simple-struct
# RUN: mojo package %S/inputs/simple_struct_package -o %t.simple-struct/simple_struct_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.simple-struct %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from simple_struct_package.simple import PlainStruct


__extension PlainStruct:
    fn sparklebark(self: PlainStruct) -> Bool:
        return True


# CHECK-LABEL: kgen.generator @"struct_extensions::test
fn test():
    var plainStruct = PlainStruct()
    # CHECK: kgen.call {{.*}}PlainStruct::sparklebark
    var result = plainStruct.sparklebark()
