# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.struct-extensions
# RUN: mojo package -disable-builtins %S/inputs/struct_extensions_package -o %t.struct-extensions/struct_extensions_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.struct-extensions %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from struct_extensions_package.simple import PlainStruct


__extension PlainStruct:
    fn sparklebark(self: PlainStruct) -> Bool:
        return True


# CHECK-LABEL: kgen.generator @"struct_extensions::test
fn test():
    var plainStruct = PlainStruct()
    # CHECK: kgen.call @"struct_extensions::extension:PlainStruct::sparklebark
    var result = plainStruct.sparklebark()


# TODO(MOCO-522): Add test for generic structs going through LowerLIT
