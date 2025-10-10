# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.extension-constructor
# RUN: mojo package -disable-builtins %S/inputs/extension_constructor_package -o %t.extension-constructor/extension_constructor_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.extension-constructor %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from extension_constructor_package import MyType, ZInt


# CHECK-LABEL: kgen.generator @"struct_extensions_constructors::test
fn test():
    # Call constructor from the struct
    var obj1 = MyType()
    # CHECK: kgen.call @"extension_constructor_package::my_type::MyType::__init__()

    # Call constructor from the extension
    var obj2 = MyType(ZInt())
    # CHECK: kgen.call @"extension_constructor_package::my_type::extension:MyType::__init__(extension_constructor_package::my_type::ZInt


# TODO(MOCO-522): Combine this with struct_extensions.mojo.
