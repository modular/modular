# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.struct-and-conforming-extension
# RUN: mojo package %S/inputs/struct_and_conforming_extension_package -o %t.struct-and-conforming-extension/struct_and_conforming_extension_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.struct-and-conforming-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

# This test verifies that extensions with trait conformances survive DCE and
# are properly processed during LowerLIT when imported from a package.

alias int = __mlir_type.index
alias `42` = __mlir_attr.`42 : index`

from struct_and_conforming_extension_package import MyStruct, Convertible


fn use_convertible[T: Convertible](x: T) -> int:
    return x.convert()


# CHECK-LABEL: kgen.generator @"extension_conformance_dce::test
fn test() -> int:
    var s = MyStruct(`42`)

    # This uses the alias defined in the extension, which requires the
    # extension to survive DCE during importing.
    alias t = MyStruct.ExtensionAlias

    # This call requires the extension conformance to be pulled in from the
    # mojopkg. Without it, this would fail in the elaborator because it can't
    # find the conformance.
    # CHECK: kgen.call @"extension_conformance_dce::use_convertible[struct_and_conforming_extension_package::my_struct::Convertible]
    return use_convertible(s)
