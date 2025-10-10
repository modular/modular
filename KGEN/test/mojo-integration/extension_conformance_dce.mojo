# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.extension-conformance
# RUN: mojo package -disable-builtins %S/inputs/extension_conformance_package -o %t.extension-conformance/extension_conformance_package.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.extension-conformance %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

# This test verifies that extensions with trait conformances survive DCE and
# are properly processed during LowerLIT when imported from a package.

alias int = __mlir_type.index
alias `42` = __mlir_attr.`42 : index`

from extension_conformance_package import MyStruct, Convertible


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
    # CHECK: kgen.call @"extension_conformance_dce::use_convertible[extension_conformance_package::my_struct::Convertible]
    return use_convertible(s)
