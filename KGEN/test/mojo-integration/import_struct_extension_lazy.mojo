# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir -p %t.lazy-test
# RUN: mojo package %S/inputs/struct_and_extension_lazy -o %t.lazy-test/lazy_test.mojopkg
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.lazy-test %s | FileCheck %s

# This imports a mojopkg's sub-package's extension.
# That extension's file loads a struct from yet another package.
# This will load the extension before its target struct.
from lazy_test import simple_extension


fn main():
    # CHECK: lit.call @lazy_test::@simple_struct::@BaseType::@"__init__()"
    var x = lazy_test.simple_struct.BaseType()
