# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.function-thunks
# RUN: mkdir -p %t.function-thunks
# RUN: mojo package %S/inputs/func_package_foo -o %t.function-thunks/func_package_foo.mojopkg
# RUN: mojo package %S/inputs/func_package_bar -o %t.function-thunks/func_package_bar.mojopkg
# RUN: kgen-opt %t.function-thunks/func_package_foo.mojopkg | FileCheck %s --check-prefix=THUNK
# RUN: kgen-opt %t.function-thunks/func_package_bar.mojopkg | FileCheck %s --check-prefix=THUNK
# RUN: kgen-translate -import-mojo %s --mojo-enable-prebuilt-packages -I %t.function-thunks | FileCheck %s
# RUN: mojo doc -Werror %s -o /dev/null -I %t.function-thunks

# THUNK-COUNT-1: lit.fn @"def

from func_package_foo.module import foo
from func_package_bar.module import bar

# CHECK-LABEL: lit.file_module @function_thunks


def thunk[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_fn
def test_fn():
    # CHECK: lit.call {{.*}}foo
    _ = foo()
    # CHECK: lit.call {{.*}}bar
    _ = bar()

    # CHECK: kgen.create_closure{{.*}}@"def(::Int, /) -> None|def(x: ::Int) -> None|{{.*}}[def(x: ::Int) -> None](::Int)"
    var f: def(Int) thin -> None = thunk[Int]


# Checking for 'postParseModule' ensures it was loaded from bytecode.

# CHECK-LABEL: lit.package @std attributes {
# CHECK-SAME: postParseModule =

# CHECK-NOT: lit.fn @"def(::Int, /) -> None|def(y: ::Int) -> None|{{.*}}[def(y: ::Int) -> None](::Int)"

# CHECK-LABEL: lit.package @func_package_foo
# CHECK-SAME: postParseModule
# CHECK: lit.fn @"foo
# CHECK: kgen.create_closure{{.*}}@"def(::Int, /) -> None|def(y: ::Int) -> None|{{.*}}[def(y: ::Int) -> None](::Int)"

# CHECK-COUNT-1: lit.fn @"def(::Int, /) -> None|def(y: ::Int) -> None|{{.*}}[def(y: ::Int) -> None](::Int)"

# CHECK-LABEL: lit.package @func_package_bar
# CHECK-SAME: postParseModule
# CHECK: lit.fn @"bar
# CHECK: kgen.create_closure{{.*}}@"def(::Int, /) -> None|def(y: ::Int) -> None|{{.*}}[def(y: ::Int) -> None](::Int)"

# CHECK-NOT: lit.fn @"def(::Int, /) -> None|def(::Int, /) -> None|{{.*}}[def(::Int, /) -> None](::Int)"

# CHECK-LABEL: dialect_resources
