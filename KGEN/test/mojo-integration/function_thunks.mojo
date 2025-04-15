# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %T/function-thunks
# RUN: mkdir -p %T/function-thunks
# RUN: mojo package %S/inputs/func_package_foo -o %T/function-thunks/func_package_foo.mojopkg
# RUN: mojo package %S/inputs/func_package_bar -o %T/function-thunks/func_package_bar.mojopkg
# RUN: kgen-opt %T/function-thunks/func_package_foo.mojopkg | FileCheck %s --check-prefix=THUNK
# RUN: kgen-opt %T/function-thunks/func_package_bar.mojopkg | FileCheck %s --check-prefix=THUNK
# RUN: kgen-translate -import-mojo %s --mojo-enable-prebuilt-packages -I %T/function-thunks | FileCheck %s
# RUN: mojo doc --validate-doc-strings %s -o /dev/null -I %T/function-thunks

# THUNK-COUNT-1: lit.fn @"fn

from func_package_foo.module import foo
from func_package_bar.module import bar

# CHECK-LABEL: lit.file_module @function_thunks


fn thunk[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test
fn test():
    _ = foo()
    _ = bar()

    # CHECK: kgen.create_closure{{.*}}@"fn(Int, /) -> None|fn(Int, /) -> None|{{.*}}[fn(Int, /) -> None](Int)"
    var f: fn (Int) -> None = thunk[Int]


# Checking for 'postParseModule' ensures it was loaded from bytecode.

# CHECK-LABEL: lit.package @stdlib attributes {postParseModule =

# CHECK-NOT: fn(Int

# CHECK-LABEL: lit.package @func_package_foo
# CHECK-SAME: postParseModule
# CHECK: lit.fn @"foo
# CHECK: kgen.create_closure{{.*}}@"fn(Int, /) -> None|fn(Int, /) -> None|{{.*}}[fn(Int, /) -> None](Int)"

# CHECK-COUNT-1: lit.fn @"fn(Int, /) -> None|fn(Int, /) -> None|{{.*}}[fn(Int, /) -> None](Int)"

# CHECK-LABEL: lit.package @func_package_bar
# CHECK-SAME: postParseModule
# CHECK: lit.fn @"bar
# CHECK: kgen.create_closure{{.*}}@"fn(Int, /) -> None|fn(Int, /) -> None|{{.*}}[fn(Int, /) -> None](Int)"

# CHECK-NOT: fn(Int

# CHECK-LABEL: dialect_resources
