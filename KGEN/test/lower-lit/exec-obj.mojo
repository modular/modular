# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo | FileCheck %s


struct SomeStructWithSize[size: __mlir_type.index]:
    pass


fn getSize() -> __mlir_type.index:
    return (0).value


# COM: Check that the mangled function name doesn't contain "@"
# CHECK: lit.func export @"testMangledName
# CHECK-NOT: {{.*}}@{{.*}}
# COM: Match the first argument of the function.
# CHECK-SAME: "(%{{.*}}:
@export
fn testMangledName(x: SomeStructWithSize[getSize()]):
    return
