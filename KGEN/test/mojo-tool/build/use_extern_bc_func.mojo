# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Test linking an external bitcode file.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen %S/inputs/define_extern_bc_func.mojo -emit-llvm -o %t.ll
# RUN: llvm-as %t.ll -o %t.bc
# RUN: %mojo --bitcode-libs=%t.bc %s | FileCheck %s


@extern("my_add_one")
fn my_add_one(x: Int32) -> Int32:
    ...


fn main():
    # CHECK: 3
    print(my_add_one(2))
