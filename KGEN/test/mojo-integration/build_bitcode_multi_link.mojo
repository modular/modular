# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# Integration test for linking multiple LLVM bitcode libraries.
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the original Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit-llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Compile the other Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/another_bitcode_impl.mojo -emit-llvm -o %t_impl_other.ll
# RUN: llvm-as %t_impl_other.ll -o %t_impl_other.bc

# Step 3: Compile and run this test file that links both bitcode libraries.
# RUN: mojo --bitcode-libs=%t_impl.bc --bitcode-libs=%t_impl_other.bc %s | FileCheck %s


@extern("extern_add")
fn extern_add(a: Int32, b: Int32) -> Int32:
    ...


@extern("extern_sub")
fn extern_sub(a: Int32, b: Int32) -> Int32:
    ...


fn main():
    var a = extern_add(10, 20)
    print("Extern add:", a)
    # CHECK: Extern add: 30

    var b = extern_sub(10, 20)
    print("Extern sub:", b)
    # # CHECK: Extern sub: -10
