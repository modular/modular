# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# Integration test for linking multiple LLVM bitcode libraries that conflicts.
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the original Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit-llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Compile the alternative Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl_alt.mojo -emit-llvm -o %t_impl_alt.ll
# RUN: llvm-as %t_impl_alt.ll -o %t_impl_alt.bc

# Step 3: Compile and run this test file that links both packages.
# RUN: mojo --bitcode-libs=%t_impl.bc --bitcode-libs=%t_impl_alt.bc %s | FileCheck %s


@extern("extern_add")
fn extern_add(a: Int32, b: Int32) -> Int32:
    ...


fn main():
    var a = extern_add(10, 20)
    print("Extern add:", a)
    # CHECK: Extern add:
