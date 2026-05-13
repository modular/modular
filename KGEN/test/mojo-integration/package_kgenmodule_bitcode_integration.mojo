# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# Variant of the integration test for packaging LLVM bitcode libraries.
# This test uses a kgen module instead of a Mojo package.
#
# ===----------------------------------------------------------------------=== #

# Step 1: Compile the Mojo bitcode implementation to LLVM bitcode
# RUN: kgen %S/inputs/bitcode_impl.mojo -emit=llvm -o %t_impl.ll
# RUN: llvm-as %t_impl.ll -o %t_impl.bc

# Step 2: Package the bitcode_package with the bitcode library
# RUN: mojo precompile %S/inputs/bitcode_package -kgenModule --bitcode-libs=%t_impl.bc -o %S/bitcode_package.mlirbc

# Step 3: Verify the package was created and contains bitcode modules
# RUN: kgen-opt %S/bitcode_package.mlirbc | FileCheck %s --check-prefix=CHECK-MODULE
# CHECK-MODULE: module attributes
# CHECK-MODULE-SAME: kgen.llvm.bitcode.libs = #kgen<llvm.bitcode.libs[<used = true, library = dense_resource<[[LLVM_BITCODE_NAME:llvm_bitcode_[[:alnum:]]+]]>
# CHECK-MODULE: dialect_resources:
# CHECK-MODULE: [[LLVM_BITCODE_NAME]]
