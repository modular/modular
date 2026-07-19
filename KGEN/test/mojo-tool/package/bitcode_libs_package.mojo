# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Test packaging external bitcode libraries into a mojo precompile.
#
# ===----------------------------------------------------------------------=== #

# First, create a bitcode library from this file's Mojo function
# RUN: kgen %s -emit=llvm -o %t.ll
# RUN: llvm-as %t.ll -o %t.bc

# Create a test package that includes the bitcode library
# RUN: mojo precompile %S/test_package --bitcode-libs=%t.bc -o %t_with_bitcode.mojoc

# Verify the package was created and contains the externLLVMBitcodeModules attribute
# RUN: kgen-opt -mlir-print-debuginfo %t_with_bitcode.mojoc | FileCheck %s

# CHECK: lit.package @{{.*}}_with_bitcode
# CHECK-SAME: externLLVMBitcodeModules = #lit<dense_resource_elements_array[dense_resource<[[LLVM_BITCODE_NAME:llvm_bitcode_[[:alnum:]]+]]
# CHECK: dialect_resources:
# CHECK: [[LLVM_BITCODE_NAME]]

# Clean up
# RUN: rm %t_with_bitcode.mojoc

# ===----------------------------------------------------------------------=== #
# Mojo function that will be compiled to bitcode and packaged
# ===----------------------------------------------------------------------=== #


@export("my_add_one")
def my_add_one(x: Int32) abi("Mojo") -> Int32:
    return x + 1
