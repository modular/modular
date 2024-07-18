# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | kgen-opt -lower-semantic-cf --check-lifetimes -lower-lit -lower-lit-types -register-custom-ops -canonicalizer -o /dev/null | FileCheck %s

import _mlir


@op_implementation("custom.with_canonicalize")
struct CustomOpWithCanonicalize:
    @staticmethod
    fn impl(x: Int32, y: Int32) -> Int32:
        return x + y

    @staticmethod
    fn canonicalize(x: _mlir.Operation):
        # We only print during canonicalization, as the MLIR C API is not yet
        # fully supported in the JIT.
        print("canonicalization is called")
        return


fn main():
    var x: Int32 = 4
    var y: Int32 = 6
    var res = __mlir_op.`custom.with_canonicalize`[_type=Int32](x, y)
    print(res)


# CHECK: canonicalization is called
