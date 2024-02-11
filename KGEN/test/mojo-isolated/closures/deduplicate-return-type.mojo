# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | FileCheck %s

# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"fn(
# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"fn(


fn make_diff_closures(m: StringLiteral, z: Int):
    fn ret_mem() escaping -> StringLiteral:
        return m

    fn ret_mlir_type() escaping -> Int:
        return z
