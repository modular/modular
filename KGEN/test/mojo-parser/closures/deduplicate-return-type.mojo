# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"_CW_
# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"_CW_


fn make_diff_closures(m: StringLiteral, z: Int):
    fn ret_mem() escaping -> StringLiteral:
        return m

    fn ret_mlir_type() escaping -> Int:
        return z
