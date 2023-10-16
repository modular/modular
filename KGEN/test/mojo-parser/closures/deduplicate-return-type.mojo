# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s

# CHECK: lit.struct.decl @"_CI_
# CHECK: lit.struct.decl @"_CW_
# CHECK: lit.struct.decl @"_CI_
# CHECK: lit.struct.decl @"_CW_


fn make_diff_closures(m: String, z: Int):
    fn ret_mem() escaping -> String:
        return m

    fn ret_mlir_type() escaping -> Int:
        return z
