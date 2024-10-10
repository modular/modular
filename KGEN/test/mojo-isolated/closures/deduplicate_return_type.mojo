# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"fn(
# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"fn(


fn make_diff_closures(m: string, z: int):
    fn ret_mem() -> string:
        return m

    fn ret_mlir_type() -> int:
        return z
