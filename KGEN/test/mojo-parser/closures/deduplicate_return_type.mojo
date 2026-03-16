# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"def(
# CHECK: lit.struct.decl @"`_CI_
# CHECK: lit.struct.decl @"def(


def make_diff_closures(m: string, z: Int):
    def ret_mem() -> string:
        return m

    def ret_mlir_type() -> Int:
        return z
