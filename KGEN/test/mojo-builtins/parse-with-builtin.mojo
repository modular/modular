# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo --allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s


# CHECK-LABEL: lit.func @"import_of_import
# CHECK-SAME: @stdlib::@builtin::@simd::@SIMD<
fn import_of_import(arg: Float64):
    pass


import builtin


# CHECK-LABEL: lit.func @"test_function_calls(stdlib::builtin::int::Int)"
fn test_function_calls(arg: builtin.int.Int):
    pass


# Test multi-return __mlir_op
# https://github.com/modularml/modular/issues/24227
fn hasMultiReturnMLIROp() -> Tuple[Int, Int]:
    # CHECK: [[MULTIRET:%.*]]:2 = "op_that_has_multiple_returns"() : () -> (!Int, !Int)
    # CHECK-NEXT: [[PACK:%.*]] = kgen.pack.create([[MULTIRET]]#0, [[MULTIRET]]#1)
    # CHECK-NEXT: [[TUPLE:%.*]] = lit.call @stdlib::@builtin::@tuple::@Tuple::@"__init__{{.*}}[!Int, !Int]{{.*}}[[PACK]]
    let r = __mlir_op.`op_that_has_multiple_returns`[_type= (Int, Int)]()
    return r
