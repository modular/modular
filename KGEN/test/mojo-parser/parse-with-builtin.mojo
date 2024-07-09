# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo --allow-unregistered-dialect %s | FileCheck %s


# CHECK-LABEL: lit.func @"import_of_import
# CHECK-SAME: #SIMD <:!DType {:dtype f64}, :!Int {1}>
fn import_of_import(arg: Float64):
    pass


import builtin


# CHECK-LABEL: lit.func @"test_function_calls(::Int)"
fn test_function_calls(arg: builtin.int.Int):
    pass


# Test multi-return __mlir_op
# https://github.com/modularml/modular/issues/24227
fn hasMultiReturnMLIROp() -> Tuple[Int, Int]:
    # CHECK: [[MULTIRET:%.*]]:2 = "op_that_has_multiple_returns"() : () -> (!Int, !Int)
    # CHECK: [[PACK:%.*]] = lit.ref.pack.create
    # CHECK: lit.call {{.*}}@Tuple::@"__init__{{.*}}[#Int1, #Int1]{{.*}}(%r,
    var r = __mlir_op.`op_that_has_multiple_returns`[_type= (Int, Int)]()
    return r^


# COM: Check that a load from a SIMD field works.
# CHECK-LABEL: lit.func @"testSIMDGetter
fn testSIMDGetter[
    type: DType
](owned a: SIMD[type, 2]) -> __mlir_type[`!pop.scalar<`, type.value, `>`]:
    # CHECK: %a_0 = lit.var.decl "a"
    # CHECK: lit.ref.store %a, %a_0
    # CHECK: %[[AVAL:.*]] = lit.ref.load %a_0
    # CHECK: %[[ZERO:.*]] = kgen.param.constant: !Int = <{0}>
    # CHECK: %[[GOT:.*]] = lit.call {{.*}}__getitem__{{.*}}(%[[AVAL]], %[[ZERO]])
    # CHECK: %[[RES:.*]] = lit.struct.extract %[[GOT]][value]
    # CHECK: lit.return %[[RES]]
    return a[0].value
