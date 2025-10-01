# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s


from utils._select import _select_register_value as select


struct IntT[x: Int](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


struct UIntT[x: UInt](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold select op
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_select_op
fn fold_select_op[B: Int = 4, C: Int = 3]() -> IntT[B]:
    # TODO(Should fold).
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@IntT<:!Int #lit<sugar aibuiltin, !Int, apply(:!lit.generator<("condition": !Bool, "lhs": !Int, "rhs": !Int) -> !Int> @stdlib::@utils::@_select::@"_select_register_value[AnyTrivialRegType](::Bool,$0,$0)"<:type !Int>, {:i1 1}, B, C), B>>, mut *"a`1">
    var a = IntT[select(True, B, C)]()
    return a


##===----------------------------------------------------------------------===##
# Fold UInt/index type ops
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_index_ceildiv
fn fold_index_ceildiv() -> UIntT[2]:
    alias A: UInt = 5
    alias B: UInt = 3
    # TODO(Should fold).
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@UIntT<:!UInt #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<("self": !UInt, "denominator": !UInt) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__ceildiv__(::UInt,::UInt)", #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<("value": !lit.struct<#IntLiteral <:!pop.int_literal 5>>) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__[__mlir_type.!pop.int_literal](::IntLiteral[$0])"<:!pop.int_literal 5>, *?), #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<(*, "mlir_value": index) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__(__mlir_type.index)", 5), {5}>>, #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<("value": !lit.struct<#IntLiteral <:!pop.int_literal 3>>) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__[__mlir_type.!pop.int_literal](::IntLiteral[$0])"<:!pop.int_literal 3>, *?), #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<(*, "mlir_value": index) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__(__mlir_type.index)", 3), {3}>>), {_mlir_value = ceil_div_u(#lit.struct.extract<:!UInt #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<("value": !lit.struct<#IntLiteral <:!pop.int_literal 5>>) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__[__mlir_type.!pop.int_literal](::IntLiteral[$0])"<:!pop.int_literal 5>, *?), #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<(*, "mlir_value": index) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__(__mlir_type.index)", 5), {5}>>, "_mlir_value">, #lit.struct.extract<:!UInt #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<("value": !lit.struct<#IntLiteral <:!pop.int_literal 3>>) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__[__mlir_type.!pop.int_literal](::IntLiteral[$0])"<:!pop.int_literal 3>, *?), #lit<sugar aibuiltin, !UInt, apply(:!lit.generator<(*, "mlir_value": index) -> !UInt> @stdlib::@builtin::@uint::@UInt::@"__init__(__mlir_type.index)", 3), {3}>>, "_mlir_value">)}>>, mut *"a`3">
    var a = UIntT[A.__ceildiv__(B)]()
    return a
