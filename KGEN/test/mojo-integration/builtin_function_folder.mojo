# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This test depends on real stdlib types and functions (utils._select, DType
# methods, UInt.__ceildiv__, StaticString) to test builtin function folding
# integration. It should remain as an integration test.

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


struct BoolT[x: Bool](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold select op
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_select_op
fn fold_select_op[B: Int = 4, C: Int = 3]() -> IntT[B]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#IntT <:!Int B>>, mut *"a`1">
    var a = IntT[select(True, B, C)]()
    return a


##===----------------------------------------------------------------------===##
# Fold UInt/index type ops
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_index_ceildiv
fn fold_index_ceildiv() -> UIntT[2]:
    comptime A: UInt = 5
    comptime B: UInt = 3
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#UIntT <:!UInt {2}>>, mut *"a`3">
    var a = UIntT[A.__ceildiv__(B)]()
    return a


##===----------------------------------------------------------------------===##
# Fold Bool ops
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_bool_init
fn fold_bool_init() -> BoolT[True]:
    comptime T = Bool(
        mlir_value=__mlir_attr.`#pop.simd<true> : !pop.scalar<bool>`
    )
    comptime F = Bool(
        mlir_value=__mlir_attr.`#pop.simd<false> : !pop.scalar<bool>`
    )
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#BoolT <:!Bool {:i1 1}>>, mut *"a`3">
    var a = BoolT[T]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!lit.struct<#BoolT <:!Bool {:i1 0}>>, mut *"b`4">
    var b = BoolT[F]()
    return a


##===----------------------------------------------------------------------===##
# Fold DType ops
##===----------------------------------------------------------------------===##


struct UInt8T[x: UInt8._mlir_type](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


comptime UI8_139 = __mlir_attr.`#pop.simd<139> : !pop.scalar<ui8>`


# CHECK-LABEL: lit.fn @"fold_dtype_as_ui8
fn fold_dtype_as_ui8() -> UInt8T[UI8_139]:
    comptime A: DType = DType.int32
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#UInt8T <:scalar<ui8> {{.*}}, 139)>>, mut *"a
    var a = UInt8T[A._as_ui8()]()
    return a


struct DTypeT[x: DType](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


# CHECK-LABEL: lit.fn @"fold_dtype_from_ui8
fn fold_dtype_from_ui8() -> DTypeT[DType.int32]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#DTypeT <:!DType {:dtype si32}>>, mut *"a
    var a = DTypeT[DType._from_ui8(UI8_139)]()
    return a


##===----------------------------------------------------------------------===##
# Ignore 'constrained' functions (MOCO-2839)
##===----------------------------------------------------------------------===##


@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    pass


fn fold_constrained_func[x: Int]() -> IntT[x]:
    constrained[False, "this is being ignored anyway"]()
    return IntT[x]()


# CHECK-LABEL: lit.fn @"fold_constrained()
fn fold_constrained() -> IntT[1]:
    return fold_constrained_func[1]()


##===----------------------------------------------------------------------===##
# Handle comptime statements
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn comptime_statements() -> Bool:
    comptime t = True
    comptime s = not t
    return s


fn fold_comptime_statements() -> BoolT[False]:
    var b = BoolT[comptime_statements()]()
    return b
