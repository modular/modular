# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


struct BoolT[x: Bool](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


struct BuiltinBoolT[x: __mlir_type.`!pop.scalar<bool>`](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


struct DTypeT[x: DType](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


struct BuiltinSI32T[x: __mlir_type.`!pop.scalar<si32>`](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold pop.cast_from_builtin
##===----------------------------------------------------------------------===##


# 139 = DType.int32
comptime MLIR_UI8_139 = __mlir_attr.`139 : ui8`
comptime POP_UI8_139 = __mlir_attr.`#pop.simd<139> : !pop.scalar<ui8>`


# 77 = DType.f8e5m2
comptime MLIR_UI8_77 = __mlir_attr.`77 : ui8`
comptime POP_UI8_77 = __mlir_attr.`#pop.simd<77> : !pop.scalar<ui8>`


@always_inline("builtin")
fn pop_cast_from_builtin_bool(
    x: __mlir_type.i1,
) -> __mlir_type.`!pop.scalar<bool>`:
    return __mlir_op.`pop.cast_from_builtin`[
        _type = __mlir_type.`!pop.scalar<bool>`
    ](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_from_builtin_bool
fn fold_pop_cast_from_builtin_bool() -> (
    BuiltinBoolT[__mlir_attr.`#pop.simd<true> : !pop.scalar<bool>`]
):
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#BuiltinBoolT <:scalar<bool>
    var a = BuiltinBoolT[pop_cast_from_builtin_bool(__mlir_attr.`true : i1`)]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!lit.struct<#BuiltinBoolT <:scalar<bool>
    var b = BuiltinBoolT[pop_cast_from_builtin_bool(__mlir_attr.`false : i1`)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.cast_to_builtin
##===----------------------------------------------------------------------===##


struct UInt8T[x: __mlir_type.ui8](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


@always_inline("builtin")
fn pop_cast_to_builtin_ui8(x: UInt8._mlir_type) -> __mlir_type.ui8:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_to_builtin_ui8
fn fold_pop_cast_to_builtin_ui8() -> UInt8T[MLIR_UI8_139]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}}<:ui8 139>>
    var a = UInt8T[pop_cast_to_builtin_ui8(POP_UI8_139)]()
    return a


@always_inline("builtin")
fn pop_cast_to_builtin_bool(
    x: __mlir_type.`!pop.scalar<bool>`,
) -> __mlir_type.i1:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i1](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_to_builtin_bool
fn fold_pop_cast_to_builtin_bool() -> BoolT[True]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}}:i1 1}>>,
    var a = BoolT[
        pop_cast_to_builtin_bool(
            __mlir_attr.`#pop.simd<true> : !pop.scalar<bool>`
        )
    ]()
    # CHECK:  %b = lit.var.decl "b" var : {{.*}}:i1 0}>>,
    var b = BoolT[
        pop_cast_to_builtin_bool(
            __mlir_attr.`#pop.simd<false> : !pop.scalar<bool>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.dtype.from_ui8
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_dtype_from_ui8(ui8: __mlir_type.ui8) -> DType:
    return DType(mlir_value=__mlir_op.`pop.dtype.from_ui8`(ui8))


# CHECK-LABEL: lit.fn @"fold_pop_dtype_from_ui8
fn fold_pop_dtype_from_ui8() -> DTypeT[DType.int32]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} si32}>>,
    var a = DTypeT[pop_dtype_from_ui8(MLIR_UI8_139)]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} f8e5m2}>>,
    var b = DTypeT[pop_dtype_from_ui8(MLIR_UI8_77)]()


##===----------------------------------------------------------------------===##
# Fold pop.cast
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_cast(
    x: __mlir_type.`!pop.scalar<si8>`,
) -> __mlir_type.`!pop.scalar<ui8>`:
    return __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<ui8>`](x)


struct POPUInt8T[x: __mlir_type.`!pop.scalar<ui8>`](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


comptime POP_SI8_N1 = __mlir_attr.`#pop.simd<-1> : !pop.scalar<si8>`
comptime POP_UI8_N1 = __mlir_attr.`#pop.simd<255> : !pop.scalar<ui8>`


# CHECK-LABEL: lit.fn @"fold_pop_cast
fn fold_pop_cast() -> POPUInt8T[POP_UI8_N1]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} 255)>>,
    var a = POPUInt8T[pop_cast(POP_SI8_N1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_splat
##===----------------------------------------------------------------------===##


struct POPUInt8x4T[x: __mlir_type.`!pop.simd<4, ui8>`](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


comptime POP_UI8x4_N1 = __mlir_attr.`#pop.simd<255, 255, 255, 255> : !pop.simd<4, ui8>`


@always_inline("builtin")
fn pop_simd_splat(
    x: __mlir_type.`!pop.scalar<ui8>`,
) -> __mlir_type.`!pop.simd<4, ui8>`:
    return __mlir_op.`pop.simd.splat`[_type = __mlir_type.`!pop.simd<4, ui8>`](
        x
    )


# CHECK-LABEL: lit.fn @"fold_pop_simd_splat
fn fold_pop_simd_splat() -> POPUInt8x4T[POP_UI8x4_N1]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} #alias_POP_UI8_N1), 255)>>,
    var a = POPUInt8x4T[pop_simd_splat(POP_UI8_N1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_and
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_simd_and(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, ui8>`:
    return __mlir_op.`pop.simd.and`(x, y)


comptime POP_UI8x4_Fold = __mlir_attr.`#pop.simd<42, 255, 1, 0> : !pop.simd<4, ui8>`


# CHECK-LABEL: lit.fn @"fold_pop_simd_and
fn fold_pop_simd_and() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" {{.*}} <42, 255, 1, 0>
    var a = POPUInt8x4T[
        pop_simd_and(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_and
@always_inline("builtin")
fn pop_unresolved_simd_and[
    dt: DType, n: Int
](x: SIMD[dt, n], y: SIMD[dt, n],) -> SIMD[dt, n]._mlir_type:
    return __mlir_op.`pop.simd.and`(x._mlir_value, y._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_xor
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_simd_xor(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, ui8>`:
    return __mlir_op.`pop.simd.xor`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_xor
fn fold_pop_simd_xor() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <9, 15, 1, 1>, <35, 240, 0, 1>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_xor(
            __mlir_attr.`#pop.simd<9, 15, 1, 1> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<35, 240, 0, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_xor
@always_inline("builtin")
fn pop_unresolved_simd_xor[
    dt: DType, n: Int
](x: SIMD[dt, n], y: SIMD[dt, n],) -> SIMD[dt, n]._mlir_type:
    return __mlir_op.`pop.simd.xor`(x._mlir_value, y._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_or
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_simd_or(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, ui8>`:
    return __mlir_op.`pop.simd.or`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_or
fn fold_pop_simd_or() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <8, 15, 1, 0>, <34, 240, 1, 0>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_or(
            __mlir_attr.`#pop.simd<8, 15, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<34, 240, 1, 0> : !pop.simd<4, ui8>`,
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_cmp
##===----------------------------------------------------------------------===##


struct POPBoolx4T[x: __mlir_type.`!pop.simd<4, bool>`](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


comptime POP_Boolx4_EQ_Fold = __mlir_attr.`#pop.simd<false, true, true, false> : !pop.simd<4, bool>`


@always_inline("builtin")
fn pop_simd_cmp_eq(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_eq(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_ne(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_ne(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_ult(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_slt(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_ule(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_sle(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_ugt(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_sgt(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_uge(
    x: __mlir_type.`!pop.simd<4, ui8>`, y: __mlir_type.`!pop.simd<4, ui8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](x, y)


@always_inline("builtin")
fn pop_simd_cmp_sge(
    x: __mlir_type.`!pop.simd<4, si8>`, y: __mlir_type.`!pop.simd<4, si8>`
) -> __mlir_type.`!pop.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_cmp
fn fold_pop_simd_cmp() -> POPBoolx4T[POP_Boolx4_EQ_Fold]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <true, false, false, false>)>>
    var a = POPBoolx4T[
        pop_simd_cmp_eq(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} <false, true, true, false>)>>
    var b = POPBoolx4T[
        pop_simd_cmp_eq(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    # CHECK: %c = lit.var.decl "c" var : {{.*}} <false, true, true, true>)>>
    var c = POPBoolx4T[
        pop_simd_cmp_ne(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %d = lit.var.decl "d" var : {{.*}} <true, false, false, true>)>>
    var d = POPBoolx4T[
        pop_simd_cmp_ne(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    # CHECK: %e = lit.var.decl "e" var : {{.*}} <false, true, false, true>)>>
    var e = POPBoolx4T[
        pop_simd_cmp_slt(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %f = lit.var.decl "f" var : {{.*}} <false, false, false, true>)>>
    var f = POPBoolx4T[
        pop_simd_cmp_ult(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    # CHECK: %g = lit.var.decl "g" var : {{.*}} <true, true, false, true>)>>
    var g = POPBoolx4T[
        pop_simd_cmp_sle(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %h = lit.var.decl "h" var : {{.*}} <false, true, true, true>)>>
    var h = POPBoolx4T[
        pop_simd_cmp_ule(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    # CHECK: %i = lit.var.decl "i" var : {{.*}} <false, false, true, false>)>>
    var i = POPBoolx4T[
        pop_simd_cmp_sgt(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %j = lit.var.decl "j" var : {{.*}} <true, false, false, false>)>>
    var j = POPBoolx4T[
        pop_simd_cmp_ugt(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    # CHECK: %k = lit.var.decl "k" var : {{.*}} <true, false, true, false>)>>
    var k = POPBoolx4T[
        pop_simd_cmp_sge(
            __mlir_attr.`#pop.simd<46, -128, 1, 0> : !pop.simd<4, si8>`,
            __mlir_attr.`#pop.simd<46, 127, -1, 1> : !pop.simd<4, si8>`,
        )
    ]()
    # CHECK: %l = lit.var.decl "l" var : {{.*}} <true, true, true, false>)>>
    var l = POPBoolx4T[
        pop_simd_cmp_uge(
            __mlir_attr.`#pop.simd<46, 255, 1, 0> : !pop.simd<4, ui8>`,
            __mlir_attr.`#pop.simd<43, 255, 1, 1> : !pop.simd<4, ui8>`,
        )
    ]()
    return b


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_cmp_sge
@always_inline("builtin")
fn pop_unresolved_simd_cmp_sge[
    dt: DType, n: Int, m: Int
](x: SIMD[dt, n + m], y: SIMD[dt, n + m]) -> SIMD[DType.bool, n + m]._mlir_type:
    return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](
        x._mlir_value, y._mlir_value
    )


@always_inline("builtin")
fn var_decls[dtype: DType](value: IntLiteral) -> Scalar[dtype]._mlir_type:
    # Convert the IntLiteral to si32
    var si32_ = __mlir_attr[
        `#pop<int_literal_convert<`, value.value, `, 0>> : si32`
    ]
    # Convert si32 to !pop.simd<si32>
    var si32 = __mlir_op.`pop.cast_from_builtin`[
        _type = __mlir_type.`!pop.scalar<si32>`
    ](si32_)
    # Convert !pop.simd<si32> to !pop.simd<X>
    var s = __mlir_op.`pop.cast`[_type = Scalar[dtype]._mlir_type](si32)
    # Convert !pop.simd<X> to !pop.simd<ui8>
    var pop_ui8 = __mlir_op.`pop.cast`[_type = Scalar[DType.uint8]._mlir_type](
        si32
    )
    # Convert !pop.simd<ui8> to ui8
    var ui8 = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.ui8](pop_ui8)
    # Convert ui8 to dtype
    var dt = __mlir_op.`pop.dtype.from_ui8`(ui8)
    # Convert dtype to ui8
    var dt_ui8 = __mlir_op.`pop.dtype.to_ui8`(dt)
    # Convert the ui8 back to !pop.simd<ui8>
    var pop_ui8_2 = __mlir_op.`pop.cast_from_builtin`[
        _type = Scalar[DType.uint8]._mlir_type
    ](dt_ui8)
    # Convert !pop.simd<ui8> back to !pop.simd<X>
    var t = __mlir_op.`pop.cast`[_type = Scalar[dtype]._mlir_type](pop_ui8_2)
    # Combine the two
    var u = __mlir_op.`pop.simd.xor`(s, t)
    return u


# CHECK-LABEL: lit.fn @"fold_var_decls
fn fold_var_decls() -> (
    BuiltinSI32T[__mlir_attr.`#pop.simd<0> : !pop.scalar<si32>`]
):
    # CHECK: %a = lit.var.decl "a" var : {{.*}} 0)>>
    var a = BuiltinSI32T[var_decls[DType.int32](42)]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} -256)>>
    var b = BuiltinSI32T[var_decls[DType.int32](-1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_reduce_or
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_simd_reduce_or(
    x: __mlir_type.`!pop.simd<4, ui8>`,
) -> __mlir_type.`!pop.scalar<ui8>`:
    return __mlir_op.`pop.simd.reduce_or`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_reduce_or
fn fold_pop_simd_reduce_or() -> POPUInt8T[POP_UI8_77]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <1, 8, 68, 0>), 77)>>
    var a = POPUInt8T[
        pop_simd_reduce_or(
            __mlir_attr.`#pop.simd<1, 8, 68, 0> : !pop.simd<4, ui8>`
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_reduce_or
@always_inline("builtin")
fn pop_unresolved_simd_reduce_or[
    dt: DType, n: Int
](x: SIMD[dt, n]) -> SIMD[dt, 1]._mlir_type:
    return __mlir_op.`pop.simd.reduce_or`(x._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_reduce_and
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn pop_simd_reduce_and(
    x: __mlir_type.`!pop.simd<4, ui8>`,
) -> __mlir_type.`!pop.scalar<ui8>`:
    return __mlir_op.`pop.simd.reduce_and`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_reduce_and
fn fold_pop_simd_reduce_and() -> POPUInt8T[POP_UI8_77]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <79, 93, 207, 221>), 77)>>
    var a = POPUInt8T[
        pop_simd_reduce_and(
            __mlir_attr.`#pop.simd<79, 93, 207, 221> : !pop.simd<4, ui8>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold kgen.param.assert
##===----------------------------------------------------------------------===##


@always_inline("builtin")
fn kgen_assert() -> Bool:
    __comptime_assert False, "Ignore this"
    return True


# CHECK-LABEL: lit.fn @"fold_kgen_assert
fn fold_kgen_assert() -> BoolT[True]:
    var a = BoolT[kgen_assert()]()
    return a
