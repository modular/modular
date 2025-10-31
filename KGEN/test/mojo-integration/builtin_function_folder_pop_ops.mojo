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


##===----------------------------------------------------------------------===##
# Fold pop.cast_from_builtin
##===----------------------------------------------------------------------===##


# 139 = DType.int32
alias MLIR_UI8_139 = __mlir_attr.`139 : ui8`
alias POP_UI8_139 = __mlir_attr.`#pop.simd<139> : !pop.scalar<ui8>`


# 77 = DType.f8e5m2
alias MLIR_UI8_77 = __mlir_attr.`77 : ui8`
alias POP_UI8_68 = __mlir_attr.`#pop.simd<68> : !pop.scalar<ui8>`


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
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder_pop_ops::@BuiltinBoolT<:scalar<bool> #kgen<sugar aibuiltin, !pop.scalar<bool>, apply(:!lit.generator<("x": i1) -> !pop.scalar<bool>> @builtin_function_folder_pop_ops::@"pop_cast_from_builtin_bool(__mlir_type.i1)", 1), true>>, mut *"a`1">
    var a = BuiltinBoolT[pop_cast_from_builtin_bool(__mlir_attr.`true : i1`)]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<@builtin_function_folder_pop_ops::@BuiltinBoolT<:scalar<bool> #kgen<sugar aibuiltin, !pop.scalar<bool>, apply(:!lit.generator<("x": i1) -> !pop.scalar<bool>> @builtin_function_folder_pop_ops::@"pop_cast_from_builtin_bool(__mlir_type.i1)", 0), false>>, mut *"b`2">
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
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder_pop_ops::@UInt8T<:ui8 #kgen<sugar aibuiltin, ui8, apply(:!lit.generator<("x": !pop.scalar<ui8>) -> ui8> @builtin_function_folder_pop_ops::@"pop_cast_to_builtin_ui8(__mlir_type.!pop.scalar<ui8>)", 139), 139>>, mut *"a`1">
    var a = UInt8T[pop_cast_to_builtin_ui8(POP_UI8_139)]()
    return a


@always_inline("builtin")
fn pop_cast_to_builtin_bool(
    x: __mlir_type.`!pop.scalar<bool>`,
) -> __mlir_type.i1:
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i1](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_to_builtin_bool
fn fold_pop_cast_to_builtin_bool() -> BoolT[True]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder_pop_ops::@BoolT<:!Bool #kgen<sugar aibuiltin, !Bool, apply(:!lit.generator<("value": i1) -> !Bool> @stdlib::@builtin::@bool::@Bool::@"__init__(__mlir_type.i1)", #kgen<sugar aibuiltin, i1, apply(:!lit.generator<("x": !pop.scalar<bool>) -> i1> @builtin_function_folder_pop_ops::@"pop_cast_to_builtin_bool(__mlir_type.!pop.scalar<bool>)", true), 1>), {:i1 #kgen<sugar aibuiltin, i1, apply(:!lit.generator<("x": !pop.scalar<bool>) -> i1> @builtin_function_folder_pop_ops::@"pop_cast_to_builtin_bool(__mlir_type.!pop.scalar<bool>)", true), 1>}>>, mut *"a`{{.*}}">
    var a = BoolT[
        pop_cast_to_builtin_bool(
            __mlir_attr.`#pop.simd<true> : !pop.scalar<bool>`
        )
    ]()
    # CHECK:  %b = lit.var.decl "b" var : !lit.ref<@builtin_function_folder_pop_ops::@BoolT<:!Bool #kgen<sugar aibuiltin, !Bool, apply(:!lit.generator<("value": i1) -> !Bool> @stdlib::@builtin::@bool::@Bool::@"__init__(__mlir_type.i1)", #kgen<sugar aibuiltin, i1, apply(:!lit.generator<("x": !pop.scalar<bool>) -> i1> @builtin_function_folder_pop_ops::@"pop_cast_to_builtin_bool(__mlir_type.!pop.scalar<bool>)", false), 0>), {:i1 #kgen<sugar aibuiltin, i1, apply(:!lit.generator<("x": !pop.scalar<bool>) -> i1> @builtin_function_folder_pop_ops::@"pop_cast_to_builtin_bool(__mlir_type.!pop.scalar<bool>)", false), 0>}>>, mut *"b`{{.*}}">
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
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder_pop_ops::@DTypeT<:!DType {:dtype si32}>, mut *"a`1">
    var a = DTypeT[pop_dtype_from_ui8(MLIR_UI8_139)]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<@builtin_function_folder_pop_ops::@DTypeT<:!DType {:dtype f8e5m2}>, mut *"b`2">
    var b = DTypeT[pop_dtype_from_ui8(MLIR_UI8_77)]()
