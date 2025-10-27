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


##===----------------------------------------------------------------------===##
# Fold pop.cast_from_builtin
##===----------------------------------------------------------------------===##


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
