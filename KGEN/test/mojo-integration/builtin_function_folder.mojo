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
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@IntT<:!Int B>, mut *"a`1">
    var a = IntT[select(True, B, C)]()
    return a


##===----------------------------------------------------------------------===##
# Fold UInt/index type ops
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_index_ceildiv
fn fold_index_ceildiv() -> UIntT[2]:
    alias A: UInt = 5
    alias B: UInt = 3
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@UIntT<:!UInt {2}>, mut *"a`3">
    var a = UIntT[A.__ceildiv__(B)]()
    return a


##===----------------------------------------------------------------------===##
# Fold DType ops
##===----------------------------------------------------------------------===##


struct UInt8T[x: UInt8._mlir_type](ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


alias UI8_139 = __mlir_attr.`#pop.simd<139> : !pop.scalar<ui8>`


# CHECK-LABEL: lit.fn @"fold_dtype_as_ui8
fn fold_dtype_as_ui8() -> UInt8T[UI8_139]:
    alias A: DType = DType.int32
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@UInt8T<{{.*}}, 139>>
    var a = UInt8T[A._as_ui8()]()
    return a
