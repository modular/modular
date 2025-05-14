# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %T %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s


from utils._select import _select_register_value as select


struct IntT[x: Int]:
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


struct UIntT[x: UInt]:
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold select op
##===----------------------------------------------------------------------===##


fn fold_select_op[B: Int = 4, C: Int = 3]() -> IntT[B]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@IntT<:!Int B>, mut *"a`1">
    var a = IntT[select(True, B, C)]()
    return a


##===----------------------------------------------------------------------===##
# Fold UInt/index type ops
##===----------------------------------------------------------------------===##


@always_inline
fn fold_index_ceildiv() -> UIntT[2]:
    alias A: UInt = 5
    alias B: UInt = 3
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@UIntT<:!UInt {2}>, mut *"a`3">
    var a = UIntT[A.__ceildiv__(B)]()
    return a


##===----------------------------------------------------------------------===##
# Fold integer division
##===----------------------------------------------------------------------===##


fn int_floordiv[A: Int, B: Int]() -> IntT[A // B]:
    var a = IntT[A // B]()
    return a


fn fold_integer_floordiv() -> IntT[3]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@IntT<:!Int {3}>, mut *"a`1">
    var a = int_floordiv[10, 3]()
    return a
