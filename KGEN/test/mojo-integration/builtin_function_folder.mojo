# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %T %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

from utils._select import _select_register_value as select


struct T[x: Int]:
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold select op
##===----------------------------------------------------------------------===##


fn fold_select_op[B: Int = 4, C: Int = 3]() -> T[B]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<@builtin_function_folder::@T<:!Int B>, mut *"a`1">
    var a = T[select(True, B, C)]()
    return a
