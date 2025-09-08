# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn materialize[T: AnyType, //, value: T](out result: T):
    # __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(result))
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


fn main():
    alias lst = [1, 2, 3]
    var dyn_lst = materialize[lst]()
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    for v in dyn_lst:
        print(v)
