# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -O0 -elaborate -S -o - %s | FileCheck %s


# CHECK-NOT: no_impl
# CHECK: kgen.func export @conditional_alias
fn no_impl() -> __mlir_type.index:
    __mlir_op.`kgen.param.assert`[
        cond = __mlir_attr.false, message = __mlir_attr.`"bad" : !kgen.string`
    ]()
    return __mlir_attr.`0 : index`


fn make_true() -> __mlir_type.i1:
    return __mlir_attr.true


@export
fn conditional_alias():
    comptime value = __mlir_attr.`1 : index` if make_true() else no_impl()
