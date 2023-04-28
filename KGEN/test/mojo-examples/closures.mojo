# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from IO import print


fn take_closure_and_print(
    g: __mlir_type.`!kgen.signature<(index borrow) capturing -> index>`,
    x: __mlir_type.index,
):
    print(g(x))


fn test_take_closure_and_print(x: __mlir_type.index):
    @parameter
    fn h(y: __mlir_type.index) -> __mlir_type.index:
        let pop_x = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](x)
        let pop_y = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](y)
        let pop_result = __mlir_op.`pop.add`(pop_x, pop_y)
        let result = __mlir_op.`pop.cast_to_builtin`[_type : __mlir_type.index](
            pop_result
        )
        return result

    @parameter
    fn thin(y: __mlir_type.index) -> __mlir_type.index:
        let pop_x = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ]((17).__as_mlir_index())
        let pop_y = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](y)
        let pop_result = __mlir_op.`pop.add`(pop_x, pop_y)
        let result = __mlir_op.`pop.cast_to_builtin`[_type : __mlir_type.index](
            pop_result
        )
        return result

    let H: __mlir_type.`!kgen.signature<(index borrow) capturing -> index>` = h
    let thin_closure: __mlir_type.`!kgen.signature<(index borrow) capturing -> index>` = thin
    let u: __mlir_type.index = (3).__as_mlir_index()
    take_closure_and_print(H, u)
    take_closure_and_print(thin_closure, u)


fn main():
    let x = (39).__as_mlir_index()
    # CHECK: 42
    # CHECK: 20
    test_take_closure_and_print(x)
