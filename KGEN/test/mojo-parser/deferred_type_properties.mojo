# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.fn @"defer_with_props[::Int]
# CHECK: kgen.deferred "test.op_with_props"
# CHECK-SAME: properties {operandSegmentSizes = array<i32: 1, 0>}
@always_inline
def defer_with_props[n: Int]() -> __mlir_deferred_type[
    `!llvm.array<`, +n._mlir_value, ` x f32>`
]:
    return __mlir_op.`test.op_with_props`[
        _type = __mlir_deferred_type[
            `!llvm.array<`, +n._mlir_value, ` x f32>`
        ],
        _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1, 0>}`,
    ]()
