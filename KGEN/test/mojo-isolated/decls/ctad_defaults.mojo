# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# COM: Verify kw-only defaults are searched during binding verification.

struct MyUnsafePointer[
    type: AnyType,
    x: Int = 3,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    exclusive: Bool = False,
    alignment: Int = 1,
    lifetime: Origin[True]._mlir_type = MutableAnyOrigin]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        type,
        `, `,
        address_space._value.value,
        `>`,
    ]
    var address: Self._mlir_type

    @always_inline
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        self.address = value

# CHECK-LABEL: lit.fn @"unsafe_ptr
fn unsafe_ptr(s: __mlir_type.`!kgen.string`):
    # CHECK:      lit.call @{{.*}}::@MyUnsafePointer::@"__init__{{.*}}"[mut *"{{.*}}"]
    # CHECK-SAME: <:!AnyType #type_value,
    # CHECK-SAME: :!Int {3},
    # CHECK-SAME: :!AddressSpace {_value: !Int = {0}},
    # CHECK-SAME: :!Bool {:i1 0},
    # CHECK-SAME: :!Int {1},
    # CHECK-SAME: :origin<1> #lit.any.origin>
    var ptr = MyUnsafePointer(__mlir_op.`pop.string.address`(s))
