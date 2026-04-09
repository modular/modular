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
    origin: Origin[mut=True],
    address_space: AddressSpace = AddressSpace.GENERIC,
    exclusive: Bool = False,
    alignment: Int = 1,
]:
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    var address: Self._mlir_type

    @always_inline
    @implicit
    def __init__(out self, value: Self._mlir_type):
        self.address = value


# CHECK-LABEL: lit.fn @"unsafe_ptr
def unsafe_ptr(s: __mlir_type.`!kgen.string`):
    # CHECK:      lit.call {{.*}}::@MyUnsafePointer::@"__init__{{.*}}"[mut *"{{.*}}"]
    # CHECK-SAME: :!AnyType #type_value,
    # CHECK-SAME: :!Int {3},
    # CHECK-SAME: :!AddressSpace {_value: !Int = {0}},
    # CHECK-SAME: :!Bool {:i1 0},
    # CHECK-SAME: :!Int {1}>>
    var ptr = MyUnsafePointer[origin=AnyOrigin[mut=True]](
        __mlir_op.`pop.string.address`(s)
    )
