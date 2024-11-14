# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct ConstantMemoryMapping:
    var name: StringLiteral
    var ptr: UnsafePointer[NoneType]
    var byte_count: Int

    fn __init__(
        inout self,
        name: StringLiteral,
        ptr: UnsafePointer[NoneType],
        byte_count: Int,
    ):
        self.name = name
        self.ptr = ptr
        self.byte_count = byte_count
