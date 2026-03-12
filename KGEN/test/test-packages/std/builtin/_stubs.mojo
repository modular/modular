# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct __MLIRType[T: __mlir_type.`!kgen.non_struct_type`](
    TrivialRegisterPassable
):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __move_ctor_is_trivial = True
    comptime __copy_ctor_is_trivial = True
