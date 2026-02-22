# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct __MLIRType[T: __TypeOfAllTypes](TrivialRegisterPassable):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __move_ctor_is_trivial = True
    comptime __copy_ctor_is_trivial = True
