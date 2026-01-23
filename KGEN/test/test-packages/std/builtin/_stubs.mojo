# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct __MLIRType[T: __TypeOfAllTypes](TrivialRegisterType):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime __copyinit__is_trivial = True
