# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](ImplicitlyCopyable):
    var value: Self.T
    comptime __del__is_trivial = True
    comptime __moveinit__is_trivial = True
    comptime __copyinit__is_trivial = True
