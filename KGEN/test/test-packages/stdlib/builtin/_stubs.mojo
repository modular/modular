# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](ImplicitlyCopyable, Movable):
    var value: Self.T
    alias __del__is_trivial = True
    alias __moveinit__is_trivial = True
    alias __copyinit__is_trivial = True
