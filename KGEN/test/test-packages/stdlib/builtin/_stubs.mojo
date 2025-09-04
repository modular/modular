# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](ImplicitlyCopyable, Movable):
    var value: T
