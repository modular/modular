# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](Copyable, Movable):
    var value: T
