# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyTrivialRegType](Movable, Copyable):
    var value: T
