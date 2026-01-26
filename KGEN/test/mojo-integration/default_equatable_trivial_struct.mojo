# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s

# Test that the default Equatable implementation works for single-field
# @register_passable("trivial") structs.

from testing import assert_true, assert_false


@fieldwise_init
@register_passable("trivial")
struct SingleFieldTrivial(Equatable):
    var value: Int


fn main() raises:
    var a = SingleFieldTrivial(42)
    var b = SingleFieldTrivial(42)
    var c = SingleFieldTrivial(10)

    assert_true(a == b, "equal values should be equal")
    assert_false(a == c, "different values should not be equal")
    assert_true(a != c, "different values should be not-equal")
    assert_false(a != b, "equal values should not be not-equal")
