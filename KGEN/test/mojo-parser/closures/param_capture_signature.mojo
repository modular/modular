# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s
# COM: Signature Capture


@fieldwise_init
struct Foo[a: Int](ImplicitlyCopyable, RegisterPassable):
    var b: Int

    def get(self) -> Int:
        return Self.a + self.b


def foo[Z: Int, W: Int]() -> Int:
    return Z + W


# COM: Closure Impl has correct parameters.
# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[b:.*]]: !Int, [[a:.*]]: !Int, Y: !Int, |>


# COM: Closure Wrapper has correct parameters and initializer parameters
# CHECK: lit.struct.decl @"fn
# CHECK-SAME: <p0: !Int, p1: !Int, |>
# CHECK: lit.fn @"__init__{{.*}}<?, Y: !Int>
# CHECK-SAME: (%impl: !lit.ref<!lit.struct<#escaping0 <:!Int p0, :!Int p1, :!Int Y>
# CHECK-SAME: %self: !lit.ref<{{.*}}@"fn{{.*}}"<:!Int p0, :!Int p1>
def test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int, Y: Int
](c: Int) -> def (x: Int, y: Foo[b]) escaping -> Foo[a]:
    comptime Y2 = foo[aa, bb]()

    def p_capture(x: Int, y: Foo[b]) -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture
