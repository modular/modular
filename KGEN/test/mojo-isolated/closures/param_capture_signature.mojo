# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s
# COM: Signature Capture


@fieldwise_init
@register_passable
struct Foo[a: Int](Copyable):
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn foo[Z: Int, W: Int]() -> Int:
    return Z + W


# COM: Closure Impl has correct parameters.
# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[b:.*]]: !Int, [[a:.*]]: !Int, [[Y:.*]]: !Int, |>


# COM: Closure Wrapper has correct parameters and initializer parameters
# CHECK: lit.struct.decl @"fn
# CHECK-SAME: <p0: !Int, p1: !Int, |>
# CHECK: lit.fn @"__init__{{.*}}<?, Y: !Int>
# CHECK-SAME: (%impl: !lit.ref<{{.*}}@"`_CI_{{.*}}"<:!Int p0, :!Int p1, :!Int [[Y]]>
# CHECK-SAME: %self: !lit.ref<{{.*}}@"fn{{.*}}"<:!Int p0, :!Int p1>
fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int, Y: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y2 = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture
