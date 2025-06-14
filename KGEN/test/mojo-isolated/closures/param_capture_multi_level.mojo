# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable
struct Foo[a: Int](Copyable):
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn bar[a: Int, b: Int]() -> Int:
    return b + a


# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping0"<X: !lit.generator<() -> !Int>, Y: {{.*}}Foo<:!Int apply(:!lit.generator<() -> !Int> X)>
# CHECK: lit.fn @"__call__{{.*}}(
# CHECK: constant: !Int = <{{.*}} X)> Y, "b">
fn parameter_capture_multiple_levels[
    a: Int, X: fn()->Int, Y: Foo[X()]
](c: Int) -> fn (x: Int) escaping -> Int:
    #alias X = bar[a, a]
    #alias Y = Foo[X()](2)

    fn p_capture(x: Int) -> Int:
        return Y.b + c

    return p_capture
