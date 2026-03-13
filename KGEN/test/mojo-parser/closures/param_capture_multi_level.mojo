# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct Foo[a: Int](ImplicitlyCopyable, RegisterPassable):
    var b: Int

    def get(self) -> Int:
        return Self.a + self.b


def bar[a: Int, b: Int]() -> Int:
    return b + a


# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping0"<X: !lit.generator<() -> !Int>, Y: {{.*}}Foo <:!Int apply(:!lit.generator<() -> !Int> X)>
# CHECK: lit.fn @"__call__{{.*}}(
# CHECK: constant: !Int = <{{.*}} X)>> Y, "b">
def parameter_capture_multiple_levels[
    a: Int, X: def () -> Int, Y: Foo[X()]
](c: Int) -> def (x: Int) escaping -> Int:
    # alias X = bar[a, a]
    # alias Y = Foo[X()](2)

    def p_capture(x: Int) -> Int:
        return Y.b + c

    return p_capture
