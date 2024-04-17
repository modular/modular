# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s


@value
@register_passable
struct Foo[a: Int]:
    var b: Int

    fn get(self) -> Int:
        return a + self.b


fn bar[a: Int, b: Int]() -> Int:
    return b + a


# CHECK: lit.struct.decl @"`_CI_{{.*}}escaping0"<[[X:\*".*"]]: !lit.signature<() -> !Int>, [[Y:\*".*"]]: {{.*}}Foo<:!Int apply(:!lit.signature<() -> !Int> [[X]])>
# CHECK: lit.func @"__call__{{.*}}(
# CHECK: constant: !Int = <{{.*}} [[X]])> [[Y]], "b">
fn parameter_capture_multiple_levels[
    a: Int
](c: Int) -> fn (x: Int) escaping -> Int:
    alias X = bar[a, a]
    alias Y = Foo[X()](2)

    fn p_capture(x: Int) -> Int:
        return Y.b + c

    return p_capture
