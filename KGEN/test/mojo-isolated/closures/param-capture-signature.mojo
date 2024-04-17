# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s
# COM: Signature Capture


@value
@register_passable
struct Foo[a: Int]:
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
# CHECK: lit.func @"__init__{{.*}}<[[Y:.*]]: !Int, |>
# CHECK-SAME: (%self: !lit.ref<{{.*}}@"fn{{.*}}"<:!Int p0, :!Int p1>
# CHECK-SAME: %impl: !lit.ref<{{.*}}@"`_CI_{{.*}}"<:!Int p0, :!Int p1, :!Int [[Y]]>
fn test_captures_are_ordered_correctly[
    aa: Int, a: Int, b: Int, bb: Int
](c: Int) -> fn (x: Int, y: Foo[b]) escaping -> Foo[a]:
    alias Y = foo[aa, bb]()

    fn p_capture(x: Int, y: Foo[b]) -> Foo[a]:
        return Foo[a](c + Y + b)

    return p_capture
