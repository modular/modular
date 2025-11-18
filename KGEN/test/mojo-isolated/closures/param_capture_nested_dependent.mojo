# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s
# COM: Check that the parameter is properly added to the ClosureWrapper and ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <B: !Int, A: !Int, [[SELFO:.*]]: origin<0>, |>(

# CHECK: lit.struct.decl @"fn
# CHECK-SAME: <p0: !Int, p1: !Int, |>

# Check that the closure impl parameter is bound to the struct parameter:
# CHECK-LABEL: lit.fn @"get_test
# CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
# CHECK-NEXT: lit.call {{.*}}@"`_CI_{{.*}}"::@"__init__{{.*}}<:!Int B, :!Int A, :origin<0> [[SELFO]]>(%self, %__call_result_tmp__)
# CHECK-SAME: !lit.generator<[2]({{.*}}"self": !lit.ref<@{{.*}}::@"`_CI_{{.*}}"<:!Int B, :!Int A, :origin<0> [[SELFO]]>

# COM: Check that the closure wrapper parameter is bound to the struct parameter:
# CHECK-NEXT: %bar = lit.var.decl
# CHECK-NEXT: lit.call @{{.*}}::@"fn{{.*}}"::@"__init__{{.*}}<:!Int B, :!Int A, :origin<0> [[SELFO]]>(%__call_result_tmp__, %bar)
# CHECK-SAME: !lit.generator<[2]({{.*}}"self": !lit.ref<@{{.*}}::@"fn{{.*}}"<:!Int B, :!Int A>


@fieldwise_init
struct Foo[C: Int, D: Int](ImplicitlyCopyable, Movable):
    var x: Int

    fn get(self) -> Int:
        return self.x + Self.C


@fieldwise_init
@register_passable
struct Bat[A: Int](ImplicitlyCopyable):
    var b: Int

    fn get_test[B: Int](self) -> fn (y: Int) escaping -> Foo[B, Self.A]:
        fn bar(y: Int) -> Foo[B, Self.A]:
            var w = B + self.b + y
            return Foo[B, Self.A](w + Self.A)

        return bar
