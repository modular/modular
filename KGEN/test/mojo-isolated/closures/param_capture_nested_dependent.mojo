# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s
# COM: Check that the parameter is properly added to the ClosureWrapper and ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[B:.*]]: !Int, [[A:.*]]: !Int, [[SELFO:.*]]: origin<0>, |>

# CHECK: lit.struct.decl @"fn
# CHECK-SAME: <p0: !Int, p1: !Int, |>

# Check that the closure impl parameter is bound to the struct parameter:
# CHECK-LABEL: lit.fn @"get_test
# CHECK-NEXT: %anonymous2A = lit.var.decl
# CHECK-NEXT: lit.call {{.*}}@"`_CI_{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]], :!Int [[ALoc:.*]], :origin<0> [[SELFO]]>(%self, %anonymous2A)
# CHECK-SAME: !lit.generator<[2]({{.*}}"self": !lit.ref<@{{.*}}::@"`_CI_{{.*}}"<:!Int [[BLoc]], :!Int [[ALoc]], :origin<0> [[SELFO]]>

# COM: Check that the closure wrapper parameter is bound to the struct parameter:
# CHECK-NEXT: %bar = lit.var.decl
# CHECK-NEXT: lit.call @{{.*}}::@"fn{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]], :!Int [[ALoc:.*]], :origin<0> [[SELFO]]>(%anonymous2A, %bar)
# CHECK-SAME: !lit.generator<[2]({{.*}}"self": !lit.ref<@{{.*}}::@"fn{{.*}}"<:!Int [[BLoc]], :!Int [[ALoc]]>


@fieldwise_init
struct Foo[C: Int, D: Int](Copyable, Movable):
    var x: Int

    fn get(self) -> Int:
        return self.x + C


@fieldwise_init
@register_passable
struct Bat[A: Int](Copyable):
    var b: Int

    fn get_test[B: Int](self) -> fn (y: Int) escaping -> Foo[B, A]:
        fn bar(y: Int) -> Foo[B, A]:
            var w = B + self.b + y
            return Foo[B, A](w + A)

        return bar
