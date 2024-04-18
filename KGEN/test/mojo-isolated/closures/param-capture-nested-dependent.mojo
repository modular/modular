# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s
# COM: Check that the parameter is properly added to the ClosureWrapper and ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[B:.*]]: !Int, [[A:.*]]: !Int, |>

# CHECK: lit.struct.decl @"fn
# CHECK-SAME: <p0: !Int, p1: !Int, |>

# Check that the closure impl parameter is bound to the struct parameter:
# CHECK-LABEL: lit.func @"get_test
# CHECK-NEXT: %anonymous2A = lit.var.dec
# CHECK-NEXT: lit.call {{.*}}@"`_CI_{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]], :!Int [[ALoc:.*]]>(%anonymous2A, %self)
# CHECK-SAME: !lit.signature<[1]("self": !lit.ref<@"{{.*}}"::@"`_CI_{{.*}}"<:!Int [[BLoc]], :!Int [[ALoc]]>

# COM: Check that the closure wrapper parameter is bound to the struct parameter:
# CHECK-NEXT: %bar = lit.var.decl
# CHECK-NEXT: lit.call @"{{.*}}"::@"fn{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]], :!Int [[ALoc:.*]]>(%bar, %anonymous2A)
# CHECK-SAME: !lit.signature<[2]("self": !lit.ref<@"{{.*}}"::@"fn{{.*}}"<:!Int [[BLoc]], :!Int [[ALoc]]>


@value
struct Foo[C: Int, D: Int]:
    var x: Int

    fn get(self) -> Int:
        return self.x + C


@value
@register_passable
struct Bat[A: Int]:
    var b: Int

    fn get_test[B: Int](self) -> fn (y: Int) escaping -> Foo[B, A]:
        fn bar(y: Int) -> Foo[B, A]:
            var w = B + self.b + y
            return Foo[B, A](w + A)

        return bar
