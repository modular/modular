# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s
# COM: Check that the parameter is properly added to the ClosureWrapper and ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[B:.*]]: !Int, [[A:.*]]: !Int, |>

# CHECK: lit.struct.decl @"_CW_
# CHECK-SAME: <p0: !Int, p1: !Int, |>

# Check that the closure impl parameter is bound to the struct parameter:
# CHECK-LABEL: lit.func @"get_test
# CHECK-NEXT: %anonymous2A = lit.varlet.dec
# CHECK-NEXT: lit.call @"${{.*}}"::@"`_CI_{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]]_B, :!Int [[ALoc:.*]]_A>(%anonymous2A, %self)
# CHECK-SAME: !lit.signature<[1]("self": !lit.ref<mut @"${{.*}}"::@"`_CI_{{.*}}"<:!Int [[BLoc]]_B, :!Int [[ALoc]]_A>

# COM: Check that the closure wrapper parameter is bound to the struct parameter:
# CHECK-NEXT:  %anonymous2A_0 = lit.varlet.decl
# CHECK-NEXT: %1 = lit.ref.to_pointer %anonymous2A
# CHECK-NEXT: lit.call @"${{.*}}"::@"_CW_{{.*}}"::@"__init__{{.*}}<:!Int [[BLoc:.*]]_B, :!Int [[ALoc:.*]]_A>(%anonymous2A_0, %1)
# CHECK-SAME: !lit.signature<[1]("self": !lit.ref<mut @"${{.*}}"::@"_CW_{{.*}}"<:!Int [[BLoc]]_B, :!Int [[ALoc]]_A>


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
        fn bar(y: Int) escaping -> Foo[B, A]:
            let w = B + self.b + y
            return Foo[B, A](w + A)

        return bar
