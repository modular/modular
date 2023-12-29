# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s


# COM: Check that the parameter is properly added to the ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[A:.*]]: !Int, |>

# COM: Check that the closure impl parameter is bound to the struct parameter:
# CHECK: lit.call @"${{.*}}"::@"`_CI_{{.*}}"::@"__init__{{.*}}<:!Int [[ALoc:.*]]_A>(%anonymous2A, %self)
# CHECK-SAME: !lit.signature<[1]("self": !lit.ref<mut @"${{.*}}"::@"`_CI_{{.*}}<:!Int [[ALoc]]_A>


@value
@register_passable
struct Foo[A: Int]:
    var b: Int

    fn get[C: Int](self) -> fn (y: Int) escaping -> Int:
        fn bar(y: Int) escaping -> Int:
            let w = A + self.b + y
            return w

        return bar
