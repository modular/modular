# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# COM: Check that the parameter is properly added to the ClosureImpl despite being defined two levels up.

# CHECK: lit.struct.decl @"`_CI_
# CHECK-SAME: <[[SELFO:.*]]: origin<0>, [[A:.*]]: !Int, |>

# COM: Check that the closure impl parameter is bound to the struct parameter:
# CHECK: lit.call {{.*}}@"`_CI_{{.*}}"::@"__init__{{.*}}<:origin<0> [[SELFO]], :!Int [[ALoc:.*]]>(%self, %anonymous2A)
# CHECK-SAME: !lit.generator<[2]({{.*}}"self": !lit.ref<@{{.*}}::@"`_CI_{{.*}}<:origin<0> *"self`2x", :!Int [[ALoc]]>


@fieldwise_init
@register_passable
struct Foo[A: Int](Copyable, Movable):
    var b: Int

    fn get[C: Int](self) -> fn (y: Int) escaping -> Int:
        fn bar(y: Int) -> Int:
            var w = A + self.b + y
            return w

        return bar
