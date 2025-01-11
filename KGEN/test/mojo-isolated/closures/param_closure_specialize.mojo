# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable
struct Thing[x: int]:
    pass


# CHECK-LABEL: lit.fn @"pass_param_closure
fn pass_param_closure():
    fn closure(x: Thing[`2`]) escaping:
        pass

    # CHECK: rebind %{{.*}} : !lit.ref<!None, mut {{.*}}> to !lit.ref<{{.*}}<2>
    take_param_closure[`2`](closure)


fn take_param_closure[dt: int](cls: fn (Thing[dt]) escaping -> None):
    pass
