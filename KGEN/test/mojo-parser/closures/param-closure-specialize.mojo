# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s


alias int = __mlir_type.index
alias two = __mlir_attr.`2 : index`


@register_passable
struct Thing[x: int]:
    pass


# CHECK-LABEL: lit.func @"pass_param_closure
fn pass_param_closure():
    fn closure(x: Thing[two]) escaping:
        pass

    # CHECK: rebind %{{.*}} : !lit.ref<mut !wrapper, {{.*}}> to !lit.ref<{{.*}}<2>
    take_param_closure[two](closure)


fn take_param_closure[dt: int](cls: fn (Thing[dt]) escaping -> None):
    pass
