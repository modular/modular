# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Thing[x: Int](RegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"pass_param_closure
def pass_param_closure():
    def closure(x: Thing[2]) escaping:
        pass

    # CHECK: rebind %{{.*}} : !lit.ref<!None, mut {{.*}}> to !lit.ref<{{.*}}<:!Int {2}>
    take_param_closure[2](closure)


def take_param_closure[dt: Int](cls: def (Thing[dt]) escaping -> None):
    pass
