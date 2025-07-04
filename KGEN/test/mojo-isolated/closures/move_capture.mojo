# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK: lit.fn @"__init__{{.*}}%fld0: !lit.ref<!Thing, mut {{.*}}> owned_in_mem
# CHECK: @Thing::@"__moveinit__


@fieldwise_init
struct Thing(Copyable, Movable):
    pass


fn use(u: Thing):
    pass


# CHECK-LABEL: lit.fn @"outer
fn outer(var x: Thing):
    # CHECK: call {{.*}}__init__{{.*}}(%x, %{{.*}})
    @__move_capture(x)
    fn nested() escaping:
        use(x)
