# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK: lit.func @"__init__{{.*}}_CI_{{.*}} %fld0: !lit.ref<!Thing, mut {{.*}}> owned_in_mem
# CHECK: @Thing::@"__moveinit__


@value
struct Thing:
    pass


fn use(u: Thing):
    pass


# CHECK-LABEL: lit.func @"outer
fn outer(owned x: Thing):
    # CHECK: call {{.*}}__init__{{.*}}(%{{.*}}, %x)
    @__move_capture(x)
    fn nested() escaping:
        use(x)
