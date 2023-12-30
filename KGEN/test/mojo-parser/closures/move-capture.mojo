# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s

# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK: lit.func @"__init__{{.*}}_CI_{{.*}} %fld0[fld0]: !lit.ref<mut !Thing, {{.*}}> owned_in_mem
# CHECK: @Thing::@"__moveinit__

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

@value
struct Thing:
    pass


fn use(u: Thing):
    pass


# CHECK-LABEL: lit.func @"outer
fn outer(owned x: Thing):
    # CHECK: [[X_TAKEN:%.*]] = lit.ownership.end_lifetime %x
    # CHECK: call {{.*}}__init__{{.*}}(%{{.*}}, [[X_TAKEN]])
    @__move_capture(x)
    fn nested() escaping:
        use(x)
