# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK: lit.fn @"__init__{{.*}}%fld0: !lit.ref<!Thing, mut {{.*}}> owned_in_mem
# CHECK: lit.call {{.*}}@Thing::@"__init__{{.*}}*, "take"


@fieldwise_init
struct Thing(ImplicitlyCopyable):
    pass


def use(u: Thing):
    pass


# CHECK-LABEL: lit.fn @"outer
def outer(var x: Thing):
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x, %{{.*}}){{.*}}owned_in_mem
    @__move_capture(x)
    def nested() escaping:
        use(x)
