# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | FileCheck %s


@value
@register_passable("trivial")
struct Int:
    pass


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
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


fn foo(x: Int, y: MemType, z: MemType):
    pass


# CHECK: lit.struct.field field0 : !Int
# CHECK: lit.struct.field field1 : !MemType
# CHECK: lit.struct.field field2 : !MemType
# CHECK: lit.func @"__init__{{.*}}(
# CHECK-SAME: %self: !lit.ref<{{.*}}> init_self,
# CHECK-SAME: %fld0: !Int borrow,
# CHECK-SAME: %fld1: !lit.ref<!MemType, {{[^>]*}}> borrow_in_mem,
# CHECK-SAME: %fld2: !lit.ref<!MemType, {{.*}}> borrow_in_mem,


# CHECK-LABEL: lit.func @"makes_escaping_closure_3
fn makes_escaping_closure_3(owned x: Int, owned y: MemType, inout z: MemType):
    fn take_owned_and_escape() escaping:
        foo(x, y, z)
