# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@value
@register_passable("trivial")
struct int:
    pass


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


fn foo(x: int, y: MemType, z: MemType):
    pass


# CHECK: lit.struct.field field0 : !int
# CHECK: lit.struct.field field1 : !MemType
# CHECK: lit.struct.field field2 : !MemType
# CHECK: lit.func @"__init__{{.*}}(
# CHECK-SAME: %self: !lit.ref<{{.*}}> init_self,
# CHECK-SAME: %fld0: !int,
# CHECK-SAME: %fld1: !lit.ref<!MemType, imm {{[^>]*}}> borrow_in_mem,
# CHECK-SAME: %fld2: !lit.ref<!MemType, imm {{.*}}> borrow_in_mem,


# CHECK-LABEL: lit.func @"makes_escaping_closure_3
fn makes_escaping_closure_3(owned x: int, owned y: MemType, inout z: MemType):
    fn take_owned_and_escape():
        foo(x, y, z)
