# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable("trivial")
struct Index:
    pass


@fieldwise_init
struct MemType(Copyable, Movable):
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


fn foo(x: Index, y: MemType, z: MemType):
    pass


# CHECK: lit.struct.field field0 : !Index
# CHECK: lit.struct.field field1 : !MemType
# CHECK: lit.struct.field field2 : !MemType
# CHECK: lit.fn @"__init__{{.*}}(
# CHECK-SAME: %fld0: !Index,
# CHECK-SAME: %fld1: !lit.ref<!MemType, imm {{[^>]*}}> read_mem,
# CHECK-SAME: %fld2: !lit.ref<!MemType, imm {{.*}}> read_mem,
# CHECK-SAME: %self: !lit.ref<{{.*}}> byref_result)


# CHECK-LABEL: lit.fn @"makes_escaping_closure_3
fn makes_escaping_closure_3(var x: Index, var y: MemType, mut z: MemType):
    fn take_owned_and_escape():
        foo(x, y, z)
