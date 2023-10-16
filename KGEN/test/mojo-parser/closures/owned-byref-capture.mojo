# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


@value
struct MemType:
    fn __add__(self, rhs: MemType) -> MemType:
        return MemType()


fn foo(x: Int, y: MemType, z: MemType):
    pass


# CHECK: lit.struct.field field0 : !Int
# CHECK: lit.struct.field field1 : !MemType
# CHECK: lit.struct.field field2 : !MemType
# CHECK: lit.func @"__init__{{.*}}"(
# CHECK-SAME: %self[self]: !kgen.pointer<{{.*}}> init_self,
# CHECK-SAME: %fld0[fld0]: !Int,
# CHECK-SAME: %fld1[fld1]: !kgen.pointer<!MemType> owned_in_mem,
# CHECK-SAME: %fld2[fld2]: !kgen.pointer<!MemType> owned_in_mem)


# CHECK-LABEL: lit.func @"makes_escaping_closure_3
fn makes_escaping_closure_3(owned x: Int, owned y: MemType, inout z: MemType):
    fn take_owned_and_escape() escaping -> NoneType:
        foo(x, y, z)
