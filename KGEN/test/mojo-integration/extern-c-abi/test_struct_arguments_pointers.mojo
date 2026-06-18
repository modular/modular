# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_test_ptr_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_pointers
# RUN: %t.dir/test_pointers | FileCheck %s

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# ============================================================================
# 16-byte struct (pointer + Int32 + padding)
# ARM64 AAPCS: IntegerPair class (two registers)
# ============================================================================
@fieldwise_init
struct PtrInt32Struct(TrivialRegisterPassable):
    @__allow_legacy_any_origin_fields
    var p: UnsafePointer[Int32, MutAnyOrigin]
    var i: Int32


def test_ptr_int32():
    var p = alloc[Int32](1)
    p[] = 999
    var s = PtrInt32Struct(p, Int32(200))
    var result = external_call["c_func_ptr_int32", PtrInt32Struct](s)
    # C increments pointer by 1 byte and int by 1
    print("ptr_int32:", result.i)
    # Verify original allocation still accessible
    print("ptr_int32_val:", p[])
    p.free()


# CHECK: ptr_int32: 201
# CHECK: ptr_int32_val: 999


# ============================================================================
# 24-byte struct (three pointers) - MEMORY class
# ============================================================================
@fieldwise_init
struct ThreePtrStruct(TrivialRegisterPassable):
    @__allow_legacy_any_origin_fields
    var a: UnsafePointer[Int32, MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var b: UnsafePointer[Int32, MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var c: UnsafePointer[Int32, MutAnyOrigin]


def test_three_ptr():
    var pa = alloc[Int32](1)
    var pb = alloc[Int32](1)
    var pc = alloc[Int32](1)
    pa[] = 111
    pb[] = 222
    pc[] = 333
    var s = ThreePtrStruct(pa, pb, pc)
    var result = external_call["c_func_three_ptr", ThreePtrStruct](s)
    # C advances each pointer by 1 byte, but original allocations are intact
    print("three_ptr:", pa[], pb[], pc[])
    pa.free()
    pb.free()
    pc.free()


# CHECK: three_ptr: 111 222 333


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    test_ptr_int32()

    test_three_ptr()
