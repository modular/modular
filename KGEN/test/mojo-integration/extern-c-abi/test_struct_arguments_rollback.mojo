# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# C reference: c_abi_test_rollback.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_rollback
# RUN: %t.dir/test_rollback | FileCheck %s

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# 12-byte struct: two eightbytes, both INTEGER class (IntegerPair).
@fieldwise_init
struct Int3(TrivialRegisterPassable):
    var x: Int32
    var y: Int32
    var z: Int32


# Control: struct passed first, so it fits in registers.
def test_struct_early():
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_early", Int32](v)
    print("struct_early:", status)


# CHECK: struct_early: 0


# Struct passed after five integer-class args: must roll back to the stack.
# Under the old per-argument lowering the struct was split, corrupting it.
def test_struct_after_five():
    var p0 = alloc[Int32](1)
    var p1 = alloc[Int32](1)
    var v = Int3(Int32(11), Int32(22), Int32(33))
    var status = external_call["check_struct_after_five", Int32](
        p0, p1, Int32(101), Int32(202), Int32(303), v
    )
    print("struct_after_five:", status)
    p0.free()
    p1.free()


# CHECK: struct_after_five: 0


def main():
    test_struct_early()
    test_struct_after_five()
