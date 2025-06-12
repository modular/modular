# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# Names of globals start with __ to avoid the "globals are deprecated" warning.

# RUN: %bare-mojo -debug-level full %s | FileCheck %s
# COM: This test fails with lld, force gnu-ld instead
# RUN: MODULAR_MOJO_MAX_SYSTEM_LIBS=$MODULAR_MOJO_MAX_SYSTEM_LIBS,-fuse-ld=ld %bare-mojo build %s -o %t
# RUN: %t | FileCheck %s


@register_passable("trivial")
struct ThreeInts:
    var x: Int
    var y: Int
    var z: Int

    fn __init__(out self):
        self.x = 0
        self.y = 0
        self.z = 0


struct OwnedInt:
    var value: Int

    fn __init__(out self, value: Int, mut ints: ThreeInts):
        if ints.x == 0:
            ints.x = value
        elif ints.y == 0:
            ints.y = value
        elif ints.z == 0:
            ints.z = value

        print("got initialized:", value)
        self.value = value

    fn __del__(owned self):
        print("got deleted: ", self.value)


var __ints = ThreeInts()

var __x = OwnedInt(10, __ints)
var __y = OwnedInt(__x.value + 20, __ints)
var __z = OwnedInt(__y.value + 30, __ints)


fn main():
    # CHECK: got initialized: 10
    # CHECK-NEXT: got initialized: 30
    # CHECK-NEXT: got initialized: 60

    # CHECK: 10 30 60
    print(__x.value, __y.value, __z.value)
    # CHECK: 10 30 60
    print(__ints.x, __ints.y, __ints.z)

    # CHECK-NEXT: got deleted: 60
    # CHECK-NEXT: got deleted: 30
    # CHECK-NEXT: got deleted: 10
