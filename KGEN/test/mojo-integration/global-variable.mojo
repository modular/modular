# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s
# RUN: mojo build %mojo_args %s -o %t
# RUN: %t | FileCheck %s


@register_passable("trivial")
struct ThreeInts:
    var x: Int
    var y: Int
    var z: Int

    fn __init__(inout self):
        self.x = 0
        self.y = 0
        self.z = 0


struct OwnedInt:
    var value: Int

    fn __init__(inout self, value: Int, inout ints: ThreeInts):
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


var ints = ThreeInts()

var x = OwnedInt(10, ints)
var y = OwnedInt(x.value + 20, ints)
var z = OwnedInt(y.value + 30, ints)


fn main():
    # CHECK: got initialized: 10
    # CHECK-NEXT: got initialized: 30
    # CHECK-NEXT: got initialized: 60

    # CHECK: 10 30 60
    print(x.value, y.value, z.value)
    # CHECK: 10 30 60
    print(ints.x, ints.y, ints.z)

    # CHECK-NEXT: got deleted: 60
    # CHECK-NEXT: got deleted: 30
    # CHECK-NEXT: got deleted: 10
