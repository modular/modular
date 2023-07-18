# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# COM: This test will pass when https://reviews.llvm.org/D154802 lands.
# XFAIL: system-linux

from IO import print


struct OwnedInt:
    var value: Int

    fn __init__(inout self, value: Int):
        print("got initialized:", value)
        self.value = value

    fn __del__(owned self):
        print("got deleted: ", self.value)


var x = OwnedInt(10)
let y = OwnedInt(20)


fn mutate(inout ref: OwnedInt):
    ref.value += 5
    x.value = y.value + ref.value


fn main():
    # CHECK-LABEL: === test_globals
    print("=== test_globals")
    # CHECK-NEXT: x: 10
    print("x:", x.value)
    # CHECK-NEXT: y: 20
    print("y:", y.value)
    mutate(x)
    # CHECK-NEXT: x: 35
    print("x:", x.value)
    # CHECK-NEXT: x: 60
    mutate(x)
    print("x:", x.value)
    # FIXME(#16605): Global destructors don't work in JIT mode.
    # XCHECK-NEXT: got deleted: 60
    # XCHECK-NEXT: got deleted: 20
