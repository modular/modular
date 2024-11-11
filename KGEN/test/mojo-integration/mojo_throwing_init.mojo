# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level full %s | FileCheck %s


struct MemType1:
    var value: Int

    fn __init__(out self, v: Int):
        self.value = v

    fn __copyinit__(out self, existing: Self):
        self.value = existing.value + 1
        print("Copy to", self.value)

    fn __del__(owned self):
        print("MemType1(", self.value, ") destroyed")


struct PartialInitType:
    var mem1: MemType1
    var mem2: MemType1
    var mem3: MemType1
    var setTwice: MemType1

    fn __init__(out self, cond: Int, other: MemType1) raises:
        self.mem1 = other
        self.mem2 = MemType1(2)

        # This copy is entirely elided since it is dead.
        self.setTwice = other
        if cond > 2:
            raise Error("bail on init")
        self.setTwice = MemType1(98)
        self.mem3 = MemType1(3)

    fn __del__(owned self):
        print("destroy PartialInitType")


fn main():
    print("start")
    # CHECK: start
    # CHECK-NOT: destroy PartialInitType
    # CHECK-NEXT: Copy to 43
    # CHECK-NEXT: MemType1( 43 ) destroyed
    # CHECK-NEXT: MemType1( 2 ) destroyed
    # CHECK-NEXT: MemType1( 42 ) destroyed
    # CHECK-NOT: MemType1( 3 ) destroyed
    # CHECK-NEXT: bail on init
    # CHECK-NEXT: done
    try:
        var m = MemType1(42)
        var x = PartialInitType(3, m)
    except e:
        print(e)

    print("done")
