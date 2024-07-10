# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import UnsafePointer


struct my_iter:
    var start: Int
    var end: Int
    var list: MyList

    fn __copyinit__(inout self, existing: Self):
        self.start = existing.start
        self.end = existing.end
        self.list = existing.list

    fn __init__(inout self, list: MyList):
        self.start = 0
        self.end = list.size
        self.list = list

    fn __next__(inout self: my_iter) -> Int:
        var result: Int = self.start
        self.start += 1
        return self.list[result]

    fn __len__(inout self: my_iter) -> Int:
        if self.start < self.end:
            return self.end - self.start
        return 0


struct MyList:
    var start: UnsafePointer[Int]
    var size: Int

    fn __copyinit__(inout self, existing: Self):
        self.start = existing.start
        self.size = existing.size

    fn __init__(inout self, ptr: UnsafePointer[Int], size: Int):
        self.start = ptr
        self.size = size

    fn __setitem__(inout self, idx: Int, val: Int):
        var ptr = self.start + idx
        ptr[] = val

    fn __getitem__(inout self, idx: Int) -> Int:
        var ptr = self.start + idx
        return ptr[]

    fn __iter__(inout self) -> my_iter:
        return my_iter(self)


fn printInt(x: Int):
    print(x)


fn main():
    var buffer = __mlir_op.`pop.stack_allocation`[
        count = Int(3).value,
        _type = __mlir_type[`!kgen.pointer<`, Int, `>`],
    ]()
    var my_pointer = UnsafePointer[Int](buffer)

    var my_list = MyList(my_pointer, 3)
    my_list[0] = 25
    my_list[1] = 23
    my_list[2] = 19

    # CHECK: 25
    # CHECK: 23
    # CHECK: 19
    for item in my_list:
        printInt(item)
