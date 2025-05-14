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

    fn __copyinit__(out self, existing: Self):
        self.start = existing.start
        self.end = existing.end
        self.list = existing.list

    @implicit
    fn __init__(out self, list: MyList):
        self.start = 0
        self.end = list.size
        self.list = list

    fn __next__(mut self: my_iter) -> Int:
        var result: Int = self.start
        self.start += 1
        return self.list[result]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self: my_iter) -> Int:
        if self.start < self.end:
            return self.end - self.start
        return 0


@fieldwise_init
struct MyList(Copyable):
    var start: UnsafePointer[Int]
    var size: Int

    fn __setitem__(mut self, idx: Int, val: Int):
        var ptr = self.start + idx
        ptr[] = val

    fn __getitem__(mut self, idx: Int) -> Int:
        var ptr = self.start + idx
        return ptr[]

    fn __iter__(mut self) -> my_iter:
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
