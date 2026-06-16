# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


struct my_iter:
    var start: Int
    var end: Int
    var list: MyList

    def __init__(out self, *, copy: Self):
        self.start = copy.start
        self.end = copy.end
        self.list = copy.list

    @implicit
    def __init__(out self, list: MyList):
        self.start = 0
        self.end = list.size
        self.list = list

    def __next__(mut self: my_iter) raises StopIteration -> Int:
        if self.__len__() <= 0:
            raise StopIteration()
        var result: Int = self.start
        self.start += 1
        return self.list[result]

    def __len__(self: my_iter) -> Int:
        if self.start < self.end:
            return self.end - self.start
        return 0


@fieldwise_init
struct MyList(ImplicitlyCopyable):
    var start: UnsafePointer[Int, MutAnyOrigin]
    var size: Int

    def __setitem__(mut self, idx: Int, val: Int):
        var ptr = self.start + idx
        ptr[] = val

    def __getitem__(mut self, idx: Int) -> Int:
        var ptr = self.start + idx
        return ptr[]

    def __iter__(mut self) -> my_iter:
        return my_iter(self)


def printInt(x: Int):
    print(x)


def main():
    var buffer = __mlir_op.`pop.stack_allocation`[
        count=Int(3).__mlir_index__(),
        _type=__mlir_type[`!kgen.pointer<`, Int, `>`],
    ]()
    var my_pointer = UnsafePointer[Int, MutAnyOrigin](buffer)

    var my_list = MyList(my_pointer, 3)
    my_list[0] = 25
    my_list[1] = 23
    my_list[2] = 19

    # CHECK: 25
    # CHECK: 23
    # CHECK: 19
    for item in my_list:
        printInt(item)
