# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s | FileCheck %s

from Pointer import Pointer
from Range import range
from IO import print


@register_passable
struct my_iter:
    var start: Int
    var end: Int
    var list: MyList

    fn __copyinit__(self) -> Self:
        return Self {start: self.start, end: self.end, list: self.list}

    fn __init__(list: MyList) -> my_iter:
        var result: my_iter
        result.start = 0
        result.end = list.size
        result.list = list
        return result

    fn __next__(self&: my_iter) -> Int:
        let result: Int = self.start
        self.start += 1
        return self.list[result]

    fn __len__(self: my_iter) -> Int:
        if self.start < self.end:
            return self.end - self.start
        return 0


@register_passable
struct MyList:
    var start: Pointer[Int]
    var size: Int

    fn __copyinit__(self) -> Self:
        return Self {start: self.start, size: self.size}

    fn __init__(ptr: Pointer[Int], size: Int) -> MyList:
        var result: MyList
        result.start = ptr
        result.size = size
        return result

    fn __setitem__(self, idx: Int, val: Int):
        let ptr = self.start + idx
        ptr.store(val)

    fn __getitem__(self, idx: Int) -> Int:
        let ptr = self.start + idx
        return ptr.load()

    fn __iter__(self) -> my_iter:
        return my_iter(self)


fn printInt(x: Int):
    print(x)


# CHECK-LABEL: lit.func @"main()"
fn main():
    let buffer = __mlir_op.`pop.stack_allocation`[
        count:(3).__as_mlir_index(), _type : __mlir_type[`!pop.pointer<`, Int, `>`]
    ]()
    var my_pointer: Pointer[Int]
    my_pointer.address = buffer

    let my_list = MyList(my_pointer, 3)
    my_list[0] = 25
    my_list[1] = 23
    my_list[2] = 19

    # CHECK: %$RANGE = lit.varlet.decl {{.*}} : <@"$for-stmt"::@my_iter>
    # CHECK: %14 = kgen.call @{{.*}}__iter__{{.*}}(%my_list)
    # CHECK: pop.store %14, %$RANGE : !pop.pointer<@"$for-stmt"::@my_iter>
    for item in my_list:
        printInt(item)


# CHECK-LABEL: @"induction_var_scope()"
fn induction_var_scope():
    # CHECK: "item"
    # CHECK: hlcf.loop
    for item in range(0):
        # CHECK: pop.load %item
        # CHECK: "g" = %{{.*}}
        let g = item
    for item in range(0):
        # CHECK: pop.load %item
        let g = item
