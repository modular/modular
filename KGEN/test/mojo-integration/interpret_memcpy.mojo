# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from memory import UnsafePointer
from sys import llvm_intrinsic, sizeof


fn memcpy(
    dst: UnsafePointer[Int],
    src: UnsafePointer[Int],
    count: Int,
):
    var byte_count = count * sizeof[Int]()

    if __mlir_op.`kgen.is_compile_time`[_type = __mlir_type.i1]():
        llvm_intrinsic["llvm.memcpy", NoneType](
            dst.bitcast[Byte](), src.bitcast[Byte](), byte_count
        )
    else:
        # Intentionally mis-match behavior between then and else
        # here for testing.
        pass


struct Data:
    var _data: UnsafePointer[Int]
    var _size: Int

    fn __init__(out self, *, size: Int):
        self._data = UnsafePointer[Int].alloc(size)
        self._size = size
        for i in range(size):
            self._data[i] = 0

    fn __init__(out self, *data: Int):
        var num_elems = len(data)
        self._data = UnsafePointer[Int].alloc(num_elems)
        self._size = num_elems
        for i in range(num_elems):
            self._data[i] = data[i]

    fn __copyinit__(out self, existing: Self):
        self._size = existing._size
        self._data = UnsafePointer[Int].alloc(self._size)
        for i in range(self._size):
            self._data[i] = existing._data[i]

    fn __str__(self) -> String:
        var str: String = ""
        for i in range(self._size):
            str += "data[" + String(i) + "] = " + String(self._data[i]) + "\n"
        return str

    fn __add__(self, rhs: Self) -> Self:
        var size = self._size + rhs._size
        var result = Self(size=size)
        memcpy(result._data, self._data, self._size)
        memcpy(result._data.offset(self._size), rhs._data, rhs._size)
        return result


fn main():
    alias d1 = Data(4, 2)
    alias d2 = Data(2, 8)
    alias d3 = d1 + d2
    var d4 = d1 + d2

    # CHECK: data[0] = 4
    # CHECK: data[1] = 2
    # CHECK: data[2] = 2
    # CHECK: data[3] = 8
    print(String(d3))

    # CHECK: data[0] = 0
    # CHECK: data[1] = 0
    # CHECK: data[2] = 0
    # CHECK: data[3] = 0
    print(String(d4))
