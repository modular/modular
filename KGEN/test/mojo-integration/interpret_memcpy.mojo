# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from std.sys import llvm_intrinsic, size_of


fn memcpy(
    dst: UnsafePointer[mut=True, Int, _],
    src: UnsafePointer[Int, _],
    count: Int,
):
    var byte_count = count * size_of[Int]()

    if __mlir_op.`kgen.is_compile_time`[_type=__mlir_type.i1]():
        llvm_intrinsic["llvm.memcpy", NoneType](
            dst.bitcast[Byte](), src.bitcast[Byte](), byte_count
        )
    else:
        # Intentionally mis-match behavior between then and else
        # here for testing.
        pass


struct Data(ImplicitlyCopyable, Writable):
    var _data: UnsafePointer[Int, MutExternalOrigin]
    var _size: Int

    fn __init__(out self, *, size: Int):
        self._data = alloc[Int](size)
        self._size = size
        for i in range(size):
            self._data[i] = 0

    fn __init__(out self, *data: Int):
        var num_elems = len(data)
        self._data = alloc[Int](num_elems)
        self._size = num_elems
        for i in range(num_elems):
            self._data[i] = data[i]

    fn __init__(out self, *, copy: Self):
        self._size = copy._size
        self._data = alloc[Int](self._size)
        for i in range(self._size):
            self._data[i] = copy._data[i]

    fn write_to(self, mut writer: Some[Writer]):
        for i in range(self._size):
            t"data[{i}] = {self._data[i]}\n".write_to(writer)

    fn __add__(self, rhs: Self) -> Self:
        var size = self._size + rhs._size
        var result = Self(size=size)
        memcpy(result._data, self._data, self._size)
        memcpy(result._data + self._size, rhs._data, rhs._size)
        return result

    fn __del__(deinit self):
        self._data.free()


fn main():
    comptime d1 = Data(4, 2)
    comptime d2 = Data(2, 8)
    comptime d3 = d1 + d2
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
