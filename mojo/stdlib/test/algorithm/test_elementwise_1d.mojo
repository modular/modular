# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv, erf, exp, tanh
from sys.info import num_physical_cores, simdwidthof

from algorithm import elementwise
from buffer import NDBuffer
from memory import UnsafePointer

from utils.index import IndexList


# CHECK-LABEL: test_elementwise_1d
fn test_elementwise_1d():
    print("== test_elementwise_1d")

    var num_work_items = num_physical_cores()

    alias num_elements = 64
    var ptr = UnsafePointer[Float32].alloc(num_elements)

    var vector = NDBuffer[DType.float32, 1, num_elements](ptr)

    for i in range(len(vector)):
        vector[i] = i

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn func[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var elem = vector.load[width=simd_width](idx[0])
        var val = exp(erf(tanh(elem + 1)))
        vector.store[width=simd_width](idx[0], val)

    elementwise[func, simdwidthof[DType.float32]()](IndexList[1](num_elements))

    # CHECK: 2.051446{{[0-9]+}}
    print(vector[0])

    ptr.free()


fn main():
    test_elementwise_1d()
