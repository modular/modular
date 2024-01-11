# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import div_ceil, erf, exp, tanh
from sys.info import simdwidthof

from Activations import gelu
from algorithm import elementwise
from memory.buffer import Buffer
from runtime.llcl import Runtime

from utils.index import StaticIntTuple
from utils.list import Dim, DimList
from collections.vector import DynamicVector


# CHECK-LABEL: test_elementwise_1d
fn test_elementwise_1d():
    print("== test_elementwise_1d")

    let num_work_items = Runtime().parallelism_level()

    alias num_elements = 64
    let ptr = DTypePointer[DType.float32].alloc(num_elements)

    let vector = Buffer[num_elements, DType.float32](ptr)

    for i in range(len(vector)):
        vector[i] = i

    let chunk_size = div_ceil(len(vector), num_work_items)

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let elem = vector.simd_load[simd_width](idx[0])
        let val = exp(erf(tanh(elem + 1)))
        vector.simd_store[simd_width](idx[0], val)

    elementwise[1, simdwidthof[DType.float32](), func](
        StaticIntTuple[1](num_elements)
    )

    # CHECK: 2.051446{{[0-9]+}}
    print(vector[0])

    ptr.free()


fn main():
    test_elementwise_1d()
