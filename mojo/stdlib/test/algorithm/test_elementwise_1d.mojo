# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import div_ceil, erf, exp, tanh
from sys.info import sizeof

from Activations import gelu
from algorithm import elementwise
from memory.buffer import Buffer
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.index import StaticIntTuple
from utils.list import Dim, DimList
from utils.vector import UnsafeFixedVector

# CHECK-LABEL: test_elementwise_1d
fn test_elementwise_1d():
    print("== test_elementwise_1d")

    with Runtime() as rt:
        let num_work_items = rt.parallelism_level()

        alias num_elements = 64
        let buf = UnsafeFixedVector[
            __mlir_type[`!pop.scalar<`, DType.float32.value, `>`]
        ](num_elements)

        let vector = Buffer[num_elements, DType.float32](buf.data)

        for i in range(vector.__len__()):
            vector[i] = i

        let chunk_size = div_ceil(vector.__len__(), num_work_items)

        @always_inline
        @parameter
        fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            let elem = vector.simd_load[simd_width](idx[0])
            let val = gelu(exp(erf(tanh(elem))))
            vector.simd_store[simd_width](idx[0], val)

        let out_chain = OwningOutputChainPtr(rt)
        elementwise[1, sizeof[DType.float32](), func](
            StaticIntTuple[1](num_elements), out_chain.borrow()
        )
        out_chain.wait()

        # CHECK: 0.84134{{[0-9]+}}
        print(vector[0])

        buf._del_old()


fn main():
    test_elementwise_1d()
