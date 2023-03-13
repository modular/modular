# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer
from DType import DType
from Functional import parallelize, map, div_ceil
from Int import Int
from IO import print
from LLCL import num_cores, Runtime
from SIMD import SIMD
from Range import range

# CHECK-LABEL: test_parallelize
fn test_parallelize():
    print("== test_parallelize\n")

    let num_work_items = 4
    let rt = Runtime(num_work_items)

    let vector = Buffer[20, DType.index].stack_allocation()

    for i in range(vector.__len__()):
        vector[i] = i.__as_mlir_index()

    let chunk_size = div_ceil(vector.__len__(), num_work_items)

    @always_inline
    fn parallel_fn(thread_id: Int):
        let start = thread_id * chunk_size
        let end = Int.min(start + chunk_size, vector.__len__())

        @always_inline
        fn add_two(idx: Int):
            vector[start + idx] = vector[start + idx] + 2

        map[add_two](end - start)

    parallelize[parallel_fn](num_cores(), 20)

    # CHECK-NOT: ERROR
    for ii in range(vector.__len__()):  # TODO(#8365) use `i`
        let expected_val = ii + 2
        if Int(vector[ii].value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")


fn main():
    test_parallelize()
