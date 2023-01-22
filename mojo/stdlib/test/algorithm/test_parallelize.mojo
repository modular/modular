# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_parallelize::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import Buffer
from DType import DType
from Functional import parallelize
from Int import Int
from IO import print
from LLCL import num_cores
from SIMD import SIMD

# CHECK-LABEL: test_parallelize
fn test_parallelize():
    print("== test_parallelize\n")

    alias size = 20

    let vector = Buffer[size, DType.index.value].stack_allocation()

    var i: Int = 0
    while i < vector.__len__():
        vector.__setitem__(i, i.__as_mlir_index())
        i += 1

    @always_inline
    fn parallel_fn(start: Int, end: Int):
        #######################################################################
        # TODO (7316): Want to be able to not have to write a loop and instead
        # write
        # @always_inline
        # fn add_two(idx: Int):
        #     vector.__setitem__(start + idx, vector.__getitem__(start + idx) + 2)
        # map[add_two](end - start)
        #######################################################################
        var i = start
        while i < end:
            vector.__setitem__(i, vector.__getitem__(i) + 2)
            i += 1

    parallelize[parallel_fn](num_cores(), size)

    i = 0
    # CHECK-NOT: ERROR
    while i < vector.__len__():
        let expected_val = i + 2
        if Int(vector.__getitem__(i).value) != expected_val:
            print("ERROR: Expecting the result to be i + 2")
        i += 1


@export
fn main() -> __mlir_type.index:
    test_parallelize()
    return 0
