# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from register import *
from compiler_internal import register
from tensor_internal import OutputTensor
from tensor_internal.managed_tensor_slice import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)

from utils.index import IndexList


@register_internal("test_indices_deduction")
fn _test_indices_deduction[num_indices: Int](indices: IndexList[num_indices]):
    """
    Used as a test to make sure we correctly deduce the size of indices.
    """
    print("Indices size: ")
    print(num_indices)
    print("Indices: ")
    print(indices)


@register_internal("test_make_indices")
fn _test_make_indices[num_indices: Int]() -> IndexList[num_indices]:
    """
    Used to return indices which we can use as a target for tests.
    """
    var out = IndexList[num_indices]()
    for i in range(num_indices):
        out[i] = i
    return out


@register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    fn execute(
        output: OutputTensor[type = DType.float32, rank=1, *_],
        inputs: MutableInputVariadicTensors[type = DType.float32, rank=1, *_],
    ) -> None:
        print("Success!")
