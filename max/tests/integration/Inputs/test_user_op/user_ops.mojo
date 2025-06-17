# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import iota
from register import *
import compiler_internal as compiler
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


@compiler.register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    fn execute(
        output: OutputTensor[dtype = DType.float32, rank=1, *_],
        inputs: MutableInputVariadicTensors[dtype = DType.float32, rank=1, *_],
    ) -> None:
        print("Success!")


@fieldwise_init
@register_passable
struct SIMDPair[S0: Int, S1: Int](Copyable, Movable):
    var x: SIMD[DType.int32, S0]
    var y: SIMD[DType.int32, S1]


@compiler.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    fn execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](
            iota[DType.int32, P0](), iota[DType.int32, P1](P0)
        )


@compiler.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    fn execute[
        P0: Int
    ](
        output: OutputTensor[dtype = DType.int32, rank=1], x: SIMDPair[P0, _]
    ) capturing:
        output.store(IndexList[1](0), x.x)
        output.store(IndexList[1](P0), x.y)

    @staticmethod
    fn shape[P0: Int](x: SIMDPair[P0, _]) -> IndexList[1]:
        return IndexList[1](x.S0 + x.S1)
