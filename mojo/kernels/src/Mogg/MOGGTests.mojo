# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import vectorize
from memory.buffer import NDBuffer
from MOGGDecorators import *
from runtime.llcl import OutputChainPtr

from utils.index import StaticIntTuple
from utils.list import DimList

# ===----------------------------------------------------------------------===#
# Special test targets just for generation tests
# ===----------------------------------------------------------------------===#


@mogg_register("test_many_ranks_and_types")
@export
fn test_many_ranks_and_types[
    type1: DType,
    rank1: Int,
    type2: DType,
    rank2: Int,
    type3: DType,
    rank3: Int,
    type4: DType,
    rank4: Int,
    type5: DType,
    rank5: Int,
](
    tensor1: NDBuffer[rank1, DimList.create_unknown[rank1](), type1],
    tensor2: NDBuffer[rank2, DimList.create_unknown[rank2](), type2],
    tensor3: NDBuffer[rank3, DimList.create_unknown[rank3](), type3],
    tensor4: NDBuffer[rank4, DimList.create_unknown[rank4](), type4],
    tensor5: NDBuffer[rank5, DimList.create_unknown[rank5](), type5],
) -> NDBuffer[rank1, DimList.create_unknown[rank1](), type1]:
    """
    Used as a test target to ensure parameter deduction works when there are
    many to deduce and also used to check errors.
    """
    return tensor1


@mogg_register("test_one_rank_many_tensor")
@export
fn test_one_rank_many_tensor[
    type: DType, rank: Int
](
    tensor1: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor2: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor3: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor4: NDBuffer[rank, DimList.create_unknown[rank](), type],
    tensor5: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:
    """
    Used as a test target to ensure we can deduce type and rank when used by
    many arguments.
    """
    return tensor1


@mogg_register("test_3D_in_out_lambda")
@export
fn test_3D_in_out_lambda[
    type: DType,
    simd_width: Int,
    input_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
](
    tensor1: NDBuffer[3, DimList.create_unknown[3](), type],
    output: NDBuffer[3, DimList.create_unknown[3](), type],
    out_chain: OutputChainPtr,
) -> NDBuffer[3, DimList.create_unknown[3](), type]:
    """
    Used as a target to test passing input and output lambdas.
    """

    for x in range(0, tensor1.dim[0]()):
        for y in range(0, tensor1.dim[1]()):

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                let indices = StaticIntTuple[3](x, y, idx)
                let result = input_0_fn[type, simd_width, 3](indices)
                output_0_fn[type, simd_width, 3](indices, result)

            vectorize[
                simd_width,
                func_wrapper,
            ](tensor1.dim[2]())

    out_chain.mark_ready()
    return output


@mogg_register("test_indices_deduction")
@export
fn _test_indices_deduction[
    num_indices: Int
](indices: StaticIntTuple[num_indices]):
    """
    Used as a test to make sure we correctly deduce the size of indices.
    """
    print("Indices size: ")
    print(num_indices)
    print("Indices: ")
    print(indices)


@mogg_register("test_make_indices")
@export
fn _test_make_indices[num_indices: Int]() -> StaticIntTuple[num_indices]:
    """
    Used to return indices which we can use as a target for tests.
    """
    var out = StaticIntTuple[num_indices]()
    for i in range(num_indices):
        out[i] = i
    return out
