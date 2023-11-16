# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import vectorize
from algorithm.functional import _elementwise_impl
from memory.buffer import NDBuffer
from runtime.llcl import OutputChainPtr

from utils._annotations import *
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
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
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
                let result = input_0_fn[simd_width, 3](indices)
                output_0_fn[simd_width, 3](indices, result)

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


@mogg_register_override("mo.sqrt", 1)
@mogg_elementwise
@export
fn sqrt_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    print("In override sqrt")
    return value


@mogg_register("test_static_shape_deduction")
@export
fn test_static_shape_deduction[
    type: DType, rank: Int, input_0_static_shape: DimList
](tensor: NDBuffer[rank, input_0_static_shape, type],):
    print("Printing shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = input_0_static_shape.at[idx]()

        @parameter
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[rank, body]()


@mogg_register("test_static_shape_output")
@export
fn test_static_shape_output[
    type: DType, rank: Int, output_0_static_shape: DimList
]() -> NDBuffer[rank, output_0_static_shape, type]:
    print("Printing output shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = output_0_static_shape.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[rank, body]()
    return NDBuffer[rank, output_0_static_shape, type](
        DTypePointer[type](), StaticIntTuple[rank](), StaticIntTuple[rank]()
    )


@mogg_register("test_int_list_param")
@export
fn test_int_list_param[length: Int, int_list: DimList]():
    print("Printing parameter: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = int_list.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[length, body]()


@mogg_register("test_custom_op")
@always_inline
@export
fn test_unary_kernel[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](
    input_shape: StaticIntTuple[rank],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    print("World!")

    if not single_thread_blocking_override:
        out_chain.mark_ready()


@mogg_register_shape_func("test_custom_op")
@always_inline
@export
fn test_unary_kernel_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](
    data: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> StaticIntTuple[rank]:
    print("Hello")

    return data.get_shape()


@mogg_register("tf.Identity")
@mogg_register("torch.aten.abs")
@mogg_register("monnx.abs_v13")
@always_inline
@export
fn test_custom_identity[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_0_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](
    input_shape: StaticIntTuple[rank],
    output_shape: StaticIntTuple[rank],
    out_chain: OutputChainPtr,
):
    print("The custom identity op is running!")

    @parameter
    @always_inline
    fn identity[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let x = input_0_fn[simd_width, rank](idx)
        output_0_fn[simd_width, rank](idx, x)

    _elementwise_impl[
        rank,
        simd_width,
        single_thread_blocking_override,
        identity,
        target="cpu",
    ](
        input_shape,
        out_chain,
    )


@mogg_register_shape_func("tf.Identity")
@mogg_register_shape_func("torch.aten.abs")
@mogg_register_shape_func("monnx.abs_v13")
@always_inline
@export
fn test_custom_identity_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](
    data: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> StaticIntTuple[rank]:
    return data.get_shape()
