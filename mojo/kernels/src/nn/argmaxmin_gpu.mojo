# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv

from nn.topk_gpu import _topk_gpu
from nn.reshape import reshape
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext

from utils import IndexList


fn argmaxmin_gpu[
    type: DType, output_type: DType, rank: Int, largest: Bool
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[output_type, rank],
) raises:
    """
    Wraps the Top-K GPU kernel with K=1 to perform argmax on the inner-most
    dimension.

    Parameters:
        type: DType - The data type of the input tensor.
        output_type: DType - The data type of the output tensor.
        rank: Int - The rank of the input tensor.
        largest: Bool - Whether to perform argmax or argmin.
    Args:
        ctx: DeviceContext - The device context.
        input: NDBuffer[type, rank] - The input tensor allocated on the device.
        output: NDBuffer[type, rank] - The output tensor allocated on the device.
    """
    constrained[rank > 0, "Input rank must be positive"]()
    alias K = 1
    alias topk_kernel = _topk_gpu[
        out_idx_type=output_type, sampling=False, largest=largest
    ]
    var orig_in_shape: IndexList[rank] = input.get_shape()
    var orig_out_shape: IndexList[rank] = output.get_shape()
    var N = orig_in_shape[rank - 1]

    # heuristic to set block size
    var block_size_: Int
    if input.size() <= 1024 * 64 * 3:
        block_size_ = 256
    elif input.size() <= 32000 * 256:
        block_size_ = 512
    else:
        block_size_ = 1024

    # Reshape the input to a 2D tensor (required by the Top-K kernel)
    var internal_bs: Int
    alias internal_rank = 2
    var internal_in_shape: IndexList[internal_rank]
    var internal_out_shape: IndexList[internal_rank]
    var internal_input: NDBuffer[type, internal_rank]
    var internal_output: NDBuffer[output_type, internal_rank]

    @parameter
    if rank == 1:
        internal_bs = 1
        internal_in_shape = IndexList[internal_rank](1, input.size())
        internal_out_shape = IndexList[internal_rank](1, K)

        internal_input = reshape(input, internal_in_shape)
        internal_output = reshape(output, internal_out_shape)
    elif rank == internal_rank:
        internal_bs = orig_in_shape[0]
        internal_in_shape = rebind[IndexList[internal_rank]](orig_in_shape)
        internal_out_shape = rebind[IndexList[internal_rank]](orig_out_shape)

        internal_input = rebind[NDBuffer[type, internal_rank]](
            input
        )  # Already in correct shape
        internal_output = rebind[NDBuffer[output_type, internal_rank]](output)
    else:  # rank > 2
        var _last_dim = orig_in_shape[rank - 1]
        internal_bs = int(orig_in_shape.flattened_length() / _last_dim)
        internal_in_shape = IndexList[internal_rank](internal_bs, _last_dim)
        internal_input = reshape(input, internal_in_shape)

        internal_out_shape = IndexList[internal_rank](internal_bs, K)
        internal_output = reshape(output, internal_out_shape)

    var num_blocks_per_input_: Int = min(ceildiv(N, block_size_), 8)
    var internal_cache_shape = DimList(internal_bs, num_blocks_per_input_ * K)

    var internal_vals_buf = ctx.create_buffer[type](
        int(internal_cache_shape.product())
    )
    var device_local_topk_vals = NDBuffer[type, internal_rank](
        internal_vals_buf.ptr, internal_cache_shape
    )

    var internal_idxs_buf = ctx.create_buffer[DType.index](
        int(internal_cache_shape.product())
    )
    var device_local_topk_idxs = NDBuffer[DType.index, internal_rank](
        internal_idxs_buf.ptr, internal_cache_shape
    )

    var out_vals_buf = ctx.create_buffer[type](
        internal_out_shape.flattened_length()
    )
    var device_out_vals = NDBuffer[type, internal_rank](
        out_vals_buf.ptr, internal_out_shape
    )

    topk_kernel(
        ctx,
        K,
        internal_input,
        device_local_topk_vals,
        device_local_topk_idxs,
        device_out_vals,
        internal_output,
        block_size=block_size_,
        num_blocks_per_input=num_blocks_per_input_,
    )

    _ = internal_vals_buf^
    _ = internal_idxs_buf^
    _ = out_vals_buf^


fn argmax_gpu[
    type: DType, output_type: DType, rank: Int
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[output_type, rank],
) raises:
    argmaxmin_gpu[largest=True](ctx, input, output)


fn argmin_gpu[
    type: DType, output_type: DType, rank: Int
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[output_type, rank],
) raises:
    argmaxmin_gpu[largest=False](ctx, input, output)
