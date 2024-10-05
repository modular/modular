# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s


from os import abort
from math import iota, ceildiv
from memory import UnsafePointer

from buffer import NDBuffer
from nn.topk_gpu import _topk_gpu
from buffer.dimlist import DimList

from internal_utils import HostNDBuffer, DeviceNDBuffer
from utils import IndexList
from random import random_float64
from gpu.host import DeviceContext

alias idx_t = DType.index  # bad practice (matches the idx_t in the topk_gpu kernel)


fn test_case[
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (inout NDBuffer[type, rank]) capturing [
        _
    ] -> None,
    sampling: Bool = True,
    rank: Int = 1,  # TODO (KERN-1016) support higher rank tensors
](
    ctx: DeviceContext,
    N: Int,
    K: Int,
    axis: Int = -1,  # TODO (low prio) need to add support for axis != -1
    block_size: Int = 256,
    # input_shape: IndexList[rank] = IndexList[rank](N), # TODO
) raises:
    # Instantiate data in host memory
    var in_buffer = HostNDBuffer[type, 1](DimList(N))
    var topk_vals = HostNDBuffer[type, 1](DimList(K))
    var topk_idxs = HostNDBuffer[idx_t, 1](DimList(K))

    # Fill the buffer with consecutive values
    fill_fn[1, type](in_buffer.tensor)
    print("Input buffer: ", in_buffer.tensor)

    # Run the Top-K kernel
    alias topk_kernel = _topk_gpu[type, sampling= (sampling)]

    # Move data to device
    var device_in = DeviceNDBuffer[type, 1](DimList(N), ctx=ctx)
    var device_out_vals = DeviceNDBuffer[type, 1](DimList(K), ctx=ctx)
    var device_out_idxs = DeviceNDBuffer[idx_t, 1](DimList(K), ctx=ctx)

    var num_blocks_stg1 = ceildiv(in_buffer.tensor.num_elements(), block_size)
    var device_local_topk_vals = DeviceNDBuffer[type, 1](
        DimList(num_blocks_stg1 * K), ctx=ctx
    )
    var device_local_topk_idxs = DeviceNDBuffer[idx_t, 1](
        DimList(num_blocks_stg1 * K), ctx=ctx
    )

    ctx.enqueue_copy_to_device(device_in.buffer, in_buffer.tensor.data)
    ctx.synchronize()

    topk_kernel(
        ctx,
        K,
        device_in.tensor,
        device_local_topk_vals.tensor,
        device_local_topk_idxs.tensor,
        device_out_vals.tensor,
        device_out_idxs.tensor,
        block_size=block_size,
    )
    ctx.synchronize()

    # Copy results back to host
    ctx.enqueue_copy_from_device(topk_vals.tensor.data, device_out_vals.buffer)
    ctx.enqueue_copy_from_device(topk_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    var _msg1: String = "Probability of chosen logit: " if sampling else "Top-K values: "
    var _msg2 = "Sample token index: " if sampling else "Top K indices: "

    @parameter
    if sampling:
        # For some reason printing the tensor directly with sampling leads to
        # a segfault in the print function so this is the current workaround
        print(_msg2, topk_idxs.tensor.data[0])
    else:
        print(_msg1, topk_vals.tensor)
        print(_msg2, topk_idxs.tensor)

    _ = topk_vals
    _ = topk_idxs
    _ = in_buffer
    _ = device_in
    _ = device_local_topk_vals
    _ = device_local_topk_idxs
    _ = device_out_vals
    _ = device_out_idxs


@parameter
fn fill_random[
    rank: Int, dtype: DType
](inout buffer: NDBuffer[dtype, rank],):
    alias min_val = 0.0
    alias max_val = 100000.0
    var total_elements = buffer.num_elements()
    for i in range(total_elements):
        var random_value = random_float64(min_val, max_val)
        buffer.data[i] = random_value.cast[dtype]()


@parameter
fn fill_iota[rank: Int, type: DType](inout buf: NDBuffer[type, rank]):
    iota(buf.data, buf.get_shape().flattened_length())


fn main() raises:
    with DeviceContext() as ctx:
        var N: Int = 4096 * 4
        var K: Int = 5
        var block_size = 256
        alias type = DType.float32

        print("==== Running Top-K without sampling")
        test_case[
            type,
            fill_iota,
            sampling=False,
        ](ctx, N, K, block_size=block_size)

        print("==== Running Top-K sampling")
        test_case[
            type,
            fill_iota,
            sampling=True,
        ](ctx, N, K, block_size=block_size)
