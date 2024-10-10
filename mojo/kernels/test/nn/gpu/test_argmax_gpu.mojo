# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from algorithm.reduction import argmax
from nn.argmax_gpu import argmax_gpu
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host.device_context import DeviceContext
from testing import assert_equal

from random import random_float64
from internal_utils import HostNDBuffer, DeviceNDBuffer


fn test_argmax_gpu[
    type: DType,
    fill_fn: fn[rank: Int, type: DType] (inout NDBuffer[type, rank]) capturing [
        _
    ] -> None,
    rank: Int = 2,
](
    ctx: DeviceContext, N: Int, batch_size: Int = 12, num_batches: Int = 6
) raises:
    # Instantiate data in host memory
    var in_shape: DimList
    var out_shape: DimList

    @parameter
    if rank == 1:
        out_shape = DimList(1)
        in_shape = DimList(N)
    elif rank == 2:
        out_shape = DimList(batch_size, 1)
        in_shape = DimList(batch_size, N)
    elif rank == 3:
        out_shape = DimList(num_batches, batch_size, 1)
        in_shape = DimList(num_batches, batch_size, N)
    else:
        raise Error("Test case doesn't support rank above 3 (just add it)")

    var in_buffer = HostNDBuffer[type, rank](in_shape)
    var out_idxs = HostNDBuffer[DType.index, rank](out_shape)

    # Fill the buffer with consecutive values
    fill_fn(in_buffer.tensor)

    var device_in = DeviceNDBuffer[type, rank](in_shape, ctx=ctx)
    var device_out_idxs = DeviceNDBuffer[DType.index, rank](out_shape, ctx=ctx)

    ctx.enqueue_copy_to_device(device_in.buffer, in_buffer.tensor.data)

    argmax_gpu(ctx, device_in.tensor, device_out_idxs.tensor)

    ctx.enqueue_copy_from_device(out_idxs.tensor.data, device_out_idxs.buffer)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var out_idxs_cpu = HostNDBuffer[DType.int64, rank](out_shape)
    argmax(in_buffer.tensor, rank - 1, out_idxs_cpu.tensor)

    for i in range(out_idxs_cpu.tensor.num_elements()):
        assert_equal(
            out_idxs.tensor.data[i],
            out_idxs_cpu.tensor.data[i].cast[DType.index](),
        )

    _ = device_in
    _ = device_out_idxs


def main():
    @parameter
    fn fill_random[
        rank: Int, dtype: DType
    ](inout buffer: NDBuffer[dtype, rank],):
        alias min_val = -1e9
        alias max_val = 1e9
        var total_elements = buffer.num_elements()
        for i in range(total_elements):
            var random_value = random_float64(min_val, max_val)
            buffer.data[i] = random_value.cast[dtype]()

    with DeviceContext() as ctx:  # argmax tests
        test_argmax_gpu[DType.float32, fill_random, rank=1](ctx, N=102_400)
        test_argmax_gpu[DType.float32, fill_random, rank=2](
            ctx, N=16_384, batch_size=32
        )
        test_argmax_gpu[DType.float32, fill_random, rank=3](
            ctx, N=1024, batch_size=12, num_batches=10
        )
