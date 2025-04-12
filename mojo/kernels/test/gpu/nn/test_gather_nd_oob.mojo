# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1738
# UNSUPPORTED: AMD-GPU
# RUN: not --crash %mojo %s 2>&1 | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from memory import stack_allocation
from nn.gather_scatter import _gather_nd_impl, gather_nd_shape
from testing import assert_equal, assert_true

from utils import IndexList


def execute_gather_nd_test[
    data_type: DType, //,
    batch_dims: Int,
](
    data_host: HostNDBuffer[data_type, **_],
    indices_host: HostNDBuffer,
    ctx: DeviceContext,
):
    # create device-side buffers and copy data to them
    var data_device = DeviceNDBuffer[
        data_host.type, data_host.rank, data_host.shape
    ](
        data_host.tensor.get_shape(),
        ctx=ctx,
    )
    var indices_device = DeviceNDBuffer[
        indices_host.type, indices_host.rank, indices_host.shape
    ](
        indices_host.tensor.get_shape(),
        ctx=ctx,
    )
    alias output_rank = 1

    var output_shape = gather_nd_shape[
        data_host.rank,
        indices_host.rank,
        output_rank,
        data_type,
        indices_host.type,
        batch_dims,
    ](
        data_host.tensor.make_dims_unknown(),
        indices_host.tensor.make_dims_unknown(),
    )

    var actual_output_device = DeviceNDBuffer[
        data_host.type,
        output_shape.size,
    ](
        output_shape,
        ctx=ctx,
    )
    ctx.enqueue_copy(data_device.buffer, data_host.tensor.data)
    ctx.enqueue_copy(indices_device.buffer, indices_host.tensor.data)

    # execute the kernel
    _gather_nd_impl[batch_dims, target="gpu"](
        data_device.tensor.make_dims_unknown(),
        indices_device.tensor.make_dims_unknown(),
        actual_output_device.tensor.make_dims_unknown(),
        ctx,
    )

    _ = data_device^
    _ = indices_device^
    _ = actual_output_device^


fn test_gather_nd_oob(ctx: DeviceContext) raises:
    # Example 1
    alias batch_dims = 0
    alias data_rank = 2
    alias data_type = DType.int32
    var data = HostNDBuffer[data_type, data_rank, DimList(2, 2)](
        IndexList[data_rank](2, 2)
    )

    data.tensor[IndexList[data_rank](0, 0)] = 0
    data.tensor[IndexList[data_rank](0, 1)] = 1
    data.tensor[IndexList[data_rank](1, 0)] = 2
    data.tensor[IndexList[data_rank](1, 1)] = 3

    alias indices_rank = 2
    var indices = HostNDBuffer[DType.int64, indices_rank, DimList(2, 2)](
        IndexList[indices_rank](2, 2)
    )

    indices.tensor[IndexList[indices_rank](0, 0)] = 0
    indices.tensor[IndexList[indices_rank](0, 1)] = 0
    indices.tensor[IndexList[indices_rank](1, 0)] = 1
    indices.tensor[IndexList[indices_rank](1, 1)] = 100  # wildly out of bounds

    execute_gather_nd_test[batch_dims](data, indices, ctx)


def main():
    with DeviceContext() as ctx:
        # CHECK: {{.*}}data index out of bounds{{.*}}
        test_gather_nd_oob(ctx)
