# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host import CacheConfig, DeviceContext
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from memory import UnsafePointer
from testing import *


fn gpu_kernel(buff: UnsafePointer[Int64]):
    var idx = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    buff[idx] = idx


def main():
    with DeviceContext() as ctx:
        var buff_host_ptr = UnsafePointer[Int64].alloc(16)
        var buff_dev = ctx.create_buffer[DType.int64](16)

        for i in range(16):
            buff_host_ptr[i] = 0

        ctx.enqueue_copy_to_device(buff_dev, buff_host_ptr)

        var kernel = ctx.compile_function[gpu_kernel](
            cache_config=CacheConfig.PREFER_SHARED
        )
        ctx.enqueue_function(kernel, buff_dev, block_dim=(4), grid_dim=(4))
        ctx.enqueue_copy_from_device(buff_host_ptr, buff_dev)

        for i in range(16):
            assert_equal(
                buff_host_ptr[i], i, msg="invalid value at index=" + str(i)
            )

        _ = buff_dev^
        buff_host_ptr.free()
