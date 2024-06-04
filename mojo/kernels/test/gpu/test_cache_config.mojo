# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from gpu.host import CacheConfig, Context, CudaInstance, Device, Function
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from testing import *


fn gpu_kernel(buff: DTypePointer[DType.int64]):
    var idx = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    buff[idx] = idx


def main():
    with CudaInstance() as instance:
        with Context(Device(instance)) as ctx:
            var buff = ctx.malloc_managed[DType.int64](16)
            for i in range(16):
                buff[i] = 0
            var kernel = Function[gpu_kernel](
                ctx, cache_config=CacheConfig.PREFER_SHARED
            )
            kernel(buff, block_dim=(4), grid_dim=(4))
            ctx.synchronize()

            for i in range(16):
                assert_equal(buff[i], i, msg="invalid value at index=" + str(i))
