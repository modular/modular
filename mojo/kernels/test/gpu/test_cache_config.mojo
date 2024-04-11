# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from gpu.host import CacheConfig, Context, Function, synchronize
from gpu.host.memory import _malloc_managed
from gpu.id import BlockDim, BlockIdx, ThreadIdx
from testing import *


fn gpu_kernel(buff: DTypePointer[DType.int64]):
    var idx = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    buff[idx] = idx


def main():
    with Context() as ctx:
        var buff = _malloc_managed[DType.int64](16)
        for i in range(16):
            buff[i] = 0
        var kernel = Function[__type_of(gpu_kernel), gpu_kernel](
            cache_config=CacheConfig.PREFER_SHARED
        )
        kernel(buff, block_dim=(4), grid_dim=(4))
        synchronize()

        for i in range(16):
            assert_equal(buff[i], i, msg="invalid value at index=" + str(i))
