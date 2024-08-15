# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: cuda
# RUN: %mojo %s %t1
# RUN: cat %t1 | FileCheck %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/cuda-test-mojo-kernel
# RUN: mojo build %s -o %t/cuda-test-mojo-kernel
# RUN: %t/cuda-test-mojo-kernel %t2
# RUN: cat %t2 | FileCheck %s

# CHECK: .target sm_80

from sys import stderr

from max.driver import (
    cpu_device,
    cuda_device,
    Tensor,
    UnsafeTensorSlice,
    Device,
    DeviceTensor,
)
from max.driver import cuda
from testing import assert_equal
from max.tensor import TensorShape
from gpu.id import ThreadIdx, BlockDim, BlockIdx
from gpu.host import Dim
from sys import env_get_string, argv
from pathlib import Path


fn vec_add[
    type: DType, rank: Int
](
    in0: UnsafeTensorSlice[type, rank],
    in1: UnsafeTensorSlice[type, rank],
    out: UnsafeTensorSlice[type, rank],
):
    var row = ThreadIdx.x()
    var col = ThreadIdx.y()
    out[row, col] = in0[row, col] + in1[row, col]


def fill(cuda_dev: Device, shape: TensorShape, val: Float32) -> DeviceTensor:
    var host = Tensor[DType.float32, 2](shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            host[i, j] = val
    return host^.to_device_tensor().copy_to(cuda_dev)


def test_vec_add():
    cuda_dev = cuda_device()
    cpu_dev = cpu_device()
    shape = TensorShape(10, 10)
    alias type = DType.float32
    in0 = fill(cuda_dev, shape, 1).to_tensor[type, 2]()
    in1 = fill(cuda_dev, shape, 2).to_tensor[type, 2]()
    out = fill(cuda_dev, shape, 0).to_tensor[type, 2]()

    kernel = cuda.compile[vec_add[type, 2], target_arch="sm_80"](
        cuda_dev, debug=True, max_registers=128, dump_ptx=Path(argv()[1])
    )
    kernel(
        cuda_dev,
        in0.unsafe_slice(),
        in1.unsafe_slice(),
        out.unsafe_slice(),
        block_dim=Dim(shape[0], shape[1]),
        grid_dim=Dim(1, 1),
    )

    out_host = (
        out.to_device_tensor().copy_to(cpu_dev).to_tensor[type, 2]()
    )  # copy blocks until the kernel is finished

    # lifetime extension required otherwise in0's last use is before the call to
    # kernel.__call__, even though the destructor is enqueued asynchronously on the stream
    _ = in0
    _ = in1

    for i in range(shape[0]):
        for j in range(shape[1]):
            assert_equal(out_host[i, j], 3)


def main():
    test_vec_add()
