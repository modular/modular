# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s

from testing import *

from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)


fn add_constant_fn(
    out: DTypePointer[DType.float32],
    input: DTypePointer[DType.float32],
    constant: Float32,
    len: Int,
):
    let tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = input[tid] + constant


def run_add_constant():
    alias length = 1024
    let stream = Stream()

    let in_host = Pointer[Float32].alloc(length)
    let out_host = Pointer[Float32].alloc(length)

    for i in range(length):
        in_host[i] = i

    let in_device = _malloc[Float32](length)
    let out_device = _malloc[Float32](length)

    _copy_host_to_device(in_device, in_host, length)

    let func = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Float32,
            Int,
        ) -> None, add_constant_fn
    ]()

    let block_dim = 32
    alias constant = FloatLiteral(33)

    func(
        stream,
        (length // block_dim),
        (block_dim),
        out_device,
        in_device,
        constant,
        length,
    )

    _copy_device_to_host(out_host, out_device, length)

    for i in range(10):
        assert_equal(out_host[i], i + constant)

    _free(in_device)
    _free(out_device)

    in_host.free()
    out_host.free()

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_add_constant()
    except e:
        print("CUDA_ERROR:", e)
