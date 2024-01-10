# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from pathlib import Path
from sys.info import triple_is_nvidia_cuda
from sys.param_env import env_get_string

from gpu import *
from gpu.host import Context, Dim, Function, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)


fn vec_func[
    op: fn (Float32, Float32) capturing -> Float32
](
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    let tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out.store(tid, op(in0.load(tid), in1.load(tid)))


# CHECK-LABEL: run_binary_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
fn run_binary_add(capture: Float32) raises:
    print("== run_binary_add")

    alias length = 1024

    let stream = Stream()

    let in0_host = Pointer[Float32].alloc(length)
    let in1_host = Pointer[Float32].alloc(length)
    let out_host = Pointer[Float32].alloc(length)

    for i in range(length):
        in0_host.store(i, i)
        in1_host.store(i, 2)

    let in0_device = _malloc[Float32](length)
    let in1_device = _malloc[Float32](length)
    let out_device = _malloc[Float32](length)

    _copy_host_to_device(in0_device, in0_host, length)
    _copy_host_to_device(in1_device, in1_host, length)

    @parameter
    fn add(lhs: Float32, rhs: Float32) -> Float32:
        return capture + lhs + rhs

    let func = Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
            Int,
        ) capturing -> None, vec_func[add]
    ]()

    let block_dim = 32
    func(
        stream,
        (length // block_dim),
        (block_dim),
        in0_device,
        in1_device,
        out_device,
        length,
    )
    # CHECK: number of captures: 1
    print("number of captures:", func._impl.num_captures)
    synchronize()

    _copy_device_to_host(out_host, out_device, length)

    # CHECK: at index 0 the value is 4.5
    # CHECK: at index 1 the value is 5.5
    # CHECK: at index 2 the value is 6.5
    # CHECK: at index 3 the value is 7.5
    # CHECK: at index 4 the value is 8.5
    # CHECK: at index 5 the value is 9.5
    # CHECK: at index 6 the value is 10.5
    # CHECK: at index 7 the value is 11.5
    # CHECK: at index 8 the value is 12.5
    # CHECK: at index 9 the value is 13.5
    for i in range(10):
        print("at index", i, "the value is", out_host.load(i))

    _free(in0_device)
    _free(in1_device)
    _free(out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = func ^
    _ = stream ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_binary_add(2.5)
    except e:
        print("CUDA_ERROR:", e)
