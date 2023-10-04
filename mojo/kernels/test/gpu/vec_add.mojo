# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from gpu import *
from gpu.nvidia_host import (
    Function,
    Context,
    Dim,
    _malloc,
    _free,
    _copy_host_to_device,
    _copy_device_to_host,
    synchronize,
)
from sys.param_env import env_get_string
from pathlib import Path
from sys.info import triple_is_nvidia_cuda


@export("vec_add")
fn vec_add(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    @parameter
    if not triple_is_nvidia_cuda():
        return

    let tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out.store(tid, in0.load(tid) + in1.load(tid))


# CHECK-LABEL: run_vec_add
fn run_vec_add() raises:
    print("== run_vec_add")

    alias length = 1024

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

    let func = Function[
        # fmt: off
      fn (DTypePointer[DType.float32],
          DTypePointer[DType.float32],
          DTypePointer[DType.float32],
          Int) -> None,
        # fmt: on
        vec_add
    ]("vec_add")

    let block_dim = 32
    func(
        (length // block_dim),
        (block_dim),
        in0_device,
        in1_device,
        out_device,
        length,
    )
    synchronize()

    _copy_device_to_host(out_host, out_device, length)

    # CHECK: at index 0 the value is 2.0
    # CHECK: at index 1 the value is 3.0
    # CHECK: at index 2 the value is 4.0
    # CHECK: at index 3 the value is 5.0
    # CHECK: at index 4 the value is 6.0
    # CHECK: at index 5 the value is 7.0
    # CHECK: at index 6 the value is 8.0
    # CHECK: at index 7 the value is 9.0
    # CHECK: at index 8 the value is 10.0
    # CHECK: at index 9 the value is 11.0
    for i in range(10):
        print("at index", i, "the value is", out_host.load(i))

    _free(in0_device)
    _free(in1_device)
    _free(out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = func ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_vec_add()
    except e:
        print("CUDA_ERROR:", e)
