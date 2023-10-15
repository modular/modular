# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: nvptx_backend
# REQUIRES: has_cuda_device
# RUN: %mojo -debug-level full -D CURRENT_DIR=%S %s | FileCheck %s


from pathlib import Path
from sys.param_env import env_get_string

from gpu.host import Context, ModuleHandle, Stream, synchronize
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from memory.unsafe import Pointer

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()


# CHECK-LABEL: run_cuda_mem_ops
fn run_cuda_mem_ops() raises:
    print("== run_cuda_mem_ops")

    alias length = 20

    let host_mem = Pointer[Int].alloc(length)
    let device_mem = _malloc[Int](length)

    _copy_host_to_device(device_mem, host_mem, length)
    _copy_device_to_host(host_mem, device_mem, length)

    _free(device_mem)
    host_mem.free()


# CHECK-LABEL: run_vec_add
fn run_vec_add() raises:
    print("== run_vec_add")

    let module = ModuleHandle((Path(CURRENT_DIR) / "vec_add.ptx"))

    let func = module.load("vec_add")

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

    @parameter
    @always_inline
    fn populate(ptr: Pointer[Pointer[NoneType]]):
        return

    let block_dim = 32
    func._call_impl[0, populate](
        (length // block_dim),
        (block_dim),
        out_device,
        in0_device,
        in1_device,
        length,
        stream=Stream[is_borrowed=False](),
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

    _ = module ^
    _ = func ^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with Context() as ctx:
            run_cuda_mem_ops()
            run_vec_add()
    except e:
        print("CUDA_ERROR:", e)
