# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo -debug-level full %s | FileCheck %s


from pathlib import Path, _dir_of_current_file

from gpu.host import Context, CudaInstance, Device, Function, Module, Stream
from memory.unsafe import Pointer


# CHECK-LABEL: run_cuda_mem_ops
fn run_cuda_mem_ops(ctx: Context) raises:
    print("== run_cuda_mem_ops")

    alias length = 20

    var host_mem = Pointer[Int].alloc(length)
    var device_mem = ctx.malloc[Int](length)

    ctx.copy_host_to_device(device_mem, host_mem, length)
    ctx.copy_device_to_host(host_mem, device_mem, length)

    ctx.free(device_mem)
    host_mem.free()


fn vec_func(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    pass


# CHECK-LABEL: run_vec_add
fn run_vec_add(ctx: Context) raises:
    print("== run_vec_add")

    var module = Module(ctx, (_dir_of_current_file() / "vec_add.ptx"))

    var func = Function[vec_func](module, "vec_add")

    alias length = 1024

    var in0_host = Pointer[Float32].alloc(length)
    var in1_host = Pointer[Float32].alloc(length)
    var out_host = Pointer[Float32].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = ctx.malloc[Float32](length)
    var in1_device = ctx.malloc[Float32](length)
    var out_device = ctx.malloc[Float32](length)

    ctx.copy_host_to_device(in0_device, in0_host, length)
    ctx.copy_host_to_device(in1_device, in1_host, length)

    @parameter
    @always_inline
    fn populate(ptr: Pointer[NoneType]):
        return

    var block_dim = 32
    func(
        out_device,
        in0_device,
        in1_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
        stream=Stream(),
    )
    ctx.synchronize()

    ctx.copy_device_to_host(out_host, out_device, length)

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

    ctx.free(in0_device)
    ctx.free(in1_device)
    ctx.free(out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = module^
    _ = func


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with CudaInstance() as instance:
            with Context(Device(instance)) as ctx:
                run_cuda_mem_ops(ctx)
                run_vec_add(ctx)
    except e:
        print("CUDA_ERROR:", e)
