# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug -debug-level full %s | FileCheck %s


from pathlib import Path, _dir_of_current_file

from gpu.host import DeviceContext, Function, Module, Stream


# CHECK-LABEL: run_cuda_mem_ops
fn run_cuda_mem_ops(ctx: DeviceContext) raises:
    print("== run_cuda_mem_ops")

    alias length = 20

    var host_mem = UnsafePointer[Scalar[DType.index]].alloc(length)
    var device_mem = ctx.create_buffer[DType.index](length)

    ctx.enqueue_copy_to_device(device_mem, host_mem)
    ctx.enqueue_copy_from_device(host_mem, device_mem)

    _ = device_mem
    host_mem.free()


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    pass


# CHECK-LABEL: run_vec_add
fn run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var module = Module(
        ctx.cuda_context, (_dir_of_current_file() / "vec_add.ptx")
    )

    var func = Function[vec_func](module, "vec_add", cuda_dll=module.cuda_dll)

    alias length = 1024

    var in0_host = UnsafePointer[Float32].alloc(length)
    var in1_host = UnsafePointer[Float32].alloc(length)
    var out_host = UnsafePointer[Float32].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = ctx.create_buffer[DType.float32](length)
    var in1_device = ctx.create_buffer[DType.float32](length)
    var out_device = ctx.create_buffer[DType.float32](length)

    ctx.enqueue_copy_to_device(in0_device, in0_host)
    ctx.enqueue_copy_to_device(in1_device, in1_host)

    @parameter
    @always_inline
    fn populate(ptr: UnsafePointer[NoneType]):
        return

    var block_dim = 32
    func(
        out_device,
        in0_device,
        in1_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
        stream=ctx.cuda_stream,
    )

    ctx.enqueue_copy_from_device(out_host, out_device)

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
        print("at index", i, "the value is", out_host[i])

    _ = in0_device
    _ = in1_device
    _ = out_device

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = module^
    _ = func


def main():
    with DeviceContext() as ctx:
        run_cuda_mem_ops(ctx)
        run_vec_add(ctx)
