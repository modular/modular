# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path

from benchmark._cuda import run as benchmark_run
from gpu import *
from gpu.host import Context, CudaInstance, Device, Dim, Function, Stream


fn vec_func(
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    var tid: UInt = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len.value:
        return
    out[tid] = in0[tid] + in1[tid]


# CHECK-LABEL: run_vec_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
fn run_vec_add(ctx: Context) raises:
    print("== run_vec_add")

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

    var func = Function[vec_func](ctx)

    var block_dim = 32

    @always_inline
    @__copy_capture(block_dim, in0_device, in1_device, out_device, func)
    @parameter
    fn run_func(stream: Stream) raises:
        func(
            in0_device,
            in1_device,
            out_device,
            length,
            grid_dim=(length // block_dim),
            block_dim=(block_dim),
            stream=stream,
        )

    var report = benchmark_run[run_func](max_iters=1000)
    # CHECK: Benchmark Report (s)
    report.print()

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
        print("at index", i, "the value is", out_host[i])

    ctx.free(in0_device)
    ctx.free(in1_device)
    ctx.free(out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = func^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with CudaInstance() as instance:
            with Context(Device(instance)) as ctx:
                run_vec_add(ctx)
    except e:
        print("CUDA_ERROR:", e)
