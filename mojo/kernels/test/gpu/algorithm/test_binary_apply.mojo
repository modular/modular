# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from sys.info import triple_is_nvidia_cuda

from gpu import *
from gpu.host import (
    CudaInstance,
    Device,
    Context,
    Dim,
    Function,
    Stream,
    synchronize,
)


fn vec_func[
    op: fn (Float32, Float32) capturing -> Float32
](
    in0: DTypePointer[DType.float32],
    in1: DTypePointer[DType.float32],
    out: DTypePointer[DType.float32],
    len: Int,
):
    var tid = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if tid >= len:
        return
    out[tid] = op(in0[tid], in1[tid])


# CHECK-LABEL: run_binary_add
# COM: Force the capture to be captured instead of inlined away.
@no_inline
fn run_binary_add(ctx: Context, capture: Float32) raises:
    print("== run_binary_add")

    alias length = 1024

    var stream = Stream(ctx)

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
    fn add(lhs: Float32, rhs: Float32) -> Float32:
        return capture + lhs + rhs

    var func = Function[vec_func[add]](ctx)

    var block_dim = 32
    func(
        in0_device,
        in1_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
        stream=stream,
    )
    # CHECK: number of captures: 1
    print("number of captures:", func._impl.num_captures)
    ctx.synchronize()

    ctx.copy_device_to_host(out_host, out_device, length)

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

    ctx.free(in0_device)
    ctx.free(in1_device)
    ctx.free(out_device)

    in0_host.free()
    in1_host.free()
    out_host.free()

    _ = func^
    _ = stream^


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with CudaInstance() as instance:
            with Context(Device(instance)) as ctx:
                run_binary_add(ctx, 2.5)
    except e:
        print("CUDA_ERROR:", e)
