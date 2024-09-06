# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from pathlib import Path
from builtin._closure import __ownership_keepalive

from gpu import *
from gpu.host import Device, Dim, Function, Stream
from gpu.host.device_context import DeviceContext

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from testing import assert_equal
from sys import env_get_int


fn vec_func(
    in0: UnsafePointer[Float32],
    in1: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    len: Int,
):
    var tid: UInt = ThreadIdx.x() + BlockDim.x() * BlockIdx.x()
    if int(tid) >= len:
        return
    out[tid] = in0[tid] + in1[tid]


@no_inline
fn bench_vec_add(
    inout b: Bench, *, block_dim: Int, length: Int, context: DeviceContext
) raises:
    alias dtype = DType.float32
    var in0_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var in1_host = UnsafePointer[Scalar[dtype]].alloc(length)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(length)

    for i in range(length):
        in0_host[i] = i
        in1_host[i] = 2

    var in0_device = context.create_buffer[dtype](length)
    var in1_device = context.create_buffer[dtype](length)
    var out_device = context.create_buffer[dtype](length)

    context.enqueue_copy_to_device(in0_device, in0_host)
    context.enqueue_copy_to_device(in1_device, in1_host)

    var func = context.compile_function[vec_func]()

    @always_inline
    @parameter
    fn run_func() raises:
        context.enqueue_function(
            func,
            in0_device,
            in1_device,
            out_device,
            length,
            grid_dim=(length // block_dim),
            block_dim=(block_dim),
        )

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            run_func()

        b.iter_custom[kernel_launch](context)

    b.bench_function[bench_func](
        BenchId("vec_add", input_id="block_dim=" + str(block_dim)),
        ThroughputMeasure(BenchMetric.flops, length),
    )
    context.synchronize()
    context.enqueue_copy_from_device(out_host, out_device)

    for i in range(length):
        assert_equal(i + 2, out_host[i])

    __ownership_keepalive(in0_device, in1_device, out_device, func)

    in0_host.free()
    in1_host.free()
    out_host.free()


# CHECK-NOT: CUDA_ERROR
def main():
    # TODO: expand to all the params
    alias phony = env_get_int["phony", 1]()
    constrained[phony == 1]()

    var b = Bench()

    try:
        with DeviceContext() as ctx:
            for block_dim in List[Int](32, 64, 128, 256, 512, 1024):
                bench_vec_add(
                    b, block_dim=block_dim[], length=32 * 1024, context=ctx
                )
    except e:
        print("CUDA_ERROR:", e)

    b.dump_report()
