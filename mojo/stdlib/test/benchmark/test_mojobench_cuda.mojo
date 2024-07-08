# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from benchmark import Bench, Bencher, BenchId
from gpu.host import DeviceContext, Dim


fn empty_kernel():
    pass


fn test_bench_empty_async(inout m: Bench, ctx: DeviceContext) raises:
    var func = ctx.compile_function[empty_kernel]()

    @parameter
    @always_inline
    fn bench_empty_async(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(func, grid_dim=Dim(1), block_dim=Dim(1))

        b.iter_custom[launch](ctx)

    m.bench_function[bench_empty_async](BenchId("bench_empty_async"))


fn test_bench_empty_sync(inout m: Bench, ctx: DeviceContext) raises:
    var func = ctx.compile_function[empty_kernel]()

    @parameter
    @always_inline
    fn bench_empty_sync(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn launch() raises:
            ctx.enqueue_function(func, grid_dim=Dim(1), block_dim=Dim(1))

        b.iter[launch]()

    m.bench_function[bench_empty_sync](BenchId("bench_empty_sync"))


def main():
    var m = Bench()
    with DeviceContext() as ctx:
        test_bench_empty_async(m, ctx)
        test_bench_empty_sync(m, ctx)
    m.dump_report()


# CHECK: bench_empty_async
# CHECK-SAME: bench_empty_sync
