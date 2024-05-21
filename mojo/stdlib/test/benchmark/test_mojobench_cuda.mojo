# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from benchmark import Bench, Bencher, BenchId
from benchmark._cuda import time_async_cuda_kernel
from gpu.host import Context, Dim, Function, Stream


fn empty_kernel():
    pass


@parameter
fn bench_empty_async(inout b: Bencher) raises:
    var func = Function[empty_kernel]()
    var stream = Stream()

    @parameter
    fn launch(stream: Stream) raises:
        func(grid_dim=Dim(1), block_dim=Dim(1), stream=stream)

    b.iter_custom[time_async_cuda_kernel[launch]]()


@parameter
fn bench_empty_sync(inout b: Bencher) raises:
    var func = Function[empty_kernel]()

    @parameter
    fn launch() raises:
        func(grid_dim=Dim(1), block_dim=Dim(1))

    b.iter[launch]()


def main():
    with Context() as ctx:
        var m = Bench()
        m.bench_function[bench_empty_async](BenchId("bench_empty_async"))
        m.bench_function[bench_empty_sync](BenchId("bench_empty_sync"))
        m.dump_report()


# CHECK: bench_empty_async
# CHECK-SAME: bench_empty_sync
