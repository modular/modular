# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from benchmark import Bench, Bencher, BenchId
from gpu.host import Context, Dim, Function, Stream
from layout import *


@parameter
fn bench_empty_launch(inout b: Bencher) raises:
    fn empty_kernel():
        pass

    var func = Function[empty_kernel]()
    var stream = Stream()

    @parameter
    fn launch() raises:
        func(grid_dim=Dim(1), block_dim=Dim(1), stream=stream)

    b.iter[launch]()
    stream.synchronize()
    _ = func^


@parameter
fn bench_empty_launch_many_params(inout b: Bencher) raises:
    fn empty_kernel[
        layout_1: Layout,
        layout_2: Layout,
        layout_3: Layout,
        layout_4: Layout,
        layout_5: Layout,
        layout_6: Layout,
        layout_7: Layout,
        layout_8: Layout,
        layout_9: Layout,
    ]():
        pass

    alias func_alias = empty_kernel[
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
        Layout(IntTuple(1, 2), IntTuple(3, 3)),
    ]
    var func = Function[func_alias]()
    var stream = Stream()

    @parameter
    fn launch() raises:
        func(grid_dim=Dim(1), block_dim=Dim(1), stream=stream)

    b.iter[launch]()
    stream.synchronize()
    _ = func^


def main():
    with Context() as ctx:
        var m = Bench()
        m.bench_function[bench_empty_launch](BenchId("bench_empty_launch"))
        m.bench_function[bench_empty_launch_many_params](
            BenchId("bench_empty_launch_many_params")
        )
        m.dump_report()
