# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s

from benchmark import Bench, Bencher, BenchId
from gpu.host import DeviceContext, Dim
from layout import *


fn empty_kernel():
    pass


fn empty_kernel_many_params[
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


fn bench_empty_launch_caller(inout m: Bench, ctx: DeviceContext) raises:
    var func = ctx.compile_function[empty_kernel]()

    @parameter
    @always_inline
    fn bench_empty_launch(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(func, grid_dim=Dim(1), block_dim=Dim(1))

        b.iter_custom[launch](ctx)

    m.bench_function[bench_empty_launch](BenchId("bench_empty_launch"))
    _ = func^


fn bench_empty_launch_many_params_caller(
    inout m: Bench, ctx: DeviceContext
) raises:
    alias func_alias = empty_kernel_many_params[
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
    var func = ctx.compile_function[func_alias]()

    @parameter
    @always_inline
    fn bench_empty_launch_many_params(inout b: Bencher) raises:
        @parameter
        fn launch() raises:
            ctx.enqueue_function(func, grid_dim=Dim(1), block_dim=Dim(1))

        b.iter[launch]()
        ctx.synchronize()

    m.bench_function[bench_empty_launch_many_params](
        BenchId("bench_empty_launch_many_params")
    )

    _ = func^


def main():
    with DeviceContext() as ctx:
        var m = Bench()
        bench_empty_launch_caller(m, ctx)
        bench_empty_launch_many_params_caller(m, ctx)
        m.dump_report()
