# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s

from random import randn
from sys import sizeof, simdwidthof

from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host.device_context import DeviceContext
from memory import UnsafePointer
from utils import IndexList


fn bench_add[
    unroll_by: Int, rank: Int
](inout b: Bench, shape: IndexList[rank], ctx: DeviceContext) raises:
    alias type = DType.float32
    var size = shape.flattened_length()
    var input0_ptr = ctx.create_buffer[type](size)
    var input1_ptr = ctx.create_buffer[type](size)
    var output_ptr = ctx.create_buffer[type](size)
    var input0_ptr_host = UnsafePointer[Scalar[type]].alloc(size)
    var input1_ptr_host = UnsafePointer[Scalar[type]].alloc(size)
    var output_ptr_host = UnsafePointer[Scalar[type]].alloc(size)
    randn(input0_ptr_host, size)
    randn(input1_ptr_host, size)
    randn(output_ptr_host, size)
    ctx.enqueue_copy_to_device(input0_ptr, input0_ptr_host)
    ctx.enqueue_copy_to_device(input1_ptr, input1_ptr_host)
    ctx.enqueue_copy_to_device(output_ptr, output_ptr_host)

    var input0 = NDBuffer[type, rank](input0_ptr.ptr, shape)
    var input1 = NDBuffer[type, rank](input1_ptr.ptr, shape)
    var output = NDBuffer[type, rank](output_ptr.ptr, shape)

    @parameter
    @always_inline
    @__copy_capture(input0, input1, output)
    fn add[simd_width: Int, _rank: Int](out_index: IndexList[_rank]):
        var idx = rebind[IndexList[rank]](out_index)
        var val = input0.load[width=simd_width](idx) + input1.load[
            width=simd_width
        ](idx)
        output.store[width=simd_width](idx, val)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher, shape: IndexList[rank]) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            elementwise[add, simd_width=unroll_by, target="cuda"](shape, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_with_input[__type_of(shape), bench_func](
        BenchId("add", str(shape)),
        shape,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(BenchMetric.elements, size * sizeof[type]() * 3),
    )

    ctx.enqueue_copy_from_device(output_ptr_host, output_ptr)

    alias nelts = simdwidthof[type]()
    for i in range(0, size, nelts):
        if not (
            output_ptr_host.load[width=nelts](i)
            == input0_ptr_host.load[width=nelts](i)
            + input1_ptr_host.load[width=nelts](i)
        ).reduce_and():
            raise Error("mismatch at flattened idx " + str(i))

    _ = input0_ptr
    _ = input1_ptr
    _ = output_ptr


fn main() raises:
    var b = Bench()
    with DeviceContext() as ctx:
        bench_add[unroll_by=4](b, IndexList[4](2, 4, 1024, 1024), ctx)
        bench_add[unroll_by=1](b, IndexList[4](2, 4, 1024, 1024), ctx)
        b.dump_report()
