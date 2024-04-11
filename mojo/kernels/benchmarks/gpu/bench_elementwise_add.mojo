# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from benchmark import Bench, Bencher, BenchId
from benchmark._cuda import time_async_cuda_kernel

from random import randn
from buffer import NDBuffer
from buffer.list import DimList
from gpu.host import Context, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from algorithm.functional import _elementwise_impl_gpu


fn bench_add[
    unroll_by: Int, rank: Int
](inout b: Bench, shape: StaticIntTuple[rank]) raises:
    alias type = DType.float32
    var size = shape.flattened_length()
    var input0_ptr = _malloc[type](size)
    var input1_ptr = _malloc[type](size)
    var output_ptr = _malloc[type](size)
    var input0_ptr_host = DTypePointer[type].alloc(size)
    var input1_ptr_host = DTypePointer[type].alloc(size)
    var output_ptr_host = DTypePointer[type].alloc(size)
    randn(input0_ptr_host, size)
    randn(input1_ptr_host, size)
    randn(output_ptr_host, size)
    _copy_host_to_device(input0_ptr, input0_ptr_host, size)
    _copy_host_to_device(input1_ptr, input1_ptr_host, size)
    _copy_host_to_device(output_ptr, output_ptr_host, size)

    var input0 = NDBuffer[type, rank](input0_ptr, shape)
    var input1 = NDBuffer[type, rank](input1_ptr, shape)
    var output = NDBuffer[type, rank](output_ptr, shape)

    @parameter
    @always_inline
    @__copy_capture(input0, input1, output)
    fn add[simd_width: Int, _rank: Int](out_index: StaticIntTuple[_rank]):
        var idx = rebind[StaticIntTuple[rank]](out_index)
        var val = input0.load[width=simd_width](idx) + input1.load[
            width=simd_width
        ](idx)
        output.store[width=simd_width](idx, val)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher, shape: StaticIntTuple[rank]):
        @parameter
        @always_inline
        fn kernel_launch(stream: Stream) raises:
            _elementwise_impl_gpu[add, simd_width=unroll_by, rank=rank](
                shape, stream
            )

        b.iter_custom[time_async_cuda_kernel[kernel_launch]]()

    b.bench_with_input[__type_of(shape), bench_func](
        BenchId("add", str(shape)),
        shape,
        throughput_elems=size * sizeof[type]() * 3,
    )

    _copy_device_to_host(output_ptr_host, output_ptr, size)

    alias nelts = simdwidthof[type]()
    for i in range(0, size, nelts):
        if output_ptr_host.load[width=nelts](i) != input0_ptr_host.load[
            width=nelts
        ](i) + input1_ptr_host.load[width=nelts](i):
            raise Error("mismatch at flattened idx " + str(i))


fn main() raises:
    var b = Bench()
    with Context() as ctx:
        bench_add[unroll_by=4](b, StaticIntTuple[4](2, 4, 1024, 1024))
        bench_add[unroll_by=1](b, StaticIntTuple[4](2, 4, 1024, 1024))
    b.dump_report()
