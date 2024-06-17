# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from random import randn

from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from benchmark._cuda import time_async_cuda_kernel
from buffer import NDBuffer
from gpu.host import Context, Stream
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
)
from nn.concat import _concat_gpu_elementwise

from utils import StaticTuple


fn bench_concat[
    num_inputs: Int, rank: Int
](inout b: Bench, shapes: List[StaticIntTuple[rank]], axis: Int) raises:
    alias type = DType.float32
    if num_inputs != len(shapes):
        raise Error("num_inputs does not match number of shapes provided")
    var inputs = StaticTuple[NDBuffer[type, rank], num_inputs]()
    var inputs_host = StaticTuple[NDBuffer[type, rank], num_inputs]()
    var out_axis = 0
    var name = String("")
    for i in range(len(shapes)):
        var shape = shapes[i]
        var size = shape.flattened_length()
        inputs[i] = NDBuffer[type, rank](_malloc[type](size), shape)
        inputs_host[i] = NDBuffer[type, rank](
            DTypePointer[type].alloc(size), shape
        )
        randn(inputs_host[i].data, size)
        _copy_host_to_device(inputs[i].data, inputs_host[i].data, size)
        name += str(shape)
        out_axis += shape[axis]

    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += "->" + str(out_shape)
    var output = NDBuffer[type, rank](
        _malloc[type](out_shape.flattened_length()), out_shape
    )
    var output_host = NDBuffer[type, rank](
        DTypePointer[type].alloc(output.size()), out_shape
    )
    randn(output_host.data, output.size())
    var output_ptr = _malloc[type](out_shape.flattened_length())
    _copy_host_to_device(output.data, output_host.data, output.size())

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher, shape: StaticIntTuple[rank]):
        @parameter
        @always_inline
        fn kernel_launch(stream: Stream) raises:
            _concat_gpu_elementwise(output, axis, inputs, stream)

        b.iter_custom[time_async_cuda_kernel[kernel_launch]]()

    b.bench_with_input[StaticIntTuple[rank], bench_func](
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(
            BenchMetric.elements,
            out_shape.flattened_length() * sizeof[type]() * 2,
        ),
    )

    _copy_device_to_host(output_host.data, output.data, output.size())

    var offset = 0
    for i in range(num_inputs):
        var input = inputs_host[i]

        @parameter
        fn check[width: Int, _rank: Int](coords: StaticIntTuple[_rank]):
            var out_coords = coords
            out_coords[axis] += offset
            if (
                output_host[rebind[StaticIntTuple[rank]](out_coords)]
                != input[rebind[StaticIntTuple[rank]](coords)]
            ):
                abort("mismatch at coords " + str(out_coords))

        elementwise[check, 1](input.get_shape())
        offset += input.get_shape()[axis]


fn main() raises:
    var b = Bench()
    with Context() as ctx:
        bench_concat[num_inputs=2](
            b,
            List(StaticIntTuple[4](1, 1, 1, 1), StaticIntTuple[4](1, 1, 1, 1)),
            axis=0,
        )
        bench_concat[num_inputs=2](
            b,
            # llama kv cache
            List(
                StaticIntTuple[4](1, 8, 1024, 128),
                StaticIntTuple[4](1, 8, 1, 128),
            ),
            axis=2,
        )
    b.dump_report()
