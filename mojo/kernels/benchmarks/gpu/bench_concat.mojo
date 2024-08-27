# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from random import randn
from os import abort
from sys import sizeof

from builtin._closure import __ownership_keepalive
from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from gpu.host.device_context import DeviceContext, DeviceBuffer
from nn.concat import _concat_gpu_elementwise

from utils import StaticTuple, StaticIntTuple


fn bench_concat[
    num_inputs: Int, rank: Int
](
    inout b: Bench,
    shapes: List[StaticIntTuple[rank]],
    ctx: DeviceContext,
    axis: Int,
) raises:
    alias type = DType.float32
    if num_inputs != len(shapes):
        raise Error("num_inputs does not match number of shapes provided")
    var inputs = StaticTuple[NDBuffer[type, rank], num_inputs]()
    var inputs_host = StaticTuple[NDBuffer[type, rank], num_inputs]()
    var out_axis = 0
    var name = String("")

    # TODO: Generalize for arbitrary num of inputs.
    var shape = shapes[0]
    var size = shape.flattened_length()
    var input0_ptr = ctx.create_buffer[type](size)
    inputs[0] = NDBuffer[type, rank](input0_ptr.ptr, shape)
    inputs_host[0] = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(size), shape
    )
    randn(inputs_host[0].data, size)
    ctx.enqueue_copy_to_device(input0_ptr, inputs_host[0].data)
    name += str(shape)
    out_axis += shape[axis]

    shape = shapes[1]
    size = shape.flattened_length()
    var input1_ptr = ctx.create_buffer[type](size)
    inputs[1] = NDBuffer[type, rank](input1_ptr.ptr, shape)
    inputs_host[1] = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(size), shape
    )
    randn(inputs_host[1].data, size)
    ctx.enqueue_copy_to_device(input1_ptr, inputs_host[1].data)
    name += str(shape)
    out_axis += shape[axis]

    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += "->" + str(out_shape)
    var output_ptr = ctx.create_buffer[type](out_shape.flattened_length())
    var output = NDBuffer[type, rank](output_ptr.ptr, out_shape)
    var output_host = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(output.size()), out_shape
    )
    randn(output_host.data, output.size())

    ctx.enqueue_copy_to_device(output_ptr, output_host.data)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher, shape: StaticIntTuple[rank]):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _concat_gpu_elementwise(output, axis, inputs, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_with_input[StaticIntTuple[rank], bench_func](
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(
            BenchMetric.elements,
            out_shape.flattened_length() * sizeof[type]() * 2,
        ),
    )

    ctx.enqueue_copy_from_device(output_host.data, output_ptr)

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

    __ownership_keepalive(
        input0_ptr, input1_ptr, output_ptr, output, axis, inputs, output_host
    )


fn main() raises:
    try:
        var b = Bench()
        with DeviceContext() as ctx:
            bench_concat[num_inputs=2](
                b,
                List(
                    StaticIntTuple[4](1, 1, 1, 1), StaticIntTuple[4](1, 1, 1, 1)
                ),
                ctx,
                axis=0,
            )
            bench_concat[num_inputs=2](
                b,
                # llama kv cache
                List(
                    StaticIntTuple[4](1, 8, 1024, 128),
                    StaticIntTuple[4](1, 8, 1, 128),
                ),
                ctx,
                axis=2,
            )
            b.dump_report()
    except e:
        print("CUDA_ERROR:", e)
