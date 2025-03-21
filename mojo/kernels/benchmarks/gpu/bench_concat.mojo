# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s

from os import abort
from random import randn
from sys import env_get_int, env_get_string, sizeof

from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import NDBuffer
from builtin._closure import __ownership_keepalive
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from nn.concat import _concat_gpu_elementwise

from utils import IndexList, StaticTuple


fn bench_concat[
    num_inputs: Int, rank: Int
](
    mut b: Bench,
    shapes: List[IndexList[rank]],
    ctx: DeviceContext,
    axis: Int,
) raises:
    alias type = DType.float32
    if num_inputs != len(shapes):
        raise Error("num_inputs does not match number of shapes provided")
    var inputs = StaticTuple[
        NDBuffer[type, rank, MutableAnyOrigin], num_inputs
    ]()
    var inputs_host = StaticTuple[
        NDBuffer[type, rank, MutableAnyOrigin], num_inputs
    ]()
    var out_axis = 0
    var name = String("")

    # TODO: Generalize for arbitrary num of inputs.
    var shape = shapes[0]
    var size = shape.flattened_length()
    var input0_ptr = ctx.enqueue_create_buffer[type](size)
    inputs[0] = NDBuffer[type, rank](input0_ptr.unsafe_ptr(), shape)
    inputs_host[0] = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(size), shape
    )
    randn(inputs_host[0].data, size)
    ctx.enqueue_copy(input0_ptr, inputs_host[0].data)
    name += String(shape)
    out_axis += shape[axis]

    shape = shapes[1]
    size = shape.flattened_length()
    var input1_ptr = ctx.enqueue_create_buffer[type](size)
    inputs[1] = NDBuffer[type, rank](input1_ptr.unsafe_ptr(), shape)
    inputs_host[1] = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(size), shape
    )
    randn(inputs_host[1].data, size)
    ctx.enqueue_copy(input1_ptr, inputs_host[1].data)
    name += String(shape)
    out_axis += shape[axis]

    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += String("->", out_shape)
    var output_ptr = ctx.enqueue_create_buffer[type](
        out_shape.flattened_length()
    )
    var output = NDBuffer[type, rank](output_ptr.unsafe_ptr(), out_shape)
    var output_host = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(output.size()), out_shape
    )
    randn(output_host.data, output.size())

    ctx.enqueue_copy(output_ptr, output_host.data)

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher, shape: IndexList[rank]) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _concat_gpu_elementwise[epilogue_fn=None](output, axis, inputs, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_with_input[IndexList[rank], bench_func](
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(
            BenchMetric.elements,
            out_shape.flattened_length() * sizeof[type]() * 2,
        ),
    )

    ctx.enqueue_copy(output_host.data, output_ptr)

    var offset = 0
    for i in range(num_inputs):
        var input = inputs_host[i]

        @parameter
        fn check[width: Int, _rank: Int](coords: IndexList[_rank]):
            var out_coords = coords
            out_coords[axis] += offset
            if (
                output_host[rebind[IndexList[rank]](out_coords)]
                != input[rebind[IndexList[rank]](coords)]
            ):
                abort(String("mismatch at coords ", out_coords))

        elementwise[check, 1](input.get_shape())
        offset += input.get_shape()[axis]

    __ownership_keepalive(
        input0_ptr, input1_ptr, output_ptr, output, axis, inputs, output_host
    )


fn main() raises:
    alias num_inputs = env_get_int["num_inputs", 2]()
    alias axis = env_get_int["axis", 0]()
    alias W0 = env_get_int["W0", 1]()
    alias X0 = env_get_int["X0", 1]()
    alias Y0 = env_get_int["Y0", 1]()
    alias Z0 = env_get_int["Z0", 1]()

    alias W1 = env_get_int["W1", 1]()
    alias X1 = env_get_int["X1", 1]()
    alias Y1 = env_get_int["Y1", 1]()
    alias Z1 = env_get_int["Z1", 1]()

    var b = Bench()
    with DeviceContext() as ctx:
        bench_concat[num_inputs=num_inputs](
            b,
            List(
                IndexList[4](W0, X0, Y0, Z0),
                IndexList[4](W1, X1, Y1, Z1),
            ),
            ctx,
            axis=axis,
        )
        b.dump_report()
