# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from os import abort
from random import randn
from sys import env_get_int, size_of

from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from builtin._closure import __ownership_keepalive
from gpu.host import DeviceContext, HostBuffer
from layout._coord import Coord, Idx
from layout._layout import Layout, row_major
from layout._tile_tensor import TileTensor
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
    comptime type = DType.float32
    if num_inputs != len(shapes):
        raise Error("num_inputs does not match number of shapes provided")

    var out_axis = 0
    var name = String()

    # Create host and device buffers for input 0
    var shape0 = shapes[0]
    var size0 = shape0.flattened_length()
    var input0_host_buffer = ctx.enqueue_create_host_buffer[type](size0)
    var input0_device_buffer = ctx.enqueue_create_buffer[type](size0)
    ctx.synchronize()
    randn(input0_host_buffer.unsafe_ptr(), size0)
    ctx.enqueue_copy(input0_device_buffer, input0_host_buffer)
    name += String(shape0)
    out_axis += shape0[axis]

    # Create host and device buffers for input 1
    var shape1 = shapes[1]
    var size1 = shape1.flattened_length()
    var input1_host_buffer = ctx.enqueue_create_host_buffer[type](size1)
    var input1_device_buffer = ctx.enqueue_create_buffer[type](size1)
    ctx.synchronize()
    randn(input1_host_buffer.unsafe_ptr(), size1)
    ctx.enqueue_copy(input1_device_buffer, input1_host_buffer)
    name += String(shape1)
    out_axis += shape1[axis]

    # Create output buffers
    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += String("->", out_shape)
    var output_size = out_shape.flattened_length()
    var output_host_buffer = ctx.enqueue_create_host_buffer[type](output_size)
    var output_device_buffer = ctx.enqueue_create_buffer[type](output_size)
    ctx.synchronize()
    randn(output_host_buffer.unsafe_ptr(), output_size)
    ctx.enqueue_copy(output_device_buffer, output_host_buffer)

    # Create TileTensors with dynamic layouts
    var input0_device = TileTensor(
        input0_device_buffer.unsafe_ptr(),
        row_major(Coord(shape0)),
    )
    var input1_device = TileTensor(
        input1_device_buffer.unsafe_ptr(),
        row_major(Coord(shape1)),
    )
    var output_device = TileTensor(
        output_device_buffer.unsafe_ptr(),
        row_major(Coord(out_shape)),
    )

    # Create input tuple for kernel
    var inputs = StaticTuple[
        TileTensor[type, input0_device.LayoutType, ImmutAnyOrigin],
        num_inputs,
    ](
        input0_device.as_any_origin().as_immut(),
        input1_device.as_any_origin().as_immut(),
    )

    # Create host TileTensors for verification
    var input0_host = TileTensor(
        input0_host_buffer,
        row_major(Coord(shape0)),
    )
    var input1_host = TileTensor(
        input1_host_buffer,
        row_major(Coord(shape1)),
    )
    var output_host = TileTensor(
        output_host_buffer,
        row_major(Coord(out_shape)),
    )

    var inputs_host = StaticTuple[
        TileTensor[type, input0_host.LayoutType, MutAnyOrigin],
        num_inputs,
    ](
        input0_host.as_any_origin(),
        input1_host.as_any_origin(),
    )

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher, shape: IndexList[rank]) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _concat_gpu_elementwise[epilogue_fn=None](
                output_device.as_any_origin(), axis, inputs, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_with_input[IndexList[rank], bench_func](
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        [
            ThroughputMeasure(
                BenchMetric.elements,
                out_shape.flattened_length() * size_of[type]() * 2,
            )
        ],
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    var offset = 0
    for i in range(num_inputs):
        var input = inputs_host[i]
        var input_shape = shapes[i]

        @parameter
        fn check[
            width: Int, _rank: Int, alignment: Int = 1
        ](coords: IndexList[_rank]):
            var out_coords = coords
            out_coords[axis] += offset
            var out_coord = Coord(out_coords)
            var in_coord = Coord(coords)
            comptime assert out_coord.flat_rank == output_host.flat_rank
            comptime assert in_coord.flat_rank == input.flat_rank
            if output_host.load[width=1](out_coord) != input.load[width=1](
                in_coord
            ):
                abort(String("mismatch at coords ", out_coords))

        elementwise[check, 1](input_shape)
        offset += input_shape[axis]

    _ = input0_device_buffer
    _ = input1_device_buffer
    _ = output_device_buffer


def main():
    comptime num_inputs = env_get_int["num_inputs", 2]()
    comptime axis = env_get_int["axis", 0]()
    comptime W0 = env_get_int["W0", 1]()
    comptime X0 = env_get_int["X0", 1]()
    comptime Y0 = env_get_int["Y0", 1]()
    comptime Z0 = env_get_int["Z0", 1]()

    comptime W1 = env_get_int["W1", 1]()
    comptime X1 = env_get_int["X1", 1]()
    comptime Y1 = env_get_int["Y1", 1]()
    comptime Z1 = env_get_int["Z1", 1]()

    var b = Bench()
    with DeviceContext() as ctx:
        bench_concat[num_inputs=num_inputs](
            b,
            [IndexList[4](W0, X0, Y0, Z0), IndexList[4](W1, X1, Y1, Z1)],
            ctx,
            axis=axis,
        )
        b.dump_report()
