# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from gpu.host import DeviceContext
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
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
    alias layout = Layout.row_major[rank]()
    var inputs = StaticTuple[
        LayoutTensor[type, layout, MutableAnyOrigin], num_inputs
    ]()
    var inputs_host = StaticTuple[
        LayoutTensor[type, layout, MutableAnyOrigin], num_inputs
    ]()
    var out_axis = 0
    var name = String()

    # TODO: Generalize for arbitrary num of inputs.
    var shape = shapes[0]
    var size = shape.flattened_length()
    var input0_ptr = ctx.enqueue_create_buffer[type](size)
    inputs[0] = LayoutTensor[type, layout](
        input0_ptr, RuntimeLayout[layout].row_major(shape)
    ).origin_cast[True, MutableAnyOrigin]()
    inputs_host[0] = LayoutTensor[type, layout, MutableAnyOrigin](
        UnsafePointer[Scalar[type]].alloc(size),
        RuntimeLayout[layout].row_major(shape),
    )
    randn(inputs_host[0].ptr, size)
    ctx.enqueue_copy(input0_ptr, inputs_host[0].ptr)
    name += String(shape)
    out_axis += shape[axis]

    shape = shapes[1]
    size = shape.flattened_length()
    var input1_ptr = ctx.enqueue_create_buffer[type](size)
    inputs[1] = LayoutTensor[type, layout, MutableAnyOrigin](
        input1_ptr, RuntimeLayout[layout].row_major(shape)
    )
    inputs_host[1] = LayoutTensor[type, layout, MutableAnyOrigin](
        UnsafePointer[Scalar[type]].alloc(size),
        RuntimeLayout[layout].row_major(shape),
    )
    randn(inputs_host[1].ptr, size)
    ctx.enqueue_copy(input1_ptr, inputs_host[1].ptr)
    name += String(shape)
    out_axis += shape[axis]

    var out_shape = shapes[0]
    out_shape[axis] = out_axis
    name += String("->", out_shape)
    var output_ptr = ctx.enqueue_create_buffer[type](
        out_shape.flattened_length()
    )
    var output = LayoutTensor[type, layout](
        output_ptr, RuntimeLayout[layout].row_major(out_shape)
    )
    var output_host = LayoutTensor[type, layout](
        UnsafePointer[Scalar[type]].alloc(output.size()),
        RuntimeLayout[layout].row_major(out_shape),
    )
    randn(output_host.ptr, output.size())

    ctx.enqueue_copy(output_ptr, output_host.ptr)

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher, shape: IndexList[rank]) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _concat_gpu_elementwise[epilogue_fn=None](
                output.origin_cast[True, MutableAnyOrigin](), axis, inputs, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_with_input[IndexList[rank], bench_func](
        BenchId("concat", name),
        out_shape,
        # TODO: Pick relevant benchmetric.
        ThroughputMeasure(
            BenchMetric.elements,
            out_shape.flattened_length() * size_of[type]() * 2,
        ),
    )

    ctx.enqueue_copy(output_host.ptr, output_ptr)

    var offset = 0
    for i in range(num_inputs):
        var input = inputs_host[i]

        @parameter
        fn check[
            width: Int, _rank: Int, alignment: Int = 1
        ](coords: IndexList[_rank]):
            var out_coords = coords
            out_coords[axis] += offset
            if output_host.load[width=1](out_coords) != input.load[width=1](
                coords
            ):
                abort(String("mismatch at coords ", out_coords))

        elementwise[check, 1](input.runtime_layout.shape.value)
        offset += input.runtime_layout.shape.value[axis]


def main():
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
            [IndexList[4](W0, X0, Y0, Z0), IndexList[4](W1, X1, Y1, Z1)],
            ctx,
            axis=axis,
        )
        b.dump_report()
