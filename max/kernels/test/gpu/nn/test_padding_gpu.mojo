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


from gpu import *
from gpu.host import DeviceContext
from layout import LayoutTensor, RuntimeLayout, Layout
from nn.pad import pad_constant as pad_cpu
from nn.pad_gpu import get_padding_output_shape, pad_constant
from testing import assert_equal

from utils.index import IndexList


@no_inline
fn test_pad_constant_gpu[
    type: DType, rank: Int
](
    input_shape: IndexList[rank],
    paddings: LayoutTensor[DType.index, Layout(2 * rank)],
    ctx: DeviceContext,
    verbose: Bool = False,
) raises:
    print("== test_pad_constant_gpu")
    alias layout = Layout.row_major[rank]()

    # Create an input matrix
    var input_data = UnsafePointer[Scalar[type]].alloc(
        input_shape.flattened_length()
    )
    var input = LayoutTensor[type, layout](
        input_data,
        RuntimeLayout[layout].row_major(input_shape),
    )

    var output_shape = get_padding_output_shape(input_shape, paddings)

    # Create an output matrix and fill it with zeros.
    var output_data = UnsafePointer[Scalar[type]].alloc(
        output_shape.flattened_length()
    )
    var output = LayoutTensor[type, layout](
        output_data, RuntimeLayout[layout].row_major(output_shape)
    ).fill(0)

    for i in range(input_shape.flattened_length()):
        var idx = input.runtime_layout(i)
        input.ptr[idx] = i

    if verbose:
        print(input)

    # create device buffers
    var in_device = ctx.enqueue_create_buffer[type](
        input_shape.flattened_length()
    )
    var out_device = ctx.enqueue_create_buffer[type](
        output_shape.flattened_length()
    )

    # copy from host to device
    ctx.enqueue_copy(in_device, input.ptr)
    ctx.enqueue_copy(out_device, output.ptr)

    # pad with constant = 5
    var constant = Scalar[type](5)

    pad_constant(
        out_device._unsafe_ptr(),
        output_shape,
        in_device._unsafe_ptr(),
        input_shape,
        paddings.ptr,
        constant,
        ctx,
    )

    ctx.enqueue_copy(output.ptr, out_device)
    ctx.synchronize()

    if verbose:
        print(output)

    # verification
    var output_data_cpu = UnsafePointer[Scalar[type]].alloc(
        output_shape.flattened_length()
    )
    var output_cpu = LayoutTensor[type, layout](
        output_data_cpu,
        RuntimeLayout[layout].row_major(output_shape),
    ).fill(0)
    pad_cpu(output_cpu, input, paddings.ptr, constant)

    if verbose:
        print(output_cpu)

    for i in range(output.size()):
        var idx = output.runtime_layout(i)
        assert_equal(output.ptr[idx], output_cpu.ptr[idx])
    print("PASS: rank=" + String(rank))
    output_data_cpu.free()

    _ = in_device
    _ = out_device

    input_data.free()
    output_data.free()


def main():
    alias type = DType.float32
    with DeviceContext() as ctx:
        var input_shape_1d = IndexList[1](32)
        # Create a padding array of the (before,after) form
        var paddings_1d = LayoutTensor[
            DType.index, Layout(2 * 1), MutableAnyOrigin
        ].stack_allocation()
        paddings_1d[0] = 2  # axis-0 pre-pad
        paddings_1d[1] = 1  # axis-0 post-pad
        test_pad_constant_gpu[type, 1](input_shape_1d, paddings_1d, ctx)
        # CHECK: PASS: rank=1

        var input_shape_2d = IndexList[2](32, 32)
        # Create a padding array of the (before,after) form
        var paddings_2d = LayoutTensor[
            DType.index, Layout(2 * 2), MutableAnyOrigin
        ].stack_allocation()
        paddings_2d[0] = 2  # axis-0 pre-pad
        paddings_2d[1] = 1  # axis-0 post-pad
        paddings_2d[2] = 3  # axis-1 pre-pad
        paddings_2d[3] = 3  # axis-1 post-pad
        test_pad_constant_gpu[type](input_shape_2d, paddings_2d, ctx)
        # CHECK: PASS: rank=2

        var input_shape_3d = IndexList[3](32, 32, 32)
        # Create a padding array of the (before,after) form
        var paddings_3d = LayoutTensor[
            DType.index, Layout(2 * 3), MutableAnyOrigin
        ].stack_allocation()
        paddings_3d[0] = 2  # axis-0 pre-pad
        paddings_3d[1] = 1  # axis-0 post-pad
        paddings_3d[2] = 3  # axis-1 pre-pad
        paddings_3d[3] = 3  # axis-1 post-pad
        paddings_3d[4] = 5  # axis-2 pre-pad
        paddings_3d[5] = 7  # axis-2 post-pad
        test_pad_constant_gpu[type](input_shape_3d, paddings_3d, ctx)
        # CHECK: PASS: rank=3
