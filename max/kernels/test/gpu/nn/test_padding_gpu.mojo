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

from gpu.host import DeviceContext
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor

from nn.pad import pad_constant as pad_cpu
from nn.pad_gpu import get_padding_output_shape, pad_constant
from testing import assert_equal

from utils.index import IndexList


@no_inline
fn test_pad_constant_gpu[
    dtype: DType, rank: Int
](
    input_shape: IndexList[rank],
    paddings: TileTensor[DType.int, address_space = AddressSpace.GENERIC, ...],
    ctx: DeviceContext,
    verbose: Bool = False,
) raises:
    print("== test_pad_constant_gpu")

    # Create host buffer for input
    var input_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_shape.flattened_length()
    )
    ctx.synchronize()
    var input = TileTensor(
        input_host_buffer,
        row_major(Coord(input_shape)),
    )

    # Initialize input with sequential values
    for i in range(input_shape.flattened_length()):
        var idx = input.layout(Idx(i))
        input.ptr[idx] = Scalar[dtype](i)

    if verbose:
        print(input)

    # Get output shape
    var output_shape = get_padding_output_shape(input_shape, paddings)

    # Create host buffer for output and fill with zeros
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_shape.flattened_length()
    )
    ctx.synchronize()
    var output = TileTensor(
        output_host_buffer,
        row_major(Coord(output_shape)),
    ).fill(0)

    # Create device buffers
    var in_device = ctx.enqueue_create_buffer[dtype](
        input_shape.flattened_length()
    )
    var out_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )

    # Copy from host to device
    ctx.enqueue_copy(in_device, input_host_buffer)
    ctx.enqueue_copy(out_device, output_host_buffer)

    # Pad with constant = 5
    var constant = Scalar[dtype](5)

    pad_constant(
        out_device.unsafe_ptr(),
        output_shape,
        in_device.unsafe_ptr(),
        input_shape,
        paddings.ptr,
        constant,
        ctx,
    )

    ctx.enqueue_copy(output_host_buffer, out_device)
    ctx.synchronize()

    if verbose:
        print(output)

    # Verification: compute CPU reference
    var output_cpu_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_shape.flattened_length()
    )
    ctx.synchronize()
    var output_cpu = TileTensor(
        output_cpu_buffer,
        row_major(Coord(output_shape)),
    ).fill(0)

    pad_cpu(
        output_cpu,
        input,
        paddings.ptr,
        constant,
    )

    if verbose:
        print(output_cpu)

    # Compare GPU and CPU results
    for i in range(output.numel()):
        var idx = output.layout(Idx(i))
        assert_equal(output.ptr[idx], output_cpu.ptr[idx])

    print("PASS: rank=" + String(rank))

    _ = in_device
    _ = out_device


def main():
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        # 1D test
        var paddings_1d_stack = InlineArray[Scalar[DType.int], 2](
            uninitialized=True
        )
        var paddings_1d = TileTensor(paddings_1d_stack, row_major[2]())
        paddings_1d[0] = 2  # axis-0 pre-pad
        paddings_1d[1] = 1  # axis-0 post-pad
        var input_shape_1d = IndexList[1](32)
        test_pad_constant_gpu[dtype, 1](input_shape_1d, paddings_1d, ctx)
        # CHECK: PASS: rank=1

        # 2D test
        var paddings_2d_stack = InlineArray[Scalar[DType.int], 4](
            uninitialized=True
        )
        var paddings_2d = TileTensor(paddings_2d_stack, row_major[4]())
        paddings_2d[0] = 2  # axis-0 pre-pad
        paddings_2d[1] = 1  # axis-0 post-pad
        paddings_2d[2] = 3  # axis-1 pre-pad
        paddings_2d[3] = 3  # axis-1 post-pad
        var input_shape_2d = IndexList[2](32, 32)
        test_pad_constant_gpu[dtype](input_shape_2d, paddings_2d, ctx)
        # CHECK: PASS: rank=2

        # 3D test
        var paddings_3d_stack = InlineArray[Scalar[DType.int], 6](
            uninitialized=True
        )
        var paddings_3d = TileTensor(paddings_3d_stack, row_major[6]())
        paddings_3d[0] = 2  # axis-0 pre-pad
        paddings_3d[1] = 1  # axis-0 post-pad
        paddings_3d[2] = 3  # axis-1 pre-pad
        paddings_3d[3] = 3  # axis-1 post-pad
        paddings_3d[4] = 5  # axis-2 pre-pad
        paddings_3d[5] = 7  # axis-2 post-pad
        var input_shape_3d = IndexList[3](32, 32, 32)
        test_pad_constant_gpu[dtype](input_shape_3d, paddings_3d, ctx)
        # CHECK: PASS: rank=3
