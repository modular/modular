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
"""Direct GPU launch coverage for ManagedTensorSlice device encoding."""

from extensibility import IOSpec, ManagedTensorSlice, StaticTensorSpec
from extensibility.managed_tensor_slice import _IndexListToTileLayout
from layout import Coord, TensorLayout, TileTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import isnan, nan
from std.memory import AddressSpace
from std.testing import assert_equal, assert_true
from std.utils import IndexList


def _copy_with_metadata[
    InputLayout: TensorLayout, OutputLayout: TensorLayout
](
    source: TileTensor[.float32, InputLayout, MutUntrackedOrigin],
    output: TileTensor[.float32, OutputLayout, MutUntrackedOrigin],
    output_alias: TileTensor[.float32, OutputLayout, MutUntrackedOrigin],
):
    var row = global_idx.y
    var col = global_idx.x
    var rows = Int(source.layout.shape[0]().value())
    var cols = Int(source.layout.shape[1]().value())
    if row >= rows or col >= cols:
        return
    var metadata = Float32(
        rows
        + 3 * cols
        + 5 * Int(output.layout.shape[0]().value())
        + 7 * Int(output.layout.shape[1]().value())
        + 11 * Int(source.layout.stride[0]().value())
        + 13 * Int(source.layout.stride[1]().value())
        + 17 * Int(output.layout.stride[0]().value())
        + 19 * Int(output.layout.stride[1]().value())
    )
    var idx = Coord(row, col)
    output.store(idx, source.load(idx) + metadata)
    output_alias.store(idx, output_alias.load(idx) + 1)


def _run_case[
    InputLayout: TensorLayout, OutputLayout: TensorLayout
](
    ctx: DeviceContext,
    rows: Int,
    cols: Int,
    input_strides: IndexList[2],
    output_strides: IndexList[2],
) raises:
    comptime input_spec = StaticTensorSpec[.float32, 2, InputLayout](
        1, AddressSpace.GENERIC
    )
    comptime output_spec = StaticTensorSpec[.float32, 2, OutputLayout](
        1, AddressSpace.GENERIC
    )
    comptime Input = ManagedTensorSlice[
        io_spec=IOSpec.Input, static_spec=input_spec
    ]
    comptime Output = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=output_spec
    ]
    comptime assert (
        Input.device_type
        == TileTensor[.float32, Input.RuntimeLayout, MutUntrackedOrigin]
    ), "ManagedTensorSlice must encode as TileTensor"
    comptime assert (
        Output.device_type
        == TileTensor[.float32, Output.RuntimeLayout, MutUntrackedOrigin]
    ), "ManagedTensorSlice must encode as TileTensor"
    var input_size = (
        (rows - 1) * input_strides[0] + (cols - 1) * input_strides[1] + 3
    )
    var output_size = (
        (rows - 1) * output_strides[0] + (cols - 1) * output_strides[1] + 3
    )
    var input_host = ctx.enqueue_create_host_buffer[.float32](input_size)
    var expected = ctx.enqueue_create_host_buffer[.float32](output_size)
    ctx.synchronize()
    for i in range(input_size):
        input_host[i] = nan[.float32]()
    for i in range(output_size):
        expected[i] = nan[.float32]()
    var metadata = Float32(
        6 * rows
        + 10 * cols
        + 11 * input_strides[0]
        + 13 * input_strides[1]
        + 17 * output_strides[0]
        + 19 * output_strides[1]
    )
    for row in range(rows):
        for col in range(cols):
            var value = Float32(100 * row + col)
            input_host[
                1 + row * input_strides[0] + col * input_strides[1]
            ] = value
            expected[1 + row * output_strides[0] + col * output_strides[1]] = (
                value + metadata + 1
            )
    var input_device = ctx.enqueue_create_buffer[.float32](input_size)
    var output_device = ctx.enqueue_create_buffer[.float32](output_size)
    ctx.enqueue_copy(input_device, input_host)
    output_device.enqueue_fill(nan[.float32]())
    var source = Input(
        input_device.unsafe_ptr() + 1, (rows, cols), input_strides
    )
    var output = Output(
        output_device.unsafe_ptr() + 1, (rows, cols), output_strides
    )
    var output_alias = Output(
        output_device.unsafe_ptr() + 1, (rows, cols), output_strides
    )

    # Passing the slices themselves exercises DevicePassable, not projection.
    ctx.enqueue_function[
        _copy_with_metadata[Input.RuntimeLayout, Output.RuntimeLayout]
    ](source, output, output_alias, grid_dim=1, block_dim=(16, 8))
    with output_device.map_to_host() as actual:
        for i in range(output_size):
            if isnan(expected[i]):
                assert_true(isnan(actual[i]))
            else:
                assert_equal(actual[i], expected[i])
    with input_device.map_to_host() as actual:
        for i in range(input_size):
            if isnan(input_host[i]):
                assert_true(isnan(actual[i]))
            else:
                assert_equal(actual[i], input_host[i])

    comptime InPlaceOutput = ManagedTensorSlice[
        io_spec=IOSpec.Output, static_spec=input_spec
    ]
    comptime MutableInput = ManagedTensorSlice[
        io_spec=IOSpec.MutableInput, static_spec=input_spec
    ]
    var inplace_source = MutableInput(
        input_device.unsafe_ptr() + 1, (rows, cols), input_strides
    )
    var inplace = InPlaceOutput(
        input_device.unsafe_ptr() + 1, (rows, cols), input_strides
    )
    var inplace_alias = InPlaceOutput(
        input_device.unsafe_ptr() + 1, (rows, cols), input_strides
    )
    ctx.enqueue_function[
        _copy_with_metadata[Input.RuntimeLayout, InPlaceOutput.RuntimeLayout]
    ](inplace_source, inplace, inplace_alias, grid_dim=1, block_dim=(16, 8))
    var inplace_metadata = Float32(
        6 * rows + 10 * cols + 28 * input_strides[0] + 32 * input_strides[1]
    )
    with input_device.map_to_host() as actual:
        for i in range(input_size):
            if isnan(input_host[i]):
                assert_true(isnan(actual[i]))
            else:
                assert_equal(actual[i], input_host[i] + inplace_metadata + 1)


def main() raises:
    comptime StaticInput = _IndexListToTileLayout[
        IndexList[2](3, 5), IndexList[2](17, 2)
    ]
    comptime StaticOutput = _IndexListToTileLayout[
        IndexList[2](3, 5), IndexList[2](19, 3)
    ]
    comptime Dynamic = _IndexListToTileLayout[
        IndexList[2](-1, -1), IndexList[2](-1, -1)
    ]
    comptime Mixed = _IndexListToTileLayout[
        IndexList[2](-1, 5), IndexList[2](-1, 2)
    ]
    with DeviceContext() as ctx:
        _run_case[StaticInput, StaticOutput](ctx, 3, 5, (17, 2), (19, 3))
        _run_case[Dynamic, Dynamic](ctx, 3, 5, (17, 2), (19, 3))
        _run_case[Dynamic, Dynamic](ctx, 7, 13, (31, 2), (41, 3))
        _run_case[Mixed, Mixed](ctx, 7, 5, (17, 2), (23, 2))
