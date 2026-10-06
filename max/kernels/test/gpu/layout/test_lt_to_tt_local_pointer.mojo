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
"""Checks adapter stores and vectorized loads through GPU-local pointers."""

from layout import (
    Layout,
    LayoutTensor,
    TileTensor,
    lt_to_tt,
    lt_to_tt_idx,
    row_major,
)
from layout._host_device_tile_tensor import HostDeviceTileTensor
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def adapter_local_pointer_kernel(
    output: TileTensor[.float32, type_of(row_major[32]()), MutAnyOrigin],
):
    var local = LayoutTensor[
        .float32, Layout.row_major(4, 4), MutAnyOrigin, address_space=.LOCAL
    ].stack_allocation()
    var converted = lt_to_tt(local)
    var indexed = lt_to_tt_idx[linear_idx_type=.int64](local)
    comptime assert converted.address_space == local.address_space
    comptime assert indexed.address_space == local.address_space
    comptime assert indexed.linear_idx_type == .int64
    for row in range(4):
        for col in range(4):
            converted.raw_store(row * 4 + col, output[row * 4 + col])
    var vectors = indexed.vectorize[1, 4]()
    var shift = Int(output[31]) % 4
    for row in range(4):
        var value = vectors[(row + shift) % 4, 0]
        for lane in range(4):
            output[16 + row * 4 + lane] = value[lane] + Float32(row + lane)
    for row in range(4):
        for col in range(4):
            indexed.raw_store(row * 4 + col, Float32(100 + row * 4 + col))
    for row in range(4):
        for col in range(4):
            output[row * 4 + col] = converted[row, col]


def main() raises:
    with DeviceContext() as ctx:
        var output = HostDeviceTileTensor[.float32](row_major[32](), ctx)
        for i in range(32):
            output.host_tensor()[i] = Float32(i * 3 - 17)
        output.host_tensor()[31] = 1
        output.to_device()
        ctx.enqueue_function[adapter_local_pointer_kernel](
            output.device_tensor().as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=1,
        )
        output.to_host()
        for row in range(4):
            for col in range(4):
                var index = row * 4 + col
                assert_equal(output.host_tensor()[index], Float32(100 + index))
                assert_equal(
                    output.host_tensor()[16 + index],
                    Float32((((row + 1) % 4) * 4 + col) * 3 - 17 + row + col),
                )
