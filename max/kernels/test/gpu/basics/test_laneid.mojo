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

import std.gpu.primitives.warp as warp
from std.gpu import global_idx_uint as global_idx, lane_id_uint as lane_id
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext
from std.testing import assert_equal


def kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var global_tid = global_idx.x
    if global_tid >= UInt(size):
        return
    output[global_tid] = Float32(lane_id())


def test_grid_dim(ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size
    var output_host = alloc[Float32](buffer_size)

    for i in range(buffer_size):
        output_host[i] = -1.0

    var output_buffer = ctx.enqueue_create_buffer[DType.float32](buffer_size)

    ctx.enqueue_copy(output_buffer, output_host)

    ctx.enqueue_function_experimental[kernel](
        output_buffer,
        buffer_size,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(buffer_size):
        assert_equal(
            output_host[i] % Float32(WARP_SIZE), Float32(i % WARP_SIZE)
        )

    output_host.free()


def main() raises:
    with DeviceContext() as ctx:
        test_grid_dim(ctx)
