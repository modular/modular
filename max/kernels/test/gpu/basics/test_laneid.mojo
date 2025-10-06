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

import gpu.warp as warp
from gpu import barrier, global_idx, lane_id
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from testing import assert_equal


fn kernel(
    output: UnsafePointer[Float32],
    size: Int,
):
    var global_tid = global_idx.x
    if global_tid >= size:
        return
    output[global_tid] = lane_id()


fn test_grid_dim(ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size
    var output_host = UnsafePointer[Float32].alloc(buffer_size)

    for i in range(buffer_size):
        output_host[i] = -1.0

    var output_buffer = ctx.enqueue_create_buffer[DType.float32](buffer_size)

    ctx.enqueue_copy(output_buffer, output_host)

    ctx.enqueue_function_checked[kernel, kernel](
        output_buffer,
        buffer_size,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(buffer_size):
        assert_equal(output_host[i] % WARP_SIZE, i % WARP_SIZE)

    output_host.free()


fn main() raises:
    with DeviceContext() as ctx:
        test_grid_dim(ctx)
