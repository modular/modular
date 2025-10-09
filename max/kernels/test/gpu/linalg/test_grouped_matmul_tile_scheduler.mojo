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

from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.id import block_idx
from linalg.grouped_matmul_tile_scheduler import TileScheduler
from internal_utils import DeviceNDBuffer, HostNDBuffer
from buffer import NDBuffer

from utils.index import Index


fn test_kernel[
    swizzle: Bool
](group_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin]):
    scheduler = TileScheduler[
        M=20,
        tile_shape = Index(4, 8, 16),
        cluster = Index(1, 1, 1),
        swizzle=swizzle,
    ](group_offsets)

    while True:
        work_info = scheduler.fetch_next_work()
        if work_info.is_done():
            break
        print(block_idx.x, work_info)


def test(ctx: DeviceContext):
    alias group_len = 3
    alias offset_shape = DimList(group_len + 1)
    host_group_offsets = HostNDBuffer[DType.uint32, 1, offset_shape](
        offset_shape
    )
    host_group_offsets.tensor[0] = 0
    host_group_offsets.tensor[1] = 18
    host_group_offsets.tensor[2] = 24
    host_group_offsets.tensor[3] = 30
    dev_group_offsets = DeviceNDBuffer[DType.uint32, 1, offset_shape](
        offset_shape, ctx=ctx
    )
    ctx.enqueue_copy(dev_group_offsets.buffer, host_group_offsets.tensor.data)

    # CHECK-DAG: 0 (0, 0, True, False)
    # CHECK-DAG: 1 (4, 0, True, False)
    # CHECK-DAG: 2 (8, 0, True, False)
    # CHECK-DAG: 3 (12, 0, True, False)
    # ----
    # CHECK-DAG: 0 (16, 0, True, False)
    # CHECK-DAG: 1 (0, 8, True, False)
    # CHECK-DAG: 2 (4, 8, True, False)
    # CHECK-DAG: 3 (8, 8, True, False)
    # ----
    # CHECK-DAG: 0 (12, 8, True, False)
    # CHECK-DAG: 1 (16, 8, True, False)
    # CHECK-DAG: 2 (0, 16, True, False)
    # CHECK-DAG: 3 (4, 16, True, False)
    # ----
    # CHECK-DAG: 0 (8, 16, True, False)
    # CHECK-DAG: 1 (12, 16, True, False)
    # CHECK-DAG: 2 (16, 16, True, False)
    # CHECK-DAG: 3 (0, 18, True, False)
    # ----
    # CHECK-DAG: 0 (4, 18, True, False)
    # CHECK-DAG: 1 (8, 18, True, False)
    # CHECK-DAG: 2 (12, 18, True, False)
    # CHECK-DAG: 3 (16, 42, False, False)
    # ----
    # CHECK-DAG: 0 (0, 24, True, False)
    # CHECK-DAG: 1 (4, 24, True, False)
    # CHECK-DAG: 2 (8, 24, True, False)
    # CHECK-DAG: 3 (12, 24, True, False)
    # ----
    # CHECK-DAG: 0 (16, 56, False, False)
    ctx.enqueue_function[test_kernel[False]](
        dev_group_offsets.tensor,
        grid_dim=(4),
        block_dim=(1),
    )

    ctx.synchronize()

    # CHECK-DAG: 0 (0, 0, True, False)
    # CHECK-DAG: 1 (4, 0, True, False)
    # CHECK-DAG: 2 (8, 0, True, False)
    # CHECK-DAG: 3 (12, 0, True, False)
    # ----
    # CHECK-DAG: 0 (16, 0, True, False)
    # CHECK-DAG: 1 (0, 8, True, False)
    # CHECK-DAG: 2 (4, 8, True, False)
    # CHECK-DAG: 3 (8, 8, True, False)
    # ----
    # CHECK-DAG: 0 (12, 8, True, False)
    # CHECK-DAG: 1 (16, 8, True, False)
    # CHECK-DAG: 2 (0, 16, True, False)
    # CHECK-DAG: 3 (4, 16, True, False)
    # ----
    # CHECK-DAG: 0 (8, 16, True, False)
    # CHECK-DAG: 1 (12, 16, True, False)
    # CHECK-DAG: 2 (16, 16, True, False)
    # CHECK-DAG: 3 (0, 18, True, False)
    # ----
    # CHECK-DAG: 0 (4, 18, True, False)
    # CHECK-DAG: 1 (8, 18, True, False)
    # CHECK-DAG: 2 (12, 18, True, False)
    # CHECK-DAG: 3 (76, 18, False, False)
    # ----
    # CHECK-DAG: 0 (0, 24, True, False)
    # CHECK-DAG: 1 (4, 24, True, False)
    # CHECK-DAG: 2 (8, 24, True, False)
    # CHECK-DAG: 3 (12, 24, True, False)
    # ----
    # CHECK-DAG: 0 (96, 24, False, False)
    ctx.enqueue_function[test_kernel[True]](
        dev_group_offsets.tensor,
        grid_dim=(4),
        block_dim=(1),
    )

    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        test(ctx)
