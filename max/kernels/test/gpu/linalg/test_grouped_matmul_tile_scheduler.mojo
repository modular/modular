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

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_idx
from gpu.host import DeviceContext
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.grouped_matmul_tile_scheduler import TileScheduler
from memory import LegacyUnsafePointer as UnsafePointer

from utils.index import Index


fn test_kernel[
    swizzle: Bool, shape: DimList
](group_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin, shape]):
    var offset_tensor = from_ndbuffer_row_major(group_offsets)
    scheduler = TileScheduler[
        static_MN=20,
        tile_shape = Index(4, 8, 16),
        cluster = Index(1, 1, 1),
        swizzle=swizzle,
    ](len(group_offsets) - 1, offset_tensor)

    while True:
        work_info = scheduler.fetch_next_work()
        if work_info.is_done():
            break
        print(block_idx.x, work_info)


def test(ctx: DeviceContext):
    comptime group_len = 3
    comptime offset_shape = DimList(group_len + 1)

    # Host allocation
    var host_group_offsets_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        group_len + 1
    )
    host_group_offsets_ptr[0] = 0
    host_group_offsets_ptr[1] = 18
    host_group_offsets_ptr[2] = 24
    host_group_offsets_ptr[3] = 30

    # Device allocation
    var dev_group_offsets_buffer = ctx.enqueue_create_buffer[DType.uint32](
        group_len + 1
    )
    var dev_group_offsets = NDBuffer[DType.uint32, 1, _, offset_shape](
        dev_group_offsets_buffer.unsafe_ptr(),
        offset_shape,
    )

    ctx.enqueue_copy(dev_group_offsets_buffer, host_group_offsets_ptr)

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
    # CHECK-DAG: 3 (16, 18, True, False)
    # ----
    # CHECK-DAG: 0 (0, 24, True, False)
    # CHECK-DAG: 1 (4, 24, True, False)
    # CHECK-DAG: 2 (8, 24, True, False)
    # CHECK-DAG: 3 (12, 24, True, False)
    # ----
    # CHECK-DAG: 0 (16, 24, True, False)
    ctx.enqueue_function_checked[
        test_kernel[False, offset_shape],
        test_kernel[False, offset_shape],
    ](
        dev_group_offsets,
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
    # CHECK-DAG: 3 (16, 18, True, False)
    # ----
    # CHECK-DAG: 0 (0, 24, True, False)
    # CHECK-DAG: 1 (4, 24, True, False)
    # CHECK-DAG: 2 (8, 24, True, False)
    # CHECK-DAG: 3 (12, 24, True, False)
    # ----
    # CHECK-DAG: 0 (16, 24, True, False)
    ctx.enqueue_function_checked[
        test_kernel[True, offset_shape], test_kernel[True, offset_shape]
    ](
        dev_group_offsets,
        grid_dim=(4),
        block_dim=(1),
    )

    ctx.synchronize()

    # Cleanup
    host_group_offsets_ptr.free()
    _ = dev_group_offsets_buffer^


def main():
    with DeviceContext() as ctx:
        test(ctx)
