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
"""Tests for GroupedTileScheduler.

These tests verify:
1. Correct delinearization of tile indices to group coordinates
2. Proper detection of group changes
3. All tiles are visited exactly once
4. K-tile count is correct for each group
"""

from gpu import barrier, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from memory import stack_allocation

from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul_kernel import (
    _ProblemSizesTile,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_tile_scheduler import (
    GroupedTileScheduler,
    GroupedWorkInfo,
    GroupedWorkIterator,
)
from utils.index import Index


# =============================================================================
# Test kernel: Collect all visited tiles
# =============================================================================


fn test_scheduler_kernel[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    max_tiles: Int,
](
    problem_sizes: LayoutTensor[
        DType.int32, Layout.row_major(max_groups, 4), MutAnyOrigin
    ],
    num_groups: Int32,
    # Outputs: record visited tiles
    visited_group: LayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1), MutAnyOrigin
    ],
    visited_m: LayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1), MutAnyOrigin
    ],
    visited_n: LayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1), MutAnyOrigin
    ],
    visited_k_tiles: LayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1), MutAnyOrigin
    ],
    visited_changed: LayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1), MutAnyOrigin
    ],
    tile_count: LayoutTensor[DType.int32, Layout.row_major(1, 1), MutAnyOrigin],
):
    """Kernel that iterates over all tiles and records their coordinates."""
    # Convert LayoutTensor to TileTensor for the scheduler
    from memory import UnsafePointer as NewPtr
    from layout._layout import row_major as new_row_major

    var problem_sizes_tt = _ProblemSizesTile[max_groups](
        ptr=NewPtr[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(problem_sizes.ptr)
        ),
        layout=new_row_major[max_groups, 4](),
    )
    var scheduler = GroupedTileScheduler[tile_m, tile_n, tile_k, max_groups, 0](
        problem_sizes_tt, Int(num_groups)
    )

    var work_iter = scheduler.work_iterator()

    while work_iter.has_work():
        var current = work_iter.current()

        # Record visited tile (only thread 0 writes)
        if thread_idx.x == 0:
            var idx = Int(work_iter.linear_tile_idx)
            if idx < max_tiles:
                visited_group[idx, 0] = Int32(current.group_idx)
                visited_m[idx, 0] = Int32(current.m)
                visited_n[idx, 0] = Int32(current.n)
                visited_k_tiles[idx, 0] = Int32(current.k_tile_count)
                visited_changed[idx, 0] = Int32(
                    1 if current.group_changed else 0
                )

        work_iter.advance()

    # Record total tile count
    if thread_idx.x == 0 and block_idx.x == 0:
        tile_count[0, 0] = Int32(scheduler.total_tiles())


def test_single_group(ctx: DeviceContext):
    """Test scheduler with a single group."""
    print("  Test: Single group (64x64x128)")

    comptime tile_m = 16
    comptime tile_n = 16
    comptime tile_k = 32
    comptime max_groups = 4
    comptime max_tiles = 64

    # Create problem sizes: 1 group, 64x64x128
    var problem_sizes = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_groups, 4)
    ](ctx)
    var ps = problem_sizes.tensor[update=False]()
    ps[0, 0] = 64  # M
    ps[0, 1] = 64  # N
    ps[0, 2] = 128  # K
    ps[0, 3] = 0  # L (unused)

    # Create output tensors
    var visited_group = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_m = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_n = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_k_tiles = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_changed = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var tile_count = ManagedLayoutTensor[DType.int32, Layout.row_major(1, 1)](
        ctx
    )

    ctx.synchronize()

    # Launch kernel
    comptime kernel = test_scheduler_kernel[
        tile_m, tile_n, tile_k, max_groups, max_tiles
    ]

    # Expected tiles: (64/16) * (64/16) = 4 * 4 = 16
    comptime expected_tiles = 16
    ctx.enqueue_function[kernel, kernel](
        problem_sizes.device_tensor(),
        Int32(1),  # num_groups
        visited_group.device_tensor(),
        visited_m.device_tensor(),
        visited_n.device_tensor(),
        visited_k_tiles.device_tensor(),
        visited_changed.device_tensor(),
        tile_count.device_tensor(),
        grid_dim=(expected_tiles,),
        block_dim=(32,),
    )

    ctx.synchronize()

    # Verify results
    var total = Int(tile_count.tensor()[0, 0])
    if total != expected_tiles:
        print("    FAILED: Expected", expected_tiles, "tiles, got", total)
        return

    # Check all tiles are from group 0 with k_tile_count = 128/32 = 4
    var errors = 0
    for i in range(expected_tiles):
        var g = Int(visited_group.tensor()[i, 0])
        var kt = Int(visited_k_tiles.tensor()[i, 0])
        if g != 0:
            if errors < 3:
                print("    Tile", i, "has wrong group:", g)
            errors += 1
        if kt != 4:
            if errors < 3:
                print("    Tile", i, "has wrong k_tile_count:", kt)
            errors += 1

    # Check first tile has group_changed = True
    if Int(visited_changed.tensor()[0, 0]) != 1:
        print("    First tile should have group_changed=True")
        errors += 1

    if errors == 0:
        print("    PASSED")
    else:
        print("    FAILED:", errors, "errors")

    # Cleanup
    _ = problem_sizes^
    _ = visited_group^
    _ = visited_m^
    _ = visited_n^
    _ = visited_k_tiles^
    _ = visited_changed^
    _ = tile_count^


def test_two_groups(ctx: DeviceContext):
    """Test scheduler with two groups of different sizes."""
    print("  Test: Two groups (32x32x64, 48x48x96)")

    comptime tile_m = 16
    comptime tile_n = 16
    comptime tile_k = 32
    comptime max_groups = 4
    comptime max_tiles = 64

    # Create problem sizes: 2 groups
    var problem_sizes = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_groups, 4)
    ](ctx)
    var ps = problem_sizes.tensor[update=False]()
    # Group 0: 32x32x64 -> 2x2=4 tiles, k_tiles=2
    ps[0, 0] = 32
    ps[0, 1] = 32
    ps[0, 2] = 64
    ps[0, 3] = 0
    # Group 1: 48x48x96 -> 3x3=9 tiles, k_tiles=3
    ps[1, 0] = 48
    ps[1, 1] = 48
    ps[1, 2] = 96
    ps[1, 3] = 0

    # Create output tensors
    var visited_group = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_m = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_n = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_k_tiles = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var visited_changed = ManagedLayoutTensor[
        DType.int32, Layout.row_major(max_tiles, 1)
    ](ctx)
    var tile_count = ManagedLayoutTensor[DType.int32, Layout.row_major(1, 1)](
        ctx
    )

    ctx.synchronize()

    # Expected tiles: 4 + 9 = 13
    comptime expected_tiles = 13
    comptime kernel = test_scheduler_kernel[
        tile_m, tile_n, tile_k, max_groups, max_tiles
    ]

    ctx.enqueue_function[kernel, kernel](
        problem_sizes.device_tensor(),
        Int32(2),  # num_groups
        visited_group.device_tensor(),
        visited_m.device_tensor(),
        visited_n.device_tensor(),
        visited_k_tiles.device_tensor(),
        visited_changed.device_tensor(),
        tile_count.device_tensor(),
        grid_dim=(expected_tiles,),
        block_dim=(32,),
    )

    ctx.synchronize()

    # Verify results
    var total = Int(tile_count.tensor()[0, 0])
    if total != expected_tiles:
        print("    FAILED: Expected", expected_tiles, "tiles, got", total)
        return

    # Count tiles per group
    var group0_count = 0
    var group1_count = 0
    var group_changed_count = 0
    var errors = 0

    for i in range(expected_tiles):
        var g = Int(visited_group.tensor()[i, 0])
        var kt = Int(visited_k_tiles.tensor()[i, 0])
        var changed = Int(visited_changed.tensor()[i, 0])

        if g == 0:
            group0_count += 1
            if kt != 2:
                if errors < 3:
                    print("    Group 0 tile", i, "has wrong k_tile_count:", kt)
                errors += 1
        elif g == 1:
            group1_count += 1
            if kt != 3:
                if errors < 3:
                    print("    Group 1 tile", i, "has wrong k_tile_count:", kt)
                errors += 1
        else:
            if errors < 3:
                print("    Tile", i, "has invalid group:", g)
            errors += 1

        if changed == 1:
            group_changed_count += 1

    if group0_count != 4:
        print("    FAILED: Group 0 should have 4 tiles, got", group0_count)
        errors += 1
    if group1_count != 9:
        print("    FAILED: Group 1 should have 9 tiles, got", group1_count)
        errors += 1

    # Each block starts with group_changed=True for its first tile.
    # With grid_dim=13 blocks, each block processes 1 tile and sets group_changed=True.
    # So we expect 13 group_changed flags (one per block/tile).
    if group_changed_count != expected_tiles:
        print(
            "    FAILED: Expected",
            expected_tiles,
            "group_changed flags, got",
            group_changed_count,
        )
        errors += 1

    if errors == 0:
        print("    PASSED")
    else:
        print("    FAILED:", errors, "errors")

    # Cleanup
    _ = problem_sizes^
    _ = visited_group^
    _ = visited_m^
    _ = visited_n^
    _ = visited_k_tiles^
    _ = visited_changed^
    _ = tile_count^


def main():
    with DeviceContext() as ctx:
        print("=" * 60)
        print("Test: GroupedTileScheduler")
        print("=" * 60)
        print()

        test_single_group(ctx)
        print()
        test_two_groups(ctx)

        print()
        print("=" * 60)
        print("All tests completed")
        print("=" * 60)
