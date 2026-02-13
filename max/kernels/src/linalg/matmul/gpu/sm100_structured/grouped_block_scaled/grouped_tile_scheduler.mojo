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
"""Grouped tile scheduler for SM100 structured block-scaled GEMM.

This scheduler extends the SM100 TileScheduler to support grouped GEMM with
variable problem sizes per group. It uses linear tile iteration instead of CLC
(Cluster Launch Control) to map a global linear tile index to group-specific
coordinates.

Key features:
- GroupedWorkInfo: Extends WorkInfo with group_idx, k_tile_count, group_changed
- delinearize_to_group(): Maps linear tile index to group + local coordinates
- Supports variable M, N, K per group
- Compatible with dynamic tensormap updates

Usage:
    var scheduler = GroupedTileScheduler[...](problem_sizes, tile_shape)
    var work_iter = scheduler.work_iterator()
    while work_iter.has_work():
        with work_iter.next() as current:
            if current.group_changed:
                update_tensormaps(current.group_idx)
            process_tile(current)
"""

from math import ceildiv

from gpu import block_idx, block_id_in_cluster, grid_dim, thread_idx
from gpu.primitives.cluster import block_rank_in_cluster, elect_one_sync
from gpu.memory import fence_async_view_proxy
from layout import Layout, LayoutTensor
from layout.tma_async import PipelineState, SharedMemBarrier
from .grouped_block_scaled_matmul_kernel import _ProblemSizesTile

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.structuring import SMemPtr, SMemArray
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from ..structured_kernels.pipeline import ProducerConsumerPipeline


# =============================================================================
# GroupedAdvanceContext - Context manager for advancing work iterator
# =============================================================================


struct GroupedAdvanceContext[
    work_origin: MutOrigin,
    idx_origin: MutOrigin,
](TrivialRegisterPassable):
    """Context manager that returns current work and advances on exit.

    This follows the same pattern as the working kernel's WaitAndAdvanceContext:
    - Pre-compute next work during construction
    - __enter__ returns current work for processing
    - __exit__ assigns pre-computed next work and updates linear index

    Usage:
        with work_iter.next() as current:
            # Process current work
        # After: work_iter.work_info updated to next work
    """

    var work_info_ptr: Pointer[GroupedWorkInfo, Self.work_origin]
    var linear_idx_ptr: Pointer[UInt32, Self.idx_origin]
    var next_work: GroupedWorkInfo
    var next_linear_idx: UInt32

    @always_inline
    fn __init__(
        out self,
        work_info_ptr: Pointer[GroupedWorkInfo, Self.work_origin],
        linear_idx_ptr: Pointer[UInt32, Self.idx_origin],
        next_work: GroupedWorkInfo,
        next_linear_idx: UInt32,
    ):
        self.work_info_ptr = work_info_ptr
        self.linear_idx_ptr = linear_idx_ptr
        self.next_work = next_work
        self.next_linear_idx = next_linear_idx

    @always_inline
    fn __enter__(self) -> GroupedWorkInfo:
        return self.work_info_ptr[]

    @always_inline
    fn __exit__(mut self):
        self.work_info_ptr[] = self.next_work
        self.linear_idx_ptr[] = self.next_linear_idx


# =============================================================================
# GroupedWorkInfo - Extended work info for grouped GEMM
# =============================================================================


@fieldwise_init
struct GroupedWorkInfo(
    ImplicitlyCopyable, Movable, Stringable, TrivialRegisterPassable, Writable
):
    """Work info for grouped GEMM with group-specific metadata.

    Extends the base WorkInfo with:
    - group_idx: Current group index
    - k_tile_count: Number of K tiles for this group
    - group_changed: True if group changed since last tile (triggers tensormap update)
    """

    # Base coordinates (compatible with WorkInfo)
    var m: UInt32
    """M-coordinate of tile within current group."""
    var n: UInt32
    """N-coordinate of tile within current group."""
    var k_start: UInt32
    """Starting K index (always 0 for grouped GEMM)."""
    var is_valid_tile: Bool
    """Whether this work tile is valid (not OOB)."""

    # Grouped extensions
    var group_idx: UInt32
    """Current group index."""
    var k_tile_count: UInt32
    """Number of K tiles for this group."""
    var group_changed: Bool
    """True if group changed since last tile (triggers tensormap update)."""

    @always_inline
    fn __init__(out self):
        """Create an invalid/empty work info."""
        self.m = 0
        self.n = 0
        self.k_start = 0
        self.is_valid_tile = False
        self.group_idx = 0
        self.k_tile_count = 0
        self.group_changed = False

    @always_inline
    fn is_valid(self) -> Bool:
        """Check if this work tile is valid."""
        return self.is_valid_tile

    @always_inline
    fn coord(self) -> Tuple[UInt, UInt]:
        """Get (m, n) tile coordinates as a tuple."""
        return (UInt(self.m), UInt(self.n))

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GroupedWorkInfo(m=",
            self.m,
            ", n=",
            self.n,
            ", k_start=",
            self.k_start,
            ", valid=",
            self.is_valid_tile,
            ", group=",
            self.group_idx,
            ", k_tiles=",
            self.k_tile_count,
            ", changed=",
            self.group_changed,
            ")",
        )


# =============================================================================
# GroupedWorkIterator - Per-warp iterator for grouped GEMM
# =============================================================================


struct GroupedWorkIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Per-warp work iterator for grouped GEMM.

    This iterator traverses tiles across all groups, tracking when groups change
    to trigger tensormap updates. It uses linear iteration instead of CLC.

    For 2SM (cta_group=2), both CTAs in a cluster work on the same logical tile.
    The cluster index (block_idx.x // cta_group) is used for tile assignment,
    and advance step is grid_dim.x // cta_group (number of clusters).

    Usage:
        var work_iter = scheduler.work_iterator()
        while work_iter.has_work():
            var current = work_iter.current()
            if current.group_changed:
                update_tensormaps(current.group_idx)
            process_tile(current)
            work_iter.advance()
    """

    var work_info: GroupedWorkInfo
    """Current work item."""
    var linear_tile_idx: UInt32
    """Current linear tile index (across all groups)."""
    var total_tiles: UInt32
    """Total number of tiles across all groups."""
    var prev_group_idx: UInt32
    """Previous group index for detecting group changes."""

    # Group metadata (cumulative tile counts)
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    """Cumulative tile count at the start of each group."""
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    """M dimension for each group."""
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    """N dimension for each group."""
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    """K dimension for each group."""
    var num_groups: UInt32
    """Number of active groups."""

    @always_inline
    fn __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        grid_size: UInt32,
    ):
        """Initialize work iterator with problem sizes.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
            grid_size: Number of blocks in the grid.
        """
        # Initialize all fields first to satisfy compiler
        self.work_info = GroupedWorkInfo()
        self.linear_tile_idx = UInt32(0)
        self.total_tiles = UInt32(0)
        self.prev_group_idx = UInt32(0)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()
        self.num_groups = UInt32(num_groups)

        # Compute cumulative tile counts
        # Explicitly zero-initialize ALL slots to avoid stale memory issues
        @parameter
        for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        # cumulative_tiles[0] is already 0 from the loop above

        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))

            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k

            # Compute tiles for this group
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            var group_tiles = UInt32(m_tiles * n_tiles)
            cumsum += group_tiles
            self.cumulative_tiles[g + 1] = cumsum

        # Initialize remaining slots
        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

        # Start at this cluster's first tile
        # For 2SM (cta_group=2), both CTAs in a cluster work on the same tile
        # Use cluster index = block_idx.x // cta_group
        self.linear_tile_idx = UInt32(block_idx.x // UInt(Self.cta_group))

        # Delinearize initial position
        self.work_info = self._delinearize_to_group(self.linear_tile_idx)
        self.work_info.group_changed = True  # First tile always triggers update

    @always_inline
    fn has_work(self) -> Bool:
        """Check if there is more work to process."""
        return self.work_info.is_valid()

    @always_inline
    fn current(self) -> GroupedWorkInfo:
        """Get current work item."""
        return self.work_info

    @always_inline
    fn advance(mut self):
        """Advance to next tile."""
        self.prev_group_idx = self.work_info.group_idx

        # For 2SM, advance by number of clusters (grid_dim.x // cta_group)
        self.linear_tile_idx += UInt32(grid_dim.x // UInt(Self.cta_group))

        if self.linear_tile_idx >= self.total_tiles:
            self.work_info = GroupedWorkInfo()  # Invalid
            return

        self.work_info = self._delinearize_to_group(self.linear_tile_idx)
        self.work_info.group_changed = (
            self.work_info.group_idx != self.prev_group_idx
        )

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> GroupedAdvanceContext[
        origin_of(self.work_info), origin_of(self.linear_tile_idx)
    ]:
        """Get context manager that returns current work and advances on exit.

        Compatible with the working kernel's pattern:
            with work_iter.next() as current:
                process_tile(current)
            # After: work_iter.work_info updated to next work

        Pre-computes next state, then on __exit__ updates work_info and linear_idx.
        """
        # Pre-compute next state (read-only computation)
        # For 2SM, advance by number of clusters (grid_dim.x // cta_group)
        var advance_step = UInt32(grid_dim.x // UInt(Self.cta_group))
        var next_linear_idx = self.linear_tile_idx + advance_step
        var next_work: GroupedWorkInfo

        if next_linear_idx >= self.total_tiles:
            next_work = GroupedWorkInfo()  # Invalid
        else:
            next_work = self._delinearize_to_group(next_linear_idx)
            next_work.group_changed = (
                next_work.group_idx != self.work_info.group_idx
            )

        return GroupedAdvanceContext(
            Pointer(to=self.work_info),
            Pointer(to=self.linear_tile_idx),
            next_work,
            next_linear_idx,
        )

    @always_inline
    fn wait_and_advance[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> GroupedAdvanceContext[
        origin_of(self.work_info), origin_of(self.linear_tile_idx)
    ]:
        """Same as next() - no CLC waiting for grouped GEMM.

        For compatibility with MMA warp pattern. Since we don't use CLC,
        this behaves identically to next().
        """
        # Pre-compute next state (same as next() for grouped GEMM)
        # For 2SM, advance by number of clusters (grid_dim.x // cta_group)
        var advance_step = UInt32(grid_dim.x // UInt(Self.cta_group))
        var next_linear_idx = self.linear_tile_idx + advance_step
        var next_work: GroupedWorkInfo

        if next_linear_idx >= self.total_tiles:
            next_work = GroupedWorkInfo()  # Invalid
        else:
            next_work = self._delinearize_to_group(next_linear_idx)
            next_work.group_changed = (
                next_work.group_idx != self.work_info.group_idx
            )

        return GroupedAdvanceContext(
            Pointer(to=self.work_info),
            Pointer(to=self.linear_tile_idx),
            next_work,
            next_linear_idx,
        )

    @always_inline
    fn _delinearize_to_group(self, linear_idx: UInt32) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates.

        Uses binary search to find the group containing this tile index.
        """
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups

        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        # Local tile index within group
        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]

        # Get group dimensions
        var m = self.problem_m[Int(group_idx)]
        var n = self.problem_n[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]

        var m_tiles = ceildiv(Int(m), Self.tile_m)
        _ = ceildiv(Int(n), Self.tile_n)  # n_tiles used only for validation
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        # Convert to M, N tile coordinates (row-major within group)
        # These are tile indices, not global coordinates
        # (load_input_tiles will multiply by BM/BN to get global coords)
        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,  # Tile index, not global coordinate
            n=n_tile,  # Tile index, not global coordinate
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=False,  # Caller sets this
        )


# =============================================================================
# GroupedTileScheduler - Main scheduler for grouped GEMM
# =============================================================================


struct GroupedTileScheduler[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_stages: Int = 0,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Tile scheduler for grouped block-scaled GEMM.

    Uses linear tile iteration to map tiles across groups. Does not use CLC
    (Cluster Launch Control) since work distribution is deterministic.

    Parameters:
        tile_m: M dimension of output tiles.
        tile_n: N dimension of output tiles.
        tile_k: K dimension of input tiles.
        max_groups: Maximum number of groups.
        num_stages: Pipeline stages (0 = single wave).
        cta_group: Number of CTAs cooperating per tile (1 or 2 for 2SM).
    """

    var num_groups: Int
    """Number of active groups."""
    var problem_sizes: _ProblemSizesTile[Self.max_groups]
    """Problem sizes tensor (num_groups, 4) with [M, N, K, L] per group."""

    @always_inline
    fn __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
    ):
        """Initialize scheduler with problem sizes.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
        """
        self.problem_sizes = problem_sizes
        self.num_groups = num_groups

    @always_inline
    fn work_iterator(
        self,
    ) -> GroupedWorkIterator[
        Self.tile_m, Self.tile_n, Self.tile_k, Self.max_groups, Self.cta_group
    ]:
        """Create a per-warp work iterator.

        Each warp should create its own work iterator. The iterator owns
        work_info and cumulative tile counts internally.

        For 2SM (cta_group=2), the iterator uses cluster-based indexing.
        """
        return GroupedWorkIterator[
            Self.tile_m,
            Self.tile_n,
            Self.tile_k,
            Self.max_groups,
            Self.cta_group,
        ](
            self.problem_sizes,
            self.num_groups,
            UInt32(grid_dim.x),
        )

    @always_inline
    fn total_tiles(self) -> Int:
        """Compute total number of tiles across all groups."""
        var total = 0
        for g in range(self.num_groups):
            var m = Int(self.problem_sizes[g, 0])
            var n = Int(self.problem_sizes[g, 1])
            var m_tiles = ceildiv(m, Self.tile_m)
            var n_tiles = ceildiv(n, Self.tile_n)
            total += m_tiles * n_tiles
        return total


# =============================================================================
# GroupedCLCWaitAndAdvanceContext - Context for CLC-based wait and advance
# =============================================================================


struct GroupedCLCWaitAndAdvanceContext[
    work_origin: MutOrigin,
](TrivialRegisterPassable):
    """Context for waiting on CLC barrier and advancing work iterator.

    Encapsulates CLC response barrier synchronization:
    - Construction: Waits for CLC response, fetches next work
    - __enter__: Returns current work_info for processing
    - __exit__: Assigns fetched work as current

    Usage:
        with work_iter.wait_and_advance() as current:
            # current is the work item to process NOW
            process(current)
        # After exit, work_iter.work_info is the NEXT work item
    """

    var work_info_ptr: Pointer[GroupedWorkInfo, Self.work_origin]
    var next_work: GroupedWorkInfo

    @always_inline
    fn __init__(
        out self,
        work_info_ptr: Pointer[GroupedWorkInfo, Self.work_origin],
        next_work: GroupedWorkInfo,
    ):
        self.work_info_ptr = work_info_ptr
        self.next_work = next_work

    @always_inline
    fn __enter__(self) -> GroupedWorkInfo:
        return self.work_info_ptr[]

    @always_inline
    fn __exit__(mut self):
        self.work_info_ptr[] = self.next_work


# =============================================================================
# GroupedCLCWorkIterator - Per-warp iterator with CLC barrier support
# =============================================================================


struct GroupedCLCWorkIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_clc_stages: Int,
    cta_group: Int = 2,
](TrivialRegisterPassable):
    """Per-warp work iterator for grouped GEMM with CLC barrier support.

    This iterator combines grouped GEMM features with CLC-based synchronization
    for 2SM support. It uses CLC barriers to ensure both CTAs in a cluster
    process the same tile at the same time.

    Key features:
    - Uses CLC barriers for inter-CTA synchronization (like working kernel)
    - Tracks group_idx, k_tile_count, group_changed (like grouped scheduler)
    - wait_and_advance() actually waits on CLC barriers

    Usage:
        var work_iter = scheduler.clc_work_iterator()
        while work_iter.has_work():
            with work_iter.wait_and_advance() as current:
                if current.group_changed:
                    update_tensormaps(current.group_idx)
                process_tile(current)
    """

    comptime ThrottlePipeline = ProducerConsumerPipeline[Self.num_clc_stages]

    var work_info: GroupedWorkInfo
    """Current work item."""
    var consumer_state: PipelineState[Self.num_clc_stages]
    """CLC consumer pipeline state."""
    var throttle_pipeline: Self.ThrottlePipeline
    """Throttle pipeline for load/scheduler sync."""

    # CLC barrier pointers
    var full_mbar: SMemPtr[SharedMemBarrier]
    """CLC full barriers (signaled by scheduler when work is ready)."""
    var empty_mbar: SMemPtr[SharedMemBarrier]
    """CLC empty barriers (signaled by workers when done)."""
    var clc_response: SMemPtr[UInt128]
    """CLC response storage (contains work info)."""

    # Group metadata (cumulative tile counts)
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    """Cumulative tile count at the start of each group."""
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    """M dimension for each group."""
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    """N dimension for each group."""
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    """K dimension for each group."""
    var num_groups: UInt32
    """Number of active groups."""
    var total_tiles: UInt32
    """Total tiles across all groups."""

    @always_inline
    fn __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        full_mbar: SMemPtr[SharedMemBarrier],
        empty_mbar: SMemPtr[SharedMemBarrier],
        clc_response: SMemPtr[UInt128],
        throttle_ptr: SMemPtr[SharedMemBarrier],
        initial_work: GroupedWorkInfo,
    ):
        """Initialize CLC work iterator.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
            full_mbar: CLC full barrier pointer.
            empty_mbar: CLC empty barrier pointer.
            clc_response: CLC response storage pointer.
            throttle_ptr: Throttle pipeline barrier pointer.
            initial_work: Initial work item (first tile).
        """
        self.work_info = initial_work
        self.consumer_state = PipelineState[Self.num_clc_stages]()
        self.throttle_pipeline = Self.ThrottlePipeline(throttle_ptr)
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.clc_response = clc_response
        self.num_groups = UInt32(num_groups)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()

        # Initialize cumulative tiles
        @parameter
        for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))
            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            cumsum += UInt32(m_tiles * n_tiles)
            self.cumulative_tiles[g + 1] = cumsum

        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

    @always_inline
    fn has_work(self) -> Bool:
        """Check if there is more work to process."""
        return self.work_info.is_valid()

    @always_inline
    fn wait_and_advance[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> GroupedCLCWaitAndAdvanceContext[
        origin_of(self.work_info)
    ]:
        """Wait for next work from CLC and advance iterator.

        This method waits on CLC full barriers to synchronize all CTAs
        in the cluster before advancing to the next work item.

        Usage:
            with work_iter.wait_and_advance() as current:
                # Process current work item
            # After exit, work_iter points to next work
        """
        var next = self._fetch_next_work()
        self.consumer_state.step()
        return GroupedCLCWaitAndAdvanceContext(Pointer(to=self.work_info), next)

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> GroupedAdvanceContext[
        origin_of(self.work_info), origin_of(self.total_tiles)
    ]:
        """Get context manager for advance-after-work pattern.

        Does NOT wait on CLC - use wait_and_advance() for MMA warp.
        """
        # For non-MMA warps, just advance without CLC wait
        var next_linear_idx = UInt32(0)  # Placeholder, not used
        var next_work = self._compute_next_work()

        return GroupedAdvanceContext(
            Pointer(to=self.work_info),
            Pointer(to=self.total_tiles),  # Placeholder
            next_work,
            next_linear_idx,
        )

    @always_inline
    fn throttle_signal(mut self, is_first_cta_in_cluster: Bool):
        """Signal CLC throttle if this is the first CTA in cluster.

        NOTE: For software CLC simulation, this is a no-op. The throttle
        pattern causes a deadlock because both Scheduler and TMA Load wait
        on each other's barriers on the first iteration. The CLC full/empty
        barriers provide sufficient synchronization without the throttle.

        Args:
            is_first_cta_in_cluster: Only first CTA signals to avoid duplicates.
        """
        # No-op for software CLC - throttle causes deadlock
        pass

    @always_inline
    fn _fetch_next_work(self) -> GroupedWorkInfo:
        """Fetch next work item with CLC barrier synchronization.

        This is the key synchronization point - waits on CLC full barrier
        to ensure all CTAs receive the same work at the same time.

        Note: For 2SM clusters, each CTA has separate SMEM so we can't read
        the clc_response written by the scheduler (CTA 0). Instead, all CTAs
        compute work locally (they get the same result since they use the same
        formula), and use barriers just for synchronization.
        """
        # Wait for scheduler to signal work is ready
        self.full_mbar[self.consumer_state.index()].wait(
            self.consumer_state.phase()
        )

        # Compute next work locally instead of reading from CLC response
        # (CLC response is in CTA 0's SMEM, not accessible to other CTAs)
        var linear_idx = self._current_linear_idx() + UInt32(
            grid_dim.x // UInt(Self.cta_group)
        )

        fence_async_view_proxy()

        # Signal that we've consumed this work item (all CTAs arrive on CTA 0)
        self.empty_mbar[self.consumer_state.index()].arrive_cluster(0)

        # Delinearize to grouped work info
        return self._delinearize_to_group(linear_idx, self.work_info.group_idx)

    @always_inline
    fn _read_linear_idx_from_clc(self) -> UInt32:
        """Read linear tile index from CLC response.

        The CLC response contains: (linear_idx, 0, 0, is_valid).
        """
        var response_ptr = self.clc_response + self.consumer_state.index()
        # Read the 128-bit response and extract linear_idx (first 32 bits)
        var response = response_ptr[].cast[DType.uint32]()
        return response
        # Note: The working kernel uses inline assembly here, but for simplicity
        # we just cast the UInt128 to UInt32 to get the first component

    @always_inline
    fn _compute_next_work(self) -> GroupedWorkInfo:
        """Compute next work item without CLC wait (for non-MMA warps)."""
        # Simple linear advance
        var linear_idx = self._current_linear_idx() + UInt32(
            grid_dim.x // UInt(Self.cta_group)
        )
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()  # Invalid
        return self._delinearize_to_group(linear_idx, self.work_info.group_idx)

    @always_inline
    fn _current_linear_idx(self) -> UInt32:
        """Compute current linear tile index from work_info."""
        var g = Int(self.work_info.group_idx)
        var m_tiles = ceildiv(Int(self.problem_m[g]), Self.tile_m)
        return (
            self.cumulative_tiles[g]
            + self.work_info.n * UInt32(m_tiles)
            + self.work_info.m
        )

    @always_inline
    fn _delinearize_to_group(
        self, linear_idx: UInt32, prev_group_idx: UInt32
    ) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates."""
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        # Local tile index within group
        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]

        # Get group dimensions
        var m = self.problem_m[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]
        var m_tiles = ceildiv(Int(m), Self.tile_m)
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=(group_idx != prev_group_idx),
        )


# =============================================================================
# GroupedCLCSchedulerIterator - For scheduler warp with CLC
# =============================================================================


struct GroupedCLCSchedulerIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_clc_stages: Int,
    cta_group: Int = 2,
](TrivialRegisterPassable):
    """Scheduler warp iterator for grouped GEMM with CLC.

    The scheduler warp produces work items for other warps via CLC.
    It iterates through all tiles across all groups and signals CLC barriers.

    Usage:
        var sched_iter = scheduler.scheduler_iterator()
        while sched_iter.has_work():
            with sched_iter.next():
                sched_iter.signal_and_advance()
        sched_iter.drain()
    """

    comptime ThrottlePipeline = ProducerConsumerPipeline[Self.num_clc_stages]

    var work_info: GroupedWorkInfo
    """Current work item."""
    var linear_tile_idx: UInt32
    """Current linear tile index."""
    var consumer_state: PipelineState[Self.num_clc_stages]
    var producer_state: PipelineState[Self.num_clc_stages]
    var throttle_pipeline: Self.ThrottlePipeline

    # CLC barrier pointers
    var full_mbar: SMemPtr[SharedMemBarrier]
    var empty_mbar: SMemPtr[SharedMemBarrier]
    var clc_response: SMemPtr[UInt128]

    # Group metadata
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    var num_groups: UInt32
    var total_tiles: UInt32
    var signal_count: UInt32
    """Number of signals sent (for pipeline fill tracking)."""

    @always_inline
    fn __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        full_mbar: SMemPtr[SharedMemBarrier],
        empty_mbar: SMemPtr[SharedMemBarrier],
        clc_response: SMemPtr[UInt128],
        throttle_ptr: SMemPtr[SharedMemBarrier],
        initial_work: GroupedWorkInfo,
    ):
        """Initialize scheduler iterator."""
        self.work_info = initial_work
        # Each cluster starts at its own linear tile index
        # block_idx.x // cta_group gives the cluster's starting tile
        self.linear_tile_idx = UInt32(Int(block_idx.x) // Self.cta_group)
        self.consumer_state = PipelineState[Self.num_clc_stages]()
        self.producer_state = PipelineState[Self.num_clc_stages](0, 1, 0)
        self.throttle_pipeline = Self.ThrottlePipeline(throttle_ptr)
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.clc_response = clc_response
        self.num_groups = UInt32(num_groups)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()
        self.signal_count = UInt32(0)

        # Initialize cumulative tiles
        @parameter
        for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))
            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            cumsum += UInt32(m_tiles * n_tiles)
            self.cumulative_tiles[g + 1] = cumsum

        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

    @always_inline
    fn has_work(self) -> Bool:
        """Check if there is more work to process."""
        return self.work_info.is_valid()

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> GroupedAdvanceContext[
        origin_of(self.work_info), origin_of(self.linear_tile_idx)
    ]:
        """Get context manager for advance-after-work pattern."""
        # For 2SM: advance by number of clusters (each cluster processes different tiles)
        var num_clusters = UInt32(grid_dim.x // UInt(Self.cta_group))
        var next_linear_idx = self.linear_tile_idx + num_clusters
        var next_work: GroupedWorkInfo

        if next_linear_idx >= self.total_tiles:
            next_work = GroupedWorkInfo()
        else:
            next_work = self._delinearize_to_group(
                next_linear_idx, self.work_info.group_idx
            )

        return GroupedAdvanceContext(
            Pointer(to=self.work_info),
            Pointer(to=self.linear_tile_idx),
            next_work,
            next_linear_idx,
        )

    @always_inline
    fn signal_and_advance(mut self):
        """Signal CLC throttle and produce next work request.

        This is called inside the work loop after processing current work.
        It signals that we've consumed the throttle and produces the next
        work item for all CTAs.

        NOTE: We skip the throttle_pipeline.consumer_signal_and_step() call
        that the hardware CLC version uses. For software CLC simulation,
        the clc_full/clc_empty barriers provide sufficient synchronization.
        The throttle pattern causes a deadlock because:
        - Scheduler waits for TMA Load via throttle full barrier
        - TMA Load waits for Scheduler via throttle empty barrier
        - Both block on first iteration since barriers start at phase 0
        """

        # Produce next work item: write linear_idx to CLC response
        # For 2SM: advance by number of clusters (each cluster processes different tiles)
        # Always signal, even for "no more work" (consumer detects via total_tiles)
        var num_clusters = grid_dim.x // UInt(Self.cta_group)
        var next_linear_idx = self.linear_tile_idx + UInt32(num_clusters)

        # Wait for empty signal (consumers done with previous)
        # Skip wait during pipeline fill phase (first num_clc_stages iterations)
        # to avoid deadlock - no previous data to wait for yet.
        if self.signal_count >= UInt32(Self.num_clc_stages):
            self.empty_mbar[self.producer_state.index()].wait(
                self.producer_state.phase()
            )

        # Write next linear index to CLC response (only one thread writes)
        # If next_linear_idx >= total_tiles, consumer will detect "no more work"
        # Signal full (work is ready) to all CTAs in the cluster
        # For 2SM: scheduler runs on CTA 0 but must signal all CTAs' barriers
        # via arrive_cluster so each CTA's MMA warp can proceed
        #
        # CRITICAL: Use elect_one_sync() to ensure only ONE thread in the warp
        # signals the barriers. Without this, all 32 threads would each do
        # arrive_cluster(), causing 32 arrivals on barriers expecting only 1.
        if elect_one_sync():
            var response_ptr = self.clc_response + self.producer_state.index()
            response_ptr[] = UInt128(Int(next_linear_idx))

            @parameter
            for cta in range(Self.cta_group):
                self.full_mbar[self.producer_state.index()].arrive_cluster(
                    UInt32(cta)
                )
        self.signal_count += 1

        self.producer_state.step()

        # Update iterator state so has_work() eventually returns False
        self.linear_tile_idx = next_linear_idx
        if next_linear_idx >= self.total_tiles:
            self.work_info = GroupedWorkInfo()  # Invalid - no more work
        else:
            self.work_info = self._delinearize_to_group(
                next_linear_idx, self.work_info.group_idx
            )

    @always_inline
    fn drain(mut self):
        """Drain all pending CLC requests before kernel exit.

        Only waits for slots that were actually signaled to avoid deadlock
        when workload is smaller than pipeline depth.

        Note: After signaling, producer_state has stepped to the NEXT stage.
        We need to wait on stages 0..slots_to_drain-1, not from producer_state.
        """
        # Number of slots actually used is min(signals_made, num_stages)
        var slots_to_drain = min(Int(self.signal_count), Self.num_clc_stages)

        # Drain stages starting from 0 (the first stage we used)
        # Phase alternates: stage 0 uses phase 0, stages wrap at num_stages
        for i in range(slots_to_drain):
            var stage = i % Self.num_clc_stages
            # Phase is 0 for first num_stages iterations, then 1, etc.
            var phase = UInt32(i // Self.num_clc_stages) & 1
            self.empty_mbar[stage].wait(phase)

    @always_inline
    fn _delinearize_to_group(
        self, linear_idx: UInt32, prev_group_idx: UInt32
    ) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates."""
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]
        var m = self.problem_m[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]
        var m_tiles = ceildiv(Int(m), Self.tile_m)
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=(group_idx != prev_group_idx),
        )
