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
"""Work scheduler for grouped 1D-1D block-scaled SM100 matmul.

Provides work iteration using offset-based addressing for the 1D-1D tensor layout.
This is a port of the TileScheduler from grouped_matmul_tile_scheduler.mojo
to the structured kernels architecture with context manager patterns.

Key characteristics:
- Uses a_offsets tensor for group boundaries (prefix sum of token counts)
- Each iteration returns (m_coord, n_coord, expert_id, expert_scale)
- Supports block swizzling for L2 cache efficiency
- 3-warp specialization (no scheduler warp)
"""

from math import ceildiv

from gpu import block_idx, grid_dim, thread_idx
from layout import Layout, LayoutTensor, RuntimeLayout
from layout._tile_tensor import TileTensor

from ..structured_kernels.tile_types import GMEMLayout1D
from memory import UnsafePointer

from utils.fast_div import FastDiv
from utils.index import Index, IndexList


# ===----------------------------------------------------------------------=== #
# Work Info for 1D-1D Grouped Matmul
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GroupedWorkInfo1D1D(Stringable, TrivialRegisterPassable, Writable):
    """Work tile information for 1D-1D grouped matmul.

    Contains the coordinates and metadata for a single work tile:
    - m, n: Output tile coordinates (m is in contiguous token space)
    - group_idx: Index into active experts (for a_offsets indexing)
    - expert_id: The actual expert ID for B tensor lookup
    - is_valid_tile: Whether this tile contains valid work
    - terminate: Whether the scheduler has no more work
    """

    var m: UInt32
    var n: UInt32
    var group_idx: UInt32
    var expert_id: Int32
    var is_valid_tile: Bool
    var terminate: Bool

    @always_inline
    fn __init__(out self):
        self.m = 0
        self.n = 0
        self.group_idx = 0
        self.expert_id = 0
        self.is_valid_tile = False
        self.terminate = False

    @always_inline
    fn is_valid(self) -> Bool:
        """Returns True if this work tile has valid work to do."""
        return self.is_valid_tile

    @always_inline
    fn is_done(self) -> Bool:
        """Returns True if the scheduler has no more work."""
        return self.terminate

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GroupedWorkInfo1D1D(m=",
            self.m,
            ", n=",
            self.n,
            ", group_idx=",
            self.group_idx,
            ", expert_id=",
            self.expert_id,
            ", valid=",
            self.is_valid_tile,
            ", terminate=",
            self.terminate,
            ")",
        )


# ===----------------------------------------------------------------------=== #
# Work Context for Context Manager Pattern
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GroupedWorkContext1D1D(ImplicitlyCopyable, Movable):
    """Context for current work tile, used with context manager pattern.

    Provides access to work tile info and expert scale factor.
    """

    var info: GroupedWorkInfo1D1D
    var expert_scale: Float32
    var m_end: UInt32  # End offset for bounds checking (exclusive upper bound)

    @always_inline
    fn m(self) -> UInt32:
        """M coordinate in contiguous token space."""
        return self.info.m

    @always_inline
    fn n(self) -> UInt32:
        """N coordinate in output space."""
        return self.info.n

    @always_inline
    fn group_idx(self) -> UInt32:
        """Index into active experts list."""
        return self.info.group_idx

    @always_inline
    fn expert_id(self) -> Int32:
        """Expert ID for B tensor indexing."""
        return self.info.expert_id

    @always_inline
    fn is_valid(self) -> Bool:
        """Whether this tile has valid work."""
        return self.info.is_valid()


# ===----------------------------------------------------------------------=== #
# Work Iterator for 1D-1D Grouped Matmul
# ===----------------------------------------------------------------------=== #


struct GroupedWorkIterator1D1D[
    static_N: Int,  # N dimension (expert output dim, static)
    tile_shape: IndexList[3],  # Block tile shape (BM, BN, BK)
    cluster: IndexList[3] = Index(1, 1, 1),
    cta_group: Int = 1,
    swizzle: Bool = False,
]:
    """Work iterator for 1D-1D grouped block-scaled matmul.

    Iterates through work tiles using offset-based addressing:
    - a_offsets: Prefix sum of token counts per active expert
    - expert_ids: Mapping from active expert index to actual expert ID
    - expert_scales: Per-expert output scaling factors
    """

    # 1D TileTensor types: dynamic shape, stride 1 (flat arrays)
    comptime OffsetsTile = TileTensor[DType.uint32, GMEMLayout1D, MutAnyOrigin]
    comptime ExpertIdsTile = TileTensor[DType.int32, GMEMLayout1D, MutAnyOrigin]
    comptime ExpertScalesTile = TileTensor[
        DType.float32, GMEMLayout1D, MutAnyOrigin
    ]

    var num_active_experts: Int
    var group_offsets: Self.OffsetsTile
    var expert_ids: Self.ExpertIdsTile
    var expert_scales: Self.ExpertScalesTile

    # Iteration state
    var current_iter: Int32
    var current_group_idx: UInt32
    var current_dynamic_dim_cumsum: UInt32
    var block_idx_start: UInt32

    # Derived constants
    comptime cta_group_tile_shape = Index(
        Self.tile_shape[0] * Self.cta_group, Self.tile_shape[1]
    )
    comptime div_dynamic_block = FastDiv[DType.uint32](
        Self.cta_group_tile_shape[0]  # M dimension is dynamic
    )
    comptime num_static_dim_blocks: UInt32 = UInt32(
        ceildiv(Self.static_N, Self.cta_group_tile_shape[1])
    )
    comptime kNum1DBlocksPerGroup: UInt32 = 16

    @always_inline
    fn __init__(
        out self,
        num_active_experts: Int,
        group_offsets: Self.OffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
    ):
        comptime assert (
            Self.cluster[1] == Self.cluster[2] == 1
        ), "Currently multicasting along non-M dimension is not supported"
        comptime assert Self.cta_group == Self.cluster[0], (
            "cta_group must be equal to cluster M size. Got cta_group = "
            + String(Self.cta_group)
            + " and cluster M size = "
            + String(Self.cluster[0])
        )

        self.num_active_experts = num_active_experts
        self.group_offsets = group_offsets
        self.expert_ids = expert_ids
        self.expert_scales = expert_scales
        self.current_iter = -1
        self.current_group_idx = 0
        self.current_dynamic_dim_cumsum = 0
        self.block_idx_start = 0

    @always_inline
    fn next(mut self) -> GroupedWorkContext1D1D:
        """Fetch next work tile and return context with work info and scale."""
        var info, m_end = self._fetch_next_work()
        var expert_scale: Float32 = 1.0
        if info.is_valid():
            expert_scale = rebind[Scalar[DType.float32]](
                self.expert_scales[Int(info.expert_id)]
            )
        return GroupedWorkContext1D1D(info, expert_scale, m_end)

    @always_inline
    fn _fetch_next_work(mut self) -> Tuple[GroupedWorkInfo1D1D, UInt32]:
        """Internal method to compute next work tile."""
        self.current_iter += 1
        # Normalize by cta_group so all CTAs in a cluster get the same
        # work tile.  For cta_group==1 this is a no-op.
        var next_block_idx = UInt32(self.current_iter) * UInt32(
            grid_dim.x // Scalar[DType.uint](Self.cta_group)
        ) + UInt32(block_idx.x // Scalar[DType.uint](Self.cta_group))
        var start_idx = rebind[Scalar[DType.uint32]](
            self.group_offsets[Int(self.current_group_idx)]
        )
        var end_idx: UInt32 = 0
        var num_dynamic_dim_blocks: UInt32 = 0
        var current_dynamic_dim: UInt32 = 0

        # Advance to the correct group
        while True:
            if self.current_group_idx >= UInt32(self.num_active_experts):
                # Finished all groups
                return (GroupedWorkInfo1D1D(0, 0, 0, 0, False, True), UInt32(0))

            end_idx = rebind[Scalar[DType.uint32]](
                self.group_offsets[Int(self.current_group_idx + 1)]
            )
            current_dynamic_dim = end_idx - start_idx
            num_dynamic_dim_blocks = UInt32(
                rebind[Scalar[Self.div_dynamic_block.uint_type]](
                    current_dynamic_dim
                    + UInt32(Self.cta_group_tile_shape[0] - 1)
                )
                / Self.div_dynamic_block
            )
            var current_dynamic_dim_block_cumsum = (
                self.current_dynamic_dim_cumsum + num_dynamic_dim_blocks
            )
            var current_dynamic_dim_block_idx_start = (
                current_dynamic_dim_block_cumsum * Self.num_static_dim_blocks
            )
            if next_block_idx < current_dynamic_dim_block_idx_start:
                break
            self.current_group_idx += 1
            self.current_dynamic_dim_cumsum = current_dynamic_dim_block_cumsum
            self.block_idx_start = current_dynamic_dim_block_idx_start
            start_idx = end_idx

        var group_local_block_idx = next_block_idx - self.block_idx_start
        var is_valid = (
            group_local_block_idx
            < num_dynamic_dim_blocks * Self.num_static_dim_blocks
        )
        if not is_valid:
            return (
                GroupedWorkInfo1D1D(
                    0, 0, self.current_group_idx, 0, False, False
                ),
                end_idx,
            )

        # Get expert_id for this group
        var expert_id = rebind[Scalar[DType.int32]](
            self.expert_ids[Int(self.current_group_idx)]
        )

        # Compute swizzled block indices
        var num_n_blocks = Self.num_static_dim_blocks
        var m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            num_n_blocks, group_local_block_idx, num_dynamic_dim_blocks
        )

        # Compute actual coordinates
        # M is in contiguous token space, offset by start_idx
        var m = m_block_idx * UInt32(Self.cta_group_tile_shape[0]) + start_idx
        var n = n_block_idx * UInt32(Self.cta_group_tile_shape[1])

        return (
            GroupedWorkInfo1D1D(
                m,
                n,
                self.current_group_idx,
                expert_id,
                True,
                False,
            ),
            end_idx,
        )

    @always_inline
    fn _get_swizzled_block_idx(
        self,
        num_n_blocks: UInt32,
        _block_idx: UInt32,
        num_dynamic_dim_blocks: UInt32,
    ) -> Tuple[UInt32, UInt32]:
        """Compute swizzled (m_block_idx, n_block_idx) for L2 cache efficiency.
        """
        var primary_num_blocks = num_dynamic_dim_blocks  # M blocks
        var div_primary_num_blocks = FastDiv[DType.uint32](
            Int(primary_num_blocks)
        )
        comptime uint_type = div_primary_num_blocks.uint_type
        var block_idx_val = rebind[Scalar[uint_type]](_block_idx)

        if not Self.swizzle:
            # Row-major order: iterate M first, then N
            return (
                UInt32(block_idx_val % div_primary_num_blocks),
                UInt32(block_idx_val / div_primary_num_blocks),
            )

        # Swizzle for better L2 usage
        var secondary_num_blocks = num_n_blocks
        var num_blocks_per_group = (
            secondary_num_blocks * Self.kNum1DBlocksPerGroup
        )
        var div_num_blocks_per_group = FastDiv[DType.uint32](
            Int(num_blocks_per_group)
        )
        var group_idx = UInt32(block_idx_val / div_num_blocks_per_group)
        var first_block_idx = group_idx * Self.kNum1DBlocksPerGroup
        var in_group_idx = block_idx_val % div_num_blocks_per_group
        var num_blocks_in_group = min(
            Self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx
        )
        var div_num_blocks_in_group = FastDiv[DType.uint32](
            Int(num_blocks_in_group)
        )
        comptime uint_type2 = div_num_blocks_in_group.uint_type
        var m_block_idx = first_block_idx + UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) % div_num_blocks_in_group
        )
        var n_block_idx = UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) / div_num_blocks_in_group
        )

        return (m_block_idx, n_block_idx)

    @always_inline
    fn current_expert_id(self) -> Int32:
        """Get the expert ID for the current group."""
        return rebind[Scalar[DType.int32]](
            self.expert_ids[Int(self.current_group_idx)]
        )
