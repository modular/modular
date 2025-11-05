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

from math import ceildiv

from gpu import block_idx, grid_dim, thread_idx

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from buffer.buffer import NDBuffer


@fieldwise_init
@register_passable("trivial")
struct RasterOrder(ImplicitlyCopyable, Movable):
    var _value: Int32

    alias AlongN = Self(0)
    alias AlongM = Self(1)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


@fieldwise_init
@register_passable("trivial")
struct WorkInfo(ImplicitlyCopyable, Movable, Stringable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool
    var terminate: Bool

    @always_inline
    fn __init__(
        out self,
    ):
        self.m = 0
        self.n = 0
        self.is_valid_tile = False
        self.terminate = False

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @always_inline
    fn is_done(self) -> Bool:
        return self.terminate

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.is_valid_tile,
            ", ",
            self.terminate,
            ")",
        )


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


# For simplicity, we always assume M is the static dimension here, because 2SM
# UMMA instructions need alignment on only the M dimension. When we use it, we
# ought to enable swapAB for grouped matmul.
@register_passable("trivial")
struct TileScheduler[
    M: Int,
    # note that the tile shape always refers to the original non-swapped AB
    # shape
    tile_shape: IndexList[3],
    cluster: IndexList[3] = Index(1, 1, 1),
    cta_group: Int = 1,
    swizzle: Bool = False,
]:
    var num_active_experts: Int
    var group_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin]
    var current_iter: Int32  # Tracks the scheduler's progress across kernel launches
    var current_group_idx: UInt32
    alias tile_n = tile_shape[1] * cta_group
    alias div_block_n = FastDiv[DType.uint32](Self.tile_n)
    var current_n_cumsum: UInt32
    var block_idx_start: UInt32

    alias num_m_blocks: UInt32 = ceildiv(M, tile_shape[0])

    alias kNum1DBlocksPerGroup: UInt32 = 16

    @always_inline
    fn __init__(
        out self,
        num_active_experts: Int,
        group_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    ):
        constrained[
            cluster[1] == cluster[2] == 1,
            "Currently multicasting along non-M dimension is not supported",
        ]()
        constrained[
            cta_group == cluster[0],
            "cta_group must be equal to cluster M size. Got cta_group = "
            + String(cta_group)
            + " and cluster M size = "
            + String(cluster[0]),
        ]()
        alias cluster_m_size = cluster[0] * tile_shape[0]
        constrained[
            cluster[0] == 1 or M % cluster_m_size == 0,
            "Problem shape M must be divisible by cluster M size. Got "
            + String(M)
            + " and cluster M size "
            + String(cluster_m_size)
            + " = cluster M size "
            + String(cluster[0])
            + " * tile M size "
            + String(tile_shape[0]),
        ]()

        self.num_active_experts = num_active_experts
        self.group_offsets = group_offsets
        self.current_iter = -1
        self.current_group_idx = 0
        self.current_n_cumsum = 0
        self.block_idx_start = 0

    @always_inline
    fn fetch_next_work(mut self) -> WorkInfo:
        self.current_iter += 1
        var next_block_idx = (
            UInt32(self.current_iter) * grid_dim.x + block_idx.x
        )
        var start_idx = self.group_offsets[Int(self.current_group_idx)]
        var end_idx: UInt32 = 0
        var num_n_blocks: UInt32 = 0
        var current_n: UInt32 = 0

        # Trim to the next group
        while True:
            if self.current_group_idx >= self.num_active_experts:
                # at this point, we finished all groups
                return WorkInfo(0, 0, False, True)

            end_idx = self.group_offsets[Int(self.current_group_idx + 1)]
            current_n = end_idx - start_idx
            num_n_blocks = UInt32(
                rebind[Scalar[Self.div_block_n.uint_type]](
                    current_n + UInt32(Self.tile_n - 1)
                )
                / Self.div_block_n
            )
            var current_n_block_cumsum = self.current_n_cumsum + num_n_blocks
            var current_block_idx_start = (
                current_n_block_cumsum * Self.num_m_blocks
            )
            if next_block_idx < current_block_idx_start:
                break
            self.current_group_idx += 1
            self.current_n_cumsum = current_n_block_cumsum
            self.block_idx_start = current_block_idx_start
            start_idx = end_idx

        var group_local_block_idx = next_block_idx - self.block_idx_start
        var is_valid = group_local_block_idx < num_n_blocks * Self.num_m_blocks
        if not is_valid:
            return WorkInfo(0, 0, False, False)

        var m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            num_n_blocks, group_local_block_idx
        )
        var m = UInt32(m_block_idx) * UInt32(Self.tile_shape[0])
        var n = UInt32(start_idx + n_block_idx * Self.tile_n)
        # In GMM scheduler, a tile may be invalid, but that is an independent
        # condition from `is_done/terminate`, that is, the CTA might have more
        # work to do in the next group. This is the consequence of not aligning
        # each group with `num_n_blocks * num_m_blocks`.
        return WorkInfo(
            m,
            n,
            True,
            False,
        )

    @always_inline
    fn _get_swizzled_block_idx(
        self, num_n_blocks: UInt32, block_idx: UInt32
    ) -> Tuple[UInt32, UInt32]:
        """
        Calculates swizzled (m_block_idx, n_block_idx) based on the overall block_idx.
        Returns a tuple (m_block_idx, n_block_idx).
        """
        alias primary_num_blocks: UInt32 = Self.num_m_blocks
        if not Self.swizzle:
            return (
                block_idx % primary_num_blocks,
                block_idx / primary_num_blocks,
            )

        var m_block_idx: UInt32
        var n_block_idx: UInt32

        # Swizzle for better L2 usages
        # Since we will only multicast on the M-dimension if any, the primary
        # dimension must be along M.
        var secondary_num_blocks: UInt32 = num_n_blocks
        var num_blocks_per_group = UInt32(
            secondary_num_blocks * Self.kNum1DBlocksPerGroup
        )
        var div_num_blocks_per_group = FastDiv[DType.uint32](
            Int(num_blocks_per_group)
        )
        alias uint_type = div_num_blocks_per_group.uint_type
        var group_idx = UInt32(
            rebind[Scalar[uint_type]](block_idx) / div_num_blocks_per_group
        )
        var first_block_idx = group_idx * Self.kNum1DBlocksPerGroup
        var in_group_idx = block_idx % num_blocks_per_group
        var num_blocks_in_group = min(
            Self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx
        )
        var div_num_blocks_in_group = FastDiv[DType.uint32](
            Int(num_blocks_in_group)
        )
        alias uint_type2 = div_num_blocks_in_group.uint_type
        m_block_idx = first_block_idx + UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) % div_num_blocks_in_group
        )
        n_block_idx = UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) / div_num_blocks_in_group
        )

        return (m_block_idx, n_block_idx)
