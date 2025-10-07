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

from gpu.id import block_idx, grid_dim

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
    swizzle: Bool = False,
]:
    var group_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var current_iter: Int32  # Tracks the scheduler's progress across kernel launches
    var current_group_idx: UInt32
    alias div_block_n = FastDiv[DType.uint32](tile_shape[1])
    var current_n_cumsum: UInt32

    alias num_m_blocks: UInt32 = ceildiv(M, tile_shape[0])

    alias kNum1DBlocksPerGroup: UInt32 = 16

    @always_inline
    fn __init__(
        out self, group_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ):
        constrained[
            cluster[1] == cluster[2] == 1,
            "Currently multicasting along non-M dimension is not supported",
        ]()
        alias cluster_m_size = cluster[0] * tile_shape[0]
        constrained[
            M % cluster_m_size == 0,
            "Problem shape M must be divisible by cluster M size. Got "
            + String(M)
            + " and cluster M size "
            + String(cluster_m_size)
            + " = cluster M size "
            + String(cluster[0])
            + " * tile M size "
            + String(tile_shape[0]),
        ]()

        self.group_offsets = group_offsets
        self.current_iter = -1
        self.current_group_idx = 0
        self.current_n_cumsum = 0

    @always_inline
    fn fetch_next_work(mut self) -> WorkInfo:
        self.current_iter += 1
        var next_block_idx = (
            UInt32(self.current_iter) * grid_dim.x + block_idx.x
        )
        var num_groups = len(self.group_offsets) - 1
        var start_idx = self.group_offsets[Int(self.current_group_idx)]
        var end_idx: UInt32 = 0
        var block_idx_start: UInt32 = 0
        var num_n_blocks: UInt32 = 0
        var current_n: UInt32 = 0

        # Trim to the next group
        while True:
            if self.current_group_idx >= num_groups:
                # at this point, we finished all groups
                return WorkInfo(0, 0, False, True)

            end_idx = self.group_offsets[Int(self.current_group_idx + 1)]
            current_n = end_idx - start_idx
            num_n_blocks = UInt32(
                rebind[Scalar[Self.div_block_n.uint_type]](
                    current_n + UInt32(Self.tile_shape[1] - 1)
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
            block_idx_start = current_block_idx_start
            start_idx = end_idx

        var group_local_block_idx = next_block_idx - block_idx_start
        var m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            num_n_blocks, group_local_block_idx
        )
        var m = UInt32(m_block_idx * Self.tile_shape[0])
        var n = UInt32(start_idx + n_block_idx * Self.tile_shape[1])
        # In GMM scheduler, a tile may be invalid, but that is an independent
        # condition from `is_done/terminate`, that is, the CTA might have more
        # work to do in the next group. This is the consequence of not aligning
        # each group with `num_n_blocks * num_m_blocks`.
        return WorkInfo(
            m,
            n,
            group_local_block_idx < num_n_blocks * Self.num_m_blocks,
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
