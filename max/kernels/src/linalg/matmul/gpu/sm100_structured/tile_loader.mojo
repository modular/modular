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

"""TileLoader for SM100 matrix multiplication.

Provides tile loading abstractions for efficient global-to-shared memory
transfers using TMA with support for:

- K-group batching (multiple tiles per barrier synchronization)
- CTA group coordination (1-SM or 2-SM cooperative loading)
- Multicast for cluster distribution

Usage:
    var loader = TileLoaderTMA[...](a_tma_op, b_tma_op, masks, peer_coord)
    loader.set_work_tile(m_coord, n_coord)

    with producer.acquire() as tiles:
        loader.load_tiles(tiles, k_iter, elect_one_cta)
"""

from gpu.cluster import elect_one_sync
from layout.tma_async import TMATensorTile
from .tile_pipeline import ProducerStage


# Partial Type Binding Pattern for Origin Inference
#
# The `//` separator below divides parameters into two groups:
#   - Before `//` (inferred): Deduced from constructor arguments
#   - After `//` (explicit): Must be bound when creating a type alias
#
# This enables callers to create a partial comptime binding only the explicit
# params, while the inferred params (especially Origins) are automatically
# deduced from constructor arguments:
#
#     # In kernel - bind explicit params only:
#     comptime TileLoaderTMA = TileLoaderTMA[a_smem_layout, b_smem_layout, ...]
#
#     # Origins are inferred from Pointer arguments:
#     var loader = Self.TileLoaderTMA(Pointer(to=a_op), Pointer(to=b_op), ...)
#
# This avoids needing a factory method for Origin parameter inference.
# See: docs/internal/mojo_parameter_gotchas.md (Section 9)


@register_passable("trivial")
struct TileLoaderTMA[
    a_tma_origin: ImmutOrigin,
    b_tma_origin: ImmutOrigin,
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    //,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    MMA_N: Int,
    cta_group: Int,
    k_group_size: Int,
    num_pipeline_stages: Int,
    num_group_stages: Int,
]:
    """TMA-based tile loader for SM100.

    Encapsulates the complete tile loading logic including:
    - K-group batching (multiple tiles per barrier)
    - CTA group coordination (1-SM or 2-SM cooperative loading)
    - Peer CTA slicing for 2-SM MMA
    - expect_bytes management

    Template Parameters:
        a_tma_origin: Origin type for A TMA pointer.
        b_tma_origin: Origin type for B TMA pointer.
        a_type: Data type for A matrix.
        b_type: Data type for B matrix.
        a_layout: Global memory layout for A.
        b_layout: Global memory layout for B.
        a_desc_layout: TMA descriptor layout for A.
        b_desc_layout: TMA descriptor layout for B.
        a_smem_layout: Shared memory tile layout for A.
        b_smem_layout: Shared memory tile layout for B.
        BM: Block tile M dimension.
        BN: Block tile N dimension.
        BK: Block tile K dimension.
        MMA_N: MMA N dimension for B coordinate calculation.
        cta_group: Number of CTAs cooperating, 1 or 2.
        k_group_size: Number of K tiles per barrier sync.
        num_pipeline_stages: Total pipeline stages.
        num_group_stages: Pipeline stages / k_group_size.
    """

    # Type aliases for TMA operations
    comptime ATmaOp = TMATensorTile[
        Self.a_type, Self.a_layout, Self.a_desc_layout
    ]
    comptime BTmaOp = TMATensorTile[
        Self.b_type, Self.b_layout, Self.b_desc_layout
    ]
    comptime ATmaOpPtr = Pointer[Self.ATmaOp, Self.a_tma_origin]
    comptime BTmaOpPtr = Pointer[Self.BTmaOp, Self.b_tma_origin]

    # Computed constants
    comptime a_expected_bytes = Self.a_smem_layout.size() * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.b_smem_layout.size() * size_of[
        Self.b_type
    ]()
    comptime expected_bytes = Self.cta_group * (
        Self.a_expected_bytes + Self.b_expected_bytes
    ) * Self.k_group_size

    comptime a_tma_load_size = Self.a_desc_layout.size()
    comptime b_tma_load_size = Self.b_desc_layout.size()
    comptime a_tma_rows = Self.a_desc_layout.shape[0].value()
    comptime b_tma_rows = Self.b_desc_layout.shape[0].value()

    # TMA descriptor pointers (referencing grid constants)
    var a_tma_op: Self.ATmaOpPtr
    var b_tma_op: Self.BTmaOpPtr

    # Multicast configuration
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16

    # Peer CTA info for 2-SM slicing
    var peer_rank_n: UInt
    var peer_rank_m: UInt
    var peer_m_rank: UInt

    # Current work tile coordinates
    var work_m_coord: UInt
    var work_n_coord: UInt

    @always_inline
    fn __init__(
        out self,
        a_tma_op: Self.ATmaOpPtr,
        b_tma_op: Self.BTmaOpPtr,
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        peer_cta_coord: Tuple[UInt, UInt, UInt],
    ):
        """Initialize the TMA tile loader.

        Args:
            a_tma_op: Pointer to A matrix TMA descriptor.
            b_tma_op: Pointer to B matrix TMA descriptor.
            a_multicast_mask: Multicast mask for A tiles.
            b_multicast_mask: Multicast mask for B tiles.
            peer_cta_coord: Peer CTA coordinates (rank_n, rank_m, peer_m_rank).
        """
        self.a_tma_op = a_tma_op
        self.b_tma_op = b_tma_op
        self.a_multicast_mask = a_multicast_mask
        self.b_multicast_mask = b_multicast_mask
        self.peer_rank_n = peer_cta_coord[0]
        self.peer_rank_m = peer_cta_coord[1]
        self.peer_m_rank = peer_cta_coord[2]
        self.work_m_coord = UInt(0)
        self.work_n_coord = UInt(0)

    @always_inline
    fn set_work_tile(mut self, m_coord: UInt, n_coord: UInt):
        """Set the current output tile coordinates.

        Args:
            m_coord: M coordinate of the output tile.
            n_coord: N coordinate of the output tile.
        """
        self.work_m_coord = m_coord
        self.work_n_coord = n_coord

    @always_inline
    fn load_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        self,
        tiles: ProducerStage[
            tiles_origin,
            Self.a_type,
            Self.b_type,
            Self.a_smem_layout,
            Self.b_smem_layout,
            Self.num_pipeline_stages,
            Self.num_group_stages,
            Self.k_group_size,
        ],
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load k_group_size A and B tiles using TMA.

        Args:
            tiles: ProducerStage context with encapsulated tile access.
            iter_idx: K iteration index (base index, not multiplied).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        # Global memory coordinates for A (M dimension) and B (N dimension)
        var a_gmem_m_coord = self.peer_m_rank * UInt(
            Self.a_tma_rows
        ) + self.work_m_coord * UInt(Self.BM)
        var b_gmem_n_coord = (
            self.peer_rank_m * UInt(Self.b_tma_rows)
            + self.peer_rank_n * UInt(Self.BN)
            + self.work_n_coord * UInt(Self.MMA_N)
        )

        if elect_one_sync():
            # Set expected bytes ONCE for all k_group tiles
            if elect_one_cta:
                tiles.expect_bytes(Self.expected_bytes)

            # Get barrier for TMA multicast loads
            var barrier = tiles.barrier()

            for j in range(Self.k_group_size):
                var a_tile, b_tile = tiles.get_tile(j)

                # Peer CTA slice offset within the tile
                var a_peer_slice = type_of(a_tile)(
                    a_tile.ptr + self.peer_m_rank * UInt(Self.a_tma_load_size)
                )
                var b_peer_slice = type_of(b_tile)(
                    b_tile.ptr + self.peer_rank_m * UInt(Self.b_tma_load_size)
                )

                var k_coord = UInt(iter_idx + j) * UInt(Self.BK)

                self.a_tma_op[].async_multicast_load[Self.cta_group](
                    a_peer_slice,
                    barrier[0],
                    (k_coord, UInt(a_gmem_m_coord)),
                    self.a_multicast_mask,
                )
                self.b_tma_op[].async_multicast_load[Self.cta_group](
                    b_peer_slice,
                    barrier[0],
                    (k_coord, UInt(b_gmem_n_coord)),
                    self.b_multicast_mask,
                )
