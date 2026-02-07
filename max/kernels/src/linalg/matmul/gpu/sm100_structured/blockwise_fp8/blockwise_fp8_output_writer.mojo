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

"""Output writer for blockwise FP8 SM100 matmul.

Handles Register → SMEM → GMEM (via TMA) flow. Unlike standard matmul which
reads from TMEM, blockwise FP8 accumulators are already in registers.
"""

from gpu import WARP_SIZE, lane_id
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace
from gpu.sync import named_barrier
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile
from utils.index import IndexList

from .blockwise_fp8_accumulator import BlockwiseFP8Accumulator
from ..structured_kernels.epilogue_components import (
    TMEMToSMemWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    tma_wait_pipelined,
)
from linalg.structuring import SMemTileArray

# TileTensor-based types for C tiles
from ..structured_kernels.tile_types import SMemTileArray2DRowMajor


# =============================================================================
# BlockwiseFP8TileWriter - Write register accumulators to GMEM
# =============================================================================


struct BlockwiseFP8TileWriter[
    c_type: DType,
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    accum_type: DType,
    accum_layout: Layout,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    is_lower_frag_required: Bool,
    cta_group: Int,
    num_output_stages: Int,
    num_output_warps: UInt,
    c_swizzle: TensorMapSwizzle,
]:
    """Write register accumulators to GMEM via SMEM and TMA."""

    # ========== Layout from dimensions ==========
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    # ========== Tile Array Types ==========
    # LayoutTensor-based C tile array (used internally for .tile[] operations)
    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]
    # TileTensor-based C tile array (for TileTensor overloads)
    comptime CTileArrayTT = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]

    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    comptime num_stages = Self.accum_layout.shape[0].value()
    comptime num_elements = Self.accum_layout.shape[1].value()

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = Self.bits // 32
    comptime fragment_size = (
        Self.data_paths * Self.num_elements_per_load
    ) // WARP_SIZE
    comptime repeats = Self.num_elements // Self.fragment_size
    comptime stageN = Self.repeats * (Self.bits // 32)
    comptime fragments_per_stage = Self.fragment_size * Self.repeats

    # Reuse TMEMToSMemWriter for fragment → SMEM path
    comptime SMEMWriter = TMEMToSMemWriter[
        Self.c_type,
        Self.accum_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.MMA_N,
        Self.stageN,
        Self.cta_group,
        Int(Self.num_output_warps),
        Self.c_swizzle,
        False,  # transpose_c (blockwise FP8 never transposes)
    ]

    # ========== Public Write Method ==========

    @staticmethod
    @always_inline
    fn write[
        c_layout: Layout,
        c_desc_layout: Layout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_layout,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
        ],
        c_tiles: Self.CTileArrayTT,
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        c_coord: Tuple[UInt, UInt],
    ):
        """Write accumulated register tiles to GMEM via double-buffered SMEM."""
        # Convert TileTensor array to LayoutTensor array via shared pointer
        var c_tiles_lt = Self.CTileArray(c_tiles.ptr)
        Self._write_impl[c_layout, c_desc_layout, cluster_size](
            accum, c_tiles_lt, c_tma_op, c_coord
        )

    # ========== Internal Implementation ==========

    @staticmethod
    @always_inline
    fn _write_impl[
        c_layout: Layout,
        c_desc_layout: Layout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_layout,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
        ],
        c_tiles: Self.CTileArray,
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        c_coord: Tuple[UInt, UInt],
    ):
        """Internal implementation for writing accumulated register tiles."""
        var warp_id = get_warp_id()
        var smem_writer = Self.SMEMWriter(UInt32(warp_id), UInt32(lane_id()))

        @parameter
        for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                stage, 0
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                stage, 0
            )

            var c_smem_tile = c_tiles[stage % 2]  # double-buffer

            # Cast from accum_type to c_type, then write to SMEM
            comptime frag_size = Self.SMEMWriter.Config.fragment_size * Self.repeats
            smem_writer.write_fragments[Self.repeats](
                rebind[SIMD[Self.c_type, frag_size]](
                    upper_frag.cast[Self.c_type]()
                ),
                rebind[SIMD[Self.c_type, frag_size]](
                    lower_frag.cast[Self.c_type]()
                ),
                c_smem_tile,
            )

            named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()

            var lane = lane_id()

            # Use shared TMA store components from epilogue_components
            comptime StoreCoords = TMAStoreCoords[
                Self.BM,
                Self.BN,
                Self.MMA_M,
                Self.MMA_N,
                Self.stageN,
                Self.cta_group,
                Self.c_smem_dim0,
                stage,
            ]
            var store_coords = StoreCoords(
                (UInt32(c_coord[0]), UInt32(c_coord[1])), UInt32(warp_id)
            )

            comptime StoreExec = TMAStoreExecutor[
                Self.c_type,
                Self.c_smem_dim0,
                Self.c_smem_dim1,
                Self.BM,
                Self.BN,
                Self.MMA_M,
                Self.MMA_N,
                Self.stageN,
                Self.stageN,  # stage_contiguous_size
                Self.cta_group,
                Self.c_swizzle,
                False,  # transpose_c
                Self.is_lower_frag_required,
            ]
            StoreExec.execute[c_layout, c_desc_layout](
                c_smem_tile,
                store_coords,
                c_tma_op,
                UInt32(warp_id),
                UInt32(lane),
            )
            tma_wait_pipelined[
                Self.c_type,
                c_layout,
                c_desc_layout,
                stage == Self.num_stages - 1,
            ](c_tma_op)

            @parameter
            if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()
