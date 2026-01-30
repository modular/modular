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

"""Output writer for blockwise FP8 SM100 matmul.

Handles Register → SMEM → GMEM (via TMA) flow. Unlike standard matmul which
reads from TMEM, blockwise FP8 accumulators are already in registers.
"""

from gpu import WARP_SIZE, lane_id
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace, fence_async_view_proxy
from gpu.sync import named_barrier
from layout import Layout, LayoutTensor
from layout.swizzle import Swizzle, make_swizzle
from layout.tma_async import TMATensorTile
from utils.index import IndexList

from .blockwise_fp8_accumulator import BlockwiseFP8Accumulator
from ..structured_kernels.tile_writer import store_fragment_to_smem
from linalg.structuring import SMemTileArray


# =============================================================================
# BlockwiseFP8TileWriter - Write register accumulators to GMEM
# =============================================================================


struct BlockwiseFP8TileWriter[
    c_type: DType,
    c_smem_layout: Layout,
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

    # ========== Tile Array Type ==========
    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
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

    comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

    # TMA tile height calculation
    comptime c_smem_shape0 = Self.c_smem_layout.shape[0].value()
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

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
        c_tiles: Self.CTileArray,
        c_tma_op: TMATensorTile[Self.c_type, c_layout, c_desc_layout],
        c_coord: Tuple[UInt, UInt],
    ):
        """Write accumulated register tiles to GMEM via double-buffered SMEM."""
        var warp_id = get_warp_id()

        @parameter
        for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                stage, 0
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                stage, 0
            )

            var c_smem_tile = c_tiles[stage % 2]  # double-buffer

            comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Int(
                Self.num_output_warps
            )
            var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
                Int(warp_id), 0
            )

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](0, 0)
            store_fragment_to_smem[Self.swizzle, Self.stageN](
                upper_frag, c_smem_warp_tile_upper
            )

            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](1, 0)

            @parameter
            if Self.is_lower_frag_required:
                store_fragment_to_smem[Self.swizzle, Self.stageN](
                    lower_frag, c_smem_warp_tile_lower
                )

            named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()

            var lane = lane_id()

            var cg2_elect_one_warp = (
                warp_id == 0 if Self.MMA_M == 256 else warp_id % 2 == 0
            )
            var cg1_elect_one_warp = warp_id == 0
            var elect_one_warp = (
                cg2_elect_one_warp if Self.cta_group
                == 2 else cg1_elect_one_warp
            )

            var coord_n_mma_m256 = c_coord[1] * UInt(Self.MMA_N) + UInt(
                stage * Self.stageN
            )
            var coord_n_mma_m128 = (
                c_coord[1] * UInt(Self.MMA_N)
                + UInt(stage * Self.stageN)
                + UInt(Self.BN * Int(warp_id // 2))
            )

            var cg2_coord_n = (
                coord_n_mma_m256 if Self.MMA_M == 256 else coord_n_mma_m128
            )
            var cg1_coord_n = coord_n_mma_m256
            var coord_n = cg2_coord_n if Self.cta_group == 2 else cg1_coord_n
            var coord_m = c_coord[0] * UInt(Self.BM)

            var cg2_c_smem_coord_m = 0 if Self.MMA_M == 256 else (warp_id // 2)
            var cg1_c_smem_coord_m = UInt(0)
            var c_smem_coord_m = (
                cg2_c_smem_coord_m if Self.cta_group
                == 2 else cg1_c_smem_coord_m
            )
            var c_smem_split = c_smem_tile.tile[Self.TMA_BM, Self.stageN](
                Int(c_smem_coord_m), 0
            )

            if elect_one_warp and lane == 0:
                fence_async_view_proxy()
                c_tma_op.async_store(
                    c_smem_split,
                    (coord_n, coord_m),
                )
                c_tma_op.commit_group()

            @parameter
            if stage < Self.num_stages - 1:
                c_tma_op.wait_group[1]()  # keep one TMA in flight
            else:
                c_tma_op.wait_group[0]()

            @parameter
            if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()
