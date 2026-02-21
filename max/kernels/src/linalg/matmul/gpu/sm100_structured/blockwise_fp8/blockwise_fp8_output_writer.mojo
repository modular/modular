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

Supports two write modes:
- write(): TMA store for standard non-grouped matmul
- write_absolute_with_bounds_check(): Element-by-element store for 1D2D
  grouped matmul with expert boundary bounds checking
"""

from sys import align_of, simd_width_of, size_of

from gpu import WARP_SIZE, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace, fence_async_view_proxy
from gpu.sync import named_barrier
from layout import Layout, LayoutTensor, RuntimeTuple, UNKNOWN_VALUE
from layout._layout import row_major
from layout.int_tuple import IntTuple
from layout.layout_tensor import zipped_divide, upcast
from layout.runtime_tuple import crd2idx as rt_crd2idx
from layout.swizzle import make_swizzle
from layout.tma_async import TMATensorTile
from utils.index import IndexList

from .blockwise_fp8_accumulator import BlockwiseFP8Accumulator
from ..structured_kernels.epilogue_components import (
    TMEMToSMemWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    tma_wait_pipelined,
)
from ..structured_kernels.barriers import WarpGroupBarrier
from layout._layout import TensorLayout
from layout._coord import Coord, Idx
from layout._tile_tensor import TileTensor
from linalg.structuring import SMemTileArray, SMemTile
from linalg.matmul.gpu.sm100.matmul import stsm_helper

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
    accum_num_stages: Int,
    accum_num_elements: Int,
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
    # LayoutTensor (for bounds-checked write path)
    comptime CTileArrayLT = SMemTileArray[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]
    comptime CTileArray = SMemTileArray2DRowMajor[
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

    comptime num_stages = Self.accum_num_stages
    comptime num_elements = Self.accum_num_elements

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
            Self.accum_num_stages,
            Self.accum_num_elements,
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
        Self._write_impl[c_layout, c_desc_layout, cluster_size](
            accum, c_tiles, c_tma_op, c_coord
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
            Self.accum_num_stages,
            Self.accum_num_elements,
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

        comptime for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                Coord(Idx(stage), Idx(0))
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                Coord(Idx(stage), Idx(0))
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

            comptime if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()

    # ========== Bounds-Checked Write for 1D2D Grouped Matmul ==========

    @staticmethod
    @always_inline
    fn write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
        ],
        c_tiles: Self.CTileArray,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
    ):
        """Write accumulated register tiles to GMEM with bounds checking.

        Args:
            accum: Blockwise FP8 accumulator with upper/lower register tiles.
            c_tiles: SMEM tile array for C output.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM (TileTensor for bounds-checked stores).
        """
        var c_tiles_lt = Self.CTileArrayLT(c_tiles.ptr)
        Self._write_absolute_impl[c_tensor_layout, cluster_size](
            accum, c_tiles_lt, m_abs, n_abs, m_end, expert_scale, c_tensor
        )

    @staticmethod
    @always_inline
    fn _write_absolute_impl[
        c_tensor_layout: TensorLayout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
        ],
        c_tiles: Self.CTileArrayLT,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
    ):
        """Internal implementation for bounds-checked register-to-GMEM write.

        Uses stsm_helper for register -> SMEM path to support all output
        tile shapes (including narrow c_smem_dim1=32 with bf16).
        """
        var warp_id = get_warp_id()
        var scale = expert_scale.cast[Self.accum_type]()

        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()
        comptime c_smem_tile_m = Self.BM // Int(Self.num_output_warps)

        comptime for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                Coord(Idx(stage), Idx(0))
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                Coord(Idx(stage), Idx(0))
            )

            # Apply expert scale
            upper_frag = upper_frag * scale

            comptime if Self.is_lower_frag_required:
                lower_frag = lower_frag * scale

            var c_smem_tile = c_tiles[stage % 2]  # double-buffer

            # Write register fragments to SMEM using stsm_helper
            # (handles bf16 correctly with stsmx instead of stmtx)
            var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
                Int(warp_id), 0
            )
            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](0, 0)
            stsm_helper[
                swizzle, UInt(Self.stageN), swizzle_mode = Self.c_swizzle
            ](upper_frag.cast[Self.c_type](), c_smem_warp_tile_upper)

            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](1, 0)

            comptime if Self.is_lower_frag_required:
                stsm_helper[
                    swizzle, UInt(Self.stageN), swizzle_mode = Self.c_swizzle
                ](lower_frag.cast[Self.c_type](), c_smem_warp_tile_lower)

            named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()

            # Bounds-checked element stores from SMEM to GMEM
            Self._store_with_bounds_check[c_tensor_layout](
                c_smem_tile,
                c_tensor,
                m_abs,
                n_abs + UInt32(stage * Self.stageN),
                m_end,
                UInt32(warp_id),
                UInt32(lane_id()),
            )

            comptime if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * UInt(WARP_SIZE))]()

    @staticmethod
    @always_inline
    fn _store_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        c_smem_tile: SMemTile[
            Self.c_type,
            Layout.row_major(Self.c_smem_dim0, Self.c_smem_dim1),
            alignment=128,
        ],
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Store SMEM tile to GMEM with per-row bounds checking.

        Used when the tile crosses the expert boundary.
        Uses element-by-element stores to avoid writing past m_end.
        """
        comptime output_threads = Int(Self.num_output_warps) * WARP_SIZE
        comptime c_smem_M = Self.c_smem_dim0
        comptime TMA_BM = c_smem_M  # blockwise FP8 uses cta_group==1 always
        comptime simd_size = simd_width_of[Self.c_type]()
        comptime alignment = align_of[SIMD[Self.c_type, simd_size]]()
        comptime thread_n = Self.stageN // simd_size
        comptime thread_layout = Layout.row_major(
            output_threads // thread_n, thread_n
        )

        # Swizzle function
        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        # Ensure fence before reading from SMEM
        if warp_id == 0 and lane == 0:
            fence_async_view_proxy()

        # Synchronize all epilogue threads
        WarpGroupBarrier[Int(Self.num_output_warps) * WARP_SIZE].sync()

        # Iterate over SMEM chunks
        comptime for i in range(c_smem_M // TMA_BM):
            var c_smem_split = c_smem_tile.tile[TMA_BM, Self.stageN](i, 0)
            comptime split_layout = c_smem_split.layout
            comptime zipped = zipped_divide(
                upcast(split_layout, simd_size), thread_layout
            )
            comptime split_layout_new = row_major[TMA_BM, Self.stageN]()

            comptime for j in range(zipped.shape[1][0].value()):
                var input_crd = RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE, j),
                    element_type = DType.uint32,
                ](Int(thread_idx.x), j)
                var linear_idx = rt_crd2idx[
                    IntTuple(UNKNOWN_VALUE, j),
                    zipped.shape,
                    zipped.stride,
                    DType.uint32,
                ](
                    input_crd,
                    RuntimeTuple[zipped.shape](),
                    RuntimeTuple[zipped.stride](),
                ) * UInt32(
                    simd_size
                )
                var cmem_crd = split_layout_new.idx2crd[
                    out_dtype = DType.uint32
                ](Int(linear_idx))
                var local_i = cmem_crd[0].value()
                var local_j = cmem_crd[1].value()
                var coord_m = m_abs + UInt32(i * TMA_BM)
                var global_i = coord_m + UInt32(local_i)
                var global_j = n_abs + UInt32(local_j)

                # Bounds check: only store if within expert boundary
                if global_i < m_end:
                    # Compute destination pointer via TileTensor layout
                    var dst_offset = c_tensor.layout(
                        Coord(Idx(Int(global_i)), Idx(Int(global_j)))
                    )
                    var dst_ptr = c_tensor.ptr + Int(dst_offset)

                    comptime if size_of[Self.c_type]() == 2:
                        var src_ptr = c_smem_split.ptr + swizzle(linear_idx)
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        dst_ptr.store[width=simd_size, alignment=alignment](src)
                    else:
                        var src_ptr = c_smem_split.ptr + linear_idx
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        dst_ptr.store[width=simd_size, alignment=alignment](src)
