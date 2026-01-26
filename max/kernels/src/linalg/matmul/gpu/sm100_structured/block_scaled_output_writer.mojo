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

"""BlockScaledTileWriter for SM100 block-scaled matmul output pipeline.

Writes accumulated results from TMEM → Registers → SMEM → GMEM (via TMA).
Uses 3D coordinates (M, N, Batch) for batched block-scaled matmul.

Uses structured building blocks from tile_writer.mojo:
- TmemArrayType / load_fragments() for TMEM load
- AccumBarrier.arrive() for barrier signaling
- TMEMToSMemWriter.write_fragments() for SMEM write
- TMAStoreExecutor with batched=True for 3D TMA stores
- tma_wait_pipelined() for TMA wait

Usage:
    var writer = BlockScaledTileWriter[...](c_tma_op)
    writer.write(c_tiles, pipeline, stage, tmem_offset, coord, shape)
"""

from memory import Pointer
from sys import size_of

from gpu import WARP_SIZE, lane_id
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout
from layout.tma_async import TMATensorTile

from linalg.structuring import SMemTileArray

from utils.index import IndexList

from .barriers import WarpGroupBarrier
from .tile_pipeline import OutputStage
from .tile_writer import (
    AccumBarrier,
    TMEMToSMemWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    tma_wait_pipelined,
)
from .tmem import TmemArrayType


struct BlockScaledTileWriter[
    # Inferred from constructor arg
    tma_origin: ImmutOrigin,
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    //,
    # Explicit config parameters
    a_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    num_accum_pipeline_stages: Int,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    # Kernel-level parameters
    c_smem_layout: Layout,
    num_output_stages: Int,
    stage_stride_cols: Int,
    num_output_warps: Int,
](TrivialRegisterType):
    """Output tile writer for SM100 block-scaled matmul epilogue.

    Uses TMAStoreExecutor with batched=True for 3D (M, N, Batch) TMA stores.
    All operations use structured building blocks from tile_writer.mojo.

    Parameters are passed explicitly to work with BlockScaledMatmulConfig.

    The stage_stride_cols parameter must match the value used when
    constructing the OutputTilePipeline that provides OutputStage
    instances to the write() method.
    """

    # Type aliases
    comptime TmaOp = TMATensorTile[
        Self.c_type, Self.c_layout, Self.c_desc_layout
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]
    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]
    comptime Stage = OutputStage[
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]

    # Derived constants
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    # FP8 uses float32 epilogue (GEX-2630), bf16 uses native type
    comptime epilogue_dtype = (
        Self.c_type if Self.a_type == DType.bfloat16 else DType.float32
    )

    # Stage dimensions
    comptime N_dim = 0 if Self.transpose_c else 1
    comptime stageN = Self.c_smem_layout.shape[Self.N_dim].value()
    comptime stage_contiguous_size = Self.c_smem_layout.shape[1].value()

    # Fragment layout constants
    comptime data_paths = 16
    comptime bits = 256
    comptime repeat = Self.stageN // (Self.bits // 32)
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // WARP_SIZE
    comptime rep_frag_size = Self.repeat * Self.fragment_size

    # CTA group determines fragment requirements
    comptime is_lower_frag_required = not (
        Self.cta_group == 1 and Self.BM == 64
    )
    comptime cg2_num_stages = (
        Self.MMA_N // Self.stageN if Self.MMA_M
        == 256 else Self.MMA_N // Self.stageN // 2
    )
    comptime cg1_num_stages = Self.MMA_N // Self.stageN
    comptime num_stages = (
        Self.cg2_num_stages if Self.cta_group == 2 else Self.cg1_num_stages
    )

    # TMEM array type for accumulator tiles
    comptime accum_tile_layout = Layout.row_major(Self.BM, Self.stageN)
    comptime AccumTmemArray = TmemArrayType[
        Self.accum_type,
        Self.accum_tile_layout,
        Self.num_stages,
        cta_group = Self.cta_group,
    ]

    # SMEM writer type
    comptime SMEMWriter = TMEMToSMemWriter[
        Self.c_type,
        Self.accum_type,
        Self.c_smem_layout,
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.MMA_N,
        Self.stageN,
        Self.cta_group,
        Self.num_output_warps,
        Self.c_swizzle,
        Self.transpose_c,
    ]

    # TMA store executor with batched=True for 3D coordinates
    comptime StoreExecutor = TMAStoreExecutor[
        Self.c_type,
        Self.c_smem_layout,
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.MMA_N,
        Self.stageN,
        Self.stage_contiguous_size,
        Self.cta_group,
        Self.c_swizzle,
        Self.transpose_c,
        Self.is_lower_frag_required,
        batched=True,
    ]

    var c_tma_op: Self.TmaOpPtr

    @always_inline
    fn __init__(out self, c_tma_op: Self.TmaOpPtr):
        """Initialize with pointer to TMA descriptor."""
        constrained[
            Self.stage_stride_cols > 0,
            "stage_stride_cols must be positive",
        ]()
        self.c_tma_op = c_tma_op

    @always_inline
    fn write(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results to global memory.

        Args:
            c_tiles: SMEM tile array for C output (double-buffered).
            stage: OutputStage with pipeline, index, and TMEM handle.
            c_coord: (m_tile, n_tile, batch) coordinates.
            c_shape: (M, N) problem dimensions.
        """
        # Fragment registers
        var upper_frag_partial: SIMD[Self.accum_type, Self.rep_frag_size]
        var lower_frag_partial = SIMD[Self.accum_type, Self.rep_frag_size]()
        var upper_frag_casted: SIMD[Self.epilogue_dtype, Self.rep_frag_size]
        var lower_frag_casted = SIMD[Self.epilogue_dtype, Self.rep_frag_size]()

        var warp_id = get_warp_id()

        # Create TMEM array at the stage's offset
        var accum_tiles = Self.AccumTmemArray(stage.tmem.offset())

        # Main stage loop
        @parameter
        for loop_stage in range(Self.num_stages):
            # ================================================================
            # PHASE 1: TMEM Load - Using TmemTensor.load_fragments()
            # ================================================================
            var frags = accum_tiles[loop_stage].load_fragments[Self.repeat]()
            Self.AccumTmemArray.Tile.wait_load()

            # Extract fragments (rebind for type compatibility)
            upper_frag_partial = rebind[
                SIMD[Self.accum_type, Self.rep_frag_size]
            ](frags.upper)

            @parameter
            if Self.is_lower_frag_required:
                lower_frag_partial = rebind[
                    SIMD[Self.accum_type, Self.rep_frag_size]
                ](frags.lower)

            # ================================================================
            # PHASE 2: Barrier Arrive - Using AccumBarrier.arrive()
            # ================================================================
            @parameter
            if loop_stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(stage.pipeline, stage.index)

            # Cast to epilogue dtype
            upper_frag_casted = upper_frag_partial.cast[Self.epilogue_dtype]()

            @parameter
            if Self.is_lower_frag_required:
                lower_frag_casted = lower_frag_partial.cast[
                    Self.epilogue_dtype
                ]()

            # ================================================================
            # PHASE 3: SMEM Write - Using TMEMToSMemWriter.write_fragments()
            # ================================================================
            var c_smem_tile = c_tiles[loop_stage % 2]
            var smem_writer = Self.SMEMWriter(
                UInt32(warp_id), UInt32(lane_id())
            )

            comptime expected_frag_size = (
                Self.SMEMWriter.Config.fragment_size * Self.repeat
            )
            smem_writer.write_fragments[Self.repeat](
                rebind[SIMD[Self.c_type, expected_frag_size]](
                    upper_frag_casted.cast[Self.c_type]()
                ),
                rebind[SIMD[Self.c_type, expected_frag_size]](
                    lower_frag_casted.cast[Self.c_type]()
                ),
                c_smem_tile,
            )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

            # ================================================================
            # PHASE 4: TMA Store - Using TMAStoreExecutor with batched=True
            # ================================================================
            var lane = lane_id()

            comptime StoreCoords = TMAStoreCoords[
                Self.BM,
                Self.BN,
                Self.MMA_M,
                Self.MMA_N,
                Self.stageN,
                Self.cta_group,
                Self.c_smem_layout.shape[0].value(),
                loop_stage,
                batched=True,
            ]
            var store_coords = StoreCoords(c_coord, UInt32(warp_id))
            Self.StoreExecutor.execute[Self.c_layout, Self.c_desc_layout](
                c_smem_tile,
                store_coords,
                self.c_tma_op[],
                UInt32(warp_id),
                UInt32(lane),
            )

            # ================================================================
            # PHASE 5: TMA Wait - Using tma_wait_pipelined()
            # ================================================================
            tma_wait_pipelined[
                Self.c_type,
                Self.c_layout,
                Self.c_desc_layout,
                loop_stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            @parameter
            if loop_stage > 0 or loop_stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
