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

"""TileWriter for SM100 matmul output pipeline.

Writes accumulated results from TMEM → Registers → SMEM → GMEM (via TMA).

Usage:
    var writer = TileWriter[config=..., ...](Pointer(to=c_tma_op))
    writer.write(smem.c_tiles(), stage, coord, shape, elect)
"""

from collections import OptionalReg
from memory import Pointer
from sys import simd_width_of

from gpu import WARP_SIZE, thread_idx
from gpu import lane_id
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile

from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.structuring import SMemTileArrayType
from linalg.utils import elementwise_compute_lambda_type

from utils.index import IndexList

from .barriers import WarpGroupBarrier
from .tile_pipeline import OutputStage
from .tile_scheduler_splitk import TileScheduler, WorkInfo
from .tile_writer import (
    AccumBarrier,
    AccumTile,
    EpilogueApplier,
    SMemEpilogueWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    TMEMToSMemWriter,
    tma_wait_pipelined,
)
from .tmem import TmemArrayType


@register_passable("trivial")
struct TileWriter[
    # Inferred from constructor arg
    tma_origin: ImmutOrigin,
    c_type: DType,
    c_layout: Layout,
    c_desc_layout: Layout,
    //,
    # From MatmulConfig
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Kernel-level parameters
    c_smem_layout: Layout,
    num_output_stages: Int,
    stage_stride_cols: UInt,
    num_output_warps: UInt,
    max_tmem_cols: UInt = 512,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
]:
    """Output tile writer for SM100 matmul epilogue.

    Stores pointer to TMA descriptor. SMEM tiles passed per-call.
    """

    # From config
    comptime cta_group = Self.config.cta_group
    comptime mma_shape = Self.config.mma_shape
    comptime block_tile_shape = Self.config.block_tile_shape
    comptime accum_type = Self.config.accum_type
    comptime num_accum_pipeline_stages = Int(
        Self.config.num_accum_pipeline_stages
    )
    comptime c_swizzle = Self.config.c_swizzle
    comptime transpose_c = Self.config.AB_swapped

    # Type aliases
    comptime TmaOp = TMATensorTile[
        Self.c_type, Self.c_layout, Self.c_desc_layout
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]
    comptime CTileArray = SMemTileArrayType[
        Self.c_type, Self.c_smem_layout, Self.num_output_stages, alignment=128
    ]
    comptime Stage = OutputStage[
        Self.num_accum_pipeline_stages,
        Int(Self.stage_stride_cols),
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
    comptime rep = Self.stageN // (Self.bits // 32)
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // WARP_SIZE
    comptime rep_frag_size = Self.rep * Self.fragment_size

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

    var c_tma_op: Self.TmaOpPtr

    @always_inline
    fn __init__(out self, c_tma_op: Self.TmaOpPtr):
        """Initialize with pointer to TMA descriptor."""
        self.c_tma_op = c_tma_op

    @always_inline
    fn write(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write accumulated results to global memory."""
        self._copy_to_gmem(c_tiles, stage, tile_coord, shape)

    @always_inline
    fn write_splitk[
        reduction_layout: Layout,
    ](
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        scheduler: TileScheduler,
        reduction_tensor: LayoutTensor[
            Self.accum_type, reduction_layout, MutAnyOrigin
        ],
        work_info: WorkInfo,
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with split-K reduction. Only last split writes to GMEM."""
        var epilogue_thread_idx = thread_idx.x

        # Perform reduction and check if this is the last split
        var is_last_split = scheduler.reduction(
            reduction_tensor,
            stage.tmem.address(),
            epilogue_thread_idx,
            work_info,
        )

        # If not last split, signal and exit early
        if not is_last_split:
            AccumBarrier[Self.cta_group].arrive(stage.pipeline, stage.index)
            return

        self._copy_to_gmem(c_tiles, stage, (work_info.m, work_info.n), shape)

    @always_inline
    fn _copy_to_gmem(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → SMEM → GMEM pipeline."""
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

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
            Int(Self.num_output_warps),
            Self.c_swizzle,
            Self.transpose_c,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

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
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id), UInt32(lane)
        )
        var c_row = UInt32(c_coord[0] * UInt(Self.BM))
        var c_col = UInt32(c_coord[1] * UInt(Self.MMA_N))

        var upper_frag_casted: SIMD[Self.epilogue_dtype, Self.rep_frag_size]
        var lower_frag_casted: SIMD[Self.epilogue_dtype, Self.rep_frag_size]

        @parameter
        for stage in range(Self.num_stages):
            # Load fragments from TMEM tile - is_lower_required handled internally
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()
            var casted = frags.cast[Self.epilogue_dtype]()
            # rebind bridges TmemTensor's (4 * rep) to our (rep * 4) type
            upper_frag_casted = rebind[type_of(upper_frag_casted)](casted.upper)
            lower_frag_casted = rebind[type_of(lower_frag_casted)](casted.lower)

            @parameter
            if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            @parameter
            if Self.elementwise_compute_lambda_fn:

                @parameter
                if Self.register_based_epilogue:
                    upper_frag_casted, lower_frag_casted = (
                        epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                    )

            var c_smem_tile = c_tiles[stage % 2]

            @parameter
            if (
                Self.register_based_epilogue
                or not Self.elementwise_compute_lambda_fn
            ):
                comptime expected_size = SMEMWriter.Config.fragment_size * Self.rep
                constrained[
                    Self.rep_frag_size == expected_size,
                    "Fragment sizes must match",
                ]()
                smem_writer.write_fragments[Self.rep](
                    rebind[SIMD[Self.c_type, expected_size]](
                        upper_frag_casted.cast[Self.c_type]()
                    ),
                    rebind[SIMD[Self.c_type, expected_size]](
                        lower_frag_casted.cast[Self.c_type]()
                    ),
                    c_smem_tile,
                )
                WarpGroupBarrier[Int(Self.num_output_warps) * WARP_SIZE].sync()
            else:
                var writer = SMemEpilogueWriter[
                    Self.epilogue_dtype,
                    Self.BM,
                    Self.BN,
                    Self.MMA_M,
                    Self.MMA_N,
                    Self.cta_group,
                    Int(Self.num_output_warps),
                    Self.c_swizzle,
                    Self.transpose_c,
                    Self.is_lower_frag_required,
                    Self.num_stages,
                    simd_size,
                    stage,
                    Self.rep_frag_size,
                    Self.elementwise_compute_lambda_fn.value(),
                ](UInt32(warp_id), c_tiles, c_shape, c_coord)
                writer.write_tile(
                    AccumTile(upper_frag_casted, lower_frag_casted)
                )

            comptime StoreCoords = TMAStoreCoords[
                Self.BM,
                Self.BN,
                Self.MMA_M,
                Self.MMA_N,
                Self.stageN,
                Self.cta_group,
                Self.c_smem_layout.shape[0].value(),
                stage,
            ]
            var store_coords = StoreCoords(c_coord, UInt32(warp_id))
            StoreExecutor.execute[Self.c_layout, Self.c_desc_layout](
                c_smem_tile,
                store_coords,
                self.c_tma_op[],
                UInt32(warp_id),
                UInt32(lane),
            )
            tma_wait_pipelined[
                Self.c_type,
                Self.c_layout,
                Self.c_desc_layout,
                stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            @parameter
            if stage > 0 or stage == Self.num_stages - 1:
                WarpGroupBarrier[Int(Self.num_output_warps) * WARP_SIZE].sync()
