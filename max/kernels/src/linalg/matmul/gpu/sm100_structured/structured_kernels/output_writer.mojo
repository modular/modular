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

"""TileWriter for SM100 matmul output pipeline.

Writes accumulated results from TMEM → Registers → SMEM → GMEM (via TMA).

Usage:
    var writer = TileWriter[config=..., ...](Pointer(to=c_tma_op))
    writer.write(smem.c_tiles(), stage, coord, shape, elect)
"""

from std.collections import Optional
from std.memory import Pointer, UnsafePointer
from std.sys import simd_width_of, size_of, align_of

from std.gpu import WARP_SIZE, thread_idx_int as thread_idx
from std.gpu import lane_id_uint as lane_id
from std.gpu import warp_id_uint as get_warp_id
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.layout import zipped_divide
from layout.layout_tensor import upcast
from layout.runtime_tuple import crd2idx as rt_crd2idx
from layout.swizzle import make_swizzle
from layout.tma_async import TMATensorTile

from linalg.utils import elementwise_compute_lambda_type

from std.utils.index import IndexList

# TileTensor-based types for C tiles
from structured_kernels.tile_types import SMemTileArray2DRowMajor

from structured_kernels.barriers import WarpGroupBarrier
from .config import OutputPipelineConfig
from .tile_pipeline import OutputStage
from .tile_scheduler_splitk import TileScheduler, WorkInfo
from .epilogue_components import (
    AccumBarrier,
    AccumTile,
    EpilogueApplier,
    EpilogueConfig,
    SMemEpilogueWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    TMEMToSMemWriter,
    tma_wait_pipelined,
)
from .tmem import TmemArrayType


struct TileWriter[
    # Inferred from constructor arg
    tma_origin: ImmutOrigin,
    c_type: DType,
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    //,
    # Explicit config parameters (works with any config type)
    a_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    opc: OutputPipelineConfig,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    # Kernel-level parameters - dimensions replace c_smem_layout
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    num_output_stages: Int,
    num_output_warps: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    batched: Bool = False,
    problem_n: Int = 0,
](TrivialRegisterPassable):
    """Output tile writer for SM100 matmul epilogue.

    Stores pointer to TMA descriptor. SMEM tiles passed per-call.

    Parameters are passed explicitly to work with both MatmulConfig
    and BlockScaledMatmulConfig.

    The opc (OutputPipelineConfig) parameter must match the config used
    when constructing the OutputTilePipeline that provides OutputStage
    instances to the write() method.
    """

    # Local aliases from OutputPipelineConfig
    comptime cta_group = Self.opc.cta_group
    comptime num_accum_pipeline_stages = Self.opc.num_stages
    comptime stage_stride_cols = Self.opc.stage_stride_cols

    # Create internal layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    # Type aliases
    comptime TmaOp = TMATensorTile[
        Self.c_type, Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]
    # C tile array (output and source tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]
    comptime Stage = OutputStage[Self.opc]

    # Derived constants
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    # FP8 uses float32 epilogue (GEX-2630), bf16 uses native type
    comptime epilogue_dtype = (
        DType.bfloat16 if (
            Self.a_type == Self.c_type == DType.bfloat16
        ) else DType.float32
    )

    # Stage dimensions - now use direct dimension access
    comptime N_dim = 0 if Self.transpose_c else 1
    comptime stageN = Self.c_smem_dim0 if Self.transpose_c else Self.c_smem_dim1
    comptime stage_contiguous_size = Self.c_smem_dim1

    # EpilogueConfig bundles common epilogue parameters
    comptime epc = EpilogueConfig.create(
        MMA_M=Self.MMA_M,
        MMA_N=Self.MMA_N,
        stageN=Self.stageN,
        cta_group=Self.cta_group,
        transpose_c=Self.transpose_c,
        BM=Self.BM,
        BN=Self.BN,
    )

    # Fragment layout constants
    comptime data_paths = 16
    comptime bits = 256
    comptime rep = Self.stageN // (Self.bits // 32)
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // WARP_SIZE
    comptime rep_frag_size = Self.fragment_size * Self.rep

    # Aliases from EpilogueConfig
    comptime is_lower_frag_required = Self.epc.is_lower_frag_required
    comptime num_stages = Self.epc.num_stages

    # TMEM array type for accumulator tiles
    comptime accum_tile_layout = Layout.row_major(Self.BM, Self.stageN)
    comptime AccumTmemArray = TmemArrayType[
        Self.accum_type,
        Self.accum_tile_layout,
        Self.num_stages,
        cta_group=Self.cta_group,
    ]

    var c_tma_op: Self.TmaOpPtr

    @always_inline
    def __init__(out self, c_tma_op: Self.TmaOpPtr):
        """Initialize with pointer to TMA descriptor."""
        comptime assert (
            Self.stage_stride_cols > 0
        ), "stage_stride_cols must be positive"
        self.c_tma_op = c_tma_op

    # ========== Public Write Methods ==========

    @always_inline
    def write(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write accumulated results to global memory (2D coords)."""
        self._copy_to_gmem(c_tiles, stage, tile_coord, shape)

    @always_inline
    def write_batched(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Write accumulated results to global memory (3D batched coords).

        Args:
            c_tiles: TileTensor-based SMEM tile array for C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile, batch) coordinates.
            shape: (M, N) problem dimensions.
            alpha: Tensor scale factor (scalar).
        """
        self._copy_to_gmem_batched(c_tiles, stage, tile_coord, shape, alpha)

    @always_inline
    def write_splitk[
        reduction_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        scheduler: TileScheduler,
        reduction_tensor: TileTensor[
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
    def write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
    ):
        """Write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute.
        """
        self._write_absolute_with_bounds_check[c_tensor_layout](
            c_tiles,
            output_stage,
            m_abs,
            n_abs,
            m_end,
            expert_scale,
            c_tensor,
        )

    @always_inline
    def _copy_to_gmem(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → SMEM → GMEM pipeline (2D coords)."""
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles, output_stage, c_coord, c_shape
            )
        else:
            self._copy_to_gmem_impl(c_tiles, output_stage, c_coord, c_shape)

    @always_inline
    def _copy_to_gmem_batched(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32,
    ):
        """TMEM → Registers → GMEM (elementwise epilogue) pipeline (3D batched coords).
           TMEM → Registers → SMEM → GMEM (compute epilogue) pipeline (3D batched coords).

        If elementwise epilogue function is provided, it will be used to write the results to global memory.
        Otherwise, the results will be written to global memory using the standard TMA based pipeline.
        """
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )
        else:
            self._copy_to_gmem_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )

    @always_inline
    def _copy_to_gmem_with_elementwise_epilogue_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → GMEM (elementwise epilogue) pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.

        In contrast to compute epilogue, elementwise epilogue input is casted to c_type, not epilogue_dtype.
        This is because elementwise epilogue writes directly to global memory, not registers.
        Therefore, we need to cast the input to c_type to match the output type.
        """

        comptime assert (
            Self.elementwise_lambda_fn is not None
        ), "Elementwise epilogue function is not provided"

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        var upper_frag_partial: InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = InlineArray[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to c_type in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions
            # (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
            var upper_simd = SIMD[Self.c_type, Self.rep_frag_size]()
            var lower_simd = SIMD[Self.c_type, Self.rep_frag_size]()

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.c_type]()
                comptime for _j in range(cast_width):
                    upper_simd[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        lower_simd[offset + _j] = dst[_j]

            epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                Self.c_type,
                Self.rep_frag_size,
                Self.elementwise_lambda_fn.value(),
                Self.is_lower_frag_required,
            ](
                upper_simd,
                lower_simd,
                UInt32(stage),
                c_row,
                c_col,
            )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _copy_to_gmem_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → SMEM → GMEM pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
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
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        var upper_frag_partial: InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = InlineArray[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to epilogue dtype in SIMD chunks
            # of at least 4 bytes for efficient hardware cast
            # instructions (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Apply epilogue lambda if provided
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    var _epilogue_result = (
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
                    upper_frag_casted = _epilogue_result[0].copy()
                    lower_frag_casted = _epilogue_result[1].copy()

            var c_smem_tile = c_tiles[stage % 2]

            comptime if (
                Self.register_based_epilogue
                or not Self.elementwise_compute_lambda_fn
            ):
                comptime expected_size = Self.epc.fragment_size * Self.rep
                # Cast from epilogue_dtype to c_type in SIMD chunks
                # of at least 4 bytes.
                var upper_c = InlineArray[Scalar[Self.c_type], expected_size](
                    uninitialized=True
                )
                var lower_c = InlineArray[Scalar[Self.c_type], expected_size](
                    uninitialized=True
                )

                comptime cast_width_c = (4 // size_of[Scalar[Self.c_type]]())
                comptime for _chunk in range(
                    Self.rep_frag_size // cast_width_c
                ):
                    comptime offset = _chunk * cast_width_c
                    var src_u = SIMD[Self.epilogue_dtype, cast_width_c]()
                    var src_l = SIMD[Self.epilogue_dtype, cast_width_c]()
                    comptime for _j in range(cast_width_c):
                        src_u[_j] = upper_frag_casted[offset + _j]
                        src_l[_j] = lower_frag_casted[offset + _j]
                    var dst_u = src_u.cast[Self.c_type]()
                    var dst_l = src_l.cast[Self.c_type]()
                    comptime for _j in range(cast_width_c):
                        upper_c[offset + _j] = dst_u[_j]
                        lower_c[offset + _j] = dst_l[_j]
                smem_writer.write_fragments[Self.rep](
                    rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                        upper_c
                    ),
                    rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                        lower_c
                    ),
                    c_smem_tile,
                )
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
            else:
                var writer = SMemEpilogueWriter[
                    Self.c_smem_dim0,
                    Self.c_smem_dim1,
                    Self.epilogue_dtype,
                    Self.epc,
                    Self.num_output_warps,
                    Self.c_swizzle,
                    simd_size,
                    stage,
                    Self.rep_frag_size,
                    Self.elementwise_compute_lambda_fn.value(),
                ](UInt32(warp_id), c_tiles, c_shape, c_coord)
                writer.write_tile(
                    AccumTile(upper_frag_casted, lower_frag_casted)
                )

            # TMA store: construct coordinates (2D or 3D based on batched flag)
            comptime StoreCoords = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                stage,
                batched=Self.batched,
            ]

            var store_coords = StoreCoords(
                (c_coord[0], c_coord[1], batch_idx if Self.batched else 0),
                UInt32(warp_id),
            )
            StoreExecutor.execute[
                Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
            ](
                c_smem_tile,
                store_coords,
                self.c_tma_op[],
                UInt32(warp_id),
                UInt32(lane),
            )

            tma_wait_pipelined[
                Self.c_type,
                Self.c_rank,
                Self.c_tile_shape,
                Self.c_desc_shape,
                stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            comptime if stage > 0 or stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
    ):
        """Internal implementation of write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute (not tile index).
        Handles partial tiles that cross expert boundaries by using element-by-element
        stores for rows that would exceed m_end.

        Args:
            c_tiles: SMEM tile array for C output (TileTensor-based).
            output_stage: OutputStage with pipeline, index, and TMEM handle.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM (for bounds-checked stores).
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())
        var warp_id = get_warp_id()
        var lane = lane_id()
        var scale = expert_scale.cast[Self.accum_type]()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutorLocal = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=False,  # Always 2D for absolute coords
        ]

        var upper_frag_partial: InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = InlineArray[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for loop_stage in range(Self.num_stages):
            # Phase 1: TMEM Load
            var frags = accum_tiles[loop_stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # rebind bridges symbolic size mismatch between
            # TmemTensor.frag_size*rep and Self.fragment_size*rep
            comptime PartialType2 = InlineArray[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType2](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType2](frags.lower).copy()

            # Phase 2: Barrier Arrive
            comptime if loop_stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale and cast to epilogue dtype in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions.
            comptime cast_width_e = (
                4 // size_of[Scalar[Self.epilogue_dtype]]()
            )

            comptime for _chunk in range(Self.rep_frag_size // cast_width_e):
                comptime offset = _chunk * cast_width_e
                var src = SIMD[Self.accum_type, cast_width_e]()
                comptime for _j in range(cast_width_e):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * scale).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width_e):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(
                    Self.rep_frag_size // cast_width_e
                ):
                    comptime offset = _chunk * cast_width_e
                    var src = SIMD[Self.accum_type, cast_width_e]()
                    comptime for _j in range(cast_width_e):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * scale).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width_e):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Phase 3: SMEM Write
            var c_smem_tile = c_tiles[loop_stage % 2]

            comptime expected_size = Self.epc.fragment_size * Self.rep
            # Cast from epilogue_dtype to c_type in SIMD chunks
            # of at least 4 bytes.
            var upper_c2 = InlineArray[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )
            var lower_c2 = InlineArray[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )

            comptime cast_width_c2 = (4 // size_of[Scalar[Self.c_type]]())
            comptime for _chunk in range(Self.rep_frag_size // cast_width_c2):
                comptime offset = _chunk * cast_width_c2
                var src_u = SIMD[Self.epilogue_dtype, cast_width_c2]()
                var src_l = SIMD[Self.epilogue_dtype, cast_width_c2]()
                comptime for _j in range(cast_width_c2):
                    src_u[_j] = upper_frag_casted[offset + _j]
                    src_l[_j] = lower_frag_casted[offset + _j]
                var dst_u = src_u.cast[Self.c_type]()
                var dst_l = src_l.cast[Self.c_type]()
                comptime for _j in range(cast_width_c2):
                    upper_c2[offset + _j] = dst_u[_j]
                    lower_c2[offset + _j] = dst_l[_j]
            smem_writer.write_fragments[Self.rep](
                rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                    upper_c2
                ),
                rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                    lower_c2
                ),
                c_smem_tile,
            )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

            # Phase 4: TMA Store with bounds checking
            comptime CG2_TMA_BM = Self.c_smem_dim0 if Self.MMA_M == 256 else Self.BM
            comptime CG1_TMA_BM = Self.c_smem_dim0
            comptime TMA_BM = CG2_TMA_BM if Self.cta_group == 2 else CG1_TMA_BM
            comptime StoreCoordsLocal = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                loop_stage,
                batched=False,
            ]

            # Transpose and non-transpose have different bounds checks
            # and coordinate conventions:
            # - Non-transpose: check m_abs (token dim), store_non_transpose
            #   swaps coords so coord_m→tc1(tokens), coord_n→tc0(weights)
            # - Transpose: check per-stage token dim (in n_abs position),
            #   store_transpose_lt passes coord_m→tc0(weights),
            #   coord_n→tc1(tokens)
            var tile_needs_bounds_check: Bool

            comptime if Self.transpose_c:
                # Per-stage bounds check on token dimension (passed as
                # n_abs). Each stage writes stageN token rows.
                var stage_token_start = m_abs + UInt32(loop_stage * Self.stageN)
                tile_needs_bounds_check = (
                    stage_token_start + UInt32(Self.stageN) > m_end
                )
            else:
                tile_needs_bounds_check = m_abs + UInt32(TMA_BM) > m_end

            if tile_needs_bounds_check:
                comptime if Self.transpose_c:
                    # CUDA core fallback for unaligned group boundaries
                    Self._store_with_bounds_check_transpose[c_tensor_layout](
                        c_smem_tile.ptr,
                        c_tensor,
                        m_abs + UInt32(loop_stage * Self.stageN),
                        n_abs,
                        m_end,
                    )
                else:
                    # Slow path: element-by-element stores with bounds check
                    Self._store_with_bounds_check[c_tensor_layout](
                        c_smem_tile,
                        c_tensor,
                        m_abs,
                        n_abs + UInt32(loop_stage * Self.stageN),
                        m_end,
                        UInt32(warp_id),
                        UInt32(lane),
                    )

                # Advance TMA group counter so wait_group[1] properly
                # drains the previous stage's in-flight TMA store.
                # Without this, the group count stays stale and
                # wait_group[1] returns immediately, allowing the next
                # double-buffer stage to overwrite SMEM while TMA reads
                # it.
                if warp_id == 0 and lane == 0:
                    self.c_tma_op[].commit_group()
            else:
                # Fast path: TMA store for tiles fully within bounds
                var n_tile = n_abs / UInt32(Self.MMA_N)
                var dummy_m_tile = UInt32(0)
                var store_coords = StoreCoordsLocal(
                    (dummy_m_tile, n_tile), UInt32(warp_id)
                )

                comptime if Self.transpose_c:
                    # Transpose: coord_m→tc0→N(weights),
                    # coord_n→tc1→M(tokens)
                    store_coords.coord_m = Int(n_abs)
                    store_coords.coord_n = Int(m_abs) + loop_stage * Self.stageN
                else:
                    # Non-transpose: coord_m→tc1→M(tokens),
                    # coord_n→tc0→N(weights)
                    store_coords.coord_m = Int(m_abs)
                    store_coords.coord_n = Int(n_abs) + loop_stage * Self.stageN

                StoreExecutorLocal.execute[
                    Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
                ](
                    c_smem_tile,
                    store_coords,
                    self.c_tma_op[],
                    UInt32(warp_id),
                    UInt32(lane),
                )

            # Phase 5: TMA Wait — unconditional to drain outstanding
            # TMA stores from earlier stages when the slow path skips
            # TMA, preventing SMEM races with double-buffered tiles.
            tma_wait_pipelined[
                Self.c_type,
                Self.c_rank,
                Self.c_tile_shape,
                Self.c_desc_shape,
                loop_stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            comptime if loop_stage > 0 or loop_stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @staticmethod
    @always_inline
    def _store_with_bounds_check[
        c_tensor_layout: TensorLayout,
        c_smem_layout: TensorLayout,
    ](
        c_smem_tile: TileTensor[
            Self.c_type,
            c_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ],
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Store SMEM tile to GMEM with per-element bounds checking.

        Used when the tile crosses the expert boundary (m_abs + TMA_BM > m_end).
        Uses element-by-element stores to avoid writing past m_end.

        Args:
            c_smem_tile: SMEM tile to store (TileTensor).
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Absolute M coordinate (start of tile).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            warp_id: Current warp ID.
            lane: Current lane ID.
        """
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime c_smem_M = Self.c_smem_dim0
        comptime TMA_BM = 64 if Self.cta_group == 1 else 128
        # Ensure enough work for all threads: need thread_rows <= TMA_BM,
        # i.e., output_threads / thread_n <= TMA_BM,
        # i.e., simd_size <= stageN * TMA_BM / output_threads.
        comptime max_simd = simd_width_of[Self.c_type]()
        comptime max_allowed_simd = Self.stageN * TMA_BM // output_threads
        comptime simd_size = min(max_simd, max(1, max_allowed_simd))
        comptime alignment = align_of[SIMD[Self.c_type, simd_size]]()
        comptime thread_n = Self.stageN // simd_size
        comptime thread_layout = Layout.row_major(
            output_threads // thread_n, thread_n
        )

        # Precompute the split layout from known dimensions
        comptime split_layout = Layout(
            IntTuple(TMA_BM, Self.stageN), IntTuple(Self.c_smem_dim1, 1)
        )

        # Swizzle function
        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        # Ensure fence before reading from SMEM
        if warp_id == 0 and lane == 0:
            fence_async_view_proxy()

        # Synchronize all epilogue threads
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

        # Iterate over SMEM chunks
        comptime for i in range(c_smem_M // TMA_BM):
            var c_smem_split = c_smem_tile.tile[TMA_BM, Self.stageN](i, 0)
            comptime zipped = zipped_divide(
                upcast(split_layout, simd_size), thread_layout
            )
            # Use new Layout for idx2crd
            comptime split_layout_new = row_major[TMA_BM, Self.stageN]()

            comptime for j in range(zipped.shape[1][0].value()):
                var input_crd = RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE, j),
                    element_type=DType.uint32,
                ](thread_idx.x, j)
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
                var cmem_crd = split_layout_new.idx2crd[out_dtype=DType.uint32](
                    Int(linear_idx)
                )
                var local_i = cmem_crd[0].value()
                var local_j = cmem_crd[1].value()
                var coord_m = m_abs + UInt32(i * TMA_BM)
                var global_i = coord_m + UInt32(local_i)
                var global_j = n_abs + UInt32(local_j)

                # Bounds check: only store if within M and N boundaries.
                # The N check prevents row-major wrap-around when the
                # last N-tile extends past the logical output width.
                var in_bounds = global_i < m_end
                comptime if Self.problem_n > 0:
                    in_bounds = in_bounds and (
                        global_j + UInt32(simd_size) <= UInt32(Self.problem_n)
                    )
                if in_bounds:
                    comptime if size_of[Self.c_type]() == 2:
                        var src_ptr = c_smem_split.ptr + swizzle(linear_idx)
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var dst_ptr = c_tensor.ptr + c_tensor.layout(
                            Coord(
                                Idx(Int(global_i)),
                                Idx(Int(global_j)),
                            )
                        )
                        dst_ptr.store[width=simd_size, alignment=alignment](src)
                    else:
                        var src_ptr = c_smem_split.ptr + linear_idx
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var dst_ptr = c_tensor.ptr + c_tensor.layout(
                            Coord(
                                Idx(Int(global_i)),
                                Idx(Int(global_j)),
                            )
                        )
                        dst_ptr.store[width=simd_size, alignment=alignment](src)

    @staticmethod
    @always_inline
    def _store_with_bounds_check_transpose[
        c_tensor_layout: TensorLayout,
    ](
        c_smem_ptr: UnsafePointer[
            Scalar[Self.c_type], _, address_space=AddressSpace.SHARED
        ],
        c_tensor: TileTensor[Self.c_type, c_tensor_layout, MutAnyOrigin],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
    ):
        """CUDA core fallback for unaligned group boundaries (transpose).

        With SWIZZLE_32B, each swizzle_width chunk (16 bf16 elements) in
        SMEM maps to one TMA row. We read flat SMEM with swizzle(simd_size
        * tidx) and decompose tidx into (vec_chunkM_idx, n_idx, chunk_idx)
        to compute global coordinates.

        This matches the reference implementation in grouped_matmul_sm100_1d1d.

        Args:
            c_smem_ptr: Raw pointer to SMEM tile.
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Token start for this stage.
            n_abs: Weight start (absolute N coordinate).
            m_end: Token boundary (exclusive).
        """
        comptime simd_size = simd_width_of[Self.c_type]()
        comptime swizzle_width = Self.c_swizzle.bytes() // size_of[
            Self.c_type
        ]()
        comptime chunkM = swizzle_width
        comptime vec_chunkM = chunkM // simd_size
        comptime chunk_num = Self.stage_contiguous_size // chunkM
        comptime logical_size = chunk_num * Self.stageN * vec_chunkM
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime assert (
            logical_size % output_threads == 0
        ), "logical_size must be divisible by output_threads"
        comptime value_shape = logical_size // output_threads
        comptime cN = c_tensor.static_shape[1]
        comptime smem_alignment = align_of[SIMD[Self.c_type, simd_size]]()

        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        var n_inbound = Int32(m_end) - Int32(m_abs)

        comptime for v in range(value_shape):
            comptime thread_offset = v * output_threads
            var tidx = UInt32(thread_idx.x) + UInt32(thread_offset)
            var rest, vec_chunkM_idx = divmod(tidx, UInt32(vec_chunkM))
            var n_idx = rest % UInt32(Self.stageN)
            if Int32(n_idx) >= min(n_inbound, Int32(Self.stageN)):
                continue
            var src_idx = UInt32(simd_size) * tidx
            var c_smem_idx = swizzle(src_idx)
            var val_vec = (c_smem_ptr + c_smem_idx).load[
                width=simd_size,
                alignment=smem_alignment,
            ]()
            var chunk_idx = rest // UInt32(Self.stageN)
            # m_abs = token index, n_abs = weight index
            var global_token = m_abs + n_idx
            var global_weight = n_abs + (
                chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
            ) * UInt32(simd_size)
            if global_token < m_end and global_weight < UInt32(cN):
                (
                    c_tensor.ptr + global_token * UInt32(cN) + global_weight
                ).store[alignment=smem_alignment](val_vec)

    # ========== Residual Add Support ==========
    # Methods for D = lambda(accum) + beta * C residual operations

    @always_inline
    def write_with_residual(
        self,
        out_tiles: Self.CTileArray,
        stage: Self.Stage,
        src_tile: Self.CTileArray,  # Source C from epilogue load SMEM
        src_stage_idx: UInt32,  # Stage index for source C tile
        beta: Scalar[Self.c_type],  # Residual scale factor
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with residual: D = lambda(accum) + beta * C.

        This method extends the standard write() to add a residual term loaded
        from source tensor C in shared memory. The epilogue load warp pre-fetches
        C tiles into src_tile before this method is called.

        Pipeline:
        1. Load accum from TMEM to registers
        2. Apply epilogue lambda (if present)
        3. Load C fragment from source SMEM
        4. Compute D = accum + beta * C
        5. Write D to output SMEM and TMA store to GMEM

        Args:
            out_tiles: Output SMEM tile array (for D output).
            stage: OutputStage with pipeline, index, and TMEM handle.
            src_tile: Source C SMEM tile array (TileTensor-based, from
                epilogue load warp via smem.src_tiles()).
            src_stage_idx: Stage index into src_tile (0 or 1 for double-buffer).
            beta: Residual scale factor.
            tile_coord: (m_tile, n_tile) coordinates.
            shape: (M, N) problem dimensions.
            elect_one_warp: Whether this warp is elected for coordination.
        """
        self._copy_to_gmem_with_residual(
            out_tiles,
            stage,
            src_tile,
            src_stage_idx,
            beta,
            tile_coord,
            shape,
        )

    @always_inline
    def _copy_to_gmem_with_residual(
        self,
        out_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        src_tiles: Self.CTileArray,
        src_stage_idx: UInt32,
        beta: Scalar[Self.c_type],
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → (+ beta*C) → SMEM → GMEM pipeline with residual.

        Internal implementation that adds residual term from source SMEM.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
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
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        var upper_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = InlineArray[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        # Get source C tile for residual add
        var src_smem_tile = src_tiles[Int(src_stage_idx) % 2]

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()
            var casted = frags.cast[Self.epilogue_dtype]()

            comptime for _i in range(Self.rep_frag_size):
                upper_frag_casted[_i] = rebind[Scalar[Self.epilogue_dtype]](
                    casted.upper[_i]
                )

            comptime for _i in range(Self.rep_frag_size):
                lower_frag_casted[_i] = rebind[Scalar[Self.epilogue_dtype]](
                    casted.lower[_i]
                )

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Apply epilogue lambda (if present)
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    var _epilogue_result = (
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
                    upper_frag_casted = _epilogue_result[0].copy()
                    lower_frag_casted = _epilogue_result[1].copy()

            # 3. Apply residual: D = accum + beta * C in registers
            # Load C from source SMEM tile using the same per-lane fragment
            # coordinate mapping as EpilogueApplier. No extra barrier syncs
            # needed since each thread loads its own C elements independently.
            comptime residual_swizzle = make_swizzle[
                Self.c_type, Self.c_swizzle
            ]()
            var _residual_result = (
                epilogue_applier.add_residual_to_both_fragments[
                    Self.epilogue_dtype,
                    Self.rep_frag_size,
                    Self.is_lower_frag_required,
                    Self.c_type,
                    Self.c_smem_dim1,
                    residual_swizzle,
                ](
                    upper_frag_casted,
                    lower_frag_casted,
                    UInt32(stage),
                    src_smem_tile.ptr,
                    beta.cast[Self.epilogue_dtype](),
                )
            )
            upper_frag_casted = _residual_result[0].copy()
            lower_frag_casted = _residual_result[1].copy()

            # 4. Write to output SMEM
            var c_smem_tile = out_tiles[stage % 2]

            comptime if (
                Self.register_based_epilogue
                or not Self.elementwise_compute_lambda_fn
            ):
                comptime expected_size = Self.epc.fragment_size * Self.rep
                comptime assert (
                    Self.rep_frag_size == expected_size
                ), "Fragment sizes must match"
                # Cast from epilogue_dtype to c_type in SIMD chunks
                # of at least 4 bytes.
                var upper_c3 = InlineArray[Scalar[Self.c_type], expected_size](
                    uninitialized=True
                )
                var lower_c3 = InlineArray[Scalar[Self.c_type], expected_size](
                    uninitialized=True
                )

                comptime cast_width_c3 = (4 // size_of[Scalar[Self.c_type]]())
                comptime for _chunk in range(
                    Self.rep_frag_size // cast_width_c3
                ):
                    comptime offset = _chunk * cast_width_c3
                    var src_u = SIMD[Self.epilogue_dtype, cast_width_c3]()
                    var src_l = SIMD[Self.epilogue_dtype, cast_width_c3]()
                    comptime for _j in range(cast_width_c3):
                        src_u[_j] = upper_frag_casted[offset + _j]
                        src_l[_j] = lower_frag_casted[offset + _j]
                    var dst_u = src_u.cast[Self.c_type]()
                    var dst_l = src_l.cast[Self.c_type]()
                    comptime for _j in range(cast_width_c3):
                        upper_c3[offset + _j] = dst_u[_j]
                        lower_c3[offset + _j] = dst_l[_j]
                smem_writer.write_fragments[Self.rep](
                    rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                        upper_c3
                    ),
                    rebind[InlineArray[Scalar[Self.c_type], expected_size]](
                        lower_c3
                    ),
                    c_smem_tile,
                )
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
            else:
                var writer = SMemEpilogueWriter[
                    Self.c_smem_dim0,
                    Self.c_smem_dim1,
                    Self.epilogue_dtype,
                    Self.epc,
                    Self.num_output_warps,
                    Self.c_swizzle,
                    simd_size,
                    stage,
                    Self.rep_frag_size,
                    Self.elementwise_compute_lambda_fn.value(),
                ](UInt32(warp_id), out_tiles, c_shape, c_coord)
                writer.write_tile(
                    AccumTile(upper_frag_casted, lower_frag_casted)
                )

            # 5. TMA store to GMEM
            comptime StoreCoords = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                stage,
                batched=Self.batched,
            ]
            var store_coords = StoreCoords(c_coord, UInt32(warp_id))
            StoreExecutor.execute[
                Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
            ](
                c_smem_tile,
                store_coords,
                self.c_tma_op[],
                UInt32(warp_id),
                UInt32(lane),
            )
            tma_wait_pipelined[
                Self.c_type,
                Self.c_rank,
                Self.c_tile_shape,
                Self.c_desc_shape,
                stage == Self.num_stages - 1,
            ](self.c_tma_op[])

            comptime if stage > 0 or stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
