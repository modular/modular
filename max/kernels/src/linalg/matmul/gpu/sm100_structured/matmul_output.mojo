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

"""SM100 Matmul Output Pipeline - TMEM → SMEM → GMEM epilogue.

This module contains the output pipeline code for SM100 matmul:
- copy_accum_to_gmem: Core epilogue pipeline (TMEM → Registers → SMEM → GMEM)
- multi_stage_store_C: Output pipeline orchestration for standard matmul
- multi_stage_store_C_split_k: Output pipeline for split-K matmul

The output pipeline handles:
- Loading accumulated results from Tensor Memory (TMEM)
- Applying optional epilogue operations (bias, activation)
- Writing to shared memory via st.matrix instructions
- Transferring to global memory via TMA async stores
"""

from collections import OptionalReg
from sys import simd_width_of

from gpu import WARP_SIZE, thread_idx
from gpu import lane_id
from gpu import warp_id as get_warp_id
from gpu.host.nvidia.tma import TensorMapSwizzle
from .barriers import WarpGroupBarrier
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile

from utils.index import IndexList

from linalg.utils import elementwise_compute_lambda_type
from .pipeline import ProducerConsumerPipeline
from .tile_pipeline import OutputStage as OutputStageType
from .tile_scheduler_splitk import (
    TileScheduler as TileSchedulerSplitK,
    WorkInfo as WorkInfoSplitK,
)
from .tile_writer import (
    AccumBarrier,
    AccumTile,
    EpilogueApplier,
    SMemEpilogueWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    TMEMToSMemWriter,
    load_tmem_fragments,
    tma_wait_pipelined,
)
from linalg.structuring import SMemTileArrayType


@always_inline
fn accum_arrive[
    cta_group: Int, num_stages: Int
](stage: OutputStageType[num_stages]):
    """Signal accumulator arrival to unblock MMA pipeline."""
    AccumBarrier[cta_group].arrive(stage.pipeline, stage.index)


@always_inline
fn copy_accum_to_gmem[
    c_type: DType,
    c_layout: Layout,
    c_smem_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    /,
    *,
    repeat: Int,
    accum_type: DType,
    cta_group: Int,
    epilogue_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    num_output_warps: UInt,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    transpose_c: Bool = False,
](
    c_tiles: SMemTileArrayType[
        c_type,
        c_smem_layout,
        num_output_stages,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    output_stage: OutputStageType[num_accum_pipeline_stages],
    c_coord: Tuple[UInt32, UInt32],
    c_shape: Tuple[UInt32, UInt32],
):
    """Epilogue pipeline: TMEM → Registers → SMEM → GMEM (via TMA).

    Args:
        c_tiles: Shared memory tiles for output staging.
        c_tma_op: TMA descriptor for C matrix.
        output_stage: Self-contained stage with pipeline, stage index, and TMEM offset.
        c_coord: (M, N) tile coordinates.
        c_shape: (M, N) matrix dimensions.
    """
    # Extract TMEM offset from self-contained OutputStage
    var tmem_offset = output_stage.tmem.offset()
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime simd_size = simd_width_of[c_type]()

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime stage_contiguous_size = c_smem_layout.shape[1].value()
    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
    comptime rep_frag_size = repeat * fragment_size

    # Fragment variables initialized by load_tmem_fragments
    var upper_frag_casted: SIMD[epilogue_dtype, rep_frag_size]
    var lower_frag_casted: SIMD[epilogue_dtype, rep_frag_size]

    comptime is_lower_frag_required = not (cta_group == 1 and BM == 64)
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    var M = c_shape[0]
    var N = c_shape[1]

    var warp_id = get_warp_id()
    var lane = lane_id()

    # Create SMEM writer for fragment → SMEM transfers (used in default path)
    comptime SMEMWriter = TMEMToSMemWriter[
        c_type,
        accum_type,
        c_smem_layout,
        BM,
        BN,
        MMA_M,
        MMA_N,
        stageN,
        cta_group,
        Int(num_output_warps),
        c_swizzle,
        transpose_c,
    ]
    var smem_writer = SMEMWriter(warp_id, lane)

    # TMA store executor type for SMEM → GMEM transfers (static methods)
    comptime StoreExecutor = TMAStoreExecutor[
        c_type,
        c_smem_layout,
        BM,
        BN,
        MMA_M,
        MMA_N,
        stageN,
        stage_contiguous_size,
        cta_group,
        c_swizzle,
        transpose_c,
        is_lower_frag_required,
    ]

    # Create epilogue applier for register-based epilogue (if lambda provided)
    comptime EpilogueApplierType = EpilogueApplier[
        MMA_M,
        stageN,
        num_stages,
        repeat,
        cta_group,
        transpose_c,
    ]
    var epilogue_applier = EpilogueApplierType(warp_id, lane)

    # Base GMEM coordinates
    var c_row = c_coord[0] * UInt(BM)
    var c_col = c_coord[1] * UInt(MMA_N)

    @parameter
    for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + (stage * stageN)

        # Load and cast fragments from TMEM using tile_writer helper
        upper_frag_casted, lower_frag_casted = load_tmem_fragments[
            accum_type,
            epilogue_dtype,
            fragment_size,
            is_lower_frag_required,
            data_paths,
            bits,
            repeat,
        ](stage_tmem_addr)

        @parameter
        if stage == num_stages - 1:
            accum_arrive[cta_group, num_accum_pipeline_stages](output_stage)

        @parameter
        if elementwise_compute_lambda_fn:

            @parameter
            if register_based_epilogue:
                # Use EpilogueApplier for register-based epilogue
                upper_frag_casted, lower_frag_casted = (
                    epilogue_applier.apply_to_both_fragments[
                        epilogue_dtype,
                        rep_frag_size,
                        elementwise_compute_lambda_fn.value(),
                        is_lower_frag_required,
                    ](
                        upper_frag_casted,
                        lower_frag_casted,
                        UInt32(stage),
                        c_row,
                        c_col,
                    )
                )

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_tiles[stage % 2]

        # Use TMEMToSMemWriter for the default path (register-based epilogue
        # or no lambda). Keep existing code for SMEM epilogue path which needs
        # access to reshaped tiles.
        @parameter
        if register_based_epilogue or not elementwise_compute_lambda_fn:
            # Default path: use TMEMToSMemWriter for clean fragment → SMEM
            comptime expected_size = SMEMWriter.Config.fragment_size * repeat
            constrained[
                rep_frag_size == expected_size,
                (
                    "Fragment sizes must match: rep_frag_size vs"
                    " SMEMWriter.Config.fragment_size * repeat"
                ),
            ]()
            smem_writer.write_fragments[repeat](
                rebind[SIMD[c_type, expected_size]](
                    upper_frag_casted.cast[c_type]()
                ),
                rebind[SIMD[c_type, expected_size]](
                    lower_frag_casted.cast[c_type]()
                ),
                c_smem_tile,
            )
            WarpGroupBarrier[Int(num_output_warps) * WARP_SIZE].sync()
        else:
            # SMEM epilogue path: create stage-specific writer
            var writer = SMemEpilogueWriter[
                epilogue_dtype,
                BM,
                BN,
                MMA_M,
                MMA_N,
                cta_group,
                Int(num_output_warps),
                c_swizzle,
                transpose_c,
                is_lower_frag_required,
                num_stages,
                simd_size,
                stage,
                rep_frag_size,
                elementwise_compute_lambda_fn.value(),
            ](warp_id, c_tiles, c_shape, c_coord)
            writer.write_tile(AccumTile(upper_frag_casted, lower_frag_casted))

        # Compute TMA store coordinates using the dedicated component
        comptime StoreCoords = TMAStoreCoords[
            BM,
            BN,
            MMA_M,
            MMA_N,
            stageN,
            cta_group,
            c_smem_layout.shape[0].value(),
            stage,
        ]
        var store_coords = StoreCoords(c_coord, warp_id)

        # Execute TMA store with proper SMEM tiling via dedicated executor
        StoreExecutor.execute[c_layout, c_desc_layout](
            c_smem_tile, store_coords, c_tma_op, warp_id, lane
        )

        # Wait with pipelining:
        tma_wait_pipelined[
            c_type, c_layout, c_desc_layout, stage == num_stages - 1
        ](c_tma_op)

        @parameter
        if stage > 0 or stage == num_stages - 1:
            # Guard the tma read from shared memory is done.
            WarpGroupBarrier[Int(num_output_warps) * WARP_SIZE].sync()


@always_inline
fn multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    /,
    *,
    input_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: UInt,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
    max_tmem_cols: UInt = 512,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    transpose_c: Bool = False,
](
    c_tiles: SMemTileArrayType[
        c_type,
        c_smem_layout,
        num_output_stages,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    stage: OutputStageType[num_accum_pipeline_stages],
    work_tile_coord: Tuple[UInt32, UInt32],
    elect_one_warp: Bool,
    M: UInt32,
    N: UInt32,
):
    """Orchestrate output from TMEM to GMEM via shared memory.

    Args:
        c_tiles: Shared memory tiles for output staging.
        c_tma_op: TMA descriptor for C matrix.
        stage: Self-contained output stage with pipeline, stage index, and TMEM offset.
        work_tile_coord: (M, N) tile coordinates.
        elect_one_warp: Whether this warp is elected.
        M: Matrix M dimension.
        N: Matrix N dimension.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    # Epilogue dtype: use c_type for bf16, float32 for FP8 (GEX-2630 workaround)
    comptime epilogue_dtype = c_type if input_type is DType.bfloat16 else DType.float32

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()

    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    comptime data_paths = 16
    comptime bits = 256
    comptime rep = stageN // (bits // 32)

    _ = get_warp_id()  # Required for warp setup

    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
    comptime rep_frag_size = rep * fragment_size
    _ = SIMD[epilogue_dtype, rep_frag_size]()  # Force type instantiation
    _ = SIMD[epilogue_dtype, rep_frag_size]()

    comptime is_lower_frag_required = not (cta_group == 1 and BM == 64)

    copy_accum_to_gmem[
        repeat=rep,
        accum_type=accum_type,
        cta_group=cta_group,
        epilogue_dtype=epilogue_dtype,
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        num_output_warps=num_output_warps,
        c_swizzle=c_swizzle,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        transpose_c=transpose_c,
    ](
        c_tiles,
        c_tma_op,
        stage,  # Self-contained OutputStage
        work_tile_coord,
        (M, N),
    )


@always_inline
fn multi_stage_store_C_split_k[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    reduction_layout: Layout,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    /,
    *,
    input_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: UInt,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
    max_tmem_cols: UInt = 512,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    transpose_c: Bool = False,
](
    scheduler: TileSchedulerSplitK,
    reduction_tensor: LayoutTensor[accum_type, reduction_layout, MutAnyOrigin],
    c_tiles: SMemTileArrayType[
        c_type,
        c_smem_layout,
        num_output_stages,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    stage: OutputStageType[num_accum_pipeline_stages],
    work_info: WorkInfoSplitK,
    elect_one_warp: Bool,
    M: UInt32,
    N: UInt32,
):
    """Split-K output pipeline with reduction.

    Args:
        scheduler: Split-K tile scheduler for reduction.
        reduction_tensor: Tensor for accumulating partial results.
        c_tiles: Shared memory tiles for output staging.
        c_tma_op: TMA descriptor for C matrix.
        stage: Self-contained output stage with pipeline, stage index, and TMEM offset.
        work_info: Current work item info.
        elect_one_warp: Whether this warp is elected.
        M: Matrix M dimension.
        N: Matrix N dimension.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    comptime epilogue_dtype = c_type if input_type is DType.bfloat16 else DType.float32

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()

    comptime data_paths = 16
    comptime bits = 256
    comptime stage_rep = stageN // (bits // 32)

    var epilogue_thread_idx = thread_idx.x

    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE

    var is_last_split = scheduler.reduction(
        reduction_tensor,
        stage.tmem.offset(),
        epilogue_thread_idx,
        work_info,
    )

    # If not last split, signal and exit
    if not is_last_split:
        accum_arrive[cta_group, num_accum_pipeline_stages](stage)
        return

    copy_accum_to_gmem[
        repeat=stage_rep,
        accum_type=accum_type,
        cta_group=cta_group,
        epilogue_dtype=epilogue_dtype,
        block_tile_shape=block_tile_shape,
        mma_shape=mma_shape,
        num_output_warps=num_output_warps,
        c_swizzle=c_swizzle,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        transpose_c=transpose_c,
    ](
        c_tiles,
        c_tma_op,
        stage,  # Self-contained OutputStage
        (work_info.m, work_info.n),
        (M, N),
    )
