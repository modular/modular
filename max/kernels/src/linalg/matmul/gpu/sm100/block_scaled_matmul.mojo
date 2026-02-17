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

from collections import OptionalReg
from math import align_up, ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import align_of, env_get_bool, simd_width_of, size_of

from bit import next_power_of_two, prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
    cluster_wait,
    cluster_arrive_relaxed,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu import block_id_in_cluster, block_idx, lane_id, thread_idx, block_idx
from gpu import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    PDLLevel,
    wait_on_dependent_grids,
)
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from gpu.compute.arch.tcgen05 import *
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.int_tuple import IntTuple
from layout.layout import blocked_product, make_layout, flatten, coalesce
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_tuple import idx2crd, crd2idx
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    tile_sf_layout_k_major,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_BlockScaled_SS
from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .config import BlockScaledMatmulConfig
from ..tile_scheduler import RasterOrder
from .tile_scheduler import (
    TileScheduler,
    WorkInfo,
)

from ..profiler import (
    MatmulProfileWarp,
    MatmulWarpSpecializationWorkSpaceManager,
)
from .pipeline import ProducerConsumerPipeline
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_K_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from .matmul import (
    WarpRole,
    RLayout32Bits,
    f32_frag_to_smem,
    stsm_helper,
    shared_memory_epilogue_transpose,
    shared_memory_epilogue,
    _compute_register_lambda_fn,
    register_epilogue,
    accum_arrive,
)


struct B200BlockScaledMatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime AType = Scalar[Self.a_type]
    comptime BType = Scalar[Self.b_type]
    comptime CType = Scalar[Self.c_type]
    comptime AScalesType = Scalar[Self.sfa_dtype]
    comptime BScalesType = Scalar[Self.sfb_dtype]

    comptime a_smem_size = (
        Self.BM * Self.BK * Int(Self.config.num_pipeline_stages)
    )
    comptime b_smem_size = (
        Self.BN * Self.BK * Int(Self.config.num_pipeline_stages)
    )
    comptime c_smem_size = (
        Self.OutputM * Self.OutputN * Int(Self.config.num_output_stages)
    )

    comptime sfa_smem_size = (
        Self.config.num_sf_k_tiles
        * (Self.BM // SF_MN_GROUP_SIZE)
        * Self.config.sf_block_atom_size
        * Int(Self.config.num_pipeline_stages)
    )
    comptime sfb_smem_size = (
        Self.config.num_sf_k_tiles
        * (align_up(Self.MMA_N, SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE)
        * Self.config.sf_block_atom_size
        * Int(Self.config.num_pipeline_stages)
    )

    comptime num_group_pipeline_stages = (
        Self.config.num_pipeline_stages // Self.config.k_group_size
    )

    # AB pipelines
    var a_smem: InlineArray[Self.AType, Self.a_smem_size]
    var b_smem: InlineArray[Self.BType, Self.b_smem_size]
    var c_smem: InlineArray[Self.CType, Self.c_smem_size]
    var sfa_smem: InlineArray[Self.AScalesType, Self.sfa_smem_size]
    var sfb_smem: InlineArray[Self.BScalesType, Self.sfb_smem_size]

    var tma_mma_mbars: InlineArray[
        SharedMemBarrier, Int(Self.num_group_pipeline_stages) * 2
    ]
    # ACCUM
    var accum_mbars: InlineArray[
        SharedMemBarrier, Int(Self.config.num_accum_pipeline_stages) * 2
    ]

    # CLC
    var clc_mbars_full: InlineArray[
        SharedMemBarrier, Int(Self.config.num_clc_pipeline_stages)
    ]
    var clc_mbars_empty: InlineArray[
        SharedMemBarrier, Int(Self.config.num_clc_pipeline_stages)
    ]
    var clc_throttle_mbars: InlineArray[
        SharedMemBarrier, Int(Self.config.num_clc_pipeline_stages) * 2
    ]
    var clc_response: InlineArray[
        UInt128, Int(Self.config.num_clc_pipeline_stages)
    ]

    # TMEM
    var tmem_dealloc_mbar: InlineArray[SharedMemBarrier, 1]
    var tmem_addr: InlineArray[UInt32, 1]


@always_inline
fn load_AB_SFA_SFB[
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    sfa_desc_layout: Layout,
    sfb_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    sfa_smem_layout: Layout,
    sfb_smem_layout: Layout,
    num_pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    num_sf_k_tiles: Int,
    cta_group: Int = 1,
    k_group_size: UInt = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    sfa_tma_op: TMATensorTile[sfa_dtype, sfa_layout, sfa_desc_layout],
    sfb_tma_op: TMATensorTile[sfb_dtype, sfb_layout, sfb_desc_layout],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    sfa_smem: LayoutTensorIter[
        sfa_dtype,
        sfa_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    sfb_smem: LayoutTensorIter[
        sfb_dtype,
        sfb_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[UInt, UInt, UInt],
    work_tile_coord: Tuple[UInt, UInt, UInt],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt32,
    elect_one_cta: Bool,
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime sfa_expected_bytes = sfa_smem_layout.size() * size_of[sfa_dtype]()
    comptime sfb_expected_bytes = sfb_smem_layout.size() * size_of[sfb_dtype]()

    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = (
        cta_group
        * (
            a_expected_bytes
            + b_expected_bytes
            + sfa_expected_bytes
            + sfb_expected_bytes
        )
    ) * Int(k_group_size)

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[1].value()
    comptime b_tma_rows = b_desc_layout.shape[1].value()

    var stage = load_mma_pipeline.producer_stage()
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)
    var a_gmem_slice_coord = peer_cta_coord[2] * UInt(
        a_tma_rows
    ) + work_tile_coord[0] * UInt(BM)
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * UInt(b_tma_rows)
        + peer_cta_coord[0] * UInt(BN)
        + work_tile_coord[1] * UInt(MMA_N)
    )
    var batch_coord = work_tile_coord[2]

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem.next(offset)[]
            var b_smem_tile = b_smem.next(offset)[]
            var sfa_smem_tile = sfa_smem.next(offset)[]
            var sfb_smem_tile = sfb_smem.next(offset)[]

            var a_smem_slice = type_of(a_smem_tile)(
                a_smem_tile.ptr + peer_cta_coord[2] * UInt(a_tma_load_size)
            )
            var b_smem_slice = type_of(b_smem_tile)(
                b_smem_tile.ptr + peer_cta_coord[1] * UInt(b_tma_load_size)
            )

            a_tma_op.async_multicast_load_3d[cta_group](
                a_smem_slice,
                tma_mbar[0],
                (
                    UInt(iter_idx + j) * UInt(BK),
                    UInt(a_gmem_slice_coord),
                    UInt(batch_coord),
                ),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load_3d[cta_group](
                b_smem_slice,
                tma_mbar[0],
                (
                    UInt(iter_idx + j) * UInt(BK),
                    UInt(b_gmem_slice_coord),
                    UInt(batch_coord),
                ),
                b_multicast_mask,
            )
            sfa_tma_op.async_copy_5d[cta_group](
                sfa_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(iter_idx + j) * num_sf_k_tiles,
                    Int(work_tile_coord[0]) * (BM // SF_MN_GROUP_SIZE),
                    Int(batch_coord),
                ),
            )

            sfb_tma_op.async_copy_5d[cta_group](
                sfb_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(iter_idx + j) * num_sf_k_tiles,
                    (Int(work_tile_coord[1]) * MMA_N) // SF_MN_GROUP_SIZE,
                    Int(batch_coord),
                ),
            )


@always_inline
fn consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    sfa_smem_layout: Layout,
    sfb_smem_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    scaling_kind: UMMAKind,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    SFA_NUM_COLS: Int,
    SFB_NUM_COLS: Int,
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: UInt = 1,
](
    tmem_addr: UInt32,
    sfa_tmem: UInt32,
    sfb_tmem: UInt32,
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    sfa_smem_iter: LayoutTensorIter[
        sfa_dtype,
        sfa_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    sfb_smem_iter: LayoutTensorIter[
        sfb_dtype,
        sfb_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_BlockScaled_SS[
        c_type,
        a_type,
        b_type,
        sfa_dtype,
        sfb_dtype,
        scaling_kind,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
    k_start: UInt32,
    work_tile_coord: Tuple[UInt, UInt],
):
    comptime BM = block_tile_shape[0]
    comptime MMA_N = mma_shape[1]

    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_iter.next(offset)[]
            var b_smem_tile = b_smem_iter.next(offset)[]
            var sfa_smem_tile = sfa_smem_iter.next(offset)[]
            var sfb_smem_tile = sfb_smem_iter.next(offset)[]

            var sfa_tmem_offset = sfa_tmem + offset * UInt32(SFA_NUM_COLS)
            var sfb_tmem_offset = sfb_tmem + offset * UInt32(SFB_NUM_COLS)

            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                sfa_smem_tile,
                sfb_smem_tile,
                tmem_addr,
                sfa_tmem_offset,
                sfb_tmem_offset,
                init_c=(
                    (iter_idx + j) == k_start
                ),  # Initialize C on first iteration
                work_tile_coord=work_tile_coord,
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


@always_inline
fn multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: Int,
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
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,  # if false it will perform epilogue on data in shared memory
    transpose_c: Bool = False,
](
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    tmem_addr: UInt32,
    alpha: Float32,
    work_tile_coord: Tuple[UInt32, UInt32, UInt32],
    elect_one_warp: Bool,
    M: UInt32,
    N: UInt32,
):
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    # TODO (GEX-2630): This is a temporary workaround to support float32 compute epilogue for FP8 models for which we use compute lambda for dequantization.
    # We should remove this once GEX-2630 is fixed.
    comptime epilogue_dtype = (
        c_type if input_type == DType.bfloat16 else DType.float32
    )

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()

    comptime bits = 256
    comptime rep = stageN // (bits // 32)

    var mma_output_stage = mma_output_pipeline.consumer_stage()
    mma_output_pipeline.wait_producer()

    var tmem_offset = mma_output_stage * UInt32(stage_stride_cols) + tmem_addr

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
        c_iter,
        c_tma_op,
        mma_output_pipeline,
        mma_output_stage,
        tmem_offset,
        work_tile_coord,
        (M, N),
        alpha,
    )


@always_inline
fn copy_accum_to_gmem[
    c_type: DType,
    c_layout: Layout,
    c_smem_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: Int,
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
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    transpose_c: Bool = False,
](
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    mma_output_stage: UInt32,
    tmem_offset: UInt32,
    c_coord: Tuple[UInt32, UInt32, UInt32],
    c_shape: Tuple[UInt32, UInt32],
    alpha: Float32 = 1.0,
):
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
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep_frag_size = repeat * fragment_size
    var upper_frag_partial: SIMD[accum_type, rep_frag_size]
    var lower_frag_partial = SIMD[accum_type, rep_frag_size]()
    var upper_frag_casted: SIMD[epilogue_dtype, rep_frag_size]
    var lower_frag_casted = SIMD[epilogue_dtype, rep_frag_size]()

    comptime is_lower_frag_required = not (cta_group == 1 and BM == 64)
    comptime cg2_num_stages = (
        MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    )
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    var M = c_shape[0]
    var N = c_shape[1]

    # stmatrix related
    comptime st_matrix_swizzle = c_swizzle
    comptime swizzle_width = c_swizzle.bytes() // size_of[c_type]()
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # lets keep track of the of the starting row and column in GMEM
    var c_row = c_coord[0] * UInt32(BM)
    var c_col = c_coord[1] * UInt32(MMA_N)

    @parameter
    for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        upper_frag_partial = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeat,
            dtype=accum_type,
            pack=False,
            width=rep_frag_size,
        ](stage_tmem_addr)

        @parameter
        if is_lower_frag_required:
            lower_frag_partial = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=repeat,
                dtype=accum_type,
                pack=False,
                width=rep_frag_size,
            ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        @parameter
        if stage == num_stages - 1:
            accum_arrive[cta_group](mma_output_pipeline, mma_output_stage)

        # Apply tensor scale factor
        upper_frag_partial = upper_frag_partial * alpha.cast[accum_type]()

        @parameter
        if is_lower_frag_required:
            lower_frag_partial = lower_frag_partial * alpha.cast[accum_type]()

        upper_frag_casted = upper_frag_partial.cast[epilogue_dtype]()

        @parameter
        if is_lower_frag_required:
            lower_frag_casted = lower_frag_partial.cast[epilogue_dtype]()

        @parameter
        if elementwise_compute_lambda_fn:

            @parameter
            if register_based_epilogue:
                register_epilogue[
                    Int(MMA_M),
                    data_paths,
                    Int(num_stages),
                    bits,
                    Int(stage),
                    Int(stageN),
                    elementwise_compute_lambda_fn.value(),
                    Int(num_output_warps),
                    epilogue_dtype,
                    upper_frag_casted.size,
                    Int(repeat),
                    transpose_c,
                    cta_group=cta_group,
                    is_lower_frag_required=is_lower_frag_required,
                ](upper_frag_casted, lower_frag_casted, c_row, c_col, N)

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_iter.next(stage % 2)[]

        @parameter
        if transpose_c:
            # if stage_contiguous_size is 128, we need to split the shared
            # memory into two stageNxswizzle_width row-major tiles due to the
            # limitation of 128B TMA swizzle. However, for easier programming,
            # we reshape the tile contiguous row_major(stageN, swizzle_width)
            # chunks.
            @parameter
            if is_lower_frag_required:
                comptime tile_width = 32
                comptime smem_swblock_layout = Layout.row_major(
                    stageN, 2, tile_width
                )
                comptime num_swblocks = stage_contiguous_size // swizzle_width
                comptime smem_logical_layout = Layout(
                    flatten([num_swblocks, smem_swblock_layout.shape]),
                    flatten(
                        [stageN * swizzle_width, smem_swblock_layout.stride]
                    ),
                )

                var new_smem = LayoutTensor[
                    c_type,
                    smem_logical_layout,
                    c_smem_tile.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = c_smem_tile.alignment,
                ](c_smem_tile.ptr)
                warp_j, warp_i = divmod(Int(warp_id), 2)
                var _c_smem_warp_tile = new_smem.tile[1, stageN, 1, tile_width](
                    warp_j, 0, warp_i, 0
                )
                var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                    coalesce(_c_smem_warp_tile.layout)
                ]()

                var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                    stageN, data_paths
                ](0, 0)
                var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                    stageN, data_paths
                ](0, 1)

                warp_offset = warp_i * tile_width
                stsm_helper[swizzle, UInt(stageN), transpose_c](
                    upper_frag_casted,
                    c_smem_warp_tile_upper,
                    UInt32(warp_offset),
                )

                warp_offset += tile_width // 2
                stsm_helper[swizzle, UInt(stageN), transpose_c](
                    lower_frag_casted,
                    c_smem_warp_tile_lower,
                    UInt32(warp_offset),
                )

                # Guard the write to shared memory is done.
                named_barrier[Int32(num_output_warps * UInt(WARP_SIZE))]()

                @parameter
                if elementwise_compute_lambda_fn:

                    @parameter
                    if not register_based_epilogue:
                        shared_memory_epilogue_transpose[
                            UInt(stage),
                            UInt(stageN),
                            new_smem.dtype,
                            new_smem.layout,
                            swizzle,
                            elementwise_compute_lambda_fn.value(),
                            Int(num_output_warps),
                            2,
                            MMA_M,
                            BN,
                            cta_group,
                        ](
                            M,
                            N,
                            UInt(c_col),
                            UInt(c_row),
                            new_smem,
                            UInt(warp_i),
                            UInt(warp_j),
                        )
            else:
                comptime tile_width = 16
                comptime smem_logical_layout = Layout.row_major(
                    stageN, 4, tile_width
                )

                var new_smem = LayoutTensor[
                    c_type,
                    smem_logical_layout,
                    c_smem_tile.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = c_smem_tile.alignment,
                ](c_smem_tile.ptr)
                var _c_smem_warp_tile = new_smem.tile[stageN, 1, tile_width](
                    0, Int(warp_id), 0
                )
                var c_smem_warp_tile = _c_smem_warp_tile.reshape[
                    coalesce(_c_smem_warp_tile.layout)
                ]()

                var c_smem_warp_tile_upper = c_smem_warp_tile
                var c_smem_warp_tile_lower = c_smem_warp_tile
                warp_offset = Int(warp_id) * tile_width
                stsm_helper[swizzle, UInt(stageN), transpose_c](
                    upper_frag_casted,
                    c_smem_warp_tile_upper,
                    UInt32(warp_offset),
                )

                # Guard the write to shared memory is done.
                named_barrier[Int32(num_output_warps * UInt(WARP_SIZE))]()

                @parameter
                if elementwise_compute_lambda_fn:

                    @parameter
                    if not register_based_epilogue:
                        shared_memory_epilogue_transpose[
                            UInt(stage),
                            UInt(stageN),
                            new_smem.dtype,
                            new_smem.layout,
                            swizzle,
                            elementwise_compute_lambda_fn.value(),
                            Int(num_output_warps),
                            1,
                            MMA_M,
                            BN,
                            cta_group,
                        ](
                            M,
                            N,
                            UInt(c_col),
                            UInt(c_row),
                            new_smem,
                            UInt(warp_id),
                            UInt(0),
                        )
        else:
            comptime c_smem_tile_m = 32 if cta_group == 2 else BM // Int(
                num_output_warps
            )
            var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, stageN](
                Int(warp_id), 0
            )

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                data_paths, stageN
            ](0, 0)
            stsm_helper[swizzle, UInt(stageN), transpose_c](
                upper_frag_casted, c_smem_warp_tile_upper
            )

            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                data_paths, stageN
            ](1, 0)

            @parameter
            if is_lower_frag_required:
                stsm_helper[swizzle, UInt(stageN), transpose_c](
                    lower_frag_casted, c_smem_warp_tile_lower
                )

            # Guard the write to shared memory is done.
            named_barrier[Int32(num_output_warps * UInt(WARP_SIZE))]()

            @parameter
            if elementwise_compute_lambda_fn:

                @parameter
                if not register_based_epilogue:
                    shared_memory_epilogue[
                        UInt(MMA_M),
                        data_paths,
                        UInt(num_stages),
                        UInt(stage),
                        UInt(stageN),
                        c_smem_warp_tile_upper.dtype,
                        UInt(c_smem_tile.shape[1]()),
                        UInt(simd_size),
                        c_smem_warp_tile_upper.layout,
                        c_smem_warp_tile_lower.layout,
                        swizzle,
                        elementwise_compute_lambda_fn.value(),
                        Int(num_output_warps),
                    ](
                        M,
                        N,
                        UInt(c_col),
                        UInt(c_row),
                        c_smem_warp_tile_upper,
                        c_smem_warp_tile_lower,
                    )

        var lane = lane_id()

        comptime CG2_TMA_BM = (
            c_smem_tile.layout.shape[0].value() if MMA_M == 256 else BM
        )
        comptime CG1_TMA_BM = c_smem_tile.layout.shape[0].value()
        comptime TMA_BM = CG2_TMA_BM if cta_group == 2 else CG1_TMA_BM

        var cg2_elect_one_warp = (
            warp_id == 0 if MMA_M == 256 else warp_id % 2 == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        var elect_one_warp = (
            cg2_elect_one_warp if cta_group == 2 else cg1_elect_one_warp
        )

        var coord_n_mma_m256 = c_coord[1] * UInt32(MMA_N) + UInt32(
            stage * stageN
        )
        var coord_n_mma_m128 = (
            c_coord[1] * UInt32(MMA_N)
            + UInt32(stage * stageN)
            + UInt32(BN * Int(warp_id // 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_coord[0] * UInt32(BM)
        var coord_b = c_coord[2]

        if elect_one_warp and lane == 0:
            fence_async_view_proxy()

            @parameter
            if transpose_c:

                @parameter
                if cta_group == 2 and MMA_M == 128:
                    var c_smem_reshaped = c_smem_tile.reshape[
                        Layout.row_major(2 * stageN, stage_contiguous_size // 2)
                    ]()
                    var c_smem_split = c_smem_reshaped.tile[
                        stageN, stage_contiguous_size // 2
                    ](Int(warp_id // 2), 0)

                    c_tma_op.async_store(
                        c_smem_split,
                        StaticTuple[UInt32, 3](
                            coord_m,
                            coord_n,
                            coord_b,
                        ),
                    )

                else:
                    comptime num_c_smem_tiles = (
                        128
                        // swizzle_width
                        // (1 if is_lower_frag_required else 2)
                    )

                    @parameter
                    for i in range(num_c_smem_tiles):
                        var c_smem_warp_tile = c_smem_tile.tile[
                            stageN * swizzle_width // stage_contiguous_size,
                            stage_contiguous_size,
                        ](i, 0).reshape[
                            Layout.row_major(stageN, swizzle_width)
                        ]()
                        c_tma_op.async_store(
                            c_smem_warp_tile,
                            StaticTuple[UInt32, 3](
                                coord_m + UInt32(i * swizzle_width),
                                coord_n,
                                coord_b,
                            ),
                        )
            else:
                var cg2_c_smem_coord_m = 0 if MMA_M == 256 else (warp_id // 2)
                var cg1_c_smem_coord_m = UInt(0)
                var c_smem_coord_m = (
                    cg2_c_smem_coord_m if cta_group == 2 else cg1_c_smem_coord_m
                )
                var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                    Int(c_smem_coord_m), 0
                )
                c_tma_op.async_store(
                    c_smem_split,
                    StaticTuple[UInt32, 3](
                        coord_n,
                        coord_m,
                        coord_b,
                    ),
                )
            c_tma_op.commit_group()

        # Keep one tma store in fly
        @parameter
        if stage < num_stages - 1:
            c_tma_op.wait_group[1]()
        # Last stage guard all tma store to finish
        else:
            c_tma_op.wait_group[0]()

        @parameter
        if stage > 0 or stage == num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * UInt(WARP_SIZE))]()


@parameter
fn _reshape_to_3d[layout: Layout]() -> Layout:
    comptime rank = len(layout.shape)

    @parameter
    if rank == 3:
        return materialize[layout]()
    else:
        return Layout.row_major(
            1,
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
        )


fn _convert_input_to_batched_tensor[
    dtype: DType,
    layout: Layout,
    reshape_layout: Layout = _reshape_to_3d[layout](),
](
    tensor: LayoutTensor[dtype, layout, ...],
) -> LayoutTensor[
    tensor.dtype,
    reshape_layout,
    tensor.origin,
    address_space = tensor.address_space,
]:
    return LayoutTensor[
        dtype,
        reshape_layout,
        tensor.origin,
        address_space = tensor.address_space,
    ](
        tensor.ptr,
        RuntimeLayout[reshape_layout].row_major(
            IndexList[3](
                1 if tensor.rank == 2 else tensor.dim(0),
                tensor.dim(0) if tensor.rank == 2 else tensor.dim(1),
                tensor.dim(1) if tensor.rank == 2 else tensor.dim(2),
            ),
        ),
    )


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
fn blackwell_block_scaled_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    sfa_desc_layout: Layout,
    sfb_desc_layout: Layout,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Need because nvvm.cluster_dim only takes StaticTuple
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    sfa_tma_op: TMATensorTile[sfa_dtype, sfa_tile_layout, sfa_desc_layout],
    sfb_tma_op: TMATensorTile[sfb_dtype, sfb_tile_layout, sfb_desc_layout],
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
    alpha: Float32 = 1.0,
):
    comptime assert c_type != DType.float32, "c_type cannot be float32"
    comptime assert transpose_b, "only support k-major B"

    comptime num_output_warps = 4

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = (
        config.cta_group * EPILOGUE_THREADS
    )

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = config.num_sf_k_tiles * (BM // 32)
    comptime SFB_NUM_COLS = config.num_sf_k_tiles * (
        align_up(MMA_N, SF_MN_GROUP_SIZE) // 32
    )
    comptime stage_stride_cols = config.mma_shape[1]

    comptime assert (
        config.num_sf_k_tiles == 1
        and config.scaling_kind == UMMAKind.KIND_MXF8F6F4
    ) or (
        config.num_sf_k_tiles == 4
        and config.scaling_kind == UMMAKind.KIND_MXF4NVF4
    ), (
        "Only support num_sf_k_tiles == 1 and scaling kind is"
        " UMMAKind.KIND_MXF8F6F4 or num_sf_k_tiles == 2 and scaling kind is"
        " UMMAKind.KIND_MXF4NVF4"
    )

    comptime assert (
        UInt(config.num_accum_pipeline_stages) * UInt(MMA_N)
        + UInt(SFA_NUM_COLS + SFB_NUM_COLS) * UInt(config.num_pipeline_stages)
        <= NUM_TMEM_COLS
    ), "sfa_tmem and sfb_tmem exceed tmem_cols"

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M = Int(config.cluster_shape[0])
    comptime CLUSTER_N = Int(config.cluster_shape[1])

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[1].value()
    comptime b_tma_rows = b_desc_layout.shape[1].value()

    # keep the physical SMEM buffer BM x MMA_N
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode = config.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]()

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        BM,
        SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles,
        config.vec_sf_size,
    ]()
    comptime sfb_smem_layout = tile_sf_layout_k_major[
        align_up(MMA_N, SF_MN_GROUP_SIZE),
        SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles,
        config.vec_sf_size,
    ]()

    comptime SmemType = B200BlockScaledMatmulSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]

    ref smem_storage = external_memory[
        Scalar[DType.uint8],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]

    ref a_smem_storage = smem_storage.a_smem
    ref b_smem_storage = smem_storage.b_smem
    ref c_smem_storage = smem_storage.c_smem
    ref sfa_smem_storage = smem_storage.sfa_smem
    ref sfb_smem_storage = smem_storage.sfb_smem
    ref tma_mma_mbars_storage = smem_storage.tma_mma_mbars
    ref accum_mbars_storage = smem_storage.accum_mbars
    ref clc_mbars_full_storage = smem_storage.clc_mbars_full
    ref clc_mbars_empty_storage = smem_storage.clc_mbars_empty
    ref clc_response_storage = smem_storage.clc_response
    ref clc_throttle_storage = smem_storage.clc_throttle_mbars
    ref tmem_addr_storage = smem_storage.tmem_addr
    ref tmem_dealloc_mbar_storage = smem_storage.tmem_dealloc_mbar

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_smem_storage.unsafe_ptr(),
        SmemType.a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        b_smem_storage.unsafe_ptr(),
        SmemType.b_smem_size,
    )

    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(
            config.output_tile_shape[0], config.output_tile_shape[1]
        ),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        c_smem_storage.unsafe_ptr(),
        SmemType.c_smem_size,
    )

    var sfa_smem = LayoutTensorIter[
        sfa_dtype,
        sfa_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        sfa_smem_storage.unsafe_ptr(),
        SmemType.sfa_smem_size,
    )
    var sfb_smem = LayoutTensorIter[
        sfb_dtype,
        sfb_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        sfb_smem_storage.unsafe_ptr(),
        SmemType.sfb_smem_size,
    )

    # Load warp as producer and mma warp as consumer
    # Dependence on MMA input in SMEM.
    # Conumer phase = 1 so that producer's wait on consumer passes trivially
    # at the start when buffer is empty.
    var load_mma_pipeline = ProducerConsumerPipeline[
        Int(config.num_pipeline_stages // config.k_group_size)
    ](
        tma_mma_mbars_storage.unsafe_ptr(),
    )

    # MMA warp as producer and Output warp as consumer.
    # Dependence on MMA output in TMEM.
    var mma_output_pipeline = ProducerConsumerPipeline[
        Int(config.num_accum_pipeline_stages)
    ](
        accum_mbars_storage.unsafe_ptr(),
    )

    # Load warp as producer and scheduler warp as consumer.
    # No data dependence. Introduce dependence to prevent CLC goes too ahead.
    # In the extreme case, all ctas keep querying next work simultaneously,
    # there will be no guarantee they get balanced number of tiles.
    var load_clc_pipeline = ProducerConsumerPipeline[
        Int(config.num_clc_pipeline_stages)
    ](
        clc_throttle_storage.unsafe_ptr(),
    )

    var ptr_tmem_addr = tmem_addr_storage.unsafe_ptr()

    clc_response = clc_response_storage.unsafe_ptr()
    clc_full_mbar = clc_mbars_full_storage.unsafe_ptr()
    clc_empty_mbar = clc_mbars_empty_storage.unsafe_ptr()

    tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()

    # hardcode to float32 for now as we only support FP32 accumulation for block scaled matmul
    # TODO: (KERN-2238) replace with get_accum_type[a_type]() when KERN-2238 is fixed and we can return FP32 for FP4-E2M1
    comptime accum_type = DType.float32

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()
        sfa_tma_op.prefetch_descriptor()
        sfb_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
            ),
        )
        mma_output_pipeline.init_mbars(
            Int32(accum_pipeline_producer_arv_count),
            Int32(accum_pipeline_consumer_arv_count),
        )
        load_clc_pipeline.init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

        @parameter
        for i in range(config.num_clc_pipeline_stages):
            clc_full_mbar[i].init(Int32(clc_producer_arv_count))
            clc_empty_mbar[i].init(Int32(clc_consumer_arv_count))

    fence_mbarrier_init()

    @parameter
    if CLUSTER_SIZE > 1:
        cluster_arrive_relaxed()

    var clc_pipe_producer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ](0, 1, 0)
    var clc_pipe_consumer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ]()

    var mma_op = MmaOpSM100_BlockScaled_SS[
        c_type,
        a_type,
        b_type,
        sfa_dtype,
        sfb_dtype,
        config.scaling_kind,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=accum_type,
        cta_group = config.cta_group,
        cluster_shape = config.cluster_shape,
        a_swizzle = config.a_swizzle,
        b_swizzle = config.b_swizzle,
        transpose_b=True,
    ]()

    var scheduler = TileScheduler[
        num_stages = Int(config.num_clc_pipeline_stages),
        cluster_shape = Index[dtype = DType.uint32](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        block_swizzle_size = config.block_swizzle_size,
        rasterize_order = config.raster_order,
    ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var work_info = scheduler.initial_work_info()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        UInt(rank_m % UInt(config.cta_group)),
        UInt(rank_m // UInt(config.cta_group)),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    @parameter
    for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    @parameter
    for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= UInt16(1 << (i * config.cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * UInt(CLUSTER_M))

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], UInt32(BK))

    comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
        warp_role, max_profiled_tiles_per_SM
    ]

    @parameter
    if CLUSTER_SIZE > 1:
        cluster_wait()
    else:
        barrier()

    if WarpRole.is_main_load():
        with MatmulProfilerType[0](workspace, 0):
            var required_clc_query = True

            @parameter
            if pdl_level > PDLLevel.OFF:
                wait_on_dependent_grids()

            while work_info.is_valid():
                # CLC throttle prevents each CTA from going a few waves ahead.
                if is_first_cta_in_cluster and required_clc_query:
                    load_clc_pipeline.wait_consumer()
                    var load_clc_producer_state = (
                        load_clc_pipeline.producer_stage()
                    )
                    _ = load_clc_pipeline.producer_mbar(
                        load_clc_producer_state
                    )[0].arrive()
                    load_clc_pipeline.producer_step()

                # DO TMA LOAD
                for i in range(num_iters // UInt32(config.k_group_size)):
                    load_AB_SFA_SFB[
                        block_tile_shape = config.block_tile_shape,
                        mma_shape = config.mma_shape,
                        num_sf_k_tiles = config.num_sf_k_tiles,
                        cta_group = config.cta_group,
                        k_group_size = UInt(config.k_group_size),
                    ](
                        a_tma_op,
                        b_tma_op,
                        sfa_tma_op,
                        sfb_tma_op,
                        a_smem,
                        b_smem,
                        sfa_smem,
                        sfb_smem,
                        load_mma_pipeline,
                        peer_cta_coord,
                        (
                            UInt(work_info.m),
                            UInt(work_info.n),
                            UInt(work_info.k_start),
                        ),
                        a_multicast_mask,
                        b_multicast_mask,
                        i * UInt32(config.k_group_size),
                        elect_one_cta,
                    )
                    load_mma_pipeline.producer_step()

                syncwarp()
                var next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )
                work_info = next_work_info
                clc_pipe_consumer_state.step()

            # Prevent CTA to exit when a peer CTA is still working on mma.
            @parameter
            for i in range(config.num_pipeline_stages // config.k_group_size):
                load_mma_pipeline.wait_consumer()
                load_mma_pipeline.producer_step()

    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        # Implies each SM will only process initial work, there is no
        # more work to schedule.
        @parameter
        if config.num_clc_pipeline_stages == 0:
            return

        with MatmulProfilerType[1](workspace, 0):
            var required_clc_query = True

            @parameter
            if pdl_level > PDLLevel.OFF:
                wait_on_dependent_grids()

            while work_info.is_valid():
                if required_clc_query:
                    load_clc_pipeline.wait_producer()
                    var load_clc_consumer_stage = (
                        load_clc_pipeline.consumer_stage()
                    )
                    _ = load_clc_pipeline.consumer_mbar(
                        load_clc_consumer_stage
                    )[0].arrive()
                    load_clc_pipeline.consumer_step()

                    # advance to next work
                    clc_pipe_producer_state = scheduler.advance_to_next_work(
                        clc_pipe_producer_state
                    )

                # scheduler fetch next work
                next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )

                work_info = next_work_info
                clc_pipe_consumer_state.step()

            # make sure all pipes are empty before kernel exit
            @parameter
            for i in range(config.num_clc_pipeline_stages):
                clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                    clc_pipe_producer_state.phase()
                )
                clc_pipe_producer_state.step()

    if WarpRole.is_mma():
        with MatmulProfilerType[2](workspace, 0):
            tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
            syncwarp()
            # non blocking, arrives and proceeds
            named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

            tmem_addr = ptr_tmem_addr[0]
            var sfa_tmem = tmem_addr + UInt32(
                UInt(config.num_accum_pipeline_stages) * UInt(MMA_N)
            )
            var sfb_tmem = sfa_tmem + UInt32(SFA_NUM_COLS) * UInt32(
                config.num_pipeline_stages
            )

            while work_info.is_valid():
                # scheduler fetch next work
                next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )
                clc_pipe_consumer_state.step()
                # DO MMA
                if elect_one_cta:
                    var mma_output_mma_stage = (
                        mma_output_pipeline.producer_stage()
                    )
                    mma_output_pipeline.wait_consumer()
                    var tmem_offset = tmem_addr + (
                        mma_output_mma_stage * UInt32(stage_stride_cols)
                    )

                    for i in range(num_iters // UInt32(config.k_group_size)):
                        consumer_main_loop[
                            block_tile_shape = config.block_tile_shape,
                            mma_shape = config.mma_shape,
                            SFA_NUM_COLS=SFA_NUM_COLS,
                            SFB_NUM_COLS=SFB_NUM_COLS,
                            cta_group = config.cta_group,
                            cluster_shape = config.cluster_shape,
                            k_group_size = UInt(config.k_group_size),
                        ](
                            tmem_offset,
                            sfa_tmem,
                            sfb_tmem,
                            a_smem,
                            b_smem,
                            sfa_smem,
                            sfb_smem,
                            load_mma_pipeline,
                            mma_op,
                            elect_one_warp,
                            i * UInt32(config.k_group_size),
                            0,
                            work_tile_coord=(
                                UInt(work_info.m),
                                UInt(work_info.n),
                            ),
                        )
                        load_mma_pipeline.consumer_step()

                    # mma arrive multicast will track completion of all mma prior to this barrier.
                    if elect_one_sync():

                        @parameter
                        if config.cta_group == 1:
                            mma_arrive[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                )
                            )
                        else:
                            mma_arrive_multicast[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                ),
                                UInt16(mma_complete_mask),
                            )
                    mma_output_pipeline.producer_step()
                work_info = next_work_info

            @parameter
            if pdl_level > PDLLevel.OFF:
                launch_dependent_grids()

            tcgen05_release_allocation_lock[Int32(config.cta_group)]()

            # wait for epilogue to finish
            tmem_dealloc_mbar[].wait()

            tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        tmem_addr = ptr_tmem_addr[0]

        var tile_idx = 0

        while work_info.is_valid():
            with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                # WAIT FOR MMA TO FINISH AND STORE RESULT
                # scheduler fetch next work
                multi_stage_store_C[
                    input_type=a_type,
                    accum_type=accum_type,
                    block_tile_shape = config.block_tile_shape,
                    mma_shape = config.mma_shape,
                    stage_stride_cols = UInt(stage_stride_cols),
                    c_swizzle = config.c_swizzle,
                    cta_group = config.cta_group,
                    num_output_warps=num_output_warps,
                    max_tmem_cols=max_tmem_cols,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    register_based_epilogue=register_based_epilogue,
                    transpose_c = config.AB_swapped,
                ](
                    c_smem_iter,
                    c_tma_op,
                    mma_output_pipeline,
                    tmem_addr,
                    alpha,
                    work_tile_coord=(
                        work_info.m,
                        work_info.n,
                        work_info.k_start,
                    ),
                    elect_one_warp=elect_one_warp,
                    M=mnk[0],
                    N=mnk[1],
                )
                mma_output_pipeline.consumer_step()

                next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )
                work_info = next_work_info
                clc_pipe_consumer_state.step()

            tile_idx += 1

        @parameter
        if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


fn _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    sfa_dtype: DType,
    sfa_layout: Layout,
    sfb_dtype: DType,
    sfb_layout: Layout,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: LayoutTensor[c_type, c_layout, ...],
    a_tensor: LayoutTensor[a_type, a_layout, ...],
    b_tensor: LayoutTensor[b_type, b_layout, ...],
    a_scales_tensor: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    b_scales_tensor: LayoutTensor[sfb_dtype, sfb_layout, MutAnyOrigin],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (MXFP8_SF_DTYPE, NVFP4_SF_DTYPE), (
        "Only support MXFP8_SF_DTYPE (F8-UE8M0) or MXFP4_SF_DTYPE (F8-E4M3) for"
        " scales"
    )

    comptime assert (
        config.scaling_kind == UMMAKind.KIND_MXF8F6F4
        or config.scaling_kind == UMMAKind.KIND_MXF4NVF4
    ), "Only support MXF8F6F4 or MXF4NVF4 for scaling kind"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime assert (
        a_tensor.rank == b_tensor.rank == c_tensor.rank
        and a_tensor.rank in (2, 3)
    ), (
        "a_tensor, b_tensor, and c_tensor must have the same rank and be 2D"
        " (non-batched) or 3D (batched) tensors"
    )

    comptime is_batched_matmul = a_tensor.rank == 3

    comptime assert (
        a_scales_tensor.rank == b_scales_tensor.rank
    ), "a_scales and b_scales must be 5D (non-batched) or 6D (batched) tensors"

    comptime assert a_scales_tensor.rank == (
        6 if is_batched_matmul else 5
    ), "a_scales must be 6D (batched) or 5D (non-batched) tensors"

    comptime assert (
        sfa_layout.shape[3 if is_batched_matmul else 2].value()
        == sfb_layout.shape[3 if is_batched_matmul else 2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        sfa_layout.shape[4 if is_batched_matmul else 3].value()
        == sfb_layout.shape[4 if is_batched_matmul else 3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        sfa_layout.shape[5 if is_batched_matmul else 4].value()
        == sfb_layout.shape[5 if is_batched_matmul else 4].value()
        == SF_ATOM_K
    ), ""

    @parameter
    if config.cta_group == 2:
        comptime assert MMA_M == 256 and MMA_N in (
            64,
            128,
            192,
            256,
        ), "Only support cta_group == 2 with MMA_M == 256"

    else:
        comptime assert MMA_M == 128 and MMA_N in (64, 128, 192, 256), (
            "Only support MMA_M == 128 and MMA_N in (64, 128, 256) when"
            " cta_group == 1"
        )

    # convert a non-batched tensor to a batched tensor if needed so we can use the same kernel for both non-batched and batched matmuls
    var a_tensor_batched = _convert_input_to_batched_tensor(a_tensor)
    var b_tensor_batched = _convert_input_to_batched_tensor(b_tensor)
    var c_tensor_batched = _convert_input_to_batched_tensor(c_tensor)

    var B = c_tensor_batched.dim[0]()
    var M = c_tensor_batched.dim[1]()
    var N = c_tensor_batched.dim[2]()
    var M_maybe_swapped = a_tensor_batched.dim[1]()
    var N_maybe_swapped = b_tensor_batched.dim[1]()

    comptime assert (
        a_tensor_batched.layout.shape[2].value()
        == b_tensor_batched.layout.shape[2].value()
    ), "A and B K dimension does not match"

    comptime K = a_tensor_batched.layout.shape[2].value()

    comptime assert (
        ceildiv(K, BK) % Int(config.k_group_size) == 0
    ), "K iterations must be a multiple of k_group_size"

    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    comptime cluster_shape = config.cluster_shape

    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tensor_tile[
        a_tma_tile_shape,
        swizzle_mode = config.a_swizzle,
        __tile_layout = Layout.row_major(a_tma_tile_shape),
    ](ctx, a_tensor_batched)

    # fmt: off
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    ) if transpose_b else Index(
        1, BK, BN // (cluster_shape[0] // config.cta_group)
    )
    var b_tma_op = create_tensor_tile[
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
        __tile_layout = Layout.row_major(b_tma_tile_shape),
    ](ctx, b_tensor_batched)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        1, 64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(1, config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    ) if (MMA_M == 256 or config.cta_group == 1) else c_tma_tile_shape_mma128

    comptime assert (not config.AB_swapped) or config.c_swizzle.bytes() == 128, "Only support 128B swizzle mode when AB_swapped is True"

    comptime c_tma_tile_shape_final = c_tma_tile_shape if not config.AB_swapped else Index(
        1, c_tma_tile_shape[1], config.c_swizzle.bytes() // size_of[c_type]()
    )
    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape_final,
        swizzle_mode = config.c_swizzle,
        __tile_layout = Layout.row_major(c_tma_tile_shape_final),
    ](ctx, c_tensor_batched)
    # fmt: on

    comptime scales_5d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value() if is_batched_matmul else 1,
        layout.shape[1]
        .value() if is_batched_matmul else layout.shape[0]
        .value(),
        layout.shape[2]
        .value() if is_batched_matmul else layout.shape[1]
        .value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime sfa_5d_layout = scales_5d_layout[sfa_layout]
    comptime sfb_5d_layout = scales_5d_layout[sfb_layout]

    var sfa_5d_tensor = LayoutTensor[sfa_dtype, sfa_5d_layout, MutAnyOrigin](
        a_scales_tensor.ptr,
        RuntimeLayout[sfa_5d_layout].row_major(
            IndexList[5](
                a_scales_tensor.dim(0) if is_batched_matmul else 1,
                a_scales_tensor.dim(
                    1
                ) if is_batched_matmul else a_scales_tensor.dim(0),
                a_scales_tensor.dim(
                    2
                ) if is_batched_matmul else a_scales_tensor.dim(1),
                a_scales_tensor.dim(
                    3
                ) if is_batched_matmul else a_scales_tensor.dim(2),
                (
                    a_scales_tensor.dim(4) * a_scales_tensor.dim(5)
                ) if is_batched_matmul else (
                    a_scales_tensor.dim(3) * a_scales_tensor.dim(4)
                ),
            ),
        ),
    )
    var sfb_5d_tensor = LayoutTensor[sfb_dtype, sfb_5d_layout, MutAnyOrigin](
        b_scales_tensor.ptr,
        RuntimeLayout[sfb_5d_layout].row_major(
            IndexList[5](
                b_scales_tensor.dim(0) if is_batched_matmul else 1,
                b_scales_tensor.dim(
                    1
                ) if is_batched_matmul else b_scales_tensor.dim(0),
                b_scales_tensor.dim(
                    2
                ) if is_batched_matmul else b_scales_tensor.dim(1),
                b_scales_tensor.dim(
                    3
                ) if is_batched_matmul else b_scales_tensor.dim(2),
                (
                    b_scales_tensor.dim(4) * b_scales_tensor.dim(5)
                ) if is_batched_matmul else (
                    b_scales_tensor.dim(3) * b_scales_tensor.dim(4)
                ),
            ),
        ),
    )

    comptime sfa_tma_tile_shape = Index(
        1,
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfa_tma_op = create_tensor_tile[
        sfa_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(sfa_tma_tile_shape),
    ](ctx, sfa_5d_tensor)

    comptime sfb_tma_tile_shape = Index(
        1,
        align_up(MMA_N, SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfb_tma_op = create_tensor_tile[
        sfb_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(sfb_tma_tile_shape),
    ](ctx, sfb_5d_tensor)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime SmemType = B200BlockScaledMatmulSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    comptime max_profiled_tiles = (
        0 if max_profiled_tiles_per_SM
        is None else max_profiled_tiles_per_SM.value()
    )
    comptime enable_profiling = max_profiled_tiles > 0

    comptime kernel = blackwell_block_scaled_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        sfa_tma_op.layout,
        sfb_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        sfa_tma_op.desc_layout,
        sfb_tma_op.desc_layout,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N_maybe_swapped, MMA_N), Int(cluster_shape[1])),
        B,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace: Span[UInt64, MutAnyOrigin]

    @parameter
    if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin](
            ptr=UnsafePointer[UInt64, origin=MutAnyOrigin](), length=0
        )

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        sfa_tma_op,
        sfb_tma_op,
        cluster_dim,
        mnk,
        workspace,
        alpha,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


fn blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    sfa_dtype: DType,
    sfa_layout: Layout,
    sfb_dtype: DType,
    sfb_layout: Layout,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: LayoutTensor[c_type, c_layout, ...],
    a_tensor: LayoutTensor[a_type, a_layout, ...],
    b_tensor: LayoutTensor[b_type, b_layout, ...],
    a_scales_tensor: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    b_scales_tensor: LayoutTensor[sfb_dtype, sfb_layout, MutAnyOrigin],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """Launch block-scaled FP8 matmul kernel on SM100.

    Computes C = scale(A) @ scale(B) where A and B are FP8 matrices with
    per-block scaling factors following MXFP8 conventions.

    When config.AB_swapped is True, internally swaps A and B operands
    (along with their scale factors) and transposes the output for better
    performance when M is small.

    Parameters:
        c_type: Output element type.
        c_layout: Output tensor layout.
        a_type: A matrix element type (FP8).
        a_layout: A matrix layout.
        b_type: B matrix element type (FP8).
        b_layout: B matrix layout.
        sfa_dtype: A scaling factor type (F8-UE8M0).
        sfa_layout: A scaling factor layout.
        sfb_dtype: B scaling factor type (F8-UE8M0).
        sfb_layout: B scaling factor layout.
        transpose_b: Whether B is transposed (must be True).
        config: Block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional epilogue lambda.
        register_based_epilogue: Whether to use register-based epilogue.
        pdl_level: Programmatic dependent launch level.
        max_profiled_tiles_per_SM: Optional profiling tile count.

    Args:
        c_tensor: Output tensor.
        a_tensor: A matrix tensor.
        b_tensor: B matrix tensor.
        a_scales_tensor: A scaling factors.
        b_scales_tensor: B scaling factors.
        ctx: Device context for kernel launch.
        alpha: Tensor scale factor (scalar).

    Raises:
        If configuration constraints are violated.
    """

    @parameter
    if config.AB_swapped:
        # When both A and B are K-major, C = A @ B'.
        # If we swap A and B: D = B @ A', and D' = (B @ A')' = A @ B' = C.
        # So swapping + transposing the output gives the same result.
        # The transpose is handled by transpose_c = config.AB_swapped in the
        # kernel.
        comptime new_config = config.swap_AB_type()
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            c_type,
            c_layout,
            b_type,
            b_layout,
            a_type,
            a_layout,
            sfb_dtype,
            sfb_layout,
            sfa_dtype,
            sfa_layout,
            transpose_b,
            config=new_config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_tensor,
            b_tensor,
            a_tensor,
            b_scales_tensor,
            a_scales_tensor,
            ctx,
            alpha,
        )
    else:
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            c_type,
            c_layout,
            a_type,
            a_layout,
            b_type,
            b_layout,
            sfa_dtype,
            sfa_layout,
            sfb_dtype,
            sfb_layout,
            transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            ctx,
            alpha,
        )
