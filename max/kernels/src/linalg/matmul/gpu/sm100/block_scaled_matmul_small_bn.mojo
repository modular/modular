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

from std.math import align_up, ceildiv
from std.sys import align_of, size_of

from std.gpu import WARP_SIZE, barrier
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync,
    elect_one_sync_with_mask,
    cluster_wait,
    cluster_arrive_relaxed,
)
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import B200
from std.gpu import block_id_in_cluster, lane_id_uint as lane_id
from std.gpu import warp_id_uint as get_warp_id
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    external_memory,
    fence_mbarrier_init,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import *
from std.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    PDLLevel,
    wait_on_dependent_grids,
)
from std.gpu.sync import (
    async_copy_arrive,
    umma_arrive_peer_cta,
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
)
from std.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeInt,
    RuntimeLayout,
    TileTensor,
)
from layout.layout_tensor import LayoutTensorIter
from layout.coord import ComptimeInt
from layout.tile_layout import Layout as TileLayout, row_major as tt_row_major
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_sf_layout_k_major,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from structured_kernels.kernel_common import _to_batched_3d
from structured_kernels.tile_types import (
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    SMemTileArrayWithLayout,
    internal_sf_k_major,
    sf_tile_dim0,
    sf_tile_dim1,
    swizzle_mode_to_bytes,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    OutputPipelineConfig,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.output_writer import (
    TileWriter,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline import (
    OutputStage,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_BlockScaled_SS
from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .config import BlockScaledMatmulConfig
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


@fieldwise_init
struct WarpRole(TrivialRegisterPassable):
    var _role: Int32

    comptime SfbTMEMLoad = Self(8)
    comptime SfbTMALoad = Self(7)
    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    def __eq__(self, other: UInt) -> Bool:
        return self._role == Int32(other)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    def __ge__(self, other: UInt) -> Bool:
        return self._role >= Int32(other)

    @always_inline
    def __le__(self, other: UInt) -> Bool:
        return self._role <= Int32(other)

    @staticmethod
    @always_inline
    def is_sfb_tmem_load() -> Bool:
        return Self.SfbTMEMLoad <= get_warp_id()

    @staticmethod
    @always_inline
    def is_sfb_gmem_to_smem() -> Bool:
        return Self.SfbTMALoad == get_warp_id()

    @staticmethod
    @always_inline
    def is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    def is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    def is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()

    @staticmethod
    @always_inline
    def is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()


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

    comptime a_smem_size = (Self.BM * Self.BK * Self.config.num_pipeline_stages)
    comptime b_smem_size = (Self.BN * Self.BK * Self.config.num_pipeline_stages)
    comptime c_smem_size = (
        Self.OutputM * Self.OutputN * Self.config.num_output_stages
    )

    comptime sfa_smem_size = (
        Self.config.num_sf_k_tiles
        * (Self.BM // SF_MN_GROUP_SIZE)
        * Self.config.sf_block_atom_size
        * Self.config.num_pipeline_stages
    )
    # cp.async packs data as num_sf_k_tiles * MMA_N * SF_ATOM_K per tile,
    # which is much smaller than the TMA atom layout (sf_block_atom_size=512).
    # TODO(2CTA): Use BN instead of MMA_N to reduce SMEM per CTA in 2SM mode.
    # Currently over-allocates for cta_group=2 but is functionally correct.
    comptime sfb_smem_size = (
        Self.config.num_sf_k_tiles
        * (
            Self.MMA_N
            * SF_ATOM_K if Self.config.use_cpasync_sfb else (
                (align_up(Self.MMA_N, SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE)
                * Self.config.sf_block_atom_size
            )
        )
        * Self.config.num_pipeline_stages
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
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]

    var sfb_load_mbars: InlineArray[
        SharedMemBarrier, Self.num_group_pipeline_stages
    ]

    # ACCUM
    var accum_mbars: InlineArray[
        SharedMemBarrier, Self.config.num_accum_pipeline_stages * 2
    ]

    # CLC
    var clc_mbars_full: InlineArray[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages
    ]
    var clc_mbars_empty: InlineArray[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages
    ]
    var clc_throttle_mbars: InlineArray[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages * 2
    ]
    var clc_response: InlineArray[UInt128, Self.config.num_clc_pipeline_stages]

    # SFB TMA pipeline (SfbTMALoad ↔ MMA)
    var sfb_tma_mbars: InlineArray[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]

    # TMEM
    var tmem_dealloc_mbar: InlineArray[SharedMemBarrier, 1]
    var tmem_addr: InlineArray[UInt32, 1]


@always_inline
def load_AB_SFA[
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    a_rank: Int,
    a_tile_shape: IndexList[a_rank],
    a_desc_shape: IndexList[a_rank],
    b_rank: Int,
    b_tile_shape: IndexList[b_rank],
    b_desc_shape: IndexList[b_rank],
    sfa_rank: Int,
    sfa_tile_shape: IndexList[sfa_rank],
    sfa_desc_shape: IndexList[sfa_rank],
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    sfa_smem_layout: Layout,
    num_pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    num_sf_k_tiles: Int,
    cta_group: Int = 1,
    k_group_size: UInt = 1,
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    sfa_tma_op: TMATensorTile[
        sfa_dtype, sfa_rank, sfa_tile_shape, sfa_desc_shape
    ],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ],
    sfa_smem: LayoutTensorIter[
        sfa_dtype,
        sfa_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
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

    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = (
        cta_group * (a_expected_bytes + b_expected_bytes + sfa_expected_bytes)
    ) * Int(k_group_size)

    comptime a_tma_load_size = _idx_product[a_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[1]
    comptime b_tma_rows = b_desc_shape[1]

    var stage = load_mma_pipeline.producer_stage()
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)
    var a_gmem_slice_coord = (
        Int(peer_cta_coord[2]) * a_tma_rows + Int(work_tile_coord[0]) * BM
    )
    var b_gmem_slice_coord = (
        Int(peer_cta_coord[1]) * b_tma_rows
        + Int(peer_cta_coord[0]) * BN
        + Int(work_tile_coord[1]) * MMA_N
    )
    var batch_coord = Int(work_tile_coord[2])

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
                    Int(iter_idx + j) * BK,
                    a_gmem_slice_coord,
                    batch_coord,
                ),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load_3d[cta_group](
                b_smem_slice,
                tma_mbar[0],
                (
                    Int(iter_idx + j) * BK,
                    b_gmem_slice_coord,
                    batch_coord,
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
                    batch_coord,
                ),
            )


@always_inline
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    scaling_kind: UMMAKind,
    num_group_pipeline_stages: Int,
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
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    sfa_smem_tiles: SMemTileArrayWithLayout[sfa_dtype, ...],
    sfb_smem_tiles: SMemTileArrayWithLayout[sfb_dtype, ...],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    load_sfb_mbars: UnsafePointer[
        SharedMemBarrier, _, address_space=AddressSpace.SHARED
    ],
    sfb_load_pipe_consumer_state: PipelineState[num_group_pipeline_stages],
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
    """TileTensor overload of consumer_main_loop for block-scaled MMA (small BN).

    Accepts SMemTileArray2D for A/B tiles and SMemTileArrayWithLayout for
    scale factor tiles, calling the TileTensor MMA overload.
    """
    comptime MMA_N = mma_shape[1]

    # Compute sfb_tmem_adj from work_tile_coord (same logic as MMA LayoutTensor path).
    var sfb_tmem_adj: UInt32
    comptime if MMA_N in (64, 192):
        sfb_tmem_adj = UInt32(work_tile_coord[1] % 2) * 2
    else:
        sfb_tmem_adj = UInt32(0)

    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    load_sfb_mbars[sfb_load_pipe_consumer_state.index()].wait(
        sfb_load_pipe_consumer_state.phase()
    )

    if elect_one_sync():
        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_tiles[offset]
            var b_smem_tile = b_smem_tiles[offset]
            var sfa_smem_tile = sfa_smem_tiles[offset]
            var sfb_smem_tile = sfb_smem_tiles[offset]

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
                sfb_tmem_adj=sfb_tmem_adj,
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


@parameter
def _reshape_to_3d[layout: Layout]() -> Layout:
    comptime rank = len(layout.shape)

    comptime if rank == 3:
        return materialize[layout]()
    else:
        return Layout.row_major(
            1,
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
        )


def _convert_input_to_batched_tensor[
    dtype: DType,
    layout: Layout,
    reshape_layout: Layout = _reshape_to_3d[layout](),
](
    tensor: LayoutTensor[dtype, layout, ...],
) -> LayoutTensor[
    tensor.dtype,
    reshape_layout,
    tensor.origin,
    address_space=tensor.address_space,
]:
    return LayoutTensor[
        dtype,
        reshape_layout,
        tensor.origin,
        address_space=tensor.address_space,
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


@always_inline
def sfb_smem_to_tmem_cpasync[
    sfb_dtype: DType,
    MMA_N: Int,
    num_sf_k_tiles: Int,
](
    sfb_smem_ptr: UnsafePointer[
        Scalar[sfb_dtype], _, address_space=AddressSpace.SHARED
    ],
    sfb_tmem_offset: UInt32,
):
    """Load SFB from SMEM (packed cp.async layout) to TMEM."""
    comptime for sf_idx in range(num_sf_k_tiles):
        var sfb_scales = SIMD[sfb_dtype, SF_ATOM_K]()
        if lane_id() < UInt(MMA_N):
            var scales_offset = UInt(
                sf_idx * MMA_N * SF_ATOM_K
            ) + lane_id() * UInt(SF_ATOM_K)
            sfb_scales = sfb_smem_ptr.load[
                width=SF_ATOM_K,
                alignment=align_of[SIMD[sfb_dtype, SF_ATOM_K]](),
            ](scales_offset)

        syncwarp()

        var _sfb_st = InlineArray[Scalar[DType.uint32], 1](uninitialized=True)
        _sfb_st[0] = bitcast[DType.uint32, 1](sfb_scales)[0]
        tcgen05_st[
            datapaths=32,
            bits=32,
            repeat=1,
            pack=False,
        ](
            sfb_tmem_offset + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
            _sfb_st,
        )
        tcgen05_store_wait()
        tcgen05_fence_before()


@always_inline
def sfb_smem_to_tmem_tma[
    sfb_dtype: DType,
    MMA_N: Int,
    num_sf_k_tiles: Int,
](
    sfb_smem_ptr: UnsafePointer[
        Scalar[sfb_dtype], _, address_space=AddressSpace.SHARED
    ],
    sfb_tmem_offset: UInt32,
    work_tile_n: UInt,
):
    """Load SFB from SMEM (TMA atom layout) to TMEM."""
    comptime SFB_TILE_BYTES = (
        SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K * size_of[sfb_dtype]()
    )

    comptime for sf_idx in range(num_sf_k_tiles):
        var sfb_scales = SIMD[sfb_dtype, SF_ATOM_K]()
        if lane_id() < UInt(MMA_N):
            var outer = (work_tile_n * UInt(MMA_N)) % UInt(
                SF_MN_GROUP_SIZE
            ) + lane_id()
            var scales_offset = (
                UInt(sf_idx) * UInt(SFB_TILE_BYTES)
                + (outer % UInt(SF_ATOM_M[0]))
                * (UInt(SF_ATOM_M[1]) * UInt(SF_ATOM_K))
                + (outer / UInt(SF_ATOM_M[0])) * UInt(SF_ATOM_K)
            )
            sfb_scales = sfb_smem_ptr.load[
                width=SF_ATOM_K,
                alignment=align_of[SIMD[sfb_dtype, SF_ATOM_K]](),
            ](scales_offset)

        syncwarp()

        var _sfb_st = InlineArray[Scalar[DType.uint32], 1](uninitialized=True)
        _sfb_st[0] = bitcast[DType.uint32, 1](sfb_scales)[0]
        tcgen05_st[
            datapaths=32,
            bits=32,
            repeat=1,
            pack=False,
        ](
            sfb_tmem_offset + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
            _sfb_st,
        )
        tcgen05_store_wait()
        tcgen05_fence_before()


@always_inline
def _sfb_cpasync_produce_tile[
    sfb_dtype: DType,
    MMA_N: Int,
    num_sf_k_tiles: Int,
    k_group_size: Int,
    num_sfb_pipeline_stages: Int,
    sfb_smem_tile_elems: Int,
](
    work_info: WorkInfo,
    num_iters: UInt32,
    mut sfb_tma_pipeline: ProducerConsumerPipeline[num_sfb_pipeline_stages],
    sfb_smem_base_ptr: UnsafePointer[
        Scalar[sfb_dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    sfb_global_ptr: UnsafePointer[Scalar[sfb_dtype], ImmutAnyOrigin],
    sfb_batch_stride: Int,
    sfb_n_stride: Int,
    sfb_k_tiles: Int,
):
    """Produce SFB data from global to shared memory using cp.async.

    Each of MMA_N lanes copies SF_ATOM_K bytes per k-atom, packed
    contiguously in SMEM.  The pipeline's producer arrive_count
    must be MMA_N.  OOB k-tiles (beyond sfb_k_tiles) are zero-filled
    since cp.async does not auto-fill zeros like TMA, and garbage
    bytes that decode as NaN in float8_e4m3fn would corrupt the
    accumulator (NaN * 0 = NaN).
    """
    comptime ROW_STRIDE = SF_ATOM_M[1] * SF_ATOM_K
    comptime K_TILE_ELEMS = SF_ATOM_M[0] * ROW_STRIDE

    # Global memory atom layout: (k_tile, row_in_atom, sub_column) → flat
    # offset within one n_group. Strides: k_tile by K_TILE_ELEMS (512),
    # row by ROW_STRIDE (16), sub_column by SF_ATOM_K (4).
    comptime sfb_global_atom_layout = TileLayout(
        Coord(
            Idx[num_sf_k_tiles](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
        ),
        Coord(
            Idx[K_TILE_ELEMS](),
            Idx[ROW_STRIDE](),
            Idx[SF_ATOM_K](),
        ),
    )

    # SMEM stage layout: pipeline stage offset → flat SMEM element offset.
    comptime sfb_smem_stage_layout = TileLayout(
        Coord(Idx[num_sfb_pipeline_stages * k_group_size]()),
        Coord(Idx[sfb_smem_tile_elems]()),
    )

    # Per-tile address components (all lanes compute).
    var outer = (UInt(work_info.n) * UInt(MMA_N)) % UInt(
        SF_MN_GROUP_SIZE
    ) + lane_id()
    var row_in_atom = outer % UInt(SF_ATOM_M[0])
    var sub_column = outer / UInt(SF_ATOM_M[0])
    var n_group = (Int(work_info.n) * MMA_N) // SF_MN_GROUP_SIZE
    var batch = Int(work_info.k_start)

    for i in range(num_iters // UInt32(k_group_size)):
        sfb_tma_pipeline.wait_consumer()

        var stage = sfb_tma_pipeline.producer_stage()
        var sfb_tma_mbar = sfb_tma_pipeline.producer_mbar(stage)

        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var sfb_smem_tile_ptr = sfb_smem_base_ptr + Int(
                sfb_smem_stage_layout(
                    Coord(RuntimeInt(Scalar[DType.int64](offset)))
                )
            )
            var k_tile_base = Int(i * UInt32(k_group_size) + j) * num_sf_k_tiles

            comptime for k_atom in range(num_sf_k_tiles):
                if lane_id() < UInt(MMA_N):
                    # Stored sequentially to avoid SMEM bank conflicts.
                    var smem_offset = (
                        k_atom * (MMA_N * SF_ATOM_K)
                        + Int(lane_id()) * SF_ATOM_K
                    )
                    # K bounds check: OOB k-atoms are zero-filled
                    # because cp.async has no auto-fill and NaN
                    # scales would corrupt the accumulator.
                    # N bounds check is unnecessary: the scale
                    # tensor is allocated with align_up(MMA_N,
                    # SF_MN_GROUP_SIZE) padding, so OOB lanes
                    # read valid memory whose values are harmless
                    # (multiplied against zero-padded B tiles).
                    if k_tile_base + k_atom < sfb_k_tiles:
                        var global_offset = (
                            batch * sfb_batch_stride
                            + n_group * sfb_n_stride
                            + Int(
                                sfb_global_atom_layout(
                                    Coord(
                                        RuntimeInt(
                                            Scalar[DType.int64](
                                                k_tile_base + k_atom
                                            )
                                        ),
                                        RuntimeInt(
                                            Scalar[DType.int64](row_in_atom)
                                        ),
                                        RuntimeInt(
                                            Scalar[DType.int64](sub_column)
                                        ),
                                    )
                                )
                            )
                        )
                        async_copy[size=SF_ATOM_K * size_of[sfb_dtype](),](
                            (sfb_global_ptr + global_offset).address_space_cast[
                                AddressSpace.GLOBAL
                            ](),
                            (
                                sfb_smem_tile_ptr + smem_offset
                            ).address_space_cast[AddressSpace.SHARED](),
                        )
                    else:
                        (sfb_smem_tile_ptr + smem_offset).store(
                            SIMD[sfb_dtype, SF_ATOM_K]()
                        )

        # async_copy_arrive: pending_count +1 (auto -1 on completion).
        # mbar.arrive: arrive_count -1.
        if lane_id() < UInt(MMA_N):
            async_copy_arrive(sfb_tma_mbar[0].unsafe_ptr())
            _ = sfb_tma_mbar[0].arrive()

        sfb_tma_pipeline.producer_step()


@always_inline
def _sfb_cpasync_produce_tile_warpwide[
    sfb_dtype: DType,
    MMA_N: Int,
    num_sf_k_tiles: Int,
    k_group_size: Int,
    num_sfb_pipeline_stages: Int,
    sfb_smem_tile_elems: Int,
](
    work_info: WorkInfo,
    num_iters: UInt32,
    mut sfb_tma_pipeline: ProducerConsumerPipeline[num_sfb_pipeline_stages],
    sfb_smem_base_ptr: UnsafePointer[
        Scalar[sfb_dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    sfb_global_ptr: UnsafePointer[Scalar[sfb_dtype], ImmutAnyOrigin],
    sfb_batch_stride: Int,
    sfb_n_stride: Int,
    sfb_k_tiles: Int,
):
    """Produce SFB data from global to shared memory using cp.async,
    utilizing the full warp (32 threads) when MMA_N * num_sf_k_tiles <= 32.

    Each lane handles one (k_atom, position) pair:
      k_atom   = lane_id / MMA_N
      position = lane_id % MMA_N
    All k-atoms are loaded in a single pass with no sequential iteration,
    and all 32 lanes issue work in parallel.

    The pipeline's producer arrive_count must be MMA_N * num_sf_k_tiles
    (= 32 in the expected configuration).
    """
    comptime assert (
        MMA_N * num_sf_k_tiles <= 32
    ), "warp-wide path requires MMA_N * num_sf_k_tiles <= 32"

    comptime ROW_STRIDE = SF_ATOM_M[1] * SF_ATOM_K
    comptime K_TILE_ELEMS = SF_ATOM_M[0] * ROW_STRIDE

    # Global memory atom layout: (k_tile, row_in_atom, sub_column) → flat
    # offset within one n_group.
    comptime sfb_global_atom_layout = TileLayout(
        Coord(
            Idx[num_sf_k_tiles](),
            Idx[SF_ATOM_M[0]](),
            Idx[SF_ATOM_M[1]](),
        ),
        Coord(
            Idx[K_TILE_ELEMS](),
            Idx[ROW_STRIDE](),
            Idx[SF_ATOM_K](),
        ),
    )

    # SMEM stage layout: pipeline stage offset → flat SMEM element offset.
    comptime sfb_smem_stage_layout = TileLayout(
        Coord(Idx[num_sfb_pipeline_stages * k_group_size]()),
        Coord(Idx[sfb_smem_tile_elems]()),
    )

    # Map lane_id to (k_atom, position) across the full warp.
    var my_k_atom = Int(lane_id()) // MMA_N
    var my_pos = Int(lane_id()) % MMA_N
    var active = lane_id() < UInt(MMA_N * num_sf_k_tiles)

    # Per-tile address components for this lane's position.
    var outer = (UInt(work_info.n) * UInt(MMA_N)) % UInt(
        SF_MN_GROUP_SIZE
    ) + UInt(my_pos)
    var row_in_atom = outer % UInt(SF_ATOM_M[0])
    var sub_column = outer / UInt(SF_ATOM_M[0])
    var n_group = (Int(work_info.n) * MMA_N) // SF_MN_GROUP_SIZE
    var batch = Int(work_info.k_start)

    for i in range(num_iters // UInt32(k_group_size)):
        sfb_tma_pipeline.wait_consumer()

        var stage = sfb_tma_pipeline.producer_stage()
        var sfb_tma_mbar = sfb_tma_pipeline.producer_mbar(stage)

        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var sfb_smem_tile_ptr = sfb_smem_base_ptr + Int(
                sfb_smem_stage_layout(
                    Coord(RuntimeInt(Scalar[DType.int64](offset)))
                )
            )
            var k_tile_base = Int(i * UInt32(k_group_size) + j) * num_sf_k_tiles

            # All k-atoms loaded in one shot — no comptime loop.
            if active:
                var smem_offset = (
                    my_k_atom * (MMA_N * SF_ATOM_K) + my_pos * SF_ATOM_K
                )
                # K bounds check: OOB k-atoms are zero-filled
                # because cp.async has no auto-fill and NaN
                # scales would corrupt the accumulator.
                # N bounds check is unnecessary: the scale
                # tensor is allocated with align_up(MMA_N,
                # SF_MN_GROUP_SIZE) padding, so OOB lanes
                # read valid memory whose values are harmless
                # (multiplied against zero-padded B tiles).
                if k_tile_base + my_k_atom < sfb_k_tiles:
                    var global_offset = (
                        batch * sfb_batch_stride
                        + n_group * sfb_n_stride
                        + Int(
                            sfb_global_atom_layout(
                                Coord(
                                    RuntimeInt(
                                        Scalar[DType.int64](
                                            k_tile_base + my_k_atom
                                        )
                                    ),
                                    RuntimeInt(
                                        Scalar[DType.int64](row_in_atom)
                                    ),
                                    RuntimeInt(Scalar[DType.int64](sub_column)),
                                )
                            )
                        )
                    )
                    async_copy[size=SF_ATOM_K * size_of[sfb_dtype](),](
                        (sfb_global_ptr + global_offset).address_space_cast[
                            AddressSpace.GLOBAL
                        ](),
                        (sfb_smem_tile_ptr + smem_offset).address_space_cast[
                            AddressSpace.SHARED
                        ](),
                    )
                else:
                    (sfb_smem_tile_ptr + smem_offset).store(
                        SIMD[sfb_dtype, SF_ATOM_K]()
                    )

        if active:
            async_copy_arrive(sfb_tma_mbar[0].unsafe_ptr())
            _ = sfb_tma_mbar[0].arrive()

        sfb_tma_pipeline.producer_step()


@always_inline
def _sfb_tma_produce_tile[
    sfb_dtype: DType,
    MMA_N: Int,
    num_sf_k_tiles: Int,
    k_group_size: Int,
    num_sfb_pipeline_stages: Int,
    sfb_smem_tile_elems: Int,
    cta_group: Int,
    sfb_rank: Int,
    sfb_tile_shape: IndexList[sfb_rank],
    sfb_desc_shape: IndexList[sfb_rank],
](
    work_info: WorkInfo,
    num_iters: UInt32,
    mut sfb_tma_pipeline: ProducerConsumerPipeline[num_sfb_pipeline_stages],
    sfb_smem_base_ptr: UnsafePointer[
        Scalar[sfb_dtype], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    elect_one_cta: Bool,
    sfb_tma_op: TMATensorTile[
        sfb_dtype, sfb_rank, sfb_tile_shape, sfb_desc_shape
    ],
):
    """Produce SFB data from global to shared memory using TMA.

    Uses TMA async_copy_5d on the elected thread, loading full
    scale-factor atoms with 16-byte row stride.
    """
    comptime SFB_TMA_ROWS = sfb_tile_shape[3]
    comptime ROW_STRIDE = SF_ATOM_M[1] * SF_ATOM_K
    comptime sfb_single_copy_bytes = (
        sfb_tile_shape[1]
        * sfb_tile_shape[3]
        * sfb_tile_shape[4]
        * size_of[sfb_dtype]()
    )
    comptime sfb_expected_bytes = (
        num_sf_k_tiles * sfb_single_copy_bytes * k_group_size
    )

    for i in range(num_iters // UInt32(k_group_size)):
        sfb_tma_pipeline.wait_consumer()

        var stage = sfb_tma_pipeline.producer_stage()
        var sfb_tma_mbar = sfb_tma_pipeline.producer_mbar(stage)

        if elect_one_sync():
            # sfb_tma_pipeline is CTA-private, so every CTA calls
            # expect_bytes on its own barrier (no elect_one_cta gate).
            sfb_tma_mbar[0].expect_bytes(Int32(sfb_expected_bytes))

            comptime sfb_atom_layout = TileLayout(
                Coord(
                    Idx[num_sf_k_tiles](),
                    Idx[SF_ATOM_M[0]](),
                ),
                Coord(
                    Idx[SF_ATOM_M[0] * ROW_STRIDE](),
                    Idx[ROW_STRIDE](),
                ),
            )

            var row_in_atom = (Int(work_info.n) * MMA_N) % SF_ATOM_M[0]
            var n_group = (Int(work_info.n) * MMA_N) // SF_MN_GROUP_SIZE
            var batch = Int(work_info.k_start)

            for jj in range(k_group_size):
                var j = UInt32(jj)
                var offset = stage * UInt32(k_group_size) + j
                var sfb_smem_tile_ptr = (
                    sfb_smem_base_ptr + Int(offset) * sfb_smem_tile_elems
                )
                var k_tile_base = (
                    Int(i * UInt32(k_group_size) + j) * num_sf_k_tiles
                )

                comptime for k_atom in range(num_sf_k_tiles):
                    var smem_offset = Int(
                        sfb_atom_layout(
                            Coord(
                                Idx[k_atom](),
                                RuntimeInt(Scalar[DType.int64](row_in_atom)),
                            )
                        )
                    )

                    var atom_dst = LayoutTensor[
                        sfb_dtype,
                        Layout.row_major(SFB_TMA_ROWS, ROW_STRIDE),
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](sfb_smem_tile_ptr + smem_offset)

                    # SFB pipeline is CTA-private: each CTA loads its
                    # own copy, no multicast (always cta_group=1).
                    sfb_tma_op.async_copy_5d[1](
                        atom_dst,
                        sfb_tma_mbar[0],
                        (
                            0,
                            row_in_atom,
                            k_tile_base + k_atom,
                            n_group,
                            batch,
                        ),
                    )

        sfb_tma_pipeline.producer_step()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
def blackwell_block_scaled_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    a_rank: Int,
    a_tile_shape: IndexList[a_rank],
    a_desc_shape: IndexList[a_rank],
    b_rank: Int,
    b_tile_shape: IndexList[b_rank],
    b_desc_shape: IndexList[b_rank],
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    sfa_rank: Int,
    sfa_tile_shape: IndexList[sfa_rank],
    sfa_desc_shape: IndexList[sfa_rank],
    sfb_rank: Int,
    sfb_tile_shape: IndexList[sfb_rank],
    sfb_desc_shape: IndexList[sfb_rank],
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Need because nvvm.cluster_dim only takes StaticTuple
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
    sfa_tma_op: TMATensorTile[
        sfa_dtype, sfa_rank, sfa_tile_shape, sfa_desc_shape
    ],
    sfb_tma_op: TMATensorTile[
        sfb_dtype, sfb_rank, sfb_tile_shape, sfb_desc_shape
    ],
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
    alpha: Float32,
    sfb_global_ptr: UnsafePointer[Scalar[sfb_dtype], ImmutAnyOrigin],
    sfb_batch_stride: Int,
    sfb_n_stride: Int,
    sfb_k_tiles: Int,
):
    comptime assert c_type != DType.float32, "c_type cannot be float32"
    comptime assert transpose_b, "only support k-major B"

    comptime num_output_warps = 4
    comptime num_sfb_load_warps = 4

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime SFB_TMA_LOAD_THREADS = WARP_SIZE
    comptime SFB_LOAD_THREADS = num_sfb_load_warps * WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS
        + SFB_TMA_LOAD_THREADS
        + MMA_THREADS
        + EPILOGUE_THREADS
        + SFB_LOAD_THREADS
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
    ), "Only support MXF8F6F4 (k=1) or MXF4NVF4 (k=4)"

    comptime assert (
        UInt(config.num_accum_pipeline_stages) * UInt(MMA_N)
        + UInt(SFA_NUM_COLS + SFB_NUM_COLS) * UInt(config.num_pipeline_stages)
        <= NUM_TMEM_COLS
    ), "sfa_tmem and sfb_tmem exceed tmem_cols"

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M = config.cluster_shape[0]
    comptime CLUSTER_N = config.cluster_shape[1]

    comptime a_tma_load_size = _idx_product[a_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[1]
    comptime b_tma_rows = b_desc_shape[1]

    # keep the physical SMEM buffer BM x MMA_N
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=config.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ]()

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        BM,
        SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles,
        config.vec_sf_size,
    ]()
    # cp.async uses a flat packed layout (num_sf_k_tiles * MMA_N * SF_ATOM_K
    # per tile), while TMA uses the hardware atom layout.
    comptime sfb_smem_layout = Layout.row_major(
        config.num_sf_k_tiles * MMA_N, SF_ATOM_K
    ) if config.use_cpasync_sfb else tile_sf_layout_k_major[
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
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]

    ref a_smem_storage = smem_storage.a_smem
    ref b_smem_storage = smem_storage.b_smem
    ref c_smem_storage = smem_storage.c_smem
    ref sfa_smem_storage = smem_storage.sfa_smem
    ref sfb_smem_storage = smem_storage.sfb_smem
    ref tma_mma_mbars_storage = smem_storage.tma_mma_mbars
    ref sfb_load_mbars_storage = smem_storage.sfb_load_mbars
    ref accum_mbars_storage = smem_storage.accum_mbars
    ref clc_mbars_full_storage = smem_storage.clc_mbars_full
    ref clc_mbars_empty_storage = smem_storage.clc_mbars_empty
    ref clc_response_storage = smem_storage.clc_response
    ref clc_throttle_storage = smem_storage.clc_throttle_mbars
    ref sfb_tma_mbars_storage = smem_storage.sfb_tma_mbars
    ref tmem_addr_storage = smem_storage.tmem_addr
    ref tmem_dealloc_mbar_storage = smem_storage.tmem_dealloc_mbar

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        a_smem_storage.unsafe_ptr(),
        SmemType.a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        b_smem_storage.unsafe_ptr(),
        SmemType.b_smem_size,
    )

    # hardcode to float32 for now as we only support FP32 accumulation for block scaled matmul
    # TODO: (KERN-2238) replace with get_accum_type[a_type]() when KERN-2238 is fixed and we can return FP32 for FP4-E2M1
    comptime accum_type = DType.float32

    # TileTensor-based view of C SMEM for TileWriter epilogue.
    comptime OutputM = config.output_tile_shape[0]
    comptime OutputN = config.output_tile_shape[1]
    var c_tiles = SMemTileArray2DRowMajor[
        c_type, OutputM, OutputN, config.num_output_stages, 128
    ](c_smem_storage.unsafe_ptr())

    # Structured epilogue configuration.
    comptime opc = OutputPipelineConfig(
        config.num_accum_pipeline_stages, stage_stride_cols, config.cta_group
    )
    comptime TileWriterType = TileWriter[
        a_type=a_type,
        accum_type=accum_type,
        block_tile_shape=config.block_tile_shape,
        mma_shape=config.mma_shape,
        opc=opc,
        c_swizzle=config.c_swizzle,
        transpose_c=config.AB_swapped,
        c_smem_dim0=OutputM,
        c_smem_dim1=OutputN,
        num_output_stages=config.num_output_stages,
        num_output_warps=num_output_warps,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        batched=True,
    ]
    comptime OutputStageType = OutputStage[opc]

    var sfa_smem = LayoutTensorIter[
        sfa_dtype,
        sfa_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        sfa_smem_storage.unsafe_ptr(),
        SmemType.sfa_smem_size,
    )
    var sfb_smem = LayoutTensorIter[
        sfb_dtype,
        sfb_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        sfb_smem_storage.unsafe_ptr(),
        SmemType.sfb_smem_size,
    )

    # TileTensor views of the same shared memory for MMA consumer path.
    # LayoutTensorIter is kept for the TMA producer (load_AB_SFA_SFB).
    # SMemTileArray2D uses internal_k_major which matches tile_layout_k_major.
    # This requires transpose_b=True (enforced by block_scaled_dispatch).
    comptime assert (
        transpose_b
    ), "SMemTileArray2D uses K-major layout; transpose_b must be True"
    comptime num_ab_tiles = config.num_pipeline_stages
    var a_smem_tt = SMemTileArray2D[
        a_type,
        BM,
        BK,
        num_ab_tiles,
        swizzle_mode_to_bytes[config.a_swizzle],
    ](a_smem_storage.unsafe_ptr())
    var b_smem_tt = SMemTileArray2D[
        b_type,
        BN,
        BK,
        num_ab_tiles,
        swizzle_mode_to_bytes[config.b_swizzle],
    ](b_smem_storage.unsafe_ptr())

    # SF tile dimensions from shared helpers (avoids duplicating atom math).
    comptime sf_bk = SF_K_GROUP_SIZE[config.vec_sf_size] * config.num_sf_k_tiles
    comptime sfa_d0 = sf_tile_dim0[BM]
    comptime sfa_d1 = sf_tile_dim1[sf_bk, config.vec_sf_size]
    comptime sfb_mn = align_up(MMA_N, SF_MN_GROUP_SIZE)
    comptime sfb_d0 = sf_tile_dim0[sfb_mn]
    comptime sfb_d1 = sfa_d1  # Same K-dim computation

    # Verify TileTensor tile sizes match the LayoutTensorIter tile sizes.
    comptime assert (
        sfa_d0 * sfa_d1 == sfa_smem_layout.size()
    ), "SFA TileTensor tile size must match LayoutTensorIter tile size"
    # cp.async uses a smaller flat SMEM layout, so skip this check —
    # the TileTensor is only needed for MMA type compatibility and is
    # never read for MMA_N < 64 (copy_sf_to_tmem is skipped).
    comptime if not config.use_cpasync_sfb:
        comptime assert (
            sfb_d0 * sfb_d1 == sfb_smem_layout.size()
        ), "SFB TileTensor tile size must match LayoutTensorIter tile size"

    comptime num_sf_tiles = config.num_pipeline_stages
    var sfa_smem_tt = SMemTileArrayWithLayout[
        sfa_dtype,
        internal_sf_k_major[sfa_d0, sfa_d1],
        num_sf_tiles,
    ](sfa_smem_storage.unsafe_ptr())
    var sfb_smem_tt = SMemTileArrayWithLayout[
        sfb_dtype,
        internal_sf_k_major[sfb_d0, sfb_d1],
        num_sf_tiles,
    ](sfb_smem_storage.unsafe_ptr())

    # Load warp as producer and mma warp as consumer
    # Dependence on MMA input in SMEM.
    # Consumer phase = 1 so that producer's wait on consumer passes trivially
    # at the start when buffer is empty.
    var load_mma_pipeline = ProducerConsumerPipeline[
        config.num_pipeline_stages // config.k_group_size
    ](
        tma_mma_mbars_storage.unsafe_ptr(),
    )

    # MMA warp as producer and Output warp as consumer.
    # Dependence on MMA output in TMEM.
    var mma_output_pipeline = ProducerConsumerPipeline[opc.num_stages](
        accum_mbars_storage.unsafe_ptr(),
    )

    # Load warp as producer and scheduler warp as consumer.
    # No data dependence. Introduce dependence to prevent CLC goes too ahead.
    # In the extreme case, all ctas keep querying next work simultaneously,
    # there will be no guarantee they get balanced number of tiles.
    var load_clc_pipeline = ProducerConsumerPipeline[
        config.num_clc_pipeline_stages
    ](
        clc_throttle_storage.unsafe_ptr(),
    )

    # SfbTMALoad warp as producer and MMA warp as consumer.
    # Dependence on SFB data in SMEM / TMEM slot consumed.
    var sfb_tma_pipeline = ProducerConsumerPipeline[
        config.num_pipeline_stages // config.k_group_size
    ](
        sfb_tma_mbars_storage.unsafe_ptr(),
    )

    var ptr_tmem_addr = tmem_addr_storage.unsafe_ptr()

    clc_response = clc_response_storage.unsafe_ptr()
    clc_full_mbar = clc_mbars_full_storage.unsafe_ptr()
    clc_empty_mbar = clc_mbars_empty_storage.unsafe_ptr()

    tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()

    load_sfb_mbars = sfb_load_mbars_storage.unsafe_ptr()

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
        comptime if not config.use_cpasync_sfb:
            sfb_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
            ),
        )

        comptime for i in range(SmemType.num_group_pipeline_stages):
            load_sfb_mbars[i].init(Int32(config.cta_group * SFB_LOAD_THREADS))

        mma_output_pipeline.init_mbars(
            Int32(accum_pipeline_producer_arv_count),
            Int32(accum_pipeline_consumer_arv_count),
        )
        load_clc_pipeline.init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )
        comptime if config.use_cpasync_sfb:
            # Each active thread does async_copy_arrive + mbar.arrive.
            # Warp-wide: all MMA_N * num_sf_k_tiles lanes are active.
            # Sequential: only MMA_N lanes are active.
            # Note: each CTA loads the full MMA_N scale factors (not split).
            comptime cpasync_arrive_count = (
                MMA_N * config.num_sf_k_tiles if MMA_N * config.num_sf_k_tiles
                <= 32 else MMA_N
            )
            sfb_tma_pipeline.init_mbars(
                Int32(cpasync_arrive_count),
                Int32(1),
            )
        else:
            sfb_tma_pipeline.init_mbars(
                Int32(1),
                Int32(1),
            )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

        comptime for i in range(config.num_clc_pipeline_stages):
            clc_full_mbar[i].init(Int32(clc_producer_arv_count))
            clc_empty_mbar[i].init(Int32(clc_consumer_arv_count))

    fence_mbarrier_init()

    comptime if CLUSTER_SIZE > 1:
        cluster_arrive_relaxed()

    var clc_pipe_producer_state = PipelineState[config.num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[
        config.num_clc_pipeline_stages
    ]()

    var sfb_load_pipe_consumer_state = PipelineState[
        SmemType.num_group_pipeline_stages
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
        cta_group=config.cta_group,
        cluster_shape=config.cluster_shape,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        transpose_b=True,
    ]()

    var scheduler = TileScheduler[
        num_stages=config.num_clc_pipeline_stages,
        cluster_shape=Index[dtype=DType.uint32](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        block_swizzle_size=config.block_swizzle_size,
        rasterize_order=config.raster_order,
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
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    comptime for i in range(CLUSTER_M // config.cta_group):
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

    comptime if CLUSTER_SIZE > 1:
        cluster_wait()
    else:
        barrier()

    if WarpRole.is_sfb_tmem_load():
        with MatmulProfilerType[0](workspace, 0):
            var required_clc_query = True

            named_barrier[
                Int32(MMA_THREADS + EPILOGUE_THREADS + SFB_LOAD_THREADS)
            ](1)

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

                var work_tile_coord = (UInt(work_info.m), UInt(work_info.n))
                # DO TMA LOAD
                for x in range(num_iters // UInt32(config.k_group_size)):
                    var stage = sfb_tma_pipeline.consumer_stage()
                    sfb_tma_pipeline.wait_producer()

                    comptime for kg in range(config.k_group_size):
                        var offset = stage * UInt32(
                            config.k_group_size
                        ) + UInt32(kg)

                        var sfb_smem_tile = sfb_smem.next(offset)[]
                        var sfb_tmem_offset = sfb_tmem + offset * UInt32(
                            SFB_NUM_COLS
                        )

                        comptime if config.use_cpasync_sfb:
                            sfb_smem_to_tmem_cpasync[
                                sfb_dtype,
                                MMA_N,
                                config.num_sf_k_tiles,
                            ](sfb_smem_tile.ptr, sfb_tmem_offset)
                        else:
                            sfb_smem_to_tmem_tma[
                                sfb_dtype,
                                MMA_N,
                                config.num_sf_k_tiles,
                            ](
                                sfb_smem_tile.ptr,
                                sfb_tmem_offset,
                                work_tile_coord[1],
                            )

                    # 2CTA: arrive on leader's barrier so MMA sees both CTAs.
                    comptime if config.cta_group == 1:
                        _ = load_sfb_mbars[
                            sfb_load_pipe_consumer_state.index()
                        ].arrive()
                    else:
                        umma_arrive_leader_cta(
                            load_sfb_mbars
                            + sfb_load_pipe_consumer_state.index()
                        )

                    sfb_tma_pipeline.consumer_step()
                    sfb_load_pipe_consumer_state.step()

                work_info = next_work_info

    if WarpRole.is_main_load():
        with MatmulProfilerType[0](workspace, 0):
            var required_clc_query = True

            comptime if pdl_level > PDLLevel.OFF:
                wait_on_dependent_grids()

            while work_info.is_valid():
                # CLC throttle prevents each CTA from going a few waves ahead.
                comptime if config.num_clc_pipeline_stages > 0:
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
                    load_AB_SFA[
                        block_tile_shape=config.block_tile_shape,
                        mma_shape=config.mma_shape,
                        num_sf_k_tiles=config.num_sf_k_tiles,
                        cta_group=config.cta_group,
                        k_group_size=UInt(config.k_group_size),
                    ](
                        a_tma_op,
                        b_tma_op,
                        sfa_tma_op,
                        a_smem,
                        b_smem,
                        sfa_smem,
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
            comptime for i in range(
                config.num_pipeline_stages // config.k_group_size
            ):
                load_mma_pipeline.wait_consumer()
                load_mma_pipeline.producer_step()

    if WarpRole.is_sfb_gmem_to_smem():
        with MatmulProfilerType[0](workspace, 0):
            while work_info.is_valid():
                next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )
                clc_pipe_consumer_state.step()

                comptime if config.use_cpasync_sfb:
                    comptime if MMA_N * config.num_sf_k_tiles <= 32:
                        _sfb_cpasync_produce_tile_warpwide[
                            sfb_dtype,
                            MMA_N,
                            config.num_sf_k_tiles,
                            config.k_group_size,
                            config.num_pipeline_stages // config.k_group_size,
                            sfb_smem_layout.size(),
                        ](
                            work_info,
                            num_iters,
                            sfb_tma_pipeline,
                            sfb_smem.ptr,
                            sfb_global_ptr,
                            sfb_batch_stride,
                            sfb_n_stride,
                            sfb_k_tiles,
                        )
                    else:
                        _sfb_cpasync_produce_tile[
                            sfb_dtype,
                            MMA_N,
                            config.num_sf_k_tiles,
                            config.k_group_size,
                            config.num_pipeline_stages // config.k_group_size,
                            sfb_smem_layout.size(),
                        ](
                            work_info,
                            num_iters,
                            sfb_tma_pipeline,
                            sfb_smem.ptr,
                            sfb_global_ptr,
                            sfb_batch_stride,
                            sfb_n_stride,
                            sfb_k_tiles,
                        )
                else:
                    _sfb_tma_produce_tile[
                        sfb_dtype,
                        MMA_N,
                        config.num_sf_k_tiles,
                        config.k_group_size,
                        config.num_pipeline_stages // config.k_group_size,
                        sfb_smem_layout.size(),
                        config.cta_group,
                        sfb_rank,
                        sfb_tile_shape,
                        sfb_desc_shape,
                    ](
                        work_info,
                        num_iters,
                        sfb_tma_pipeline,
                        sfb_smem.ptr,
                        elect_one_cta,
                        sfb_tma_op,
                    )

                work_info = next_work_info

            # Drain: prevent exit while SfbTMEMLoad is still working.
            comptime for i in range(
                config.num_pipeline_stages // config.k_group_size
            ):
                sfb_tma_pipeline.wait_consumer()
                sfb_tma_pipeline.producer_step()

    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        # Implies each SM will only process initial work, there is no
        # more work to schedule.
        comptime if config.num_clc_pipeline_stages == 0:
            return

        with MatmulProfilerType[1](workspace, 0):
            var required_clc_query = True

            comptime if pdl_level > PDLLevel.OFF:
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
            comptime for i in range(config.num_clc_pipeline_stages):
                clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                    clc_pipe_producer_state.phase()
                )
                clc_pipe_producer_state.step()

    if WarpRole.is_mma():
        with MatmulProfilerType[2](workspace, 0):
            tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
            syncwarp()
            # non blocking, arrives and proceeds
            named_barrier_arrive[
                Int32(MMA_THREADS + EPILOGUE_THREADS + SFB_LOAD_THREADS)
            ](1)

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
                            block_tile_shape=config.block_tile_shape,
                            mma_shape=config.mma_shape,
                            SFA_NUM_COLS=SFA_NUM_COLS,
                            SFB_NUM_COLS=SFB_NUM_COLS,
                            cta_group=config.cta_group,
                            cluster_shape=config.cluster_shape,
                            k_group_size=UInt(config.k_group_size),
                        ](
                            tmem_offset,
                            sfa_tmem,
                            sfb_tmem,
                            a_smem_tt,
                            b_smem_tt,
                            sfa_smem_tt,
                            sfb_smem_tt,
                            load_mma_pipeline,
                            load_sfb_mbars,
                            sfb_load_pipe_consumer_state,
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
                        sfb_load_pipe_consumer_state.step()

                        # Signal SfbTMALoad that this TMEM slot is consumed.
                        # 2CTA: leader signals both CTAs' private pipelines.
                        if elect_one_sync():
                            # Arrive on own (leader) CTA's barrier.
                            _ = sfb_tma_pipeline.consumer_mbar(
                                sfb_tma_pipeline.consumer_stage()
                            )[0].arrive()
                            # Arrive on peer CTA's barrier.
                            comptime if config.cta_group == 2:
                                umma_arrive_peer_cta(
                                    sfb_tma_pipeline.consumer_mbar(
                                        sfb_tma_pipeline.consumer_stage()
                                    )[0].unsafe_ptr()
                                )
                        sfb_tma_pipeline.consumer_step()

                    # mma arrive multicast will track completion of all mma prior to this barrier.
                    if elect_one_sync():
                        comptime if config.cta_group == 1:
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

            tcgen05_release_allocation_lock[Int32(config.cta_group)]()

            # wait for epilogue to finish
            tmem_dealloc_mbar[].wait()

            comptime if pdl_level > PDLLevel.OFF:
                launch_dependent_grids()

            tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS + SFB_LOAD_THREADS)](
            1
        )
        tmem_addr = ptr_tmem_addr[0]
        var tile_writer = TileWriterType(Pointer(to=c_tma_op))

        var tile_idx = 0

        while work_info.is_valid():
            with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                # Wait for MMA to finish this stage.
                var stage_idx = mma_output_pipeline.consumer_stage()
                mma_output_pipeline.wait_producer()
                var tmem_offset = (
                    stage_idx * UInt32(stage_stride_cols) + tmem_addr
                )

                # Create OutputStage from existing pipeline state.
                var output_stage = OutputStageType.from_raw(
                    mma_output_pipeline, stage_idx, tmem_offset
                )

                # TileWriter handles: TMEM load -> alpha scale -> SMEM write
                # -> TMA store -> AccumBarrier.arrive()
                tile_writer.write_batched(
                    c_tiles,
                    output_stage,
                    (work_info.m, work_info.n, work_info.k_start),
                    (mnk[0], mnk[1]),
                    alpha,
                )
                mma_output_pipeline.consumer_step()

                next_work_info = scheduler.fetch_next_work(
                    work_info, clc_pipe_consumer_state
                )
                work_info = next_work_info
                clc_pipe_consumer_state.step()

            tile_idx += 1

        comptime if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


# =============================================================================
# TMA + Kernel Launch: operates on already-reshaped 3D TileTensors (A/B/C)
# and 5D LayoutTensors (scale factors)
# =============================================================================


def _create_tma_and_launch[
    transpose_b: Bool,
    *,
    K: Int,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    a_3d: TileTensor,
    b_3d: TileTensor,
    c_3d: TileTensor,
    sfa_5d_tensor: TileTensor,
    sfb_5d_tensor: TileTensor,
    ctx: DeviceContext,
    alpha: Float32,
) raises:
    """Create TMA descriptors and launch the small-BN block-scaled matmul kernel.

    Takes 3D TileTensors for A/B/C and 5D LayoutTensors for scale factors.
    TMA descriptors and kernel launch live in the same scope to avoid
    lifetime issues with scoped TMA references.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime sfa_dtype = config.sfa_dtype
    comptime sfb_dtype = config.sfb_dtype

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]
    comptime cluster_shape = config.cluster_shape

    var B = Int(c_3d.dim[0]())
    var M = Int(c_3d.dim[1]())
    var N = Int(c_3d.dim[2]())
    var M_maybe_swapped = Int(a_3d.dim[1]())
    var N_maybe_swapped = Int(b_3d.dim[1]())

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    # A matrix TMA (from TileTensor)
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tensor_tile[
        a_tma_tile_shape,
        swizzle_mode=config.a_swizzle,
        __tile_shape=a_tma_tile_shape,
    ](ctx, a_3d)

    # fmt: off
    # B matrix TMA (from TileTensor)
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    ) if transpose_b else Index(
        1, BK, BN // (cluster_shape[0] // config.cta_group)
    )
    var b_tma_op = create_tensor_tile[
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
        __tile_shape = b_tma_tile_shape,
    ](ctx, b_3d)

    # C matrix TMA (from TileTensor)
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
        __tile_shape = c_tma_tile_shape_final,
    ](ctx, c_3d)
    # fmt: on

    # Scale factor TMAs (from LayoutTensor)
    comptime sfa_tma_tile_shape = Index(
        1,
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfa_tma_op = create_tensor_tile[
        sfa_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __tile_shape=sfa_tma_tile_shape,
    ](ctx, sfa_5d_tensor)

    # Copy one k-atom at a time, loading only the rows the MMA actually
    # needs (min(MMA_N, SF_ATOM_M[0])).  The TMA load loop issues
    # num_sf_k_tiles separate copies per k_group, placing each at the
    # correct atom-stride offset in SMEM.
    comptime SFB_TMA_ROWS = MMA_N if MMA_N < SF_ATOM_M[0] else SF_ATOM_M[0]
    comptime sfb_tma_tile_shape = Index(
        1,
        align_up(MMA_N, SF_MN_GROUP_SIZE) // SF_MN_GROUP_SIZE,
        1,
        SFB_TMA_ROWS,
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfb_tma_op = create_tensor_tile[
        sfb_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __tile_shape=sfb_tma_tile_shape,
    ](ctx, sfb_5d_tensor)

    # SFB global pointer and strides for cp.async path.
    # 5D layout: (batch, n_groups, k_tiles, SF_ATOM_M[0], SF_ATOM_M[1]*SF_ATOM_K)
    # Row-major strides: stride[2]=512, stride[1]=k_tiles*512,
    # stride[0]=n_groups*stride[1].
    var sfb_ptr = sfb_5d_tensor.ptr
    comptime K_TILE_ELEMS = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var sfb_k_tiles = Int(sfb_5d_tensor.dim[2]())
    var sfb_n_stride_val = sfb_k_tiles * K_TILE_ELEMS
    var sfb_batch_stride_val = Int(sfb_5d_tensor.dim[1]()) * sfb_n_stride_val

    # Shared memory
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

    # Kernel instantiation
    comptime kernel = blackwell_block_scaled_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
        type_of(sfa_tma_op).rank,
        type_of(sfa_tma_op).tile_shape,
        type_of(sfa_tma_op).desc_shape,
        type_of(sfb_tma_op).rank,
        type_of(sfb_tma_op).tile_shape,
        type_of(sfb_tma_op).desc_shape,
        transpose_b,
        config=config,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        elementwise_lambda_fn=elementwise_lambda_fn,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    # Grid and block dimensions
    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        B,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    comptime load_warps = 1
    comptime sfb_tma_load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4
    comptime sfb_tmem_load_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    # Profiling workspace
    var workspace: Span[UInt64, MutAnyOrigin]

    comptime if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = {}

    # Launch kernel
    ctx.enqueue_function[kernel, kernel, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        sfa_tma_op,
        sfb_tma_op,
        cluster_dim,
        mnk,
        workspace,
        Float32(alpha),
        sfb_ptr,
        sfb_batch_stride_val,
        sfb_n_stride_val,
        sfb_k_tiles,
        grid_dim=grid_dim,
        # 1 TMA, 1 SFB_TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE, 4 SFB_TMEM_LOAD warps
        block_dim=(
            32
            * (
                load_warps
                + sfb_tma_load_warps
                + mma_warps
                + scheduler_warps
                + epilogue_warps
                + sfb_tmem_load_warps
            )
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    comptime if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


# =============================================================================
# Internal host function: validates, reshapes, and dispatches to
# _create_tma_and_launch
# =============================================================================


def _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    K: Int,
    config: BlockScaledMatmulConfig[_, _, _, sfa_dtype, sfb_dtype, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: TileTensor,
    a_tensor: TileTensor,
    b_tensor: TileTensor,
    a_scales_tensor: TileTensor[sfa_dtype, ...],
    b_scales_tensor: TileTensor[sfb_dtype, ...],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    comptime assert (
        a_tensor.rank in (2, 3)
        and a_tensor.rank == b_tensor.rank == c_tensor.rank
    ), (
        "a_tensor, b_tensor, and c_tensor must have the same rank and be 2D"
        " (non-batched) or 3D (batched) TileTensors"
    )

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (
        MXFP8_SF_DTYPE,
        NVFP4_SF_DTYPE,
    ), "Only support float8_e8m0fnu (MXFP8) or float8_e4m3fn (NVFP4) for scales"

    comptime assert (
        config.scaling_kind == UMMAKind.KIND_MXF8F6F4
        or config.scaling_kind == UMMAKind.KIND_MXF4NVF4
    ), "Only support MXF8F6F4 or MXF4NVF4 for scaling kind"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime assert (
        a_scales_tensor.rank == b_scales_tensor.rank
    ), "a_scales and b_scales must have the same rank"

    comptime is_batched_matmul = a_scales_tensor.rank == 6

    comptime assert a_scales_tensor.rank in (
        5,
        6,
    ), "a_scales must be 5D (non-batched) or 6D (batched) tensors"

    comptime assert (
        a_scales_tensor.static_shape[3 if is_batched_matmul else 2]
        == b_scales_tensor.static_shape[3 if is_batched_matmul else 2]
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales_tensor.static_shape[4 if is_batched_matmul else 3]
        == b_scales_tensor.static_shape[4 if is_batched_matmul else 3]
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales_tensor.static_shape[5 if is_batched_matmul else 4]
        == b_scales_tensor.static_shape[5 if is_batched_matmul else 4]
        == SF_ATOM_K
    ), ""

    comptime if config.cta_group == 2:
        comptime assert MMA_M == 256 and MMA_N in (
            16,
            24,
            32,
            64,
            128,
            192,
            256,
        ), (
            "Only support cta_group == 2 with MMA_M == 256 and MMA_N in (16,"
            " 24, 32, 64, 128, 192, 256)"
        )

    else:
        comptime assert MMA_M == 128 and MMA_N in (
            8,
            16,
            24,
            32,
            64,
            128,
            192,
            256,
        ), (
            "Only support MMA_M == 128 and MMA_N in (8, 16, 24, 32, 64, 128,"
            " 192, 256) when cta_group == 1"
        )
    comptime register_based_epilogue = config.register_based_epilogue

    # Reshape scale factors to 5D TileTensor for TMA.
    # create_tensor_tile reads .layout.shape/stride from the TileTensor.
    @parameter
    def _scales_5d_shape(
        scales: TileTensor,
    ) -> Coord[
        RuntimeInt[DType.int64],
        RuntimeInt[DType.int64],
        RuntimeInt[DType.int64],
        ComptimeInt[SF_ATOM_M[0]],
        ComptimeInt[SF_ATOM_M[1] * SF_ATOM_K],
    ]:
        comptime if is_batched_matmul:
            return Coord(
                RuntimeInt[DType.int64](Int64(Int(scales.dim[0]()))),
                RuntimeInt[DType.int64](Int64(Int(scales.dim[1]()))),
                RuntimeInt[DType.int64](Int64(Int(scales.dim[2]()))),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )
        else:
            return Coord(
                RuntimeInt[DType.int64](Int64(1)),
                RuntimeInt[DType.int64](Int64(Int(scales.dim[0]()))),
                RuntimeInt[DType.int64](Int64(Int(scales.dim[1]()))),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )

    var sfa_5d_shape = _scales_5d_shape(a_scales_tensor)
    var sfa_5d_layout = tt_row_major(sfa_5d_shape)
    var sfa_5d_tensor = TileTensor[
        sfa_dtype, type_of(sfa_5d_layout), ImmutAnyOrigin
    ](
        rebind[UnsafePointer[Scalar[sfa_dtype], ImmutAnyOrigin]](
            a_scales_tensor.ptr
        ),
        sfa_5d_layout,
    )
    var sfb_5d_shape = _scales_5d_shape(b_scales_tensor)
    var sfb_5d_layout = tt_row_major(sfb_5d_shape)
    var sfb_5d_tensor = TileTensor[
        sfb_dtype, type_of(sfb_5d_layout), ImmutAnyOrigin
    ](
        rebind[UnsafePointer[Scalar[sfb_dtype], ImmutAnyOrigin]](
            b_scales_tensor.ptr
        ),
        sfb_5d_layout,
    )

    comptime if is_batched_matmul:
        _create_tma_and_launch[
            K=K,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            elementwise_lambda_fn=elementwise_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_5d_tensor,
            sfb_5d_tensor,
            ctx,
            alpha,
        )
    else:
        _create_tma_and_launch[
            K=K,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            elementwise_lambda_fn=elementwise_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            _to_batched_3d(a_tensor),
            _to_batched_3d(b_tensor),
            _to_batched_3d(c_tensor),
            sfa_5d_tensor,
            sfb_5d_tensor,
            ctx,
            alpha,
        )


def blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    K: Int,
    config: BlockScaledMatmulConfig[_, _, _, sfa_dtype, sfb_dtype, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(1),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: TileTensor,
    a_tensor: TileTensor,
    b_tensor: TileTensor,
    a_scales_tensor: TileTensor[sfa_dtype, ...],
    b_scales_tensor: TileTensor[sfb_dtype, ...],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """Launch small-BN block-scaled FP8 matmul kernel on SM100.

    A, B, C, and scale factors are all passed as TileTensors.
    A/B/C are 2D (non-batched) or 3D (batched).
    Scale factors are 5D (non-batched) or 6D (batched).

    When config.AB_swapped is True, internally swaps A and B operands
    (along with their scale factors) and transposes the output.
    """

    comptime if config.AB_swapped:
        # When both A and B are K-major, C = A @ B'.
        # If we swap A and B: D = B @ A', and D' = (B @ A')' = A @ B' = C.
        # So swapping + transposing the output gives the same result.
        # The transpose is handled by transpose_c = config.AB_swapped in the
        # kernel.
        comptime new_config = config.swap_AB_type()
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            sfb_dtype,
            sfa_dtype,
            transpose_b,
            K=K,
            config=new_config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            elementwise_lambda_fn=elementwise_lambda_fn,
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
            sfa_dtype,
            sfb_dtype,
            transpose_b,
            K=K,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            elementwise_lambda_fn=elementwise_lambda_fn,
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
