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
from memory import LegacyUnsafePointer as UnsafePointer
from sys import align_of, simd_width_of, size_of

from bit import next_power_of_two, prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu import block_id_in_cluster, block_idx, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.mma import st_matrix
from gpu.mma_sm100 import *
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
from gpu.tcgen05 import *
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
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .config import MatmulConfig
from ..tile_scheduler import RasterOrder
from .tile_scheduler import (
    TileScheduler,
    WorkInfo,
)
from .tile_scheduler_splitk import (
    get_required_locks_buffer_size_bytes,
    get_num_tiles,
    TileScheduler as TileSchedulerSplitK,
    WorkInfo as WorkInfoSplitK,
)
from ..profiler import (
    MatmulProfileWarp,
    MatmulWarpSpecializationWorkSpaceManager,
)
from .pipeline import ProducerConsumerPipeline


@fieldwise_init
@register_passable("trivial")
struct WarpRole(ImplicitlyCopyable):
    var _role: Int32

    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    fn __eq__(self, other: UInt) -> Bool:
        return self._role == other

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    fn __ge__(self, other: UInt) -> Bool:
        return self._role >= other

    @staticmethod
    @always_inline
    fn is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()

    @staticmethod
    @always_inline
    fn is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()


struct B200MatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime AType = Scalar[Self.a_type]
    comptime BType = Scalar[Self.b_type]
    comptime CType = Scalar[Self.c_type]

    comptime a_smem_size = Self.BM * Self.BK * Int(
        Self.config.num_pipeline_stages
    )
    comptime b_smem_size = Self.BN * Self.BK * Int(
        Self.config.num_pipeline_stages
    )
    comptime c_smem_size = Self.OutputM * Self.OutputN * Int(
        Self.config.num_output_stages
    )
    comptime num_group_pipeline_stages = Self.config.num_pipeline_stages // Self.config.k_group_size

    # AB pipelines
    var a_smem: InlineArray[Self.AType, Self.a_smem_size]
    var b_smem: InlineArray[Self.BType, Self.b_smem_size]
    var c_smem: InlineArray[Self.CType, Self.c_smem_size]
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
fn load_AB[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    num_pipeline_stages: UInt,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    k_group_size: UInt = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
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
    load_mma_pipeline: ProducerConsumerPipeline[Int(num_pipeline_stages)],
    peer_cta_coord: Tuple[UInt, UInt, UInt],
    work_tile_coord: Tuple[UInt, UInt],
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
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes
    ) * Int(k_group_size)

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[0].value()
    comptime b_tma_rows = b_desc_layout.shape[0].value()

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

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(expected_bytes)

        for j in range(k_group_size):
            var a_smem_tile = a_smem.next(stage * k_group_size + j)[]
            var b_smem_tile = b_smem.next(stage * k_group_size + j)[]

            var a_smem_slice = type_of(a_smem_tile)(
                a_smem_tile.ptr + peer_cta_coord[2] * UInt(a_tma_load_size)
            )
            var b_smem_slice = type_of(b_smem_tile)(
                b_smem_tile.ptr + peer_cta_coord[1] * UInt(b_tma_load_size)
            )

            a_tma_op.async_multicast_load[cta_group](
                a_smem_slice,
                tma_mbar[0],
                (UInt(iter_idx + j) * UInt(BK), UInt(a_gmem_slice_coord)),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_slice,
                tma_mbar[0],
                (UInt(iter_idx + j) * UInt(BK), UInt(b_gmem_slice_coord)),
                b_multicast_mask,
            )


@always_inline
fn consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: UInt = 1,
](
    tmem_addr: UInt32,
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
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
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
):
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        for j in range(k_group_size):
            var a_smem_tile = a_smem_iter.next(stage * k_group_size + j)[]
            var b_smem_tile = b_smem_iter.next(stage * k_group_size + j)[]
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(
                    (iter_idx + j) == k_start
                ),  # Initialize C on first iteration
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
    layout, element_type = DType.uint32, linear_idx_type = DType.uint32
]


@always_inline
fn f32_frag_to_smem[
    swizzle_mode: TensorMapSwizzle,
    stageN: UInt,
](
    vec: SIMD[_, _],
    dst: LayoutTensor[
        mut=True, _, _, address_space = AddressSpace.SHARED, *_, **_
    ],
):
    # TODO: apply swizzle. Somehow swizzle+distribute results in wrong values.
    # alias swizzle = make_swizzle[DType.float64, swizzle_mode]() # hack
    # var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4), swizzle=swizzle](lane_id())
    var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane_id()
    )
    constrained[
        2 * dst_frag.layout.size() == vec.size,
        "2*dst_frag.layout.size() must be equal to vec.size",
    ]()

    @parameter
    for i in range(dst_frag.layout.shape[0].value()):

        @parameter
        for j in range(dst_frag.layout.shape[1].value()):
            comptime i_vec = i + j * dst_frag.layout.shape[0].value()
            val = SIMD[dst.dtype, 2](
                rebind[Scalar[dst.dtype]](vec[2 * i_vec]),
                rebind[Scalar[dst.dtype]](vec[2 * i_vec + 1]),
            )
            dst_frag[i, j] = rebind[dst_frag.element_type](val)


@always_inline
fn stsm_helper[
    swizzle: Swizzle,
    stageN: UInt,
    transpose_c: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    vec: SIMD[_, _],
    dst: LayoutTensor[
        mut=True, _, _, address_space = AddressSpace.SHARED, *_, **_
    ],
    warp_offset: UInt32 = 0,
):
    @parameter
    if size_of[dst.dtype]() == 4:
        constrained[not transpose_c, "transpose_c must be False"]()
        return f32_frag_to_smem[swizzle_mode, stageN](vec, dst)
    # Number of elements in one row is 32B and 16B per stsmx4 and stmtx2 tile, respectively.
    comptime stsmx_row_size = 32 // size_of[
        dst.dtype
    ]() if stageN % 16 == 0 else 16 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    comptime stsmx_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    comptime stride0 = dst.layout.stride[0].value()
    comptime stride1 = dst.layout.stride[1].value()
    constrained[
        stride1 == 1,
        "stride1 must be 1. Got: "
        + String(stride1)
        + " for layout: "
        + String(dst.layout),
    ]()
    comptime shape0 = dst.layout.shape[
        1
    ].value() if not transpose_c else dst.layout.shape[0].value()
    # the layout looks like
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    # but transposed and coalesced by 8 elements.
    comptime trans_st_matrix_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(stride0, 8 * stride1, 8 * stride0)
    )
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane = lane_id()
    var stsm_lane_offset: UInt32 = (lane & 15) * UInt(stride0) + (
        lane >> 4
    ) * 8 if not transpose_c else RLayout32Bits[trans_st_matrix_layout]()(
        Int(lane)
    )

    # Helper function to slice a range of SIMD vector.
    # LLVM extract intrinsic generates bad code on GPU.
    @always_inline
    fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
        var tmp = SIMD[v.dtype, size]()

        @parameter
        for i in range(size):
            tmp[i] = v[i + offset]
        return tmp

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    @parameter
    for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        @parameter
        if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + n_offset + warp_offset) - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + n_offset)
        comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
        var v = slice[i * stsmx_lane_size, 2 * stmtx_simd_width](vec).cast[
            dst.dtype
        ]()
        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr + offset, bitcast[DType.float32, stmtx_simd_width](v)
        )


@always_inline
fn shared_memory_epilogue_transpose[
    stage: UInt,
    stageN: UInt,
    c_type: DType,
    c_smem_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: UInt,
    warp_dim: UInt,
    MMA_M: Int,
    BN: Int,
    cta_group: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem: LayoutTensor[c_type, c_smem_layout, MutAnyOrigin, *_, **_],
    warp_i: UInt,
    warp_j: UInt,
):
    var c_i = c_col + stage * stageN
    var c_j = c_row
    # this function write the shared memory tile to global memory starting at
    # (c_i, c_j). When `warp_dim` is 2, the layout modes are:
    # (warp_j, stageN, warp_i, UL),
    # else, `warp_dim` is 1, the layout modes are:
    # (stageN, warp_i, U), where U denotes upper and L denotes lower.
    comptime simd_size = simd_width_of[c_type]()
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    comptime swizzle_dim = 64

    @parameter
    if warp_dim == 2:
        comptime layout_3d = Layout.row_major(2, Int(stageN), swizzle_dim)
        var rt_layout_3d = RLayout32Bits[layout_3d]()
        constrained[c_smem_layout.rank() == 4, "c_smem_layout must be 4D"]()
        comptime thread_layout = Layout.row_major(1, 8, 1, 4)
        comptime result = zipped_divide(
            upcast(c_smem_layout, simd_size), thread_layout
        )
        var rt_thread_layout = RLayout32Bits[thread_layout]()
        var lane = lane_id()
        var crd = idx2crd(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE), element_type = DType.uint32](
                Int(lane)
            ),
            rt_thread_layout.shape,
            rt_thread_layout.stride,
        )
        comptime thread_shape = IntTuple(0, UNKNOWN_VALUE, 0, UNKNOWN_VALUE)

        @parameter
        for iter_i in range(result.shape[1][3].value()):

            @parameter
            for iter_j in range(result.shape[1][1].value()):
                comptime rest_shape = IntTuple(
                    UNKNOWN_VALUE, iter_j, UNKNOWN_VALUE, iter_i
                )
                var coord = RuntimeTuple[
                    [thread_shape, rest_shape], element_type = DType.uint32
                ](
                    Int(0),
                    Int(crd[1].get_int()),
                    Int(0),
                    Int(crd[3].get_int()),
                    Int(warp_j),
                    Int(iter_j),
                    Int(warp_i),
                    Int(iter_i),
                )
                var offset = simd_size * RLayout32Bits[result]()(coord)
                var logical_crd = idx2crd(
                    RuntimeTuple[
                        IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                    ](Int(offset)),
                    rt_layout_3d.shape,
                    rt_layout_3d.stride,
                )
                var local_i: UInt32
                var local_j: UInt32

                var ci = logical_crd[0].get_int()
                var cj = logical_crd[1].get_int()
                var ck = logical_crd[2].get_int()

                @parameter
                if cta_group == 2 and MMA_M == 128:
                    # logical shared memory -> global layout Layout B:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
                    local_i = cj + ci * BN
                    local_j = ck
                else:
                    # logical shared memory -> global layout Layout A:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
                    local_i = cj
                    local_j = ci * swizzle_dim + ck

                # undo swizzle to get logical `c_smem[logical_crd]` value.
                var ptr = (
                    c_smem.ptr
                    + swizzle(cj * swizzle_dim + ck)
                    + ci * swizzle_dim * Int(stageN)
                )
                var global_i = local_i + c_i
                var global_j = local_j + c_j
                if global_i < Int(M) and global_j < Int(N):
                    var val = ptr.load[width=simd_size, alignment=alignment]()
                    var reg_val = compute_lambda_fn[alignment=alignment](
                        (Int(global_i), Int(global_j)),
                        val,
                    )
                    ptr.store[width=simd_size, alignment=alignment](reg_val)
    else:
        # Layout F: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f
        constrained[c_smem_layout.rank() == 3, "c_smem_layout must be 3D"]()
        comptime thread_layout = Layout.row_major(min(16, Int(stageN)), 1, 2)
        comptime thread_bound = UInt(thread_layout.cosize())
        var lane = lane_id()
        if lane < thread_bound:
            comptime result = zipped_divide(
                upcast(c_smem_layout, simd_size), thread_layout
            )
            var rt_thread_layout = RLayout32Bits[thread_layout]()
            var crd = idx2crd(
                RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                ](Int(lane)),
                rt_thread_layout.shape,
                rt_thread_layout.stride,
            )
            comptime thread_shape = IntTuple(UNKNOWN_VALUE, 0, UNKNOWN_VALUE)
            comptime layout_2d = Layout.row_major(Int(stageN), swizzle_dim)
            var rt_layout_2d = RLayout32Bits[layout_2d]()

            @parameter
            for iter_i in range(result.shape[1][2].value()):

                @parameter
                for iter_j in range(result.shape[1][0].value()):
                    comptime rest_shape = IntTuple(
                        iter_j,
                        UNKNOWN_VALUE,
                        iter_i,
                    )
                    var coord = RuntimeTuple[
                        [thread_shape, rest_shape], element_type = DType.uint32
                    ](
                        Int(crd[0].get_int()),
                        Int(0),
                        Int(crd[2].get_int()),
                        Int(iter_j),
                        Int(warp_i),
                        Int(iter_i),
                    )
                    var offset = simd_size * RLayout32Bits[result]()(coord)
                    var logical_crd = idx2crd(
                        RuntimeTuple[
                            IntTuple(UNKNOWN_VALUE), element_type = DType.uint32
                        ](Int(offset)),
                        rt_layout_2d.shape,
                        rt_layout_2d.stride,
                    )

                    var local_i = logical_crd[0].get_int()
                    var local_j = logical_crd[1].get_int()

                    # undo swizzle to get logical `c_smem[logical_crd]` value.
                    var ptr = c_smem.ptr + swizzle(offset)
                    var global_i = local_i + c_i
                    var global_j = local_j + c_j
                    if global_i < Int(M) and global_j < Int(N):
                        var val = ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var reg_val = compute_lambda_fn[alignment=alignment](
                            (Int(global_i), Int(global_j)),
                            val,
                        )
                        ptr.store[width=simd_size, alignment=alignment](reg_val)

    named_barrier[num_output_warps * UInt(WARP_SIZE)]()


@always_inline
fn shared_memory_epilogue[
    MMA_M: UInt,
    data_paths: UInt,
    num_stages: UInt,
    stage: UInt,
    stageN: UInt,
    c_type: DType,
    shared_n: UInt,
    simd_size: UInt,
    c_smem_upper_layout: Layout,
    c_smem_lower_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: UInt,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem_warp_tile_upper: LayoutTensor[
        c_type, c_smem_upper_layout, MutAnyOrigin, *_, **_
    ],
    c_smem_warp_tile_lower: LayoutTensor[
        c_type, c_smem_lower_layout, MutAnyOrigin, *_, **_
    ],
):
    # Here we start keeping track of the index / indices this thread is
    # responsible for in shared memory. This is represented with shared_memory_row
    # and shared_memory_column and the children of these values shared_memory_row_upper_half
    # shared_memory_row_lower_half. We also need to update the global memory column c_col by
    # stageN since we are sliding through the overall compute block.

    var staged_c_col = c_col + stage * stageN

    var warp_id = get_warp_id()
    var shared_memory_row = warp_id * 32

    var shared_memory_row_upper_half = shared_memory_row
    var shared_memory_row_lower_half = shared_memory_row + 16

    # This distribute layout allocates vectors to corresponding threads. If stageN is 32, 8 x 4 is used since each row of
    # 4 threads can access 8 elements (8 x 4 = 32). If stageN is 16 then 16 x 2 is used. Since each fragment contains 16 rows,
    # there will be 2 chunks created when using 8x4.

    comptime distribute_cols = stageN // simd_size
    comptime distribute_rows = WARP_SIZE // Int(distribute_cols)

    comptime distribute_layout = Layout.row_major(
        distribute_rows, Int(distribute_cols)
    )
    var c_smem_upper_frag = c_smem_warp_tile_upper.vectorize[
        1, Int(simd_size)
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    var c_smem_lower_frag = c_smem_warp_tile_lower.vectorize[
        1, Int(simd_size)
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    comptime fragment_size = c_smem_upper_frag.layout.size()

    var local_row, local_col = divmod(lane_id(), distribute_cols)

    var shared_memory_col = local_col * simd_size
    shared_memory_row_lower_half += local_row
    shared_memory_row_upper_half += local_row

    @parameter
    for i in range(fragment_size):
        comptime alignment = align_of[SIMD[c_type, Int(simd_size)]]()

        # these offsets are swizzled so to retrieve the corresponding gmem offset we need to remove the swizzle
        # luckily removing the swizzle is as simple as swizzling a second time
        var swz_offset_upper = (
            shared_memory_row_upper_half * shared_n + shared_memory_col
        )
        var swz_offset_lower = (
            shared_memory_row_lower_half * shared_n + shared_memory_col
        )

        var offset_upper = swizzle(Int(swz_offset_upper))
        var offset_lower = swizzle(Int(swz_offset_lower))

        var shared_upper_row: Int64
        var shared_upper_col: Int64
        var shared_lower_row: Int64
        var shared_lower_col: Int64

        # Now that we have the true index we, need to add the global tile index to find the corresponding
        # index, in gmem. However the data will be stored in tensor memory differently depending on
        # MMA_M size, we take that into account here.

        @parameter
        if MMA_M != 256:
            comptime blocked_m_128_layout = blocked_product(
                Layout.row_major(Int(data_paths * 2), Int(stageN)),
                Layout.col_major(2, 2),
                coalesce_output=True,
            )

            var upper_coord = idx2crd(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_upper),
                RuntimeTuple[
                    blocked_m_128_layout.shape,
                    element_type = DType.int64,
                ](),
                RuntimeTuple[
                    blocked_m_128_layout.stride,
                    element_type = DType.int64,
                ](),
            )

            var lower_coord = idx2crd(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_lower),
                RuntimeTuple[
                    blocked_m_128_layout.shape,
                    element_type = DType.int64,
                ](),
                RuntimeTuple[
                    blocked_m_128_layout.stride,
                    element_type = DType.int64,
                ](),
            )

            shared_upper_row = upper_coord[0].get_int()
            shared_lower_row = lower_coord[0].get_int()

            var section_offset_upper = upper_coord[1][1].get_int()
            var col_offset_upper = upper_coord[1][0].get_int()

            var section_offset_lower = lower_coord[1][1].get_int()
            var col_offset_lower = lower_coord[1][0].get_int()

            shared_upper_col = (
                section_offset_upper * (num_stages * stageN) + col_offset_upper
            )
            shared_lower_col = (
                section_offset_lower * (num_stages * stageN) + col_offset_lower
            )

        else:
            # can't cast to uint64 as it's not supported yet
            # this will cost us slightly in performance
            comptime fast_div = FastDiv[DType.uint32](Int(shared_n))

            shared_upper_row = (
                Scalar[DType.int](offset_upper).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_upper_col = offset_upper % Int(shared_n)

            shared_lower_row = (
                Scalar[DType.int](offset_lower).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_lower_col = offset_lower % Int(shared_n)

        # now we need to add the global tile offset
        var global_upper_row = shared_upper_row + c_row
        var global_upper_col = shared_upper_col + staged_c_col
        var global_lower_row = shared_lower_row + c_row
        var global_lower_col = shared_lower_col + staged_c_col

        if global_upper_row < Int(M) and global_upper_col < Int(N):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_upper_row), Int(global_upper_col)),
                c_smem_upper_frag[i, 0],
            )
            c_smem_upper_frag[i, 0] = reg_val

        if global_lower_row < Int(M) and global_lower_col < Int(N):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_lower_row), Int(global_lower_col)),
                c_smem_lower_frag[i, 0],
            )
            c_smem_lower_frag[i, 0] = reg_val

        # If more than one chunk is created (happens when 8x4 is used)
        # they will be spaced 8 rows away from each other

        shared_memory_row_upper_half += UInt(distribute_rows)
        shared_memory_row_lower_half += UInt(distribute_rows)

    named_barrier[num_output_warps * UInt(WARP_SIZE)]()


fn _blackwell_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: OptionalReg[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, *_, **_],
    a_device: LayoutTensor[a_type, a_layout, *_, **_],
    b_device: LayoutTensor[b_type, b_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    constrained[
        config.cta_group in (1, 2), "Only support cta_group == 1 or 2"
    ]()

    constrained[
        config.num_pipeline_stages % config.k_group_size == 0,
        "num_pipeline_stages must be a multiple of k_group_size",
    ]()

    @parameter
    if config.cta_group == 2:
        constrained[
            (MMA_M == 256 or MMA_M == 128),
            "Only support cta_group == 2 with MMA_M == 128 or 256",
        ]()
        constrained[
            (MMA_M != 256) or (MMA_N % 16 == 0),
            "MMA_N must be a multiple of 16 when MMA_M is 256",
        ]()
        constrained[
            (
                config.AB_swapped
                or MMA_M != 128
                or register_based_epilogue
                or elementwise_compute_lambda_fn is None
            )
            or (MMA_N % 32 == 0),
            (
                "SM100 doesn't support shared memory based epilogue when MMA_M"
                " == 128 and MMA_N is not a multiple of 32"
            ),
        ]()
    else:
        constrained[
            MMA_M == 128 or MMA_M == 64,
            "Only support MMA_M == 128 or 64 when cta_group == 1",
        ]()

    comptime cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var M_maybe_swapped = a_device.dim[0]()
    var N_maybe_swapped = b_device.dim[0]()
    comptime K = a_layout.shape[1].value()

    constrained[
        ceildiv(K, BK) % Int(config.k_group_size) == 0,
        "K iterations must be a multiple of k_group_size",
    ]()

    a_tma_op = create_tma_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode = config.a_swizzle
    ](ctx, a_device)

    b_tma_op = create_tma_tile[
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        is_k_major=transpose_b,
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    constrained[
        (not config.AB_swapped) or config.c_swizzle.bytes() == 128,
        "Only support 128B swizzle mode when AB_swapped is True",
    ]()
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime smem_size = size_of[SmemType]()

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    comptime kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N_maybe_swapped, MMA_N), Int(cluster_shape[1])),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        ceildiv(grid_dim[0], cluster_shape[0]),
        ceildiv(grid_dim[1], cluster_shape[1]),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](M, N, K)

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

    ctx.enqueue_function_checked[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(b200_smem),
        attributes=pdl_launch_attributes(pdl_level),
    )

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


@always_inline
fn _compute_register_lambda_fn[
    epilogue_dtype: DType,
    frag_size: UInt,
    inc: UInt,
    offset: UInt,
    compute_lambda_fn: elementwise_compute_lambda_type,
    transpose_c: Bool,
](
    top_coord: StaticTuple[UInt32, 2],
    bottom_coord: StaticTuple[UInt32, 2],
    mut frag: SIMD[epilogue_dtype, Int(frag_size)],
    staged_c_row: UInt32,
    staged_c_col: UInt32,
):
    # update local coordinates w/ global memory offsets
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + top_coord[0], staged_c_col + top_coord[1] + inc
    )

    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + bottom_coord[0], staged_c_col + bottom_coord[1] + inc
    )

    # slice the fragment to get the current repeat top and bottom fragments
    var simd_top = frag.slice[2, offset = Int(offset)]()
    var simd_bottom = frag.slice[2, offset = Int(offset + 2)]()

    # In normal case, simd_top and simd_bottom are elements on the M dimension
    # when transpose_c is true, they are on the N dimension. We change the index order
    # when we do the transpose and pass the SIMD sector one-by-one to the lambda function.
    @parameter
    for i in range(simd_top.size):

        @parameter
        if not transpose_c:
            simd_top[i] = compute_lambda_fn(
                IndexList[2](
                    Int(top_frag_upper_coord[0]),
                    Int(top_frag_upper_coord[1] + i),
                ),
                simd_top[i],
            )

            simd_bottom[i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[0]),
                    Int(bottom_frag_upper_coord[1] + i),
                ),
                simd_bottom[i],
            )
        else:
            simd_top[i] = compute_lambda_fn(
                IndexList[2](
                    Int(top_frag_upper_coord[1] + i),
                    Int(top_frag_upper_coord[0]),
                ),
                simd_top[i],
            )

            simd_bottom[i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[1] + i),
                    Int(bottom_frag_upper_coord[0]),
                ),
                simd_bottom[i],
            )

    # store the results back into the fragment
    frag[Int(offset)] = simd_top[0]
    frag[Int(offset + 1)] = simd_top[1]
    frag[Int(offset + 2)] = simd_bottom[0]
    frag[Int(offset + 3)] = simd_bottom[1]


@always_inline
fn register_epilogue[
    MMA_M: UInt,
    data_paths: UInt,
    num_stages: UInt,
    bits: UInt,
    stage: UInt,
    stageN: UInt,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: UInt,
    epilogue_dtype: DType,
    frag_size: UInt,
    repeats: UInt,
    transpose_c: Bool,
    cta_group: Int,
    is_lower_frag_required: Bool,
](
    mut upper_frag_casted: SIMD[epilogue_dtype, Int(frag_size)],
    mut lower_frag_casted: SIMD[epilogue_dtype, Int(frag_size)],
    c_row: UInt32,
    c_col: UInt32,
    N: UInt32,
):
    constrained[
        bits == 256 and data_paths == 16,
        "Only 16x256b tensor memory load is supported",
    ]()

    comptime load_width = 2

    var warp_id = get_warp_id()

    # get global memory offset based on tile coordinates

    # we update the column offset to include the current stage
    var staged_c_col = c_col + stage * stageN
    var staged_c_row = c_row

    @parameter
    if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row += warp_id * 32
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row += warp_id * 16
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row += (warp_id % 2) * 32
        staged_c_col += (warp_id // 2) * num_stages * stageN

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = stageN // repeats // load_width
    var top_frag_upper_coord_left = StaticTuple[UInt32, 2](
        lane_id() // threads_per_row, lane_id() % threads_per_row * load_width
    )

    # getting the other 3 coordinates is straightforward. Each fragment is spaced out by 16 rows
    # and within each fragment the elements are spaced out by 8 rows(this can be seen by the tv layout).
    var bottom_frag_upper_coord_left = StaticTuple[UInt32, 2](
        top_frag_upper_coord_left[0] + 8, top_frag_upper_coord_left[1]
    )

    var top_frag_lower_coord_left = StaticTuple[UInt32, 2](
        top_frag_upper_coord_left[0] + 16, top_frag_upper_coord_left[1]
    )

    var bottom_frag_lower_coord_left = StaticTuple[UInt32, 2](
        top_frag_lower_coord_left[0] + 8, top_frag_lower_coord_left[1]
    )

    @parameter
    for i in range(repeats):
        # each tensor memory load (16x256b) may be repeated based on our desired size.
        # if thats the case our fragment will be repeated as well. So process it in chunks i.e
        # one 16x256b at a time.
        # inc represents the shift in global memory offset for each chunk, based on the repeat, and
        # offset represents the offset into the fragment for each chunk.

        comptime inc = i * 8
        comptime offset = i * 4

        comptime helper = _compute_register_lambda_fn[
            epilogue_dtype=epilogue_dtype,
            frag_size=frag_size,
            compute_lambda_fn=compute_lambda_fn,
            inc=inc,
            offset=offset,
            transpose_c=transpose_c,
        ]

        helper(
            top_frag_upper_coord_left,
            bottom_frag_upper_coord_left,
            upper_frag_casted,
            staged_c_row,
            staged_c_col,
        )

        @parameter
        if is_lower_frag_required:
            helper(
                top_frag_lower_coord_left,
                bottom_frag_lower_coord_left,
                lower_frag_casted,
                staged_c_row,
                staged_c_col,
            )


@always_inline
fn accum_arrive[
    cta_group: Int
](mma_output_pipeline: ProducerConsumerPipeline, mma_output_stage: UInt32):
    @parameter
    if cta_group == 1:
        _ = mbarrier_arrive(mma_output_pipeline.consumer_mbar(mma_output_stage))
    else:
        umma_arrive_leader_cta(
            mma_output_pipeline.consumer_mbar(mma_output_stage)
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
    elementwise_compute_lambda_fn: OptionalReg[
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
    c_coord: Tuple[UInt32, UInt32],
    c_shape: Tuple[UInt32, UInt32],
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
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
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
    var c_row = c_coord[0] * UInt(BM)
    var c_col = c_coord[1] * UInt(MMA_N)

    @parameter
    for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + (stage * stageN)
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

        upper_frag_casted = upper_frag_partial.cast[epilogue_dtype]()

        @parameter
        if is_lower_frag_required:
            lower_frag_casted = lower_frag_partial.cast[epilogue_dtype]()

        @parameter
        if elementwise_compute_lambda_fn:

            @parameter
            if register_based_epilogue:
                register_epilogue[
                    UInt(MMA_M),
                    data_paths,
                    UInt(num_stages),
                    bits,
                    UInt(stage),
                    UInt(stageN),
                    elementwise_compute_lambda_fn.value(),
                    UInt(num_output_warps),
                    epilogue_dtype,
                    UInt(upper_frag_casted.size),
                    UInt(repeat),
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
                    upper_frag_casted, c_smem_warp_tile_upper, warp_offset
                )

                warp_offset += tile_width // 2
                stsm_helper[swizzle, UInt(stageN), transpose_c](
                    lower_frag_casted, c_smem_warp_tile_lower, warp_offset
                )

                # Guard the write to shared memory is done.
                named_barrier[num_output_warps * UInt(WARP_SIZE)]()

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
                            UInt(num_output_warps),
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
                    upper_frag_casted, c_smem_warp_tile_upper, warp_offset
                )

                # Guard the write to shared memory is done.
                named_barrier[num_output_warps * UInt(WARP_SIZE)]()

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
                            UInt(num_output_warps),
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
            named_barrier[num_output_warps * UInt(WARP_SIZE)]()

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
                        UInt(num_output_warps),
                    ](
                        M,
                        N,
                        UInt(c_col),
                        UInt(c_row),
                        c_smem_warp_tile_upper,
                        c_smem_warp_tile_lower,
                    )

        var lane = lane_id()

        comptime CG2_TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 else BM
        comptime CG1_TMA_BM = c_smem_tile.layout.shape[0].value()
        comptime TMA_BM = CG2_TMA_BM if cta_group == 2 else CG1_TMA_BM

        var cg2_elect_one_warp = (
            warp_id == 0 if MMA_M == 256 else warp_id % 2 == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        var elect_one_warp = (
            cg2_elect_one_warp if cta_group == 2 else cg1_elect_one_warp
        )

        var coord_n_mma_m256 = c_coord[1] * UInt(MMA_N) + UInt(stage * stageN)
        var coord_n_mma_m128 = (
            c_coord[1] * UInt(MMA_N)
            + UInt(stage * stageN)
            + UInt(BN * Int(warp_id // 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_coord[0] * UInt(BM)

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
                        (
                            UInt(coord_m),
                            UInt(coord_n),
                        ),
                    )

                else:
                    comptime num_c_smem_tiles = 128 // swizzle_width // (
                        1 if is_lower_frag_required else 2
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
                            (
                                UInt(coord_m + UInt(i * swizzle_width)),
                                UInt(coord_n),
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
                    (
                        UInt(coord_n),
                        UInt(coord_m),
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
            named_barrier[num_output_warps * UInt(WARP_SIZE)]()


@always_inline
fn multi_stage_store_C_split_k[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    reduction_layout: Layout,
    num_accum_pipeline_stages: UInt,
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
    register_based_epilogue: Bool = True,  # if false it will perform epilogue on data in shared memory
    transpose_c: Bool = False,
](
    scheduler: TileSchedulerSplitK,
    reduction_tensor: LayoutTensor[accum_type, reduction_layout, MutAnyOrigin],
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    mma_output_pipeline: ProducerConsumerPipeline[
        Int(num_accum_pipeline_stages)
    ],
    tmem_addr: UInt32,
    work_info: WorkInfoSplitK,
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
    comptime epilogue_dtype = c_type if input_type is DType.bfloat16 else DType.float32

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    # typically repeated 4 times to get the desired 32 elements
    # stageN is how many elements we want to load at once

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var mma_output_stage = mma_output_pipeline.consumer_stage()
    mma_output_pipeline.wait_producer()

    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = mma_output_stage * stage_stride_cols + tmem_addr
    var epilogue_thread_idx = thread_idx.x

    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE

    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    # typically repeated 4 times to get the desired 32 elements
    # stageN is how many elements we want to load at once

    # repetitions per stage
    comptime stage_rep = stageN // (bits // 32)

    var is_last_split = scheduler.reduction(
        reduction_tensor,
        tmem_offset,
        epilogue_thread_idx,
        work_info,
    )

    # Do not copy to c_tile since they are in reduction workspace already.
    # If it is the last split, accumulators will already be in tmem.
    if not is_last_split:
        # signal accumulator arrival and exit.
        accum_arrive[cta_group](mma_output_pipeline, mma_output_stage)
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
        c_iter,
        c_tma_op,
        mma_output_pipeline,
        mma_output_stage,
        tmem_offset,
        (work_info.m, work_info.n),
        (M, N),
    )


@always_inline
fn multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: UInt,
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
    mma_output_pipeline: ProducerConsumerPipeline[
        Int(num_accum_pipeline_stages)
    ],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[UInt32, UInt32],
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

    # assume N dimension is static
    comptime simd_size = simd_width_of[c_type]()

    # TODO (GEX-2630): This is a temporary workaround to support float32 compute epilogue for FP8 models for which we use compute lambda for dequantization.
    # We should remove this once GEX-2630 is fixed.
    comptime epilogue_dtype = c_type if input_type is DType.bfloat16 else DType.float32

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime stage_contiguous_size = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages

    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep = stageN // (bits // 32)  # repetitions per stage
    # typically repeated 4 times to get the desired 32 elements

    # stageN is how many elements we want to load at once

    # stmatrix related
    comptime st_matrix_swizzle = c_swizzle
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # lets keep track of the of the starting row and column in GMEM
    var c_row = work_tile_coord[0] * UInt(BM)
    var c_col = work_tile_coord[1] * UInt(MMA_N)

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var mma_output_stage = mma_output_pipeline.consumer_stage()
    mma_output_pipeline.wait_producer()

    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = mma_output_stage * stage_stride_cols + tmem_addr

    comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
    comptime rep_frag_size = rep * fragment_size
    var upper_frag_casted = SIMD[epilogue_dtype, rep_frag_size]()
    var lower_frag_casted = SIMD[epilogue_dtype, rep_frag_size]()

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
        c_iter,
        c_tma_op,
        mma_output_pipeline,
        mma_output_stage,
        tmem_offset,
        work_tile_coord,
        (M, N),
    )


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn blackwell_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Need because nvvm.cluster_dim only takes StaticTuple
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
):
    constrained[c_type is not DType.float32, "c_type cannot be float32"]()
    constrained[transpose_b, "only support k-major B"]()

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

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // Int(
        config.num_accum_pipeline_stages
    )

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M = Int(config.cluster_shape[0])
    comptime CLUSTER_N = Int(config.cluster_shape[1])

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[0].value()
    comptime b_tma_rows = b_desc_layout.shape[0].value()

    # keep the physical SMEM buffer BM x MMA_N
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode = config.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]()

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]

    ref smem_storage = external_memory[
        Scalar[DType.uint8],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]

    ref a_smem_storage = smem_storage.a_smem
    ref b_smem_storage = smem_storage.b_smem
    ref c_smem_storage = smem_storage.c_smem
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

    comptime accum_type = get_accum_type[a_type]()

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            config.cluster_shape[0] // config.cta_group
            + config.cluster_shape[1]
            - 1,
        )
        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            accum_pipeline_consumer_arv_count,
        )
        load_clc_pipeline.init_mbars(
            clc_throttle_producer_arv_count,
            clc_throttle_consumer_arv_count,
        )

        tmem_dealloc_mbar[].init(EPILOGUE_THREADS * config.cta_group)

        @parameter
        for i in range(config.num_clc_pipeline_stages):
            clc_full_mbar[i].init(clc_producer_arv_count)
            clc_empty_mbar[i].init(clc_consumer_arv_count)

    fence_mbarrier_init()
    cluster_sync()

    var clc_pipe_producer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ](0, 1, 0)
    var clc_pipe_consumer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
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
        a_multicast_mask |= 1 << (i * CLUSTER_M)
    # they all have the same v and m, but different n,

    @parameter
    for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= 1 << (i * config.cta_group)

    a_multicast_mask <<= rank_m
    b_multicast_mask <<= peer_cta_coord[0]
    b_multicast_mask <<= rank_n * UInt(CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], BK)

    comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
        warp_role, max_profiled_tiles_per_SM
    ]

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
                for i in range(num_iters // config.k_group_size):
                    load_AB[
                        block_tile_shape = config.block_tile_shape,
                        mma_shape = config.mma_shape,
                        cta_group = config.cta_group,
                        k_group_size = config.k_group_size,
                    ](
                        a_tma_op,
                        b_tma_op,
                        a_smem,
                        b_smem,
                        load_mma_pipeline,
                        peer_cta_coord,
                        (UInt(work_info.m), UInt(work_info.n)),
                        a_multicast_mask,
                        b_multicast_mask,
                        i * config.k_group_size,
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
            tcgen05_alloc[config.cta_group](ptr_tmem_addr, max_tmem_cols)
            syncwarp()
            # non blocking, arrives and proceeds
            named_barrier_arrive[MMA_THREADS + EPILOGUE_THREADS](1)

            tmem_addr = ptr_tmem_addr[0]

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
                        mma_output_mma_stage * stage_stride_cols
                    )

                    for i in range(num_iters // config.k_group_size):
                        consumer_main_loop[
                            block_tile_shape = config.block_tile_shape,
                            mma_shape = config.mma_shape,
                            cta_group = config.cta_group,
                            cluster_shape = config.cluster_shape,
                            k_group_size = config.k_group_size,
                        ](
                            tmem_offset,
                            a_smem,
                            b_smem,
                            load_mma_pipeline,
                            mma_op,
                            elect_one_warp,
                            i * config.k_group_size,
                            0,
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
                                mma_complete_mask,
                            )
                    mma_output_pipeline.producer_step()
                work_info = next_work_info

            @parameter
            if pdl_level > PDLLevel.OFF:
                launch_dependent_grids()

            tcgen05_release_allocation_lock[config.cta_group]()

            # wait for epilogue to finish
            tmem_dealloc_mbar[].wait()

            tcgen05_dealloc[config.cta_group](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[MMA_THREADS + EPILOGUE_THREADS](1)
        tmem_addr = ptr_tmem_addr[0]

        var tile_idx = 0

        while work_info.is_valid():
            with MatmulProfilerType[3](workspace, tile_idx):
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
                    work_tile_coord=(work_info.m, work_info.n),
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


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn blackwell_tma_umma_warp_specialized_split_k_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    reduction_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Need because nvvm.cluster_dim only takes StaticTuple
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    max_profiled_tiles_per_SM: UInt32 = 0,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    reduction_tensor: LayoutTensor[
        config.accum_type, reduction_layout, MutAnyOrigin
    ],
    lock_ptr: UnsafePointer[UInt8],
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
):
    constrained[c_type is not DType.float32, "c_type cannot be float32"]()
    constrained[transpose_b, "only support k-major B"]()

    comptime num_output_warps = 4
    comptime num_split_k = config.num_split_k

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // Int(
        config.num_accum_pipeline_stages
    )

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M = Int(config.cluster_shape[0])
    comptime CLUSTER_N = Int(config.cluster_shape[1])

    comptime a_tma_load_size = a_desc_layout.size()
    comptime b_tma_load_size = b_desc_layout.size()
    comptime a_tma_rows = a_desc_layout.shape[0].value()
    comptime b_tma_rows = b_desc_layout.shape[0].value()

    # keep the physical SMEM buffer BM x MMA_N
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode = config.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode = config.b_swizzle
    ]()

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]

    ref smem_storage = external_memory[
        Scalar[DType.uint8],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]

    ref a_smem_storage = smem_storage.a_smem
    ref b_smem_storage = smem_storage.b_smem
    ref c_smem_storage = smem_storage.c_smem
    ref tma_mma_mbars_storage = smem_storage.tma_mma_mbars
    ref clc_response_storage = smem_storage.clc_response
    ref clc_mbars_full_storage = smem_storage.clc_mbars_full
    ref clc_mbars_empty_storage = smem_storage.clc_mbars_empty
    ref clc_throttle_storage = smem_storage.clc_throttle_mbars
    ref accum_mbars_storage = smem_storage.accum_mbars
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
    ](c_smem_storage.unsafe_ptr(), SmemType.c_smem_size)

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

    clc_response = clc_response_storage.unsafe_ptr()
    clc_full_mbar = clc_mbars_full_storage.unsafe_ptr()
    clc_empty_mbar = clc_mbars_empty_storage.unsafe_ptr()
    ptr_tmem_addr = tmem_addr_storage.unsafe_ptr()

    tmem_dealloc_mbar = tmem_dealloc_mbar_storage.unsafe_ptr()

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            config.cluster_shape[0] // config.cta_group
            + config.cluster_shape[1]
            - 1,
        )
        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            accum_pipeline_consumer_arv_count,
        )
        load_clc_pipeline.init_mbars(
            clc_throttle_producer_arv_count,
            clc_throttle_consumer_arv_count,
        )

        tmem_dealloc_mbar[].init(EPILOGUE_THREADS * config.cta_group)

        @parameter
        for i in range(config.num_clc_pipeline_stages):
            clc_full_mbar[i].init(clc_producer_arv_count)
            clc_empty_mbar[i].init(clc_consumer_arv_count)

    fence_mbarrier_init()
    cluster_sync()

    var clc_pipe_producer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ](0, 1, 0)
    var clc_pipe_consumer_state = PipelineState[
        Int(config.num_clc_pipeline_stages)
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        config.block_tile_shape,
        config.mma_shape,
        accum_type = config.accum_type,
        cta_group = config.cta_group,
        cluster_shape = config.cluster_shape,
        a_swizzle = config.a_swizzle,
        b_swizzle = config.b_swizzle,
        transpose_b=True,
    ]()

    var scheduler = TileSchedulerSplitK[
        num_stages = Int(config.num_clc_pipeline_stages),
        reduction_tile_shape = Index(BM, MMA_N, BK),
        cluster_shape = Index[dtype = DType.uint32](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        block_swizzle_size = config.block_swizzle_size,
        rasterize_order = config.raster_order,
        num_split_k = config.num_split_k,
    ](cluster_dim, mnk, clc_response, clc_full_mbar, clc_empty_mbar, lock_ptr)

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
        a_multicast_mask |= 1 << (i * CLUSTER_M)
    # they all have the same v and m, but different n,

    @parameter
    for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= 1 << (i * config.cta_group)

    a_multicast_mask <<= rank_m
    b_multicast_mask <<= peer_cta_coord[0]
    b_multicast_mask <<= rank_n * UInt(CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
        warp_role, max_profiled_tiles_per_SM
    ]

    if WarpRole.is_main_load():
        with MatmulProfilerType[0](workspace, 0):
            var required_clc_query = True

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

                var start = work_info.k_start
                var end = start + work_info.num_k_tiles

                # DO TMA LOAD
                for i in range(start, end, config.k_group_size):
                    load_AB[
                        block_tile_shape = config.block_tile_shape,
                        mma_shape = config.mma_shape,
                        cta_group = config.cta_group,
                        k_group_size = config.k_group_size,
                    ](
                        a_tma_op,
                        b_tma_op,
                        a_smem,
                        b_smem,
                        load_mma_pipeline,
                        peer_cta_coord,
                        (UInt(work_info.m), UInt(work_info.n)),
                        a_multicast_mask,
                        b_multicast_mask,
                        i,
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
            tcgen05_alloc[config.cta_group](ptr_tmem_addr, max_tmem_cols)
            syncwarp()
            # non blocking, arrives and proceeds
            named_barrier_arrive[MMA_THREADS + EPILOGUE_THREADS](1)

            tmem_addr = ptr_tmem_addr[0]

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
                        mma_output_mma_stage * stage_stride_cols
                    )

                    var start = work_info.k_start
                    var end = start + work_info.num_k_tiles

                    for i in range(start, end, config.k_group_size):
                        consumer_main_loop[
                            block_tile_shape = config.block_tile_shape,
                            mma_shape = config.mma_shape,
                            cta_group = config.cta_group,
                            cluster_shape = config.cluster_shape,
                            k_group_size = config.k_group_size,
                        ](
                            tmem_offset,
                            a_smem,
                            b_smem,
                            load_mma_pipeline,
                            mma_op,
                            elect_one_warp,
                            i,
                            start,
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
                                mma_complete_mask,
                            )
                    mma_output_pipeline.producer_step()
                work_info = next_work_info

            tcgen05_release_allocation_lock[config.cta_group]()

            # wait for epilogue to finish
            tmem_dealloc_mbar[].wait()

            tcgen05_dealloc[config.cta_group](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[MMA_THREADS + EPILOGUE_THREADS](1)
        tmem_addr = ptr_tmem_addr[0]

        var tile_idx = 0

        while work_info.is_valid():
            with MatmulProfilerType[3](workspace, tile_idx):
                # WAIT FOR MMA TO FINISH AND STORE RESULT
                # scheduler fetch next work

                multi_stage_store_C_split_k[
                    input_type=a_type,
                    accum_type = config.accum_type,
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
                    scheduler,
                    reduction_tensor,
                    c_smem_iter,
                    c_tma_op,
                    mma_output_pipeline,
                    tmem_addr,
                    work_info=work_info,
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


fn blackwell_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: OptionalReg[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, *_, **_],
    a_device: LayoutTensor[a_type, a_layout, *_, **_],
    b_device: LayoutTensor[b_type, b_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    @parameter
    if config.AB_swapped:
        # Swap the a_type, b_type in signature
        # TODO: Do this without creating a new instance.
        comptime new_config = config.swap_AB_type()

        # When both A and B are K-major, then the matrix multiplication math is
        # C = A @ B'
        # If we swap A and B, we have
        # D = B @ A'
        # Note that D' = (B @ A')' = A'' @ B' = A @ B' which is the same as the
        # original math. Therefore, when we swap A and B, we need to transpose
        # the result for consistency and correctness.
        @parameter
        if config.num_split_k > 1:
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                c_type,
                c_layout,
                b_type,
                b_layout,
                a_type,
                a_layout,
                transpose_b,
                config=new_config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, b_device, a_device, ctx)
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                c_type,
                c_layout,
                b_type,
                b_layout,
                a_type,
                a_layout,
                transpose_b,
                config=new_config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, b_device, a_device, ctx)
    else:

        @parameter
        if config.num_split_k > 1:
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                c_type,
                c_layout,
                a_type,
                a_layout,
                b_type,
                b_layout,
                transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                c_type,
                c_layout,
                a_type,
                a_layout,
                b_type,
                b_layout,
                transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)


fn _blackwell_matmul_tma_umma_warp_specialized_split_k[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    max_profiled_tiles_per_SM: OptionalReg[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, *_, **_],
    a_device: LayoutTensor[a_type, a_layout, *_, **_],
    b_device: LayoutTensor[b_type, b_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    constrained[
        config.cta_group in (1, 2), "Only support cta_group == 1 or 2"
    ]()

    @parameter
    if config.cta_group == 2:
        constrained[
            (MMA_M == 256 or MMA_M == 128),
            "Only support cta_group == 2 with MMA_M == 128 or 256",
        ]()
        constrained[
            (MMA_M != 256) or (MMA_N % 16 == 0),
            "MMA_N must be a multiple of 16 when MMA_M is 256",
        ]()

        # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
        constrained[
            (not config.AB_swapped) or MMA_M == 256,
            "swapAB is only supported for MMA_M == 256",
        ]()

    else:
        constrained[
            MMA_M == 128 or MMA_M == 64,
            "Only support MMA_M == 128 or 64 when cta_group == 1",
        ]()
        constrained[
            register_based_epilogue or elementwise_compute_lambda_fn is None,
            "only register-based epilogue is supported for cta_group == 1",
        ]()

    comptime cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var M_maybe_swapped = a_device.dim[0]()
    var N_maybe_swapped = b_device.dim[0]()
    comptime K = a_layout.shape[1].value()

    constrained[
        ceildiv(K, BK) % Int(config.k_group_size) == 0,
        "K iterations must be a multiple of k_group_size",
    ]()

    constrained[
        config.num_pipeline_stages % config.k_group_size == 0,
        "num_pipeline_stages must be a multiple of k_group_size",
    ]()

    a_tma_op = create_tma_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode = config.a_swizzle
    ](ctx, a_device)

    b_tma_op = create_tma_tile[
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        is_k_major=transpose_b,
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    # c_swizzle is set to 32B mode when swapAB is enabled so we need to adjust
    # the tile shape with 128B swizzle mode, there should always be 64 elements
    # on the contiguous dim.
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime smem_size = size_of[SmemType]()
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    comptime reduction_layout = Layout.row_major(UNKNOWN_VALUE, BM, MMA_N)

    comptime kernel = blackwell_tma_umma_warp_specialized_split_k_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        reduction_layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N_maybe_swapped, MMA_N), Int(cluster_shape[1])),
        config.num_split_k,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        ceildiv(grid_dim[0], cluster_shape[0]),
        ceildiv(grid_dim[1], cluster_shape[1]),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](M, N, K)

    var workspace: Span[UInt64, MutAnyOrigin]

    var output_tiles = get_num_tiles(
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )
    var num_output_tiles = output_tiles[0] * output_tiles[1]
    var lock_buffer_size_bytes = get_required_locks_buffer_size_bytes[
        config.accum_type
    ](
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )

    var locks_buffer = ctx.enqueue_create_buffer[DType.uint8](
        lock_buffer_size_bytes
    )
    var reduction_workspace = ctx.enqueue_create_buffer[config.accum_type](
        num_output_tiles * BM * MMA_N
    )

    var reduction_tensor = LayoutTensor[config.accum_type, reduction_layout](
        reduction_workspace,
        RuntimeLayout[reduction_layout].row_major(
            Index(num_output_tiles, BM, MMA_N)
        ),
    )

    ctx.enqueue_memset(locks_buffer, 0)

    @parameter
    if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin]()

    ctx.enqueue_function_checked[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        reduction_tensor,
        locks_buffer,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(b200_smem),
    )

    _ = reduction_workspace^
    _ = locks_buffer^

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn matmul_sm100_fallback_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]  # BM
    comptime MMA_N = mma_shape[1]  # BN
    comptime MMA_K = mma_shape[2]  # 16
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    a_smem = rebind[
        UnsafePointer[Scalar[a_type], address_space = AddressSpace.SHARED]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )

    # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // Int(
        num_threads
    )  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    # Create MmaOpSM100_SS instance to handle MMA operations
    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=1,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    for i in range(num_iters):
        # so only one thread per CTA does the copy
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (UInt(i) * UInt(BK), block_idx.y * UInt(BM)),
            )
            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (
                    UInt(i) * UInt(BK),
                    block_idx.x * UInt(BN),
                ) if transpose_b else (
                    block_idx.x * UInt(BN),
                    UInt(i) * UInt(BK),
                ),
            )

        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:
            # Use MmaOpSM100_SS to perform the MMA operation
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(i == 0),  # Initialize C on first iteration
            )

            mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()  # wait for the load to finish

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // UInt(WARP_SIZE)
    warp_id = thread_idx.x // UInt(WARP_SIZE)

    ctile, ctile_coords, _ = c.tile_with_offset[BM, BN](
        Int(block_idx.y), Int(block_idx.x)
    )
    comptime c_coord_type = type_of(ctile_coords)

    var M = c.dim[0]()
    comptime N = c.layout.shape[1].value()

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // Int(num_warps), MMA_N](
                    4 * m_mma + Int(warp_id), n_mma
                )
            )
            c_gmem_warp_tile_coords = ctile_coords + rebind[c_coord_type](
                _c_gmem_warp_tile_coords
            )

            c_gmem_frag, _c_gmem_frag_coords, _ = c_gmem_warp_tile.vectorize[
                1, 2
            ]().distribute_with_offset[Layout.row_major(8, 4)](lane_id())
            new_c_gmem_frag_coords = rebind[c_coord_type](_c_gmem_frag_coords)
            new_c_gmem_frag_coords[1] *= 2
            c_gmem_frag_coords = (
                c_gmem_warp_tile_coords + new_c_gmem_frag_coords
            )

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    comptime dst_idx = type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    comptime dst_m_offset = dst_idx // N
                    comptime dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < N:
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        @parameter
                        if elementwise_lambda_fn:
                            comptime alignment = align_of[SIMD[c_type, 2]]()
                            comptime epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (Int(m), Int(n)), c_mn
                            )
                        else:
                            c_gmem_frag[m_vec, n_vec] = rebind[
                                c_gmem_frag.element_type
                            ](c_mn)


fn matmul_sm100_fallback[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, *_, **_],
    a: LayoutTensor[a_type, a_layout, *_, **_],
    b: LayoutTensor[b_type, b_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    constrained[
        a_type == b_type and a_type in (DType.bfloat16, DType.float8_e4m3fn),
        "Only support bfloat16 and float8_e4m3fn",
    ]()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tma_op = create_tma_tile[Index(BM, BK), swizzle_mode=a_swizzle](ctx, a)
    b_tma_op = create_tma_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime block_dim = 128

    comptime kernel = matmul_sm100_fallback_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).layout,
        type_of(b_tma_op).layout,
        type_of(c).layout,
        type_of(a_tma_op).desc_layout,
        type_of(b_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    ctx.enqueue_function_checked[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c,
        UInt(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
