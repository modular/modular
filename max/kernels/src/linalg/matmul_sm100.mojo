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
from sys import size_of, align_of, simd_width_of
from math import ceildiv, align_up
from bit import prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import B200
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import (
    block_idx,
    lane_id,
    thread_idx,
    warp_id as get_warp_id,
    block_id_in_cluster,
)
from gpu.memory import (
    AddressSpace,
    fence_async_view_proxy,
    fence_mbarrier_init,
    external_memory,
)
from gpu.mma import st_matrix
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
)
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    RuntimeTuple,
)

from layout.layout import blocked_product
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_tuple import idx2crd
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    st_matrix_n_layout,
    tile_to_descriptor,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.swizzle import make_swizzle, make_ldmatrix_swizzle, Swizzle
from gpu.cluster import (
    elect_one_sync,
    elect_one_sync_with_mask,
    block_rank_in_cluster,
    cluster_sync,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
    PipelineState,
)
from linalg.mmaop_sm100 import MmaOpSM100_SS
from linalg.matmul_tile_scheduler_sm100 import TileScheduler, WorkInfo
from linalg.matmul_tile_scheduler import RasterOrder

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from .utils import elementwise_epilogue_type, elementwise_compute_lambda_type
from .utils_gpu import MatmulConfig
from utils.fast_div import FastDiv
from bit import next_power_of_two, prev_power_of_two


@fieldwise_init
@register_passable("trivial")
struct WarpRole(ImplicitlyCopyable, Movable):
    var _role: Int32

    alias Mma = Self(6)
    alias MainLoad = Self(5)
    alias Scheduler = Self(4)
    alias Epilogue = Self(3)

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
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    producer_phase: PipelineState[num_pipeline_stages],
    peer_cta_coord: Tuple[UInt, UInt, UInt],
    work_tile_coord: Tuple[UInt, UInt],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt,
    elect_one_cta: Bool,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    var stage = producer_phase.index()
    var phase = producer_phase.phase()
    mma_mbar[stage].wait(phase)

    var a_gmem_slice_coord = (
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0] * BM
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1] * MMA_N
    )

    var a_smem_tile = a_smem.next(stage)[]
    var b_smem_tile = b_smem.next(stage)[]

    var a_smem_slice = __type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
    )
    var b_smem_slice = __type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
    )

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[stage].expect_bytes(expected_bytes)

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[stage],
            (UInt(iter_idx) * UInt(BK), UInt(a_gmem_slice_coord)),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[stage],
            (UInt(iter_idx) * UInt(BK), UInt(b_gmem_slice_coord)),
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
](
    tmem_addr: UInt32,
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    consumer_phase: PipelineState[pipeline_stages],
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
    iter_idx: UInt,
):
    var stage = consumer_phase.index()
    var phase = consumer_phase.phase()

    tma_mbar[stage].wait(phase)

    var a_smem_tile = a_smem_iter.next(stage)[]
    var b_smem_tile = b_smem_iter.next(stage)[]
    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        mma_op.mma(
            a_smem_tile,
            b_smem_tile,
            tmem_addr,
            init_c=(iter_idx == 0),  # Initialize C on first iteration
        )

        mma_op.commit(mma_mbar + stage)


@always_inline
fn stsm_helper[
    swizzle: Swizzle
](
    vec: SIMD[_, _],
    dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_],
):
    # Number of elements in one row per stsmx4 tile, a row is 32B.
    alias stsmx4_row_size = 32 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    alias stsmx4_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    alias stride0 = dst.layout.stride[0].value()
    alias shape0 = dst.layout.shape[1].value()

    var lane = lane_id()
    var stsm_lane_offset = (lane & 15) * stride0 + (lane >> 4) * 8

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
    for i in range(shape0 // stsmx4_row_size):
        alias n_offset = i * stsmx4_row_size
        var offset = swizzle(stsm_lane_offset + n_offset)
        var v = slice[i * stsmx4_lane_size, stsmx4_lane_size](vec).cast[
            dst.dtype
        ]()
        st_matrix[simd_width=4](dst.ptr + offset, bitcast[DType.float32, 4](v))


@always_inline
fn elementwise_helper[
    N: UInt,
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
    M: UInt,
    c_col: UInt,
    c_row: UInt,
    c_smem_warp_tile_upper: LayoutTensor[
        c_type, c_smem_upper_layout, MutableAnyOrigin, *_, **_
    ],
    c_smem_warp_tile_lower: LayoutTensor[
        c_type, c_smem_lower_layout, MutableAnyOrigin, *_, **_
    ],
):
    # Here we start keeping track of the index / indices this thread is
    # responsbile for in shared memory. This is represented with shared_memory_row
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

    alias distribute_cols = stageN // simd_size
    alias distribute_rows = WARP_SIZE // distribute_cols

    alias distribute_layout = Layout.row_major(distribute_rows, distribute_cols)
    var c_smem_upper_frag = c_smem_warp_tile_upper.vectorize[
        1, simd_size
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    var c_smem_lower_frag = c_smem_warp_tile_lower.vectorize[
        1, simd_size
    ]().distribute[distribute_layout, swizzle=swizzle](lane_id())

    alias fragment_size = c_smem_upper_frag.layout.size()

    var local_col = lane_id() % distribute_cols
    var local_row = lane_id() // distribute_cols

    var shared_memory_col = local_col * simd_size
    shared_memory_row_lower_half += local_row
    shared_memory_row_upper_half += local_row

    @parameter
    for i in range(fragment_size):
        alias alignment = align_of[SIMD[c_type, simd_size]]()

        # these offsets are swizzled so to reteive the corresponding gmem offset we need to remove the swizzle
        # luckily removing the swizzle is as simple as swizzling a second time
        var swz_offset_upper = (
            shared_memory_row_upper_half * shared_n + shared_memory_col
        )
        var swz_offset_lower = (
            shared_memory_row_lower_half * shared_n + shared_memory_col
        )

        var offset_upper = swizzle(swz_offset_upper)
        var offset_lower = swizzle(swz_offset_lower)

        var shared_upper_row: SIMD[DType.int64, 1]
        var shared_upper_col: SIMD[DType.int64, 1]
        var shared_lower_row: SIMD[DType.int64, 1]
        var shared_lower_col: SIMD[DType.int64, 1]

        # Now that we have the true index we, need to add the global tile index to find the correlating
        # index, in gmem. However the data will be stored in tensor memory differently depending on
        # MMA_M size, we take that into account here.

        @parameter
        if MMA_M != 256:
            alias blocked_m_128_layout = blocked_product(
                Layout.row_major(data_paths * 2, stageN),
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
            # cant cast to uint64 as it's not supported yet
            # this will cost us slightly in performance
            alias fast_div = FastDiv[DType.uint32](shared_n)

            shared_upper_row = (
                Scalar[DType.index](offset_upper).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_upper_col = offset_upper % shared_n

            shared_lower_row = (
                Scalar[DType.index](offset_lower).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_lower_col = offset_lower % shared_n

        # now we need to add the global tile offset
        var global_upper_row = shared_upper_row + c_row
        var global_upper_col = shared_upper_col + staged_c_col
        var global_lower_row = shared_lower_row + c_row
        var global_lower_col = shared_lower_col + staged_c_col

        if global_upper_row < M and global_upper_col < N:
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_upper_row), Int(global_upper_col)),
                c_smem_upper_frag[i, 0],
            )
            c_smem_upper_frag[i, 0] = reg_val

        if global_lower_row < M and global_lower_col < N:
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_lower_row), Int(global_lower_col)),
                c_smem_lower_frag[i, 0],
            )
            c_smem_lower_frag[i, 0] = reg_val

        # If more than one chunk is created (happens when 8x4 is used)
        # they will be spaced 8 rows away from each other

        shared_memory_row_upper_half += distribute_rows
        shared_memory_row_lower_half += distribute_rows

    named_barrier[num_output_warps * WARP_SIZE]()


@always_inline
fn multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: UInt,
    /,
    *,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: UInt,
    n: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
    max_tmem_cols: UInt = 512,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    accum_pipeline_consumer_state: PipelineState[num_accum_pipeline_stages],
    accum_full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    accum_empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[UInt, UInt],
    elect_one_warp: Bool,
    m: Int,
):
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    # scheduler fetch next work
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    # assume N dimension is static
    alias simd_size = simd_width_of[c_type]()

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    alias stageN = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    alias num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    alias data_paths = 16  # same as lanes
    alias bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    alias rep = stageN // (bits // 32)  # repetitions per stage

    # stmatrix related
    alias st_matrix_swizzle = TensorMapSwizzle.SWIZZLE_64B if stageN == 32 else TensorMapSwizzle.SWIZZLE_32B
    alias swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # lets keep track of the of the starting row and column in GMEM
    var c_row = work_tile_coord[0] * BM
    var c_col = work_tile_coord[1] * MMA_N

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    accum_full_mbar[index].wait(phase)
    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = index * stage_stride_cols + tmem_addr

    @parameter
    for stage in range(num_stages):
        # column offset, moving right by 32 columns each time, since each num_stage stores two, 16 column submatrices
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_offset + (stage * stageN)
        var upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr)

        var lower_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        @parameter
        if stage == num_stages - 1:
            umma_arrive_leader_cta(accum_empty_mbar + index)

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_iter.next(stage % 2)[]
        var c_smem_warp_tile = c_smem_tile.tile[32, stageN](warp_id, 0)

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[data_paths, stageN](
            0, 0
        )
        var c_smem_warp_tile_lower = c_smem_warp_tile.tile[data_paths, stageN](
            1, 0
        )

        # Pack the upper frag to shared memory
        stsm_helper[swizzle](upper_frag, c_smem_warp_tile_upper)
        stsm_helper[swizzle](lower_frag, c_smem_warp_tile_lower)

        # Guard the write to shared memory is done.
        named_barrier[num_output_warps * WARP_SIZE]()

        @parameter
        if elementwise_compute_lambda_fn:
            elementwise_helper[
                n,
                MMA_M,
                data_paths,
                num_stages,
                stage,
                stageN,
                c_smem_warp_tile_upper.dtype,
                c_smem_tile.shape[1](),
                simd_size,
                c_smem_warp_tile_upper.layout,
                c_smem_warp_tile_lower.layout,
                swizzle,
                elementwise_compute_lambda_fn.value(),
                num_output_warps,
            ](
                m,
                c_col,
                c_row,
                c_smem_warp_tile_upper,
                c_smem_warp_tile_lower,
            )

        var lane = lane_id()

        alias TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 else BM

        var elect_one_warp = warp_id == 0 if MMA_M == 256 else warp_id % 2 == 0
        var coord_n_mma_m256 = work_tile_coord[1] * MMA_N + stage * stageN
        var coord_n_mma_m128 = (
            work_tile_coord[1] * MMA_N + stage * stageN + BN * (warp_id // 2)
        )

        var coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var c_smem_coord_m = 0 if MMA_M == 256 else (warp_id // 2)

        var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](c_smem_coord_m, 0)

        if elect_one_warp and lane == 0:
            fence_async_view_proxy()
            c_tma_op.async_store(
                c_smem_split,
                (
                    UInt(coord_n),
                    UInt(work_tile_coord[0] * BM),
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
            named_barrier[num_output_warps * WARP_SIZE]()


@always_inline
fn store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: UInt,
    /,
    *,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
    max_tmem_cols: UInt = 512,
](
    c_smem_tile: LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    accum_pipeline_consumer_state: PipelineState[num_accum_pipeline_stages],
    accum_full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    accum_empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[UInt, UInt],
    elect_one_warp: Bool,
    stage_stride_cols: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)

    alias TMA_BN = c_layout.shape[1].value()
    var warp_id = get_warp_id()

    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    # Calculate how many elements we need to load based on MMA_N
    alias elements_per_row = BN if MMA_M == 128 else MMA_N
    # this is the main load, most cases use this, power of 2
    alias main_load_elements = prev_power_of_two(elements_per_row)
    # this is remainder load, executed only if MMA_N is not power of 2
    alias num_remainder = elements_per_row - main_load_elements

    # if i do have non-power of 2, then remainder_elements must be divisible by 32 (can extend to support more values later)
    constrained[
        num_remainder % 32 == 0,
        "num_remainder must be divisible by 32",
    ]()

    alias main_repeats = main_load_elements // 8
    alias remainder_repeats = num_remainder // 8

    alias data_paths = 16
    alias bits = 256
    alias num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    alias fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE

    alias NUM_TMA_TILES = MMA_N // TMA_BN
    alias NUM_ST_MATRIX = BN // TMA_BN if MMA_M == 128 else MMA_N // TMA_BN
    alias C_SPLIT_ROWS = BM * NUM_TMA_TILES // 2 if MMA_M == 128 else BM * NUM_TMA_TILES

    # NOTE: Every load is 8 elements (256 bits), repetitions is row size / 8
    # We load 16 lanes by 8 elements so 128 elements total
    # 1 warp or 32 threads does this, each thread storing 128/32=4 elements on every load
    # and total number of register usage is num_regs_per_thread * main_repeats

    # Load c_frag_upper
    # Load once if MMA_N is power of 2, otherwise load twice

    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    accum_full_mbar[index].wait(phase)

    var c_upper_pow_2_main: SIMD[accum_type, main_repeats * fragment_size]

    var c_lower_pow_2_main: SIMD[accum_type, main_repeats * fragment_size]

    # dummy registers in case there's no remainder. We still need to
    # satisfy power-of-2 when using SIMD.
    alias remainder_reg_size = max(2, remainder_repeats * fragment_size)

    var c_upper_pow_2_rem = SIMD[accum_type, remainder_reg_size](0)
    var c_lower_pow_2_rem = SIMD[accum_type, remainder_reg_size](0)

    var tmem_offset = index * stage_stride_cols
    # Primary Load
    c_upper_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repeats,
        dtype=accum_type,
        pack=False,
        width = c_upper_pow_2_main.size,
    ](tmem_addr | tmem_offset | ((warp_id * 32) << 16))

    # Load c_frag_lower
    # Primary load
    c_lower_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repeats,
        dtype=accum_type,
        pack=False,
        width = c_lower_pow_2_main.size,
    ](tmem_addr | tmem_offset | ((warp_id * 32 + 16) << 16))

    @parameter
    if not MMA_N.is_power_of_two():
        # no mma_n can be larger than 256, so if there's a remainder,
        # we've loaded the smallest power of 2, 128, and the rem is after
        # 128. this is why tmem address is offset by 128
        c_upper_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repeats,
            dtype=accum_type,
            pack=False,
            width = c_upper_pow_2_rem.size,
        ](tmem_addr + 128 | tmem_offset | ((warp_id * WARP_SIZE) << 16))

        c_lower_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repeats,
            dtype=accum_type,
            pack=False,
            width = c_lower_pow_2_rem.size,
        ](tmem_addr + 128 | tmem_offset | ((warp_id * WARP_SIZE + 16) << 16))

    # Remainder load happens later, only if needed
    tcgen05_load_wait()

    umma_arrive_leader_cta(accum_empty_mbar + index)

    # Create a layout for everything
    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    alias st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()

    # 128*160 = 20,480 and is same as (128 * 5) * 32 = 20,480
    var c_smem_tile_reshaped = c_smem_tile.reshape[
        Layout.row_major(BM * NUM_TMA_TILES, TMA_BN)
    ]()

    var split_coord_x = warp_id // 2 if MMA_M == 128 else 0
    var c_smem_split = c_smem_tile_reshaped.tile[C_SPLIT_ROWS, TMA_BN](
        split_coord_x, 0
    )

    @parameter
    for tma_n in range(NUM_ST_MATRIX):
        var c_smem_iter = c_smem_split.tile[BM, TMA_BN](tma_n, 0)
        var c_smem_warp_tile = c_smem_iter.tile[32, TMA_BN](
            warp_id % 2 if MMA_M == 128 else warp_id, 0
        )
        var upper = c_smem_warp_tile.tile[16, TMA_BN](0, 0)
        var lower = c_smem_warp_tile.tile[16, TMA_BN](1, 0)

        var d_reg_upper: SIMD[DType.bfloat16, 8]
        var d_reg_lower: SIMD[DType.bfloat16, 8]

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for i in range((TMA_BN // 16)):
                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](lane_id(), i, m_mma, 0)

                # i,0,0                # if MMA_N is a power of 2, then just use the main load for all iterations
                # if it's not a power of 2, then go till NUM_ST_MATRIX -1 using the main regists
                # and for last iteration we load remainder registers (for the remainder 32 )
                @parameter
                if MMA_N.is_power_of_two() or tma_n < NUM_ST_MATRIX - 1:
                    # every iteration of tma_n is a motion across BM * 32 elements
                    # and we agree that each of those has 2 rows * 8 elements in the register
                    d_reg_upper = c_upper_pow_2_main.slice[
                        8, offset = (i * 8) + tma_n * (TMA_BN // 16) * 8
                    ]().cast[DType.bfloat16]()
                    d_reg_lower = c_lower_pow_2_main.slice[
                        8, offset = (i * 8) + tma_n * (TMA_BN // 16) * 8
                    ]().cast[DType.bfloat16]()
                else:
                    d_reg_upper = c_upper_pow_2_rem.slice[
                        8, offset = (i * 8)
                    ]().cast[DType.bfloat16]()
                    d_reg_lower = c_lower_pow_2_rem.slice[
                        8, offset = (i * 8)
                    ]().cast[DType.bfloat16]()

                var d_reg_upper_packed = bitcast[DType.float32, 4](d_reg_upper)
                var d_reg_lower_packed = bitcast[DType.float32, 4](d_reg_lower)

                st_matrix[simd_width=4](
                    upper.ptr.offset(
                        st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    ),
                    d_reg_upper_packed,
                )
                st_matrix[simd_width=4](
                    lower.ptr.offset(
                        st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    ),
                    d_reg_lower_packed,
                )

    named_barrier[num_output_warps * WARP_SIZE]()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    # #           c_frag                   c_smem_tile      c_tma_op
    if elect_one_warp and thread_idx.x < UInt(NUM_TMA_TILES):
        var row_start = work_tile_coord[0] * BM
        var col_start = work_tile_coord[1] * MMA_N + thread_idx.x * TMA_BN

        fence_async_view_proxy()
        var c_smem_offset = c_smem_tile.ptr.offset(BM * TMA_BN * thread_idx.x)

        var c_tma_tile = LayoutTensor[
            c_type,
            c_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ](c_smem_offset)

        c_tma_op.async_store(c_tma_tile, (UInt(col_start), UInt(row_start)))
        c_tma_op.commit_group()
        c_tma_op.wait_group[0]()


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
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: UInt,
    num_clc_pipeline_stages: UInt,
    num_accum_pipeline_stages: UInt,
    n: Int,
    num_output_stages: UInt = 2,
    output_tile_shape: IndexList[2] = Index(128, 32),
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 2,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    block_swizzle_size: Int = 0,
    rasterize_order: RasterOrder = RasterOrder.AlongM,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    cluster_dim: StaticTuple[Int32, 3],
    num_iters: UInt,
    m: Int,
):
    constrained[c_type is not DType.float32, "c_type cannot be float32"]()

    alias num_output_warps = 4

    alias SCHEDULER_THREADS = WARP_SIZE
    alias TMA_LOAD_THREADS = WARP_SIZE
    alias MMA_THREADS = WARP_SIZE
    alias EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    alias CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]
    alias clc_producer_arv_count = 1
    alias clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    alias NUM_TMEM_COLS = 512
    alias stage_stride_cols = NUM_TMEM_COLS // num_accum_pipeline_stages

    alias clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    alias clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    alias accum_pipeline_producer_arv_count = 1
    alias accum_pipeline_consumer_arv_count = cta_group * EPILOGUE_THREADS

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)
    alias num_k_mmas = BK // mma_shape[2]

    alias CLUSTER_M = Int(cluster_shape[0])
    alias CLUSTER_N = Int(cluster_shape[1])

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()
    alias c_smem_layout = Layout.row_major(BM, MMA_N)

    # keep the physical SMEM buffer BM × MMA_N
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]()

    alias a_smem_size = a_smem_layout.size() * num_pipeline_stages
    alias b_smem_size = b_smem_layout.size() * num_pipeline_stages
    alias c_smem_size = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_smem_base,
        a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        b_smem_base,
        b_smem_size,
    )

    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(output_tile_shape[0], output_tile_shape[1]),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem_base, c_smem_size)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    var tma_mbar_ptr = smem_pool
    var mma_mbar_ptr = tma_mbar_ptr + num_pipeline_stages
    var accum_full_mbar_ptr = mma_mbar_ptr + num_pipeline_stages
    var accum_empty_mbar_ptr = accum_full_mbar_ptr + num_accum_pipeline_stages

    var clc_full_mbar_ptr = accum_empty_mbar_ptr + num_accum_pipeline_stages
    var clc_empty_mbar_ptr = clc_full_mbar_ptr + num_clc_pipeline_stages
    var clc_throttle_full_mbar_ptr = (
        clc_empty_mbar_ptr + num_clc_pipeline_stages
    )
    var clc_throttle_empty_mbar_ptr = (
        clc_throttle_full_mbar_ptr + num_clc_pipeline_stages
    )

    var clc_response_ptr = (
        clc_throttle_empty_mbar_ptr + num_clc_pipeline_stages
    ).bitcast[Int128]()

    var tmem_dealloc_mbar_ptr = (
        clc_response_ptr + num_clc_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    accum_full_mbar = accum_full_mbar_ptr.bitcast[SharedMemBarrier]()
    accum_empty_mbar = accum_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    clc_response = clc_response_ptr.bitcast[UInt128]()
    clc_full_mbar = clc_full_mbar_ptr.bitcast[SharedMemBarrier]()
    clc_empty_mbar = clc_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()
    clc_throttle_full_mbar = clc_throttle_full_mbar_ptr.bitcast[
        SharedMemBarrier
    ]()
    clc_throttle_empty_mbar = clc_throttle_empty_mbar_ptr.bitcast[
        SharedMemBarrier
    ]()

    alias accum_type = get_accum_type[a_type]()

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    alias max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        @parameter
        for i in range(num_pipeline_stages):
            tma_mbar[i].init()
            # we need to have 5 arrivals, 2 M, 4 N, top left M/N is shared
            mma_mbar[i].init(
                cluster_shape[0] // cta_group + cluster_shape[1] - 1
            )

        @parameter
        for i in range(num_accum_pipeline_stages):
            accum_full_mbar[i].init(accum_pipeline_producer_arv_count)
            accum_empty_mbar[i].init(accum_pipeline_consumer_arv_count)

        tmem_dealloc_mbar[].init(EPILOGUE_THREADS * cta_group)

    @parameter
    for i in range(num_clc_pipeline_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(clc_consumer_arv_count)
        clc_throttle_full_mbar[i].init(clc_throttle_producer_arv_count)
        clc_throttle_empty_mbar[i].init(clc_throttle_consumer_arv_count)

    fence_mbarrier_init()
    cluster_sync()

    var consumer_phase = PipelineState[num_pipeline_stages]()
    var producer_phase = PipelineState[num_pipeline_stages](0, 1, 0)

    var clc_pipe_producer_state = PipelineState[num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[num_clc_pipeline_stages]()

    var clc_throttle_producer_state = PipelineState[num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_throttle_consumer_state = PipelineState[num_clc_pipeline_stages]()

    var accum_pipeline_producer_state = PipelineState[
        num_accum_pipeline_stages
    ](0, 1, 0)
    var accum_pipeline_consumer_state = PipelineState[
        num_accum_pipeline_stages
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape = Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var scheduler = TileScheduler[
        num_stages=num_clc_pipeline_stages,
        cluster_shape = Index[dtype = DType.uint32](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        block_swizzle_size=block_swizzle_size,
        rasterize_order=rasterize_order,
    ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var work_info = scheduler.initial_work_info()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        UInt(rank_m % cta_group),
        UInt(rank_m // cta_group),
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
    for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= 1 << (i * cta_group)

    a_multicast_mask <<= rank_m
    b_multicast_mask <<= peer_cta_coord[0]
    b_multicast_mask <<= rank_n * CLUSTER_M

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    if WarpRole.is_main_load():
        var required_clc_query = True

        while work_info.is_valid():
            # CLC throuttle prevents each CTA from going a few waves ahead.
            if is_first_cta_in_cluster and required_clc_query:
                var index = clc_throttle_producer_state.index()
                var phase = clc_throttle_producer_state.phase()
                clc_throttle_empty_mbar[index].wait(phase)
                _ = clc_throttle_full_mbar[index].arrive()

                clc_throttle_producer_state.step()

            # DO TMA LOAD
            for i in range(num_iters):
                load_AB[
                    block_tile_shape=block_tile_shape,
                    mma_shape=mma_shape,
                    cta_group=cta_group,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_smem,
                    b_smem,
                    mma_mbar,
                    tma_mbar,
                    producer_phase,
                    peer_cta_coord,
                    (UInt(work_info.m), UInt(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                )
                producer_phase.step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        @parameter
        for i in range(num_pipeline_stages):
            mma_mbar[producer_phase.index()].wait(producer_phase.phase())
            producer_phase.step()

    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        var required_clc_query = True

        while work_info.is_valid():
            if required_clc_query:
                var index = clc_throttle_consumer_state.index()
                var phase = clc_throttle_consumer_state.phase()
                clc_throttle_full_mbar[index].wait(phase)
                _ = clc_throttle_empty_mbar[index].arrive()

                clc_throttle_consumer_state.step()

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
        for i in range(num_clc_pipeline_stages):
            clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                clc_pipe_producer_state.phase()
            )
            clc_pipe_producer_state.step()

    if WarpRole.is_mma():
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)
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
                var accum_index = accum_pipeline_producer_state.index()
                var accum_phase = accum_pipeline_producer_state.phase()

                accum_empty_mbar[accum_index].wait(accum_phase)
                var tmem_offset = tmem_addr + (accum_index * stage_stride_cols)

                for i in range(num_iters):
                    consumer_main_loop[
                        block_tile_shape=block_tile_shape,
                        mma_shape=mma_shape,
                        cta_group=cta_group,
                        cluster_shape = Index(
                            cluster_shape[0], cluster_shape[1], cluster_shape[2]
                        ),
                    ](
                        tmem_offset,
                        a_smem,
                        b_smem,
                        mma_mbar,
                        tma_mbar,
                        consumer_phase,
                        mma_op,
                        elect_one_warp,
                        i,
                    )
                    consumer_phase.step()

                # mma arrive multicast will track completion of all mma prior to this barrier.
                if elect_one_sync():
                    mma_arrive_multicast[cta_group](
                        accum_full_mbar + accum_index,
                        mma_complete_mask,
                    )
                accum_pipeline_producer_state.step()
            work_info = next_work_info

        tcgen05_release_allocation_lock[cta_group]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[MMA_THREADS + EPILOGUE_THREADS](1)
        tmem_addr = ptr_tmem_addr[0]

        while work_info.is_valid():
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            # scheduler fetch next work
            multi_stage_store_C[
                accum_type=accum_type,
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                stage_stride_cols = UInt(stage_stride_cols),
                n=n,
                c_swizzle=c_swizzle,
                cta_group=cta_group,
                num_output_warps=num_output_warps,
                max_tmem_cols=max_tmem_cols,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](
                c_smem_iter,
                c_tma_op,
                accum_pipeline_consumer_state,
                accum_full_mbar,
                accum_empty_mbar,
                tmem_addr,
                work_tile_coord=(UInt(work_info.m), UInt(work_info.n)),
                elect_one_warp=elect_one_warp,
                m=m,
            )
            accum_pipeline_consumer_state.step()

            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


fn blackwell_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    cta_group: Int = 1,
    num_clc_pipeline_stages: UInt = 2,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    block_swizzle_size: Int = 0,
    rasterize_order: RasterOrder = RasterOrder.AlongM,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    alias BM = MMA_M // cta_group
    alias BN = MMA_N // cta_group
    alias BK = config.block_tile_shape[2]

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B

    alias cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var K = a_device.dim[1]()

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a)

    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(
            BN // (cluster_shape[0] // cta_group), BK
        ) if transpose_b else Index(BK, BN // (cluster_shape[0] // cta_group)),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    # If MMA_M is 256, the warps read the entire MMA_N.
    # That MMA_N to be multiple of 32 for me to use large N dim on C buf write out
    # If MMA_M is 128, the warps read 1/2 of MMA_N (BN), so now *that* has to be multiple of 32
    # Otherwise, we just use 16
    alias width = 32 if (MMA_M == 256 and MMA_N % 32 == 0) or (
        MMA_M == 128 and BN % 32 == 0
    ) else 16
    alias output_tile_shape = Index(128, width)
    alias split_tile_shape = Index(64, width)
    alias c_tma_tile_shape = output_tile_shape if MMA_M == 256 else split_tile_shape
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_64B if width == 32 else TensorMapSwizzle.SWIZZLE_32B
    var c_tma_op = create_tma_tile[
        c_type,
        2,
        c_tma_tile_shape,
        swizzle_mode=c_swizzle,
    ](ctx, c)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
    alias a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    alias b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    # A and B per pipeline stage
    alias AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
    # Support double-buffer for output stages.
    alias num_output_stages = 2

    alias c_smem_bytes = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * size_of[c_type]()

    alias MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    alias CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    alias TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address
    # the 'N' dimension of tensor memory is 512
    alias TMEM_N = 512
    # the maximum different number of mma's that can be run in parallel is TMEM_N/MMA_N
    alias max_accum_pipeline_stages = TMEM_N // next_power_of_two(MMA_N)
    # Mainloop barrier
    alias accum_full_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages
    alias accum_empty_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages

    alias clc_response_bytes = CLC_RESPONSE_BYTES * num_clc_pipeline_stages
    alias clc_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_throttle_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    alias clc_throttle_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages

    alias tmem_addr_bytes = TMEM_ADDR_BYTES
    alias tmem_dealloc_mbar_bytes = MBAR_BYTES

    alias tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    alias accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    alias clc_smem = (
        clc_response_bytes
        + clc_full_mbar_bytes
        + clc_empty_mbar_bytes
        + clc_throttle_full_mbar_bytes
        + clc_throttle_empty_mbar_bytes
    )
    alias smem_leftover = (b200_smem) - (
        clc_smem + accum_smem + tmem_writeout_smem
    )

    alias tma_mbar_bytes_per_stage = MBAR_BYTES
    alias mma_mbar_bytes_per_stage = MBAR_BYTES

    alias producer_consumer_smem_per_stage = (
        AB_smem_per_stage + tma_mbar_bytes_per_stage + mma_mbar_bytes_per_stage
    )

    alias max_pipeline_stages = smem_leftover // producer_consumer_smem_per_stage

    alias producer_consumer_smem = producer_consumer_smem_per_stage * max_pipeline_stages

    alias smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    alias kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=transpose_b,
        cluster_shape = StaticTuple[Int32, 3](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=cta_group,
        num_pipeline_stages = UInt(max_pipeline_stages),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        num_accum_pipeline_stages = UInt(max_accum_pipeline_stages),
        n = c.shape[1](),
        num_output_stages = UInt(num_output_stages),
        output_tile_shape=output_tile_shape,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        block_swizzle_size=block_swizzle_size,
        rasterize_order=rasterize_order,
    ]

    var grid_dim = (
        align_up(ceildiv(M, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N, MMA_N), Int(cluster_shape[1])),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        ceildiv(grid_dim[0], cluster_shape[0]),
        ceildiv(grid_dim[1], cluster_shape[1]),
        1,
    )

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        cluster_dim,
        K // BK,
        c.dim[0](),
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )


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
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]  # BM
    alias MMA_N = mma_shape[1]  # BN
    alias MMA_K = mma_shape[2]  # 16
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
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
    alias a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()

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

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

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

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    ctile, ctile_coords, _ = c.tile_with_offset[BM, BN](
        block_idx.y, block_idx.x
    )
    alias c_coord_type = __type_of(ctile_coords)

    var M = c.dim[0]()
    alias N = c.layout.shape[1].value()

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            alias mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                ctile.tile_with_offset[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
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

            alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
            alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

            @parameter
            for n_vec in range(num_vecs_n):

                @parameter
                for m_vec in range(num_vecs_m):
                    alias i_vec = n_vec * num_vecs_m + m_vec
                    alias dst_idx = __type_of(c_gmem_frag).layout(
                        IntTuple(m_vec, n_vec)
                    )
                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                    var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                    if m < M and n < N:
                        var c_mn = SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()

                        @parameter
                        if elementwise_lambda_fn:
                            alias alignment = align_of[SIMD[c_type, 2]]()
                            alias epilogue = elementwise_lambda_fn.value()
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

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, BK), swizzle_mode=a_swizzle
    ](ctx, a)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    alias smem_use = (BM * size_of[a_type]() + BN * size_of[b_type]()) * BK + 24

    alias block_dim = 128

    alias kernel = matmul_sm100_fallback_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
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

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        ceildiv(K, BK),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
