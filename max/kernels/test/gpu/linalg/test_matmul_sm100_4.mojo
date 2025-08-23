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

from math import align_up
from sys import argv, sizeof
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from layout.layout_tensor import LayoutTensorIter
from gpu import WARP_SIZE, barrier
from gpu.sync import named_barrier

from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from gpu.id import block_idx, lane_id, thread_idx, block_id_in_cluster
from gpu.id import warp_id as get_warp_id
from gpu.memory import AddressSpace, fence_async_view_proxy
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from internal_utils import ndbuffer_to_str
from bit import next_power_of_two, prev_power_of_two

from gpu.mma import st_matrix
from layout import (
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    LayoutTensor,
    IntTuple,
    UNKNOWN_VALUE,
)
from layout.swizzle import make_swizzle, make_ldmatrix_swizzle, Swizzle

from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    st_matrix_n_layout,
    tile_to_descriptor,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
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

from linalg import vendor_blas
from linalg.mmaop_sm100 import MmaOpSM100_SS
from math import ceildiv

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from internal_utils import ndbuffer_to_str


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


@fieldwise_init
@register_passable("trivial")
struct WarpRole(Copyable, Movable):
    var _role: Int32

    alias MainLoad = Self(4)
    alias Mma = Self(5)
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
        circular=False,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=16
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=16
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

    alias a_expected_bytes = a_smem_layout.size() * sizeof[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * sizeof[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    var stage = producer_phase.index()
    var phase = producer_phase.phase()
    mma_mbar[stage].wait(phase)

    if elect_one_cta:
        tma_mbar[stage].expect_bytes(expected_bytes)

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

    a_tma_op.async_multicast_load[cta_group](
        a_smem_slice,
        tma_mbar[stage],
        (UInt(iter_idx) * BK, a_gmem_slice_coord),
        a_multicast_mask,
    )

    b_tma_op.async_multicast_load[cta_group](
        b_smem_slice,
        tma_mbar[stage],
        (UInt(iter_idx) * BK, b_gmem_slice_coord),
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
        circular=False,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=16
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED, alignment=16
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

    var a_smem_tile = a_smem_iter.next_unsafe(stage)[]
    var b_smem_tile = b_smem_iter.next_unsafe(stage)[]

    if elect_one_sync():
        mma_op.mma(
            a_smem_tile,
            b_smem_tile,
            tmem_addr,
            init_c=(iter_idx == 0),  # Initialize C on first iteration
        )

        mma_op.commit(mma_mbar + stage)


@always_inline
fn store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
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
    tmem_addr: UInt32,
    elect_one_warp: Bool,
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
    alias remainder_elements = elements_per_row - main_load_elements

    # if i do have non-power of 2, then remainder_elements must be divisible by 32 (can extend to support more values later)
    constrained[
        remainder_elements % 32 == 0,
        "remainder_elements must be divisible by 32",
    ]()

    alias main_repetition = main_load_elements // 8
    alias remainder_repetitions = remainder_elements // 8

    alias data_paths = 16
    alias bits = 256
    alias num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    alias num_regs_per_thread = (
        data_paths * num_elements_per_load
    ) // WARP_SIZE

    alias NUM_TMA_TILES = MMA_N // TMA_BN
    alias NUM_ST_MATRIX = BN // TMA_BN if MMA_M == 128 else MMA_N // TMA_BN
    alias C_SPLIT_ROWS = BM * NUM_TMA_TILES // 2 if MMA_M == 128 else BM * NUM_TMA_TILES

    # NOTE: Every load is 8 elements (256 bits), repetitions is row size / 8
    # We load 16 lanes by 8 elements so 128 elements total
    # 1 warp or 32 threads does this, each thread storing 128/32=4 elements on every load
    # and total number of register usage is num_regs_per_thread * main_repetition

    # Load c_frag_upper
    # Load once if MMA_N is power of 2, otherwise load twice

    var c_upper_pow_2_main = SIMD[
        accum_type, main_repetition * num_regs_per_thread
    ](0)

    var c_lower_pow_2_main = SIMD[
        accum_type, main_repetition * num_regs_per_thread
    ](0)

    # dummy registers in case there's no remainder. We still need to
    # satisfy power-of-2 when using SIMD.
    alias remainder_reg_size = max(
        2, remainder_repetitions * num_regs_per_thread
    )

    var c_upper_pow_2_rem = SIMD[accum_type, remainder_reg_size](0)
    var c_lower_pow_2_rem = SIMD[accum_type, remainder_reg_size](0)

    # Primary Load
    c_upper_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repetition,
        dtype=accum_type,
        pack=False,
        width = c_upper_pow_2_main.size,
    ](tmem_addr | ((warp_id * 32) << 16))

    # Load c_frag_lower
    # Primary load
    c_lower_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repetition,
        dtype=accum_type,
        pack=False,
        width = c_lower_pow_2_main.size,
    ](tmem_addr | ((warp_id * 32 + 16) << 16))

    @parameter
    if MMA_N != prev_power_of_two(MMA_N):
        # no mma_n can be larger than 256, so if there's a remainder,
        # we've loaded the smallest power of 2, 128, and the rem is after
        # 128. this is why tmem address is offset by 128
        c_upper_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repetitions,
            dtype=accum_type,
            pack=False,
            width = c_upper_pow_2_rem.size,
        ](tmem_addr + 128 | ((warp_id * WARP_SIZE) << 16))

        c_lower_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repetitions,
            dtype=accum_type,
            pack=False,
            width = c_lower_pow_2_rem.size,
        ](tmem_addr + 128 | ((warp_id * WARP_SIZE + 16) << 16))

    # Remainder load happens later, only if needed
    tcgen05_load_wait()

    # Create a layout for everything
    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    # For 32-column tiles, we need a different swizzle pattern
    alias st_matrix_swizzle = make_swizzle[
        c_type,
        TensorMapSwizzle.SWIZZLE_64B if TMA_BN
        == 32 else TensorMapSwizzle.SWIZZLE_128B,
    ]()

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
                # i,0,0

                var d_reg_upper = SIMD[DType.bfloat16, 8](0)
                var d_reg_lower = SIMD[DType.bfloat16, 8](0)

                # if MMA_N is a power of 2, then just use the main load for all iterations
                # if it's not a power of 2, then go till NUM_ST_MATRIX -1 using the main regists
                # and for last iteration we load remainder registers (for the remainder 32 )
                @parameter
                if (
                    MMA_N == prev_power_of_two(MMA_N)
                    or tma_n < NUM_ST_MATRIX - 1
                ):
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

    if elect_one_warp and thread_idx.x < NUM_TMA_TILES:
        var row_start = block_idx.x * BM
        var col_start = block_idx.y * MMA_N + thread_idx.x * TMA_BN

        fence_async_view_proxy()
        var c_smem_offset = c_smem_tile.ptr.offset(BM * TMA_BN * thread_idx.x)

        var c_tma_tile = LayoutTensor[
            c_type,
            c_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ](c_smem_offset)

        c_tma_op.async_store(c_tma_tile, (col_start, row_start))
        c_tma_op.commit_group()
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[cta_group]()
        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)


@always_inline
fn stsm_helper[
    swizzle: Swizzle
](
    vec: SIMD,
    dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_],
):
    # Number of elements in one row per stsmx4 tile, a row is 32B.
    alias stsmx4_row_size = 32 // sizeof[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    alias stsmx4_lane_size = 16 // sizeof[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    alias stride0 = dst.layout.stride[0].value()
    alias shape0 = dst.layout.shape[1].value()

    var lane = lane_id()
    var stsm_lane_offset = (lane & 15) * stride0 + (lane >> 4) * 8

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    @parameter
    for i in range(shape0 // stsmx4_row_size):
        alias n_offset = i * stsmx4_row_size
        var offset = swizzle(stsm_lane_offset + n_offset)
        var v = vec.slice[
            stsmx4_lane_size, offset = i * stsmx4_lane_size
        ]().cast[dst.dtype]()
        st_matrix[simd_width=4](dst.ptr + offset, bitcast[DType.float32, 4](v))


@always_inline
fn store_C_v1[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    c_layout_complete: Layout,
    c_desc_layout_complete: Layout,
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
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op_split: TMATensorTile[c_type, c_layout, c_desc_layout],
    c_tma_op_complete: TMATensorTile[
        c_type, c_layout_complete, c_desc_layout_complete
    ],
    tmem_addr: UInt32,
    elect_one_warp: Bool,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    # We break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # For MMA_M=128, we use 64x16 tiles, otherwise 64x32
    alias stageM = c_smem_layout.shape[0].value()  # 128
    alias stageN = c_smem_layout.shape[1].value()  # 32
    # For MMA_M=128, we have 8 logical stages (128/16) but process left and right separately
    alias num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    alias tmem_cell_bytes = 4
    alias data_paths = 16
    alias bits = 256
    alias rep = stageN // (bits // 32)

    # stmatrix related
    alias stsmx4N_bytes = 32
    alias stsmx4N = stsmx4N_bytes // sizeof[c_type]()
    alias stsmx4_size_per_lane = (16 * stsmx4N) // WARP_SIZE
    # if the tile is not 32, it is assumed to be 16
    alias st_matrix_swizzle = TensorMapSwizzle.SWIZZLE_64B if stageN == 32 else TensorMapSwizzle.SWIZZLE_32B
    alias swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    @parameter
    for stage in range(num_stages):
        stage_tmem_addr = tmem_addr + stage * stageN

        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_addr + stage * stageN

        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr | ((warp_id * 32) << 16))
        var lower_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr | (((warp_id * 32) + 16) << 16))

        tcgen05_load_wait()

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_iter.next(stage % 2)[]
        var c_smem_warp_tile = c_smem_tile.tile[32, stageN](warp_id, 0)

        # Pack the upper frag to shared memory
        stsm_helper[swizzle](
            upper_frag, c_smem_warp_tile.tile[16, stageN](0, 0)
        )
        stsm_helper[swizzle](
            lower_frag, c_smem_warp_tile.tile[16, stageN](1, 0)
        )

        # Guard the write to shared memory is done.
        named_barrier[num_output_warps * WARP_SIZE]()

        var lane = lane_id()

        @parameter
        if MMA_M == 256:
            if warp_id == 0 and lane == 0:
                fence_async_view_proxy()
                c_tma_op_complete.async_store(
                    c_smem_tile,
                    (block_idx.y * MMA_N + stage * stageN, block_idx.x * BM),
                )
                c_tma_op_complete.commit_group()
        else:
            var c_smem_tile_left_right = c_smem_tile.tile[BM, stageN](
                (warp_id // 2), 0
            )
            if warp_id % 2 == 0 and lane == 0:
                fence_async_view_proxy()
                c_tma_op_split.async_store(
                    c_smem_tile_left_right,
                    (
                        block_idx.y * MMA_N
                        + stage * stageN
                        + BN * (warp_id // 2),
                        block_idx.x * BM,
                    ),
                )
                c_tma_op_split.commit_group()

        @parameter
        # Keep one tma store in fly
        if stage < num_stages - 1:
            c_tma_op_split.wait_group[1]()
        # Last stage guard all tma store to finish
        else:
            c_tma_op_split.wait_group[0]()

        @parameter
        if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            # E.g. stage = 1, this guards the TMA store using buffer 0 is done.
            named_barrier[num_output_warps * WARP_SIZE]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[cta_group]()
        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op_split, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op_complete, `nvvm.grid_constant`)
fn blackwell_tma_pair_umma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,  # must pass mma_m by mma_n as this layout, since that's how much each output has to be
    c_layout_complete: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    c_desc_layout_complete: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: UInt,
    num_output_stages: UInt = 2,
    output_tile_shape: IndexList[2] = Index(128, 32),
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 2,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op_split: TMATensorTile[c_type, c_layout, c_desc_layout],
    c_tma_op_complete: TMATensorTile[
        c_type, c_layout_complete, c_desc_layout_complete
    ],
    num_iters: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)
    alias num_k_mmas = BK // mma_shape[2]
    alias num_output_warps = 4

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

    base_ptr_smem = rebind[
        UnsafePointer[
            Scalar[a_type], address_space = AddressSpace.SHARED, alignment=128
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]()
    )  # pointer to first byte of scratchpad

    alias a_smem_size = a_smem_layout.size()
    alias b_smem_size = b_smem_layout.size()
    alias c_smem_size = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages

    var a_smem_base = base_ptr_smem  # need space for 4096 (64 x 64) elements by 2 bytes or 8192 total, which is 0x2000
    var b_smem_base = (a_smem_base + a_smem_size * num_pipeline_stages).bitcast[
        Scalar[b_type]
    ]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ](
        a_smem_base.static_alignment_cast[128](),
        a_smem_size * num_pipeline_stages,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=False,
    ](
        b_smem_base.static_alignment_cast[128](),
        b_smem_size * num_pipeline_stages,
    )

    var c_smem_base = (
        (b_smem_base + b_smem_size * num_pipeline_stages)
        .bitcast[Scalar[c_type]]()
        .static_alignment_cast[128]()
    )
    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(output_tile_shape[0], output_tile_shape[1]),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem_base, c_smem_size)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    alias accum_type = get_accum_type[a_type]()

    # this gets 8 bytes of space
    # var ptr_tmem_addr = smem_pool.bitcast[UInt32]()
    # adding 8 bytes for ptr_tmem_addr (smem poll is 8 byte casted)
    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    # + num_pipeline_stages is 1 * num_pipeline_stage so 8 bytes for each barrier at each stage
    var mma_mbar_ptr = tma_mbar_ptr + (num_pipeline_stages)
    var math_barrier_base = mma_mbar_ptr + (num_pipeline_stages)
    var ptr_tmem_addr = (math_barrier_base + 1).bitcast[UInt32]()

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    math_barrier = math_barrier_base.bitcast[SharedMemBarrier]()

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var warp_id = get_warp_id()
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    if elect_one_warp and elect_one_thread:

        @parameter
        for i in range(num_pipeline_stages):
            tma_mbar[i].init()
            # we need to have 5 arrivals, 2 M, 4 N, top left M/N is shared
            mma_mbar[i].init(
                cluster_shape[0] // cta_group + cluster_shape[1] - 1
            )
        math_barrier[].init()

    cluster_sync()

    var consumer_phase = PipelineState[num_pipeline_stages]()
    var producer_phase = PipelineState[num_pipeline_stages](0, 1, 0)

    tmem_addr = ptr_tmem_addr[0]

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

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        rank_m % cta_group,
        rank_m // cta_group,
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

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
        if elect_one_sync():
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
                    (block_idx.x, block_idx.y),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                )
                producer_phase.step()

    if elect_one_cta and WarpRole.is_mma():
        for i in range(num_iters):
            consumer_main_loop[
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                cta_group=cta_group,
                cluster_shape = Index(
                    cluster_shape[0], cluster_shape[1], cluster_shape[2]
                ),
            ](
                tmem_addr,
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
            mma_arrive_multicast[cta_group](math_barrier, mma_complete_mask)

    if WarpRole.is_epilogue():
        math_barrier[].wait()

        store_C_v1[
            accum_type=accum_type,
            block_tile_shape=block_tile_shape,
            mma_shape=mma_shape,
            c_swizzle=c_swizzle,
            cta_group=cta_group,
            num_output_warps=num_output_warps,
            max_tmem_cols=max_tmem_cols,
        ](
            c_smem_iter,
            c_tma_op_split,
            c_tma_op_complete,
            tmem_addr,
            elect_one_warp,
        )


fn blackwell_matmul_tma_pair_mma[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias MMA_M = umma_shape[0]
    alias MMA_N = umma_shape[1]
    alias MMA_K = umma_shape[2]

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

    # Create two TMA descriptors for left and right halves when MMA_M=128
    # Each descriptor handles 64x16 tiles
    # If MMA_M is 256, the warps read the entire MMA_N.
    # That MMA_N to be multiple of 32 for me to use large N dim on C buf write out
    # If MMA_M is 128, the warps read 1/2 of MMA_N (BN), so now *that* has to be multiple of 32
    # Otherwise, we just use 16
    alias width = 32 if (MMA_M == 256 and MMA_N % 32 == 0) or (
        MMA_M == 128 and BN % 32 == 0
    ) else 16
    alias output_tile_shape = Index(128, width)
    alias split_tile_shape = Index(64, width)
    var c_tma_op_split = create_tma_tile[
        c_type,
        2,
        split_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_64B if width
        == 32 else TensorMapSwizzle.SWIZZLE_32B,
    ](ctx, c)

    # For the right half, we need a separate descriptor
    var c_tma_op_complete = create_tma_tile[
        c_type,
        2,
        output_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_64B if width
        == 32 else TensorMapSwizzle.SWIZZLE_32B,
    ](ctx, c)

    # Configure shared memory usage
    # Total size = capacity - 1KB_reserved_by_L1
    alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
    # A and B per pipeline stage
    alias AB_smem_per_stage = BM * BK * sizeof[a_type]() + BN * BK * sizeof[
        b_type
    ]()
    # Support double-buffer for output stages.
    alias num_output_stages = 2
    alias C_smem = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * sizeof[c_type]()
    # Usage reserved for mbar and others
    # - tma_mbar_ptr: 8 bytes per pipeline stage
    # - mma_mbar_ptr: 8 bytes per pipeline stage
    # - math_barrier: 8 bytes
    # - tmem addr in shared memory: 4B
    alias mbar_per_stage = 16
    alias other_usage = 8 + 4
    alias per_stage_usage = AB_smem_per_stage + mbar_per_stage
    # Compute the max number of pipeline stages supported
    alias num_pipeline_stages = (
        b200_smem - C_smem - other_usage
    ) // per_stage_usage
    # Total smem usage
    alias smem_size = per_stage_usage * num_pipeline_stages + C_smem + other_usage

    alias kernel = blackwell_tma_pair_umma_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op_split.layout,
        c_tma_op_complete.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op_split.desc_layout,
        c_tma_op_complete.desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        num_output_stages=num_output_stages,
        output_tile_shape=output_tile_shape,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op_split,
        c_tma_op_complete,
        K // BK,
        grid_dim=(
            align_up(ceildiv(M, BM), Int(cluster_shape[0])),
            align_up(ceildiv(N, MMA_N), Int(cluster_shape[1])),
            1,
        ),
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * 6),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )


def test_blackwell_matmul_tma_pair_mma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    if not benchmark:
        print(
            String(
                M,
                "x",
                N,
                "x",
                K,
                " mma_shape=",
                mma_shape,
                " block_tile_shape=",
                block_tile_shape,
            )
        )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Perf varies with initial values. Simple values have lower noise for
    # the current benchmark comparing to random initial values.
    if simple_init():
        var at = a_host.tensor
        var bt = b_host.tensor
        for m in range(M):
            for k in range(K):
                at[m, k] = k
        for n in range(N):
            for k in range(K):
                bt[n, k] = 1 if n == k else 0
    else:
        random(a_host.tensor)
        random(b_host.tensor)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    blackwell_matmul_tma_pair_mma[
        transpose_b=transpose_b,
        umma_shape=mma_shape,
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=2,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    if benchmark:
        alias num_runs = 100
        alias num_warmup = 100

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            # vendor_blas.matmul(
            #     ctx,
            #     c_device_ref.tensor,
            #     a_device.tensor,
            #     b_device.tensor,
            #     c_row_major=True,
            #     transpose_b=transpose_b,
            # )
            blackwell_matmul_tma_pair_mma[
                transpose_b=transpose_b,
                umma_shape=mma_shape,
                block_tile_shape=block_tile_shape,
                cluster_shape=cluster_shape,
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                cta_group=2,
            ](
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                M,
                N,
                K,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()

        var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        # Round TFLOPS to two decimal places for cleaner output
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 2)
        print(String(M, "x", N, "x", K), tflops_rounded)
    else:
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()

        alias rtol = 1e-2
        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=0.0001,
            rtol=rtol,
        )
        print("\n=== TEST PASSED ===\n")

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


fn get_shapes_dict(
    index: Int, shapes_dict: Dict[Int, Tuple[Int, Int, Int], *_, **_]
) -> Tuple[Int, Int, Int]:
    try:
        return shapes_dict[index]
    except error:
        print("error")
        return (128, 128, 128)


fn make_shapes_dict() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic: Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher] = {
        0: (4096, 4096, 4096),
        1: (512, 2560, 8192),
        2: (512, 8192, 2048),
        3: (512, 14336, 8192),
        4: (512, 8192, 7168),
        5: (4096, 2560, 8192),
        6: (4096, 8192, 2048),
        7: (4096, 14336, 8192),
        8: (4096, 8192, 7168),
        9: (8192, 2560, 8192),
        10: (8192, 8192, 2048),
        11: (8192, 14336, 8192),
        12: (8192, 8192, 7168),
    }
    return dic


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    alias a_type = DType.bfloat16
    alias b_type = DType.bfloat16
    alias c_type = DType.bfloat16
    alias block_tile_shape = Index(128, 128, 64)
    alias umma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, 16
    )
    alias shapes_dict = make_shapes_dict()

    print("Benchmarking blackwell_matmul_tma_umma_kernel")
    print("============================================")
    print("M, N, K, time(ms), TFLOPS")

    @parameter
    for i in range(len(shapes_dict)):
        alias shape = get_shapes_dict(i, shapes_dict)
        try:
            test_blackwell_matmul_tma_pair_mma[
                a_type,
                b_type,
                c_type,
                block_tile_shape,
                umma_shape,
                cluster_shape = StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                benchmark=True,
            ](ctx, dynamic(shape[0]), static[shape[1]](), static[shape[2]]())
        except error:
            print("error")


def main():
    with DeviceContext() as ctx:
        # Run the benchmark
        if is_benchmark():
            benchmark_blackwell_matmul(ctx)
            return

        @parameter
        for mma_m_scale in range(1, 3):

            @parameter
            for mma_n_scale in range(1, 5):
                alias block_tile_shape = Index(
                    64 * mma_m_scale, 32 * mma_n_scale, 64
                )
                alias umma_shape = Index(
                    128 * mma_m_scale, 64 * mma_n_scale, 16
                )

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, dynamic(1000), static[1024](), static[1024]())

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, dynamic(512), static[4096](), static[1024]())

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, dynamic(500), static[2048](), static[4096]())

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](8, 2, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, dynamic(1024), static[256](), static[128]())

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](2, 2, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, static[1024](), static[1024](), static[2048]())

                test_blackwell_matmul_tma_pair_mma[
                    DType.bfloat16,
                    DType.bfloat16,
                    DType.bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape = StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
                ](ctx, dynamic(8192), static[2560](), static[8192]())
