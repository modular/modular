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
from math import align_up, ceildiv, gcd
from sys import size_of

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
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from gpu.id import block_id_in_cluster, block_idx, lane_id, thread_idx
from gpu.id import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.mma import st_matrix
from gpu.mma_sm100 import *
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
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
from layout.layout_tensor import LayoutTensorIter
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

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from .matmul import WarpRole, consumer_main_loop, stsm_helper
from .tile_scheduler import TileScheduler, WorkInfo


@always_inline
fn _get_accumulator_size[
    *,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
]() -> IndexList[2]:
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    alias stageN = c_smem_layout.shape[1].value()
    alias num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    alias data_paths = 16
    alias bits = 256
    alias repeats = stageN // (bits // 32)

    alias num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    alias fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    alias num_elements = repeats * fragment_size

    return Index(num_stages, num_elements)


@always_inline
fn load_AB[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_scales_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_scales_smem_layout: Layout,
    num_pipeline_stages: UInt,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_layout, a_scales_desc_layout
    ],
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
    a_scales_smem: LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
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
    alias a_scales_expected_bytes = a_scales_smem_layout.size() * size_of[
        a_scales_type
    ]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes + a_scales_expected_bytes
    )

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_scales_tma_load_size = a_scales_desc_layout.size()
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
    var a_scales_smem_tile = a_scales_smem.next(stage)[]

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
            (UInt(UInt(iter_idx) * BK), UInt(a_gmem_slice_coord)),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[stage],
            (UInt(UInt(iter_idx) * BK), UInt(b_gmem_slice_coord)),
            b_multicast_mask,
        )

        a_scales_tma_op.async_copy[cta_group](
            a_scales_smem_tile,
            tma_mbar[stage],
            (UInt(work_tile_coord[0] * BM), UInt(iter_idx)),
        )


@always_inline
fn multi_stage_reg_epilogue[
    c_smem_layout: Layout,
    c_layout: Layout,
    c_desc_layout: Layout,
    accum_type: DType,
    accum_layout: Layout,
    /,
    *,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
](
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    work_tile_coord: Tuple[UInt, UInt],
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

    alias num_stages = accum_layout.shape[0].value()
    alias num_elements = accum_layout.shape[1].value()

    alias data_paths = 16
    alias bits = 256
    alias num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    alias fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    alias repeats = num_elements // fragment_size
    alias stageN = repeats * (bits // 32)
    alias fragments_per_stage = fragment_size * repeats

    # stmatrix related
    alias stsmx4N_bytes = 32
    alias stsmx4N = stsmx4N_bytes // size_of[c_type]()  # 16
    alias stsmx4_size_per_lane = (16 * stsmx4N) // WARP_SIZE  # 8
    alias st_matrix_swizzle = TensorMapSwizzle.SWIZZLE_64B if stageN == 32 else TensorMapSwizzle.SWIZZLE_32B
    alias swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    @parameter
    for stage in range(num_stages):
        var upper_frag = c_upper_main_tile.load[fragments_per_stage](stage, 0)
        var lower_frag = c_lower_main_tile.load[fragments_per_stage](stage, 0)

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
        if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            # E.g. stage = 1, this guards the TMA store using buffer 0 is done.
            named_barrier[num_output_warps * WARP_SIZE]()


@always_inline
fn promote_accumulators[
    pipeline_stages: UInt,
    num_accum_pipeline_stages: UInt,
    accum_type: DType,
    accum_layout: Layout,
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: Layout,
    a_scales_smem_layout: Layout,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    CLUSTER_SIZE: Int32,
    num_output_warps: UInt = 4,
](
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutableAnyOrigin],
    a_scales_smem_iter: LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_upper_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    c_lower_main_tile: LayoutTensor[
        accum_type,
        accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        *_, **_,
    ],
    accum_pipeline_consumer_state: PipelineState[num_accum_pipeline_stages],
    accum_full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    accum_empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tmem_addr: UInt32,
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    consumer_phase: PipelineState[pipeline_stages],
    work_tile_coord: Tuple[UInt, UInt],
    elect_one_warp: Bool,
    stage_stride_cols: UInt,
    k_iter: UInt,
    problem_shape: StaticTuple[Int32, 3],
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

    constrained[
        a_scales_type == b_scales_type and accum_type == DType.float32,
        "Only support float32 for a_scales, b_scales, and accum_type",
    ]()
    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    var M = problem_shape[0]
    var N = problem_shape[1]
    var K = problem_shape[2]

    alias num_stages = accum_layout.shape[0].value()
    alias num_elements = accum_layout.shape[1].value()

    alias data_paths = 16
    alias bits = 256
    alias num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    alias fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    alias repeats = num_elements // fragment_size
    alias stageN = repeats * (bits // 32)

    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    var tmem_offset = index * stage_stride_cols + tmem_addr

    var bm = work_tile_coord[0]
    var bn = work_tile_coord[1]

    # scale_b index calculation when MMA_N != BK(128)
    var b_scale_idx0 = 0
    var b_scale_next_n = 0
    var b_scale_0: Scalar[accum_type]
    var b_scale_1: Scalar[accum_type]

    @parameter
    if MMA_N != BK:
        constrained[
            stageN <= gcd(MMA_N, BK) and (gcd(MMA_N, BK) % stageN == 0),
            (
                "gcd(MMA_N, BK) must be divisible by stageN. If not then this"
                " step should be updated to support non-divisible case"
                " accordingly"
            ),
        ]()

        var global_bn_start = bn * MMA_N
        var begin_n = min(MMA_N, BK - global_bn_start % BK)
        var end_n = min(MMA_N, N - global_bn_start)

        # find the first b_scale index just by dividing by block size (128)
        # we use `b_scale_next_n` to find the second b_scale index later
        b_scale_idx0 = global_bn_start // BK
        # If MMA_N > BK (128) then we should use two scales_b in each block. `next_n` determines the border between the two scales_b.
        # Example: N = 960, MMA_N = 192, num_of_b_scales: ceildiv(960, BK) = 8
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[idx0]--------------->|<--block_scales[idx0+1]-->
        #                                                           next_n(128)

        # this condition determines the border between the two scale_b and whether we have two scale_b in this block or one
        b_scale_next_n = begin_n if begin_n < Int(end_n) else MMA_N
        # Example 1: N = 896, MMA_N = 192, num_of_b_scales: ceildiv(896, BK) = 7
        # This will be the last block on the horizaontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <------------------------------------------------------------------------------------->|<
        # <-----------------------------------block_scales[6]----------------------------------->|<
        #                                                                                     next_n (192)

        # Example 2: N = 904, MMA_N = 192, num_of_b_scales: ceildiv(N, BK) = 8
        # This will be the last block on the horizaontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[6]------------------>|<-----block_scales[7]---->
        #                                                           next_n(128)

        # prefetch b scales
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[b_scale_idx0, k_iter].cast[accum_type]()
        )
        # this mean in this block we have two scale_b
        if b_scale_next_n < MMA_N:
            b_scale_1 = rebind[Scalar[accum_type]](
                b_scales[b_scale_idx0 + 1, k_iter].cast[accum_type]()
            )
        else:
            b_scale_1 = 0.0

    else:
        # when MMA_N == BK == 128 we only have one scale_b per block
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[bn, k_iter].cast[accum_type]()
        )
        b_scale_1 = 0.0

    # load a scales
    var warp_id = get_warp_id()
    var coord_m_warp_level_offset = (
        warp_id * WARP_SIZE if MMA_M == 256 else (warp_id % 2) * WARP_SIZE
    )
    var upper_local_m_offset = lane_id() // 4
    var lower_local_m_offset = lane_id() // 4 + 16

    accum_full_mbar[index].wait(phase)

    var tma_load_stage_index = consumer_phase.index()
    var a_scales_smem = a_scales_smem_iter.next(tma_load_stage_index)[]

    var upper_sfa0_smem = a_scales_smem[
        0, upper_local_m_offset + coord_m_warp_level_offset
    ].cast[accum_type]()
    var upper_sfa1_smem = a_scales_smem[
        0, upper_local_m_offset + coord_m_warp_level_offset + 8
    ].cast[accum_type]()

    var lower_sfa0_smem = a_scales_smem[
        0, lower_local_m_offset + coord_m_warp_level_offset
    ].cast[accum_type]()
    var lower_sfa1_smem = a_scales_smem[
        0, lower_local_m_offset + coord_m_warp_level_offset + 8
    ].cast[accum_type]()

    syncwarp()
    if lane_id() < UInt(CLUSTER_SIZE):
        _ = mma_mbar[tma_load_stage_index].arrive()
    syncwarp()

    @parameter
    for stage in range(num_stages):
        # column offset, moving right by 32 columns each time, since each num_stage stores two, 16 column submatrices
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_offset + (stage * stageN)
        var upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeats,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr)

        var lower_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeats,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        @parameter
        if stage == num_stages - 1:
            umma_arrive_leader_cta(accum_empty_mbar + index)

        var coord_n_mma_m256 = stage * stageN
        var coord_n_mma_m128 = stage * stageN + BN * (warp_id // 2)
        var coord_n_warp_level_offset = (
            coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        )

        var b_scale: Scalar[accum_type]

        @parameter
        if MMA_N != BK:
            # check if we cross the border between the two scale_b
            b_scale = (
                b_scale_0 if coord_n_warp_level_offset
                < b_scale_next_n else b_scale_1
            )
        else:
            b_scale = b_scale_0

        @parameter
        for ld_iter in range(stageN // 8):

            @parameter
            for j in range(fragment_size // 2):
                var upper_elems = upper_frag.slice[
                    2, offset = ld_iter * fragment_size + j * 2
                ]()
                var lower_elems = lower_frag.slice[
                    2, offset = ld_iter * fragment_size + j * 2
                ]()

                var upper_a_scale = (
                    upper_sfa0_smem if j % 2 == 0 else upper_sfa1_smem
                )
                var lower_a_scale = (
                    lower_sfa0_smem if j % 2 == 0 else lower_sfa1_smem
                )

                var upper_scale = upper_a_scale * b_scale
                var lower_scale = lower_a_scale * b_scale

                c_upper_main_tile[
                    stage, ld_iter * fragment_size + j * 2
                ] += rebind[Scalar[accum_type]](upper_elems[0]) * rebind[
                    Scalar[accum_type]
                ](
                    upper_scale
                )
                c_upper_main_tile[
                    stage, ld_iter * fragment_size + j * 2 + 1
                ] += rebind[Scalar[accum_type]](upper_elems[1]) * rebind[
                    Scalar[accum_type]
                ](
                    upper_scale
                )
                c_lower_main_tile[
                    stage, ld_iter * fragment_size + j * 2
                ] += rebind[Scalar[accum_type]](lower_elems[0]) * rebind[
                    Scalar[accum_type]
                ](
                    lower_scale
                )
                c_lower_main_tile[
                    stage, ld_iter * fragment_size + j * 2 + 1
                ] += rebind[Scalar[accum_type]](lower_elems[1]) * rebind[
                    Scalar[accum_type]
                ](
                    lower_scale
                )


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
fn blackwell_tma_umma_warp_specialized_blockwise_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,  # must pass mma_m by mma_n as this layout, since that's how much each output has to be
    a_scales_tile_layout: Layout,
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: UInt,
    num_clc_pipeline_stages: UInt,
    num_accum_pipeline_stages: UInt,
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
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_tile_layout, a_scales_desc_layout
    ],
    cluster_dim: StaticTuple[Int32, 3],
    num_iters: UInt,
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutableAnyOrigin],
    problem_shape: StaticTuple[Int32, 3],
):
    alias num_output_warps = 4

    alias accum_type = get_accum_type[a_type]()

    constrained[
        b_scales_type == a_scales_type and accum_type == DType.float32,
        "Only support float32 for a_scales and b_scales",
    ]()

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

    constrained[cta_group == 2, "Only support cta_group = 2"]()
    constrained[BK == 128, "Only support BK = 128"]()
    constrained[
        MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK,
        "MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK. MMA_N="
        + String(MMA_N)
        + ", GCD="
        + String(gcd(MMA_N, BK)),
    ]()

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

    # keep the physical SMEM buffer BM x MMA_N
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    alias a_scales_smem_layout = Layout.row_major(1, BM)

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

    alias a_scales_smem_size = a_scales_smem_layout.size() * num_pipeline_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()
    var a_scales_smem_base = (c_smem_base + c_smem_size).bitcast[
        Scalar[a_scales_type]
    ]()

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

    var a_scales_smem = LayoutTensorIter[
        a_scales_type,
        a_scales_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_scales_smem_base,
        a_scales_smem_size,
    )
    var smem_pool = (a_scales_smem_base + a_scales_smem_size).bitcast[Int64]()

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

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    alias max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:

        @parameter
        for i in range(num_pipeline_stages):
            tma_mbar[i].init()
            # we need to have 5 arrivals, 2 M, 4 N, top left M/N is shared
            mma_mbar[i].init(
                cluster_shape[0] // cta_group
                + cluster_shape[1]
                - 1
                + CLUSTER_SIZE * (EPILOGUE_THREADS // 32)
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
                    a_scales_tma_op,
                    a_smem,
                    b_smem,
                    a_scales_smem,
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
                for i in range(num_iters):
                    var accum_index = accum_pipeline_producer_state.index()
                    var accum_phase = accum_pipeline_producer_state.phase()
                    accum_empty_mbar[accum_index].wait(accum_phase)
                    var tmem_offset = tmem_addr + (
                        accum_index * stage_stride_cols
                    )

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
                        0,
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
            alias reg_info = _get_accumulator_size[
                c_smem_layout = c_smem_iter.layout,
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                cta_group=cta_group,
            ]()
            # final results accumulator regs for C
            var c_upper_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()

            var c_lower_main_tile = LayoutTensor[
                accum_type,
                Layout.row_major(reg_info[0], reg_info[1]),
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ].stack_allocation()

            _ = c_upper_main_tile.fill(0.0)
            _ = c_lower_main_tile.fill(0.0)

            for k_iter in range(num_iters):
                promote_accumulators[
                    block_tile_shape=block_tile_shape,
                    mma_shape=mma_shape,
                    cta_group=cta_group,
                    CLUSTER_SIZE = Int32(CLUSTER_SIZE),
                ](
                    b_scales,
                    a_scales_smem,
                    c_upper_main_tile,
                    c_lower_main_tile,
                    accum_pipeline_consumer_state,
                    accum_full_mbar,
                    accum_empty_mbar,
                    tmem_addr,
                    mma_mbar,
                    consumer_phase,
                    work_tile_coord=(UInt(work_info.m), UInt(work_info.n)),
                    elect_one_warp=elect_one_warp,
                    stage_stride_cols=UInt(stage_stride_cols),
                    k_iter=k_iter,
                    problem_shape=problem_shape,
                )
                consumer_phase.step()
                accum_pipeline_consumer_state.step()

            # TODO (KERN-2081): investigate why this barrier is needed and if we can move/remove it
            named_barrier[num_output_warps * WARP_SIZE]()
            # wait for CUDA core promotion to finish and store result
            # scheduler fetch next work
            multi_stage_reg_epilogue[
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                c_swizzle=c_swizzle,
                cta_group=cta_group,
                num_output_warps=num_output_warps,
            ](
                c_upper_main_tile,
                c_lower_main_tile,
                c_smem_iter,
                c_tma_op,
                work_tile_coord=(UInt(work_info.m), UInt(work_info.n)),
                elect_one_warp=elect_one_warp,
            )

            next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


fn sm100_warp_specialized_blockwise_fp8[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    cta_group: Int = 1,
    num_clc_pipeline_stages: UInt = 2,
    num_pipeline_stages: Optional[UInt] = None,
](
    c: LayoutTensor[c_type, c_layout, *_, **_],
    a: LayoutTensor[a_type, a_layout, *_, **_],
    b: LayoutTensor[b_type, b_layout, *_, **_],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, *_, **_],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    constrained[
        a_type == b_type and a_type is DType.float8_e4m3fn,
        "Only support float8_e4m3fn",
    ]()

    constrained[
        a_scales_type == b_scales_type,
        "Only support float32 for scales",
    ]()

    if (a_scales.dim(1) * size_of[a_scales_type]()) % 16 != 0:
        raise Error(
            "a_scales should be a multiple of 16 bytes on the M dimension"
        )

    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    alias BM = MMA_M // cta_group
    alias BN = MMA_N // cta_group
    alias BK = config.block_tile_shape[2]

    constrained[
        (MMA_M != 128) or (MMA_N % 32 == 0),
        "if MMA_M is 128, then MMA_N must be a multiple of 32, MMA_N="
        + String(MMA_N)
        + ", MMA_M="
        + String(MMA_M),
    ]()

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B

    alias cluster_shape = config.cluster_shape

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1)

    a_tma_op = create_tma_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a)

    b_tma_op = create_tma_tile[
        Index(
            BN // (cluster_shape[0] // cta_group), BK
        ) if transpose_b else Index(BK, BN // (cluster_shape[0] // cta_group)),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    a_scales_tma_op = create_tma_tile[1, BM](ctx, a_scales)

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
    var c_tma_op = create_tma_tile[c_tma_tile_shape, swizzle_mode=c_swizzle](
        ctx, c
    )

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
    alias a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    alias b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    alias a_scales_smem_bytes_per_stage = BM * size_of[a_scales_type]()
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
        AB_smem_per_stage
        + a_scales_smem_bytes_per_stage
        + tma_mbar_bytes_per_stage
        + mma_mbar_bytes_per_stage
    )

    alias max_pipeline_stages = smem_leftover // producer_consumer_smem_per_stage

    constrained[
        max_pipeline_stages >= 1,
        "not enough smem even for one pipeline stage!",
    ]()

    @parameter
    if num_pipeline_stages:
        constrained[
            num_pipeline_stages.value() <= UInt(max_pipeline_stages),
            "num_pipeline_stages <= max_pipeline_stages",
        ]()

    alias pipeline_stages = num_pipeline_stages.value() if num_pipeline_stages else UInt(
        max_pipeline_stages
    )
    alias producer_consumer_smem = producer_consumer_smem_per_stage * pipeline_stages

    alias smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    alias kernel = blackwell_tma_umma_warp_specialized_blockwise_fp8_kernel[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        a_scales_tma_op.layout,
        a_scales_type,
        b_scales_type,
        b_scales_layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        a_scales_tma_op.desc_layout,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=transpose_b,
        cluster_shape = StaticTuple[Int32, 3](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=cta_group,
        num_pipeline_stages=pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        num_accum_pipeline_stages = UInt(max_accum_pipeline_stages),
        num_output_stages=num_output_stages,
        output_tile_shape=output_tile_shape,
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

    var problem_shape = StaticTuple[Int32, 3](M, N, K)

    ctx.enqueue_function[kernel, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_scales_tma_op,
        cluster_dim,
        ceildiv(K, BK),
        b_scales,
        problem_shape,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )
