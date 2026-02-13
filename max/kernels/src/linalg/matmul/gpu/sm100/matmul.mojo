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
"""SM100 matmul - COMPATIBILITY LAYER for grouped_matmul.

NOTE: This module is maintained for backward compatibility with grouped_matmul
implementations that depend on internal functions (WarpRole, consumer_main_loop,
stsm_helper, shared_memory_epilogue, register_epilogue, accum_arrive).

For new code, use sm100_structured directly:
- Import configs from: linalg.matmul.gpu.sm100_structured.config
- Import matmul from: linalg.matmul.gpu.sm100_structured.matmul
"""

from sys import simd_width_of, size_of

from gpu import WARP_SIZE, lane_id, thread_idx, warp_id
from gpu.primitives.cluster import elect_one_sync
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.compute.mma import st_matrix
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.sync import (
    named_barrier,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from gpu.compute.arch.tcgen05 import *
from layout import (
    UNKNOWN_VALUE,
    Layout,
    RuntimeTuple,
)
from layout.int_tuple import IntTuple
from layout.layout import coalesce
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_tuple import idx2crd
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_core_async import st_matrix_n_layout
from layout.tma_async import TMATensorTile

from utils.fast_div import FastDiv
from utils.index import IndexList
from utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_compute_lambda_type
from .pipeline import ProducerConsumerPipeline


@fieldwise_init
struct WarpRole[has_scheduler: Bool = True](TrivialRegisterPassable):
    var _role: Int32

    comptime Mma = Self(6) if Self.has_scheduler else Self(5)
    comptime MainLoad = Self(5) if Self.has_scheduler else Self(4)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    fn __eq__(self, other: UInt) -> Bool:
        return self._role == Int32(other)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    fn __ge__(self, other: UInt) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @always_inline
    fn is_main_load() -> Bool:
        return Self.MainLoad == warp_id()

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        return Self.Mma == warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        return Self.Epilogue >= warp_id()

    @staticmethod
    @always_inline
    fn is_scheduler() -> Bool:
        constrained[Self.has_scheduler, "Scheduler warp is not enabled"]()
        return Self.Scheduler == warp_id()


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
    k_group_size: Int = 1,
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
        for j in range(UInt32(k_group_size)):
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_iter.next(offset)[]
            var b_smem_tile = b_smem_iter.next(offset)[]
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
    dst: LayoutTensor[mut=True, _, _, address_space = AddressSpace.SHARED, ...],
):
    # TODO: apply swizzle. Somehow swizzle+distribute results in wrong values.
    # alias swizzle = make_swizzle[DType.float64, swizzle_mode]() # hack
    # var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4), swizzle=swizzle](lane_id())
    var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane_id()
    )
    comptime assert (
        2 * dst_frag.layout.size() == vec.size
    ), "2*dst_frag.layout.size() must be equal to vec.size"

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
    dst: LayoutTensor[mut=True, _, _, address_space = AddressSpace.SHARED, ...],
    warp_offset: UInt32 = 0,
):
    @parameter
    if size_of[dst.dtype]() == 4:
        comptime assert not transpose_c, "transpose_c must be False"
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
    comptime assert stride1 == 1, (
        "stride1 must be 1. Got: "
        + String(stride1)
        + " for layout: "
        + String(dst.layout)
    )
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
    var stsm_lane_offset = UInt32(
        (lane & 15) * UInt(stride0) + (lane >> 4) * 8
    ) if not transpose_c else RLayout32Bits[trans_st_matrix_layout]()(Int(lane))

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
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))
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
    num_output_warps: Int,
    warp_dim: Int,
    MMA_M: Int,
    BN: Int,
    cta_group: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem: LayoutTensor[c_type, c_smem_layout, MutAnyOrigin, ...],
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
        comptime assert c_smem_layout.rank() == 4, "c_smem_layout must be 4D"
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
                    iter_j,
                    Int(warp_i),
                    iter_i,
                )
                var offset = UInt32(simd_size) * RLayout32Bits[result]()(coord)
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
                    local_i = cj + ci * UInt32(BN)
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
                    + ci * swizzle_dim * UInt32(Int(stageN))
                )
                var global_i = local_i + UInt32(c_i)
                var global_j = local_j + UInt32(c_j)
                if global_i < UInt32(Int(M)) and global_j < UInt32(Int(N)):
                    var val = ptr.load[width=simd_size, alignment=alignment]()
                    var reg_val = compute_lambda_fn[alignment=alignment](
                        (Int(global_i), Int(global_j)),
                        val,
                    )
                    ptr.store[width=simd_size, alignment=alignment](reg_val)
    else:
        # Layout F: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f
        comptime assert c_smem_layout.rank() == 3, "c_smem_layout must be 3D"
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
                        iter_j,
                        Int(warp_i),
                        iter_i,
                    )
                    var offset = UInt32(simd_size) * RLayout32Bits[result]()(
                        coord
                    )
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
                    var global_i = local_i + UInt32(c_i)
                    var global_j = local_j + UInt32(c_j)
                    if global_i < UInt32(Int(M)) and global_j < UInt32(Int(N)):
                        var val = ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        var reg_val = compute_lambda_fn[alignment=alignment](
                            (Int(global_i), Int(global_j)),
                            val,
                        )
                        ptr.store[width=simd_size, alignment=alignment](reg_val)

    named_barrier[Int32(num_output_warps * WARP_SIZE)]()


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
    num_output_warps: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: UInt,
    c_row: UInt,
    c_smem_warp_tile_upper: LayoutTensor[
        c_type, c_smem_upper_layout, MutAnyOrigin, ...
    ],
    c_smem_warp_tile_lower: LayoutTensor[
        c_type, c_smem_lower_layout, MutAnyOrigin, ...
    ],
):
    # Here we start keeping track of the index / indices this thread is
    # responsible for in shared memory. This is represented with shared_memory_row
    # and shared_memory_column and the children of these values shared_memory_row_upper_half
    # shared_memory_row_lower_half. We also need to update the global memory column c_col by
    # stageN since we are sliding through the overall compute block.

    var staged_c_col = c_col + stage * stageN

    var warp_id = warp_id()
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
                section_offset_upper * Int64(num_stages * stageN)
                + col_offset_upper
            )
            shared_lower_col = (
                section_offset_lower * Int64(num_stages * stageN)
                + col_offset_lower
            )

        else:
            # can't cast to uint64 as it's not supported yet
            # this will cost us slightly in performance
            comptime fast_div = FastDiv[DType.uint32](Int(shared_n))

            shared_upper_row = (
                Scalar[DType.int](offset_upper).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_upper_col = Int64(offset_upper % Int(shared_n))

            shared_lower_row = (
                Scalar[DType.int](offset_lower).cast[fast_div.uint_type]()
                / fast_div
            ).cast[DType.int64]()
            shared_lower_col = Int64(offset_lower % Int(shared_n))

        # now we need to add the global tile offset
        var global_upper_row = shared_upper_row + Int64(c_row)
        var global_upper_col = shared_upper_col + Int64(staged_c_col)
        var global_lower_row = shared_lower_row + Int64(c_row)
        var global_lower_col = shared_lower_col + Int64(staged_c_col)

        if global_upper_row < Int64(Int(M)) and global_upper_col < Int64(
            Int(N)
        ):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_upper_row), Int(global_upper_col)),
                c_smem_upper_frag[i, 0],
            )
            c_smem_upper_frag[i, 0] = reg_val

        if global_lower_row < Int64(Int(M)) and global_lower_col < Int64(
            Int(N)
        ):
            var reg_val = compute_lambda_fn[alignment=alignment](
                (Int(global_lower_row), Int(global_lower_col)),
                c_smem_lower_frag[i, 0],
            )
            c_smem_lower_frag[i, 0] = reg_val

        # If more than one chunk is created (happens when 8x4 is used)
        # they will be spaced 8 rows away from each other

        shared_memory_row_upper_half += UInt(distribute_rows)
        shared_memory_row_lower_half += UInt(distribute_rows)

    named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
fn _compute_register_lambda_fn[
    epilogue_dtype: DType,
    frag_size: Int,
    inc: Int,
    offset: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
    transpose_c: Bool,
](
    top_coord: StaticTuple[UInt32, 2],
    bottom_coord: StaticTuple[UInt32, 2],
    mut frag: SIMD[epilogue_dtype, frag_size],
    staged_c_row: UInt32,
    staged_c_col: UInt32,
):
    # update local coordinates w/ global memory offsets
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + top_coord[0], staged_c_col + top_coord[1] + UInt32(inc)
    )

    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        staged_c_row + bottom_coord[0],
        staged_c_col + bottom_coord[1] + UInt32(inc),
    )

    # slice the fragment to get the current repeat top and bottom fragments
    var simd_top = frag.slice[2, offset=offset]()
    var simd_bottom = frag.slice[2, offset = offset + 2]()

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
                    Int(top_frag_upper_coord[1] + UInt32(i)),
                ),
                simd_top[i],
            )

            simd_bottom[i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[0]),
                    Int(bottom_frag_upper_coord[1] + UInt32(i)),
                ),
                simd_bottom[i],
            )
        else:
            simd_top[i] = compute_lambda_fn(
                IndexList[2](
                    Int(top_frag_upper_coord[1] + UInt32(i)),
                    Int(top_frag_upper_coord[0]),
                ),
                simd_top[i],
            )

            simd_bottom[i] = compute_lambda_fn(
                IndexList[2](
                    Int(bottom_frag_upper_coord[1] + UInt32(i)),
                    Int(bottom_frag_upper_coord[0]),
                ),
                simd_bottom[i],
            )

    # store the results back into the fragment
    frag[offset] = simd_top[0]
    frag[offset + 1] = simd_top[1]
    frag[offset + 2] = simd_bottom[0]
    frag[offset + 3] = simd_bottom[1]


@always_inline
fn register_epilogue[
    MMA_M: Int,
    data_paths: Int,
    num_stages: Int,
    bits: Int,
    stage: Int,
    stageN: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: Int,
    epilogue_dtype: DType,
    frag_size: Int,
    repeats: Int,
    transpose_c: Bool,
    cta_group: Int,
    is_lower_frag_required: Bool,
](
    mut upper_frag_casted: SIMD[epilogue_dtype, frag_size],
    mut lower_frag_casted: SIMD[epilogue_dtype, frag_size],
    c_row: UInt32,
    c_col: UInt32,
    N: UInt32,
):
    comptime assert (
        bits == 256 and data_paths == 16
    ), "Only 16x256b tensor memory load is supported"

    comptime load_width = UInt(2)

    var warp_id = warp_id()

    # get global memory offset based on tile coordinates

    # we update the column offset to include the current stage
    var staged_c_col = c_col + UInt32(stage * stageN)
    var staged_c_row = c_row

    @parameter
    if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row += UInt32(warp_id * 32)
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row += UInt32(warp_id * 16)
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row += UInt32((warp_id % 2) * 32)
        staged_c_col += UInt32((warp_id // 2) * UInt(num_stages * stageN))

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = UInt(stageN) // UInt(repeats) // load_width
    var top_frag_upper_coord_left = StaticTuple[UInt32, 2](
        UInt32(lane_id() // threads_per_row),
        UInt32(lane_id() % threads_per_row * load_width),
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
