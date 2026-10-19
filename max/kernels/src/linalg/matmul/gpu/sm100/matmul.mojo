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
"""SM100 matmul - TileTensor-native compatibility layer for grouped_matmul.

NOTE: This module is maintained for backward compatibility with grouped_matmul
implementations that depend on internal functions (WarpRole, consumer_main_loop,
stsm_helper, accum_arrive).
The helper surface is TileTensor-native; legacy callers should adapt at their
own boundary before entering this module.

For new code, use sm100_structured directly:
- Import configs from: linalg.matmul.gpu.sm100_structured.config
- Import matmul from: linalg.matmul.gpu.sm100_structured.matmul
"""

from std.sys import align_of, size_of

from max.gpu import lane_id, warp_id
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import (
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    DefaultEngine,
    Idx,
    TileTensor,
    row_major,
)
from layout.swizzle import Swizzle
from layout.tile_layout import Layout as TileLayout, TensorLayout
from structured_kernels.tile_types import SMemTileArray2D

from std.utils.index import Index, IndexList

from ....arch.sm100 import MmaOpSM100_SS
from .pipeline import ProducerConsumerPipeline, MbarPtr


@fieldwise_init
struct WarpRole[has_scheduler: Bool = True](TrivialRegisterPassable):
    """Warp-role assignments for the SM100 warp-specialized matmul kernel.

    Each warp is statically assigned to one of four roles: `Mma` computes
    UMMA instructions using TCGEN05; `MainLoad` issues TMA requests to fill
    the A/B shared-memory pipeline; `Scheduler` advances the tile work queue;
    `Epilogue` writes accumulator results to global memory. The role indices
    shift by one when `has_scheduler` is `True`.

    Parameters:
        has_scheduler: Whether a dedicated scheduler warp is present.
    """

    var _role: Int32

    comptime Mma = Self(6) if Self.has_scheduler else Self(5)
    comptime MainLoad = Self(5) if Self.has_scheduler else Self(4)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @inline(.always)
    def __eq__(self, other: Int) -> Bool:
        return self._role == Int32(other)

    @inline(.always)
    def __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @inline(.always)
    def __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @inline(.always)
    def __ge__(self, other: Int) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @inline(.always)
    def is_main_load() -> Bool:
        return Self.MainLoad == warp_id()

    @staticmethod
    @inline(.always)
    def is_mma() -> Bool:
        return Self.Mma == warp_id()

    @staticmethod
    @inline(.always)
    def is_epilogue() -> Bool:
        return Self.Epilogue >= warp_id()

    @staticmethod
    @inline(.always)
    def is_scheduler() -> Bool:
        comptime assert Self.has_scheduler, "Scheduler warp is not enabled"
        return Self.Scheduler == warp_id()


@inline(.always)
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
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
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: Int = 1,
](
    tmem_addr: UInt32,
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
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
    """TileTensor overload of `consumer_main_loop`.

    Accepts `SMemTileArray2D` instead of iterator-based shared-memory tiles, indexing directly
    into the tile arrays to get TileTensor tiles for MMA. The tile dimension
    and swizzle parameters (a_dim0, a_dim1, etc.) are explicit because Mojo
    requires them for overload resolution with parametric struct arguments.
    """
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        for j in range(UInt32(k_group_size)):
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_tiles[offset]
            var b_smem_tile = b_smem_tiles[offset]
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=(
                    (iter_idx + j) == k_start
                ),  # Initialize C on first iteration
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


@inline(.always)
def f32_frag_to_smem[
    swizzle_mode: TensorMapSwizzle,
    vec_dtype: DType,
    vec_size: Int,
    DstLayout: TensorLayout,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[
        _,
        DstLayout,
        MutAnyOrigin,
        # Shared memory is raw-pointer addressed so pin DefaultEngine.
        Engine=DefaultEngine[element_width=1],
        address_space=.SHARED,
    ],
):
    """Writes an FP32 TCGEN05 accumulator fragment to a swizzled shared-memory tile.

    Implements a manual 8×4 thread-to-element distribution for FP32 fragments,
    since the compiler cannot prove `all_dims_known` through migrated layout types.
    Each lane writes a pair of FP32 values at the correct swizzled row/column offset.

    Parameters:
        swizzle_mode: Shared memory swizzle mode to apply.
        vec_dtype: Element type of the source fragment (must be FP32).
        vec_size: Number of elements in the fragment (= 2 × frag_rows × frag_cols).
        DstLayout: Layout of the destination shared-memory tile.

    Args:
        vec: Source accumulator fragment as a flat inline array.
        dst: Destination tile in shared memory.
    """
    # Manual implementation of dst.vectorize[1, 2]().distribute[row_major(8, 4)]
    # because the compiler can't prove `all_dims_known` through migrated layout types. See MSTDL-2422.
    comptime stride0: Int = dst.static_stride[0]
    comptime shape0 = dst.static_shape[0]
    comptime shape1 = dst.static_shape[1]

    comptime frag_rows = shape0 // 8
    comptime frag_cols = shape1 // 8
    comptime assert (
        2 * frag_rows * frag_cols == vec_size
    ), "2*frag_rows*frag_cols must be equal to vec_size"

    var lane = lane_id()
    var thread_row = (lane >> 2) & 7
    var thread_col = lane & 3
    var base_offset = thread_row * stride0 + thread_col * 2
    var dst_flat = dst.reshape(row_major[shape0 * stride0]())

    comptime for i in range(frag_rows):
        comptime for j in range(frag_cols):
            comptime i_vec = i + j * frag_rows
            var val = SIMD[dst.dtype, 2](
                rebind[Scalar[dst.dtype]](vec[2 * i_vec]),
                rebind[Scalar[dst.dtype]](vec[2 * i_vec + 1]),
            )
            var offset = base_offset + i * 8 * stride0 + j * 8
            dst_flat.store[alignment=align_of[dst.dtype]()]((offset,), val)


@inline(.always)
def stsm_helper[
    swizzle: Swizzle,
    stageN: Int,
    vec_dtype: DType,
    vec_size: Int,
    DstLayout: TensorLayout,
    transpose_c: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[
        _,
        DstLayout,
        MutAnyOrigin,
        # Shared memory is raw-pointer addressed so pin DefaultEngine.
        Engine=DefaultEngine[element_width=1],
        address_space=.SHARED,
    ],
    warp_offset: UInt32 = 0,
):
    """Stores a TCGEN05 accumulator fragment to shared memory using st.matrix or scalar stores.

    Routes to `f32_frag_to_smem` for FP32 fragments, or uses `st_matrix` hardware
    instructions (`stsmx4` or `stsmx2`) for narrower types. Applies swizzle addressing
    so the destination layout matches subsequent TMA-store or register-epilogue consumers.

    Parameters:
        swizzle: Swizzle descriptor for shared memory address computation.
        stageN: Column stage dimension used to select stsmx4 vs stsmx2 path.
        vec_dtype: Fragment element type.
        vec_size: Number of elements in the fragment.
        DstLayout: Static layout of the destination tile in shared memory.
        transpose_c: Whether to transpose the C tile before writing (FP32 only).
        swizzle_mode: TMA swizzle mode for FP32 fallback path.

    Args:
        vec: Accumulator fragment to write.
        dst: Destination tile in shared memory.
        warp_offset: Per-warp row offset within the tile (default 0).
    """
    comptime if size_of[dst.dtype]() == 4:
        comptime assert not transpose_c, "transpose_c must be False"
        return f32_frag_to_smem[swizzle_mode](vec, dst)
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
    comptime stride0 = dst.static_stride[0]
    comptime stride1 = dst.static_stride[1]
    comptime assert stride1 == 1, (
        "stride1 must be 1. Got: "
        + String(stride1)
        + " for strides ("
        + String(stride0)
        + ", "
        + String(stride1)
        + ")"
    )
    comptime shape0 = dst.static_shape[
        1
    ] if not transpose_c else dst.static_shape[0]
    # the layout looks like
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    # but transposed and coalesced by 8 elements.
    comptime trans_st_matrix_layout = TileLayout(
        Coord(Idx[8], Idx[2], Idx[2]),
        Coord(Idx[stride0], Idx[8 * stride1], Idx[8 * stride0]),
    )
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane = lane_id()
    var stsm_lane_offset = UInt32(
        (lane & 15) * stride0 + (lane >> 4) * 8
    ) if not transpose_c else trans_st_matrix_layout.__call__[
        linear_idx_type=.uint32
    ](
        Coord(lane)
    )

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    comptime for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        comptime if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))
        comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
        comptime cast_width = 4 // size_of[Scalar[dst.dtype]]()
        var v = SIMD[dst.dtype, stmtx_simd_width * cast_width]()
        comptime for k in range(stmtx_simd_width):
            var src = SIMD[vec_dtype, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = vec[i * stsmx_lane_size + k * cast_width + _j]
            var casted = src.cast[dst.dtype]()
            comptime for _j in range(cast_width):
                v[k * cast_width + _j] = casted[_j]
        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst.ptr.unsafe_offset(offset),
            bitcast[.float32, stmtx_simd_width](v),
        )


@inline(.always)
def accum_arrive[
    cta_group: Int
](mma_output_pipeline: ProducerConsumerPipeline, mma_output_stage: UInt32):
    """Signals arrival at the MMA output pipeline barrier for a given stage.

    Dispatches between a plain `mbarrier_arrive` for single-CTA groups and
    `umma_arrive_leader_cta` for multi-CTA UMMA groups, where only the leader
    CTA signals arrival.

    Parameters:
        cta_group: Number of CTAs in the UMMA group.

    Args:
        mma_output_pipeline: Producer-consumer pipeline tracking MMA completion.
        mma_output_stage: Pipeline stage index to arrive at.
    """
    comptime if cta_group == 1:
        _ = mbarrier_arrive(
            rebind[MbarPtr](mma_output_pipeline.consumer_mbar(mma_output_stage))
        )
    else:
        umma_arrive_leader_cta(
            rebind[MbarPtr](mma_output_pipeline.consumer_mbar(mma_output_stage))
        )
