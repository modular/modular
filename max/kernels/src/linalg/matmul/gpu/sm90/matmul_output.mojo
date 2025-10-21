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
from math import ceildiv
from sys import simd_width_of, size_of

from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import lane_id
from gpu.memory import fence_async_view_proxy
from gpu.mma import st_matrix
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import (
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core_async import st_matrix_n_layout
from layout.tma_async import TMATensorTile
from memory import bitcast
from stdlib.bit import log2_floor

from utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....structuring import (
    SMemTileType,
    RegTileType,
)

# Helper structures and functions for GEMM output processing
# These break down the complex output logic into simpler, reusable components.


# Simple struct to hold thread mapping information
@register_passable("trivial")
struct ThreadInfo:
    """Thread identification within the warp group."""

    var warp_id: UInt
    var lane_id: UInt
    var lane_row: UInt32
    var lane_col: UInt32

    fn __init__(
        out self,
        warp_id: UInt,
        lane_id: UInt,
        lane_row: UInt32,
        lane_col: UInt32,
    ):
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.lane_row = lane_row
        self.lane_col = lane_col

    @staticmethod
    fn from_warp_group_thread_idx(thread_idx: UInt) -> ThreadInfo:
        var warp_id = thread_idx // UInt(WARP_SIZE)
        var lane_id = UInt(lane_id())
        var lane_row = UInt32(lane_id // 4)
        var lane_col = UInt32(lane_id % 4)
        return ThreadInfo(warp_id, lane_id, lane_row, lane_col)


# Simple struct for MMA tile coordinates
@register_passable("trivial")
struct MMATileCoords:
    """Coordinates for an MMA tile in the output."""

    var m_mma: Int
    var n_mma: Int
    var mma_id: Int

    fn __init__(out self, m_mma: Int, n_mma: Int, mma_id: Int):
        self.m_mma = m_mma
        self.n_mma = n_mma
        self.mma_id = mma_id

    @staticmethod
    fn compute(m_mma: Int, n_mma: Int, num_m_mmas: Int) -> MMATileCoords:
        return MMATileCoords(m_mma, n_mma, n_mma * num_m_mmas + m_mma)


@always_inline
fn calculate_output_tile_bounds[
    c_layout: Layout, //, BM: Int, BN: Int
](
    c: LayoutTensor[_, c_layout, MutableAnyOrigin, *_, **_],
    block_y: Int,
    block_x: Int,
) -> Tuple[UInt32, UInt32]:
    """Calculate the output bounds for the current thread block.

    Returns the maximum valid row and column indices for this block's output tile,
    accounting for edge cases where the tile extends beyond the matrix dimensions.
    """
    alias N = c_layout.shape[1].value()  # Output matrix column dimension
    var M = c.dim[0]()  # Output matrix row dimension (can be dynamic)

    # Calculate bounds for this tile, clamping to matrix dimensions
    var M_bound = min(UInt32((block_y + 1) * BM), UInt32(M))
    var N_bound = min(UInt32((block_x + 1) * BN), UInt32(N))
    return M_bound, N_bound


@always_inline
fn store_accumulator_fragments_to_shared_memory[
    c_type: DType,
    c_tile_layout: Layout,
    accum_type: DType,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
    num_m_mmas: Int,
    sub_wg_bn_id: Int,
    use_x2_for_last_iter: Bool,
    num_consumer: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    st_matrix_swizzle: Swizzle,
    st_matrix_rt_layout: RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    """Store accumulator fragments from registers to shared memory using st.matrix instructions.

    This function uses NVIDIA's st.matrix instruction for efficient bf16 storage,
    handling the register-to-shared-memory transfer with proper swizzling for
    bank conflict avoidance.
    """

    # Iterate over TMA-sized chunks in the N dimension
    @parameter
    for tma_n in range(WG_BN // TMA_BN):

        @parameter
        for m_mma in range(num_m_mmas):
            # Each st.matrix instruction handles 16 bytes at a time
            @parameter
            for i in range(TMA_BN // 16):
                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](
                    Int(warp_group_thread_idx),
                    i,
                    m_mma,
                    Int(local_warp_group_idx),
                )
                # Calculate which MMA tile we're storing in the N dimension
                var n_mma = (
                    i + tma_n * (TMA_BN // 16) + sub_wg_bn_id * (WG_BN // 16)
                )

                # Calculate shared memory offset with swizzling for bank conflict avoidance
                var offset = c_tile.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    + WG_BM * TMA_BN * tma_n
                )

                @always_inline
                @parameter
                fn st_matrix_frag[x2: Bool = False]():
                    alias xn = 1 if x2 else 2
                    alias xf = 2 if x2 else 1
                    var c_frag = c_reg_tile.tile[1, 4 * xf](
                        m_mma,
                        xn * n_mma,
                    )
                    var d_reg = c_frag.load[4 * xf](0, 0).cast[DType.bfloat16]()
                    var d_reg_f32_packed = bitcast[DType.float32, 2 * xf](d_reg)
                    st_matrix[simd_width = 2 * xf](offset, d_reg_f32_packed)

                @parameter
                if use_x2_for_last_iter:
                    st_matrix_frag()
                else:
                    st_matrix_frag[True]()


# Mutable fragment lambda applicator used for both compute and non-compute cases

alias elementwise_lambda_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
] (IndexList[2], mut SIMD[dtype, width]) capturing -> None


@always_inline
fn apply_epilogue_to_output_tile[
    c_type: DType,
    c_tile_layout: Layout, //,
    elementwise_lambda_fn: elementwise_lambda_type,
    N: Int,
    WG_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_wg_coord_m: Int,
    c_gmem_wg_coord_n: Int,
    local_thread_idx: UInt,
    M_bound: UInt32,
    N_bound: UInt32,
):
    """Apply the epilogue lambda function to the output data.

    This function reads data from shared memory, applies the user-provided
    epilogue function (e.g., bias addition, activation), and handles bounds
    checking for edge tiles.
    """
    alias epilogue = elementwise_lambda_fn
    alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[c_type, WG_BN]()
    alias thread_layout = Layout.row_major(
        num_consumer_threads // (WG_BN // simd_size),
        WG_BN // simd_size,
    )

    var c_gmem_frag, c_gmem_offset_coords, _ = c_gmem_wg_tile.vectorize[
        1, simd_size
    ]().distribute_with_offset[thread_layout](local_thread_idx)
    var coord_m = c_gmem_wg_coord_m + c_gmem_offset_coords[0]
    var coord_n = c_gmem_wg_coord_n + c_gmem_offset_coords[1] * simd_size

    var c_smem_frag = c_tile.vectorize[1, simd_size]().distribute[
        thread_layout, swizzle=st_matrix_vec_swizzle
    ](local_thread_idx)

    alias num_stores_per_thread = c_gmem_frag.layout.size()

    @parameter
    for i in range(num_stores_per_thread):
        alias src_idx = c_smem_frag.layout(i)
        alias dst_idx = c_gmem_frag.layout(i)
        alias dst_m_offset = dst_idx // N
        alias dst_n_offset = dst_idx % N
        var m = UInt32(coord_m + dst_m_offset)
        var n = UInt32(coord_n + dst_n_offset)
        alias alignment = align_of[SIMD[c_type, simd_size]]()

        if m < M_bound and n < N_bound:
            epilogue(
                (Int(m), Int(n)),
                c_smem_frag[i, 0],
            )


@always_inline
fn store_output_tile_via_tma[
    c_type: DType,
    c_tma_layout: Layout,
    c_tile_layout: Layout, //,
    BM: Int,
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, _],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    local_thread_idx: UInt,
    block_x: Int,
    block_y: Int,
    sub_wg_bn_id: Int,
):
    """Store output tile to global memory using Tensor Memory Accelerator (TMA).

    Uses NVIDIA's TMA hardware for efficient async memory transfers from
    shared memory to global memory with automatic address generation.
    """

    fence_async_view_proxy()

    if local_thread_idx < UInt(WG_BN // TMA_BN):
        var smem_offset = c_tile.ptr.offset(WG_BM * TMA_BN * local_thread_idx)
        var c_tma_tile = SMemTileType[
            c_type,
            c_tma_layout,
            alignment=128,
        ](smem_offset)

        c_tma_op.async_store(
            c_tma_tile,
            (
                UInt(
                    block_x * BN
                    + sub_wg_bn_id * WG_BN
                    + local_thread_idx * UInt(TMA_BN)
                ),
                UInt(block_y * BM),
            ),
        )
        c_tma_op.commit_group()
        c_tma_op.wait_group()


@always_inline
fn store_output_tile_direct[
    c_type: DType,
    c_tile_layout: Layout, //,
    use_x2_for_last_iter: Bool,
    WG_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
    st_matrix_swizzle: Swizzle,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    local_thread_idx: UInt,
):
    alias thread_layout = Layout.row_major(
        num_consumer_threads // (WG_BN // simd_size),
        WG_BN // simd_size,
    )

    @parameter
    if use_x2_for_last_iter:
        var masked_c_tile = c_tile.slice[
            Slice(0, Int(c_tile.layout.shape[0])),
            Slice(0, Int(c_tile.layout.shape[1]) // 2),
        ]()
        var masked_c_gmem_wg_tile = c_gmem_wg_tile.slice[
            Slice(0, Int(c_gmem_wg_tile.layout.shape[0])),
            Slice(0, Int(c_gmem_wg_tile.layout.shape[1]) // 2),
        ]()
        alias thread_layout_v2 = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size // 2,
        )
        if local_thread_idx < UInt(num_consumer_threads // 2):
            copy_sram_to_dram[
                thread_layout=thread_layout_v2,
                swizzle=st_matrix_swizzle,
            ](
                masked_c_gmem_wg_tile.vectorize[1, simd_size](),
                masked_c_tile.vectorize[1, simd_size](),
            )
    else:
        copy_sram_to_dram[
            thread_layout=thread_layout,
            swizzle=st_matrix_swizzle,
        ](
            c_gmem_wg_tile.vectorize[1, simd_size](),
            c_tile.vectorize[1, simd_size](),
        )


@always_inline
fn handle_optimized_bfloat16_output[
    c_type: DType,
    c_tma_layout: Layout,
    c_desc_layout: Layout,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    num_consumer: Int,
    use_tma_store: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type],
    BM: Int,
    BN: Int,
    num_m_mmas: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    c_gmem_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_corner_coords: c.CornerCoordsType,
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Handle output using st.matrix instructions for optimized bf16 output."""
    # Calculate output bounds
    var M_bound, N_bound = calculate_output_tile_bounds[BM, BN](
        c, block_y, block_x
    )

    # Layout dimensions
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    alias st_matrix_swizzle = make_ldmatrix_swizzle[
        c_type, TMA_BN, log2_floor(16 // size_of[c_type]())
    ]()

    alias num_sub_wg_bn_iters = ceildiv(BN, WG_BN)
    alias last_iter = BN // WG_BN
    alias needs_x2 = BN % WG_BN != 0
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE

    constrained[
        needs_x2 == (c_frag_size % 4 == 0 and c_frag_size % 8 != 0),
        "stmatrix and wgmma register count conflict: needs_x2 = "
        + String(needs_x2)
        + " c_frag_size ="
        + String(c_frag_size),
    ]()

    @parameter
    for sub_wg_bn_id in range(num_sub_wg_bn_iters):
        alias use_x2_for_last_iter = needs_x2 and sub_wg_bn_id == last_iter

        # Store fragments to shared memory
        store_accumulator_fragments_to_shared_memory[
            wgmma_shape,
            BN,
            WG_BM,
            WG_BN,
            TMA_BN,
            num_m_mmas,
            sub_wg_bn_id,
            use_x2_for_last_iter,
            num_consumer,
        ](
            c_tile,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            st_matrix_swizzle,
            st_matrix_rt_layout,
        )

        named_barrier[num_consumer_threads](10)

        alias thread_layout = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size,
        )

        var c_gmem_wg_tile, wg_tile_coords, _ = c_gmem_tile.tile_with_offset[
            BM, WG_BN
        ](0, sub_wg_bn_id)
        var c_gmem_wg_coords = (
            rebind[c.CornerCoordsType](wg_tile_coords) + c_gmem_corner_coords
        )

        # Handle compute lambda
        @parameter
        if elementwise_compute_lambda_fn:
            alias lambda_fn = elementwise_compute_lambda_fn.value()

            @parameter
            fn _compute_lambda[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](
                index: IndexList[2], mut val: SIMD[dtype, width]
            ) capturing -> None:
                var res = lambda_fn[alignment=alignment](index, val)
                val = res

            apply_epilogue_to_output_tile[
                _compute_lambda,
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )
            named_barrier[num_consumer_threads](10)

        # Handle epilogue lambda
        @parameter
        if elementwise_lambda_fn:
            alias lambda_fn = elementwise_lambda_fn.value()

            @parameter
            fn _epilogue_lambda[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](
                index: IndexList[2], mut val: SIMD[dtype, width]
            ) capturing -> None:
                _ = lambda_fn[alignment=alignment](index, val)

            apply_epilogue_to_output_tile[
                _epilogue_lambda,
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )

        else:
            # Regular store path
            @parameter
            if use_tma_store and not use_x2_for_last_iter:
                store_output_tile_via_tma[BM, BN, WG_BM, WG_BN, TMA_BN,](
                    c_tma_op,
                    c_tile,
                    local_thread_idx,
                    block_x,
                    block_y,
                    sub_wg_bn_id,
                )
            else:
                store_output_tile_direct[
                    use_x2_for_last_iter,
                    WG_BN,
                    num_consumer_threads,
                    simd_size,
                    st_matrix_swizzle,
                ](
                    c_tile,
                    c_gmem_wg_tile,
                    local_thread_idx,
                )

        named_barrier[num_consumer_threads](10)


@always_inline
fn write_gemm_output_to_global_memory[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_tma_layout: Layout,
    c_reg_layout: Layout,
    c_desc_layout: Layout, //,
    /,
    *,
    c_tile_shape: IndexList[2],
    c_swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[
        c_type,
        c_tile_layout,
        alignment=128,
    ],
    c_reg_tile: RegTileType[
        accum_type,
        c_reg_layout,
        _,
    ],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Write matrix multiplication output from registers to global memory.

    This is the main orchestrator function that selects the optimal output path:
    1. Optimized bf16 path: Uses st.matrix instructions for efficient bf16 storage
       when data types and alignment constraints are met
    2. Aligned path: Fast path when output dimension N is divisible by tile size,
       avoiding bounds checking on the column dimension
    3. General path: Handles arbitrary matrix dimensions with full bounds checking

    The function also supports optional epilogue operations (bias, activation) via
    lambda functions that are applied during the output write.
    """
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias simd_size = simd_width_of[c_type]()
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]

    # Use helper to compute tile coordinates
    var c_gmem_tile, c_gmem_corner_coords, c_gmem_offset = c.tile_with_offset[
        BM, BN
    ](block_y, block_x)
    var c_gmem_split, split_coords, split_offset = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)

    alias c_coord_type = c.CornerCoordsType
    # Calculate which warp this thread belongs to within the warp group
    var warp_id = warp_group_thread_idx // UInt(WARP_SIZE)

    alias N = c_layout.shape[1].value()
    alias is_N_multiple_of_16B = N * size_of[c_type]() % 16 == 0
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN
    # Determine which output path to use based on data types and alignment.
    # The st.matrix path provides significant performance benefits for bf16 output
    # by using specialized hardware instructions, but requires:
    # - FP32 accumulator and BF16 output types
    # - Register count divisible by 4
    # - Proper alignment of M dimension, shared memory, and output row size
    # - Limited consumer threads and specific tile size constraints
    # fmt: off
    alias use_stmatrix = (accum_type is DType.float32
            and c_type is DType.bfloat16                # BF16 output
            and c_frag_size % 4 == 0                    # Register count constraint
            and BM % wgmma_shape[0] == 0                # M dimension alignment
            and WG_BN % 16 == 0                         # Shared memory alignment
            and num_consumer <= 2                       # Consumer thread limit
            and BN == wgmma_shape[1]                    # Tile size constraint
            and BM == WG_BM                             # Block size constraint
            and N * size_of[c_type]() % 16 == 0)        # Output row size alignment
    # fmt: on

    @parameter
    if use_stmatrix:
        # Path 1: Optimized bf16 output using st.matrix instructions
        # This path converts fp32 accumulator to bf16 and uses specialized
        # hardware for efficient register-to-memory transfers
        handle_optimized_bfloat16_output[
            wgmma_shape,
            num_consumer,
            use_tma_store,
            elementwise_lambda_fn,
            elementwise_compute_lambda_fn,
            BM,
            BN,
            num_m_mmas,
            num_consumer_threads,
            simd_size,
        ](
            c_tma_op,
            c,
            c_tile,
            c_reg_tile,
            c_gmem_tile,
            c_gmem_corner_coords,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )

    # Path 2: Aligned output when N is divisible by tile size
    # This avoids bounds checking on columns for better performance
    elif N % BN == 0:
        write_gemm_output_aligned[
            c_tile_shape=c_tile_shape,
            wgmma_shape=wgmma_shape,
            num_consumer=num_consumer,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](
            c,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            block_y,
            block_x,
        )

    # Path 3: General case with full bounds checking
    # Handles arbitrary matrix dimensions safely but with some performance cost
    else:
        write_gemm_output_with_bounds_check[
            c_tile_shape=c_tile_shape,
            wgmma_shape=wgmma_shape,
            num_consumer=num_consumer,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            block_y,
            block_x,
        )


# Simplified aligned output function
@always_inline
fn write_gemm_output_aligned[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_reg_layout: Layout,
    /,
    *,
    c_tile_shape: IndexList[2],
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Simplified aligned output - N divisible by BN, no column bounds check needed.
    """
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]
    alias N = c_layout.shape[1].value()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    # Get tile coordinates
    var c_gmem_tile, c_gmem_corner_coords, c_gmem_offset = c.tile_with_offset[
        BM, BN
    ](block_y, block_x)

    # Split tile for consumer groups
    var c_gmem_split, split_coords, split_offset = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)

    # Get thread info
    var thread_info = ThreadInfo.from_warp_group_thread_idx(
        warp_group_thread_idx
    )
    var M_bound = UInt(c.dim[0]())

    @parameter
    if (
        elementwise_lambda_fn is not None
        or elementwise_compute_lambda_fn is not None
    ):
        # With epilogue: process each element individually
        var c_frag_vec2 = c_reg_tile.vectorize[1, 2]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                var mma_tile = MMATileCoords.compute(m_mma, n_mma, num_m_mmas)

                # Get warp tile and coordinates
                var warp_tile, warp_tile_coords, warp_tile_offset = (
                    c_gmem_split.tile_with_offset[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](Int(m_mma * 4 + thread_info.warp_id), n_mma)
                )

                # Calculate final coordinates
                var warp_coords = IndexList[2](
                    warp_tile_coords[0]
                    + c_gmem_corner_coords[0]
                    + split_coords[0],
                    warp_tile_coords[1]
                    + c_gmem_corner_coords[1]
                    + split_coords[1],
                )
                var warp_offset = (
                    warp_tile_offset + c_gmem_offset + split_offset
                )

                # Process each element in the warp tile
                var gmem_frag, gmem_offset_coords_raw, gmem_offset = (
                    warp_tile.vectorize[1, 2]().distribute_with_offset[
                        Layout.row_major(8, 4)
                    ](thread_info.lane_id)
                )

                var gmem_offset_coords = rebind[IndexList[2]](
                    gmem_offset_coords_raw
                )
                gmem_offset_coords[1] *= 2
                var coords = gmem_offset_coords + rebind[IndexList[2]](
                    warp_coords
                )
                var c_block_offset = gmem_offset + warp_offset

                alias num_vecs = gmem_frag.layout.size()

                @parameter
                for i in range(num_vecs):
                    alias dst_idx = gmem_frag.layout(i)
                    alias dst_m_offset = dst_idx // N
                    alias dst_n_offset = dst_idx % N
                    var m = Int(coords[0] + dst_m_offset)
                    var n = Int(coords[1] + dst_n_offset)

                    alias alignment = align_of[SIMD[c_type, 2]]()
                    if m < Int(M_bound) and n < N:

                        @parameter
                        if elementwise_lambda_fn:
                            alias epilogue = elementwise_lambda_fn.value()
                            epilogue[alignment=alignment](
                                (m, n),
                                c_frag_vec2[mma_tile.mma_id, i].cast[c_type](),
                            )
                        else:
                            alias compute_lambda = elementwise_compute_lambda_fn.value()
                            var reg_val = compute_lambda[alignment=alignment](
                                (m, n),
                                c_frag_vec2[mma_tile.mma_id, i].cast[c_type](),
                            )
                            c.ptr.store[alignment=alignment](
                                c_block_offset + dst_idx, reg_val
                            )
    else:
        # Without epilogue: direct copy
        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                var mma_tile = MMATileCoords.compute(m_mma, n_mma, num_m_mmas)

                # Get tiles
                var warp_tile = c_gmem_split.tile[
                    wgmma_shape[0] // 4, wgmma_shape[1]
                ](Int(m_mma * 4 + thread_info.warp_id), n_mma)

                var c_frag = c_reg_tile.tile[1, c_frag_size](mma_tile.mma_id, 0)

                # Direct copy using hardware layout
                copy_local_to_dram[Layout.row_major(8, 4)](
                    warp_tile.vectorize[1, 2](),
                    c_frag.vectorize[1, 2](),
                )


# Simplified bounds-checked output function
@always_inline
fn write_gemm_output_with_bounds_check[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_reg_layout: Layout,
    /,
    *,
    c_tile_shape: IndexList[2],
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Simplified bounds-checked output - handles arbitrary matrix dimensions.
    """
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    # Get tile coordinates
    var c_gmem_tile, c_gmem_corner_coords, _ = c.tile_with_offset[BM, BN](
        block_y, block_x
    )
    var c_gmem_split, _, _ = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)

    # Get thread info
    var thread_info = ThreadInfo.from_warp_group_thread_idx(
        warp_group_thread_idx
    )

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):
            var mma_tile = MMATileCoords.compute(m_mma, n_mma, num_m_mmas)

            # Get warp tile with bounds checking
            var warp_tile, warp_tile_coords_raw, _ = (
                c_gmem_split.tile_with_offset[
                    wgmma_shape[0] // 4, wgmma_shape[1]
                ](Int(m_mma * 4 + thread_info.warp_id), n_mma, 0, 0)
            )

            var warp_tile_coords = rebind[IndexList[2]](warp_tile_coords_raw)
            warp_tile_coords[0] += c_gmem_corner_coords[0] + wgmma_shape[
                0
            ] * Int(local_warp_group_idx)
            warp_tile_coords[1] += c_gmem_corner_coords[1]

            # Fragment matrix handling
            alias num_n_frag_mat = wgmma_shape[1] // 8
            alias num_m_frag_mat = wgmma_shape[0] // 4 // 8

            @parameter
            for n_frag in range(num_n_frag_mat):

                @parameter
                for m_frag in range(num_m_frag_mat):
                    alias frag_mat_id = n_frag * num_m_frag_mat + m_frag
                    var frag_mat_gmem = warp_tile.tile[8, 8](m_frag, n_frag)

                    # Get bounds for this fragment
                    var bound0 = UInt32(
                        frag_mat_gmem.runtime_layout.shape[0].value[0]
                    )
                    var bound1 = UInt32(
                        frag_mat_gmem.runtime_layout.shape[1].value[0]
                    )

                    # Process each element with bounds checking
                    @parameter
                    for i in range(2):
                        if (
                            thread_info.lane_row < bound0
                            and thread_info.lane_col * 2 + i < bound1
                        ):

                            @parameter
                            if elementwise_lambda_fn:
                                alias epilogue = elementwise_lambda_fn.value()
                                var frag_m = Int(warp_tile_coords[0]) + Int(
                                    m_frag * 8 + thread_info.lane_row
                                )
                                var frag_n = Int(warp_tile_coords[1]) + Int(
                                    n_frag * 8 + thread_info.lane_col * 2 + i
                                )
                                epilogue[
                                    alignment = align_of[Scalar[c_type]]()
                                ](
                                    (frag_m, frag_n),
                                    c_reg_tile[
                                        mma_tile.mma_id, frag_mat_id * 2 + i
                                    ].cast[c_type](),
                                )
                            else:
                                frag_mat_gmem[
                                    Int(thread_info.lane_row),
                                    Int(thread_info.lane_col * 2 + i),
                                ] = rebind[frag_mat_gmem.element_type](
                                    c_reg_tile[
                                        mma_tile.mma_id, frag_mat_id * 2 + i
                                    ].cast[c_type]()
                                )
