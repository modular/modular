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
"""AMD RDNA matmul kernel with WMMA for RDNA 3+ and naive fallback for older.

RDNA 3+ (gfx11xx/gfx12xx): Uses 16x16x16 WMMA instructions with Wave32.
RDNA 1/2 (gfx10xx): Falls back to a per-thread naive matmul.

Block configuration (shared by both paths):
4 warps x 32 threads = 128 threads per workgroup.
BLOCK_M=64, BLOCK_N=64, BLOCK_K=16.
"""

from sys.info import _is_amd_rdna2_or_earlier

from gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.compute.mma import mma as _mma_intrinsic
from layout import Layout, LayoutTensor
from memory import stack_allocation
from utils import Index, IndexList
from utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type

# Block/grid constants (shared by both paths)
comptime BLOCK_M = 64
comptime BLOCK_N = 64
comptime BLOCK_K = 16
comptime NUM_WARPS = 4

# WMMA-specific constants
comptime MMA_M = 16
comptime MMA_N = 16
comptime MMA_K = 16
comptime WARPS_M = 2
comptime WARPS_N = 2
comptime WARP_M = BLOCK_M // WARPS_M  # 32
comptime WARP_N = BLOCK_N // WARPS_N  # 32
comptime NUM_M_MMAS = WARP_M // MMA_M  # 2
comptime NUM_N_MMAS = WARP_N // MMA_N  # 2
comptime NUM_C_TILES = NUM_M_MMAS * NUM_N_MMAS  # 4
comptime AB_FRAG_SIZE = 16
comptime CD_FRAG_SIZE = 8


fn gemm_kernel_rdna[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """GEMM kernel for AMD RDNA GPUs.

    On RDNA 3+ (gfx11xx/gfx12xx), uses 16x16x16 WMMA instructions with shared
    memory tiling. On older RDNA (gfx10xx), falls back to a per-thread naive
    matmul that iterates over the K dimension with scalar accumulation.
    """

    comptime if _is_amd_rdna2_or_earlier() or a_type not in (
        DType.float16,
        DType.bfloat16,
    ):
        _naive_matmul_kernel[
            c_type,
            a_type,
            b_type,
            c_layout,
            a_layout,
            b_layout,
            transpose_b,
            elementwise_lambda_fn,
            s_type,
        ](c, a, b, m, n, k)
    else:
        _wmma_matmul_kernel[
            c_type,
            a_type,
            b_type,
            c_layout,
            a_layout,
            b_layout,
            transpose_b,
            elementwise_lambda_fn,
            s_type,
        ](c, a, b, m, n, k)


fn _naive_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """Per-thread naive matmul for RDNA 1/2 (no WMMA support).

    Each thread in the 128-thread workgroup iterates over output elements
    assigned to it within the block's 64x64 tile. With 128 threads covering
    4096 elements, each thread handles 32 output elements.
    """
    var block_m_offset = Int(block_idx.y) * BLOCK_M
    var block_n_offset = Int(block_idx.x) * BLOCK_N
    var tid = Int(thread_idx.x)

    # 128 threads handle 64*64 = 4096 elements → 32 elements per thread
    comptime for elem in range(32):
        var linear = tid * 32 + elem
        var local_row = linear // BLOCK_N
        var local_col = linear % BLOCK_N
        var global_row = block_m_offset + local_row
        var global_col = block_n_offset + local_col

        if global_row < m and global_col < n:
            var accum = Scalar[s_type](0)

            comptime if transpose_b:
                for i in range(k):
                    accum += rebind[Scalar[s_type]](
                        a[global_row, i].cast[s_type]()
                    ) * rebind[Scalar[s_type]](b[global_col, i].cast[s_type]())
            else:
                for i in range(k):
                    accum += rebind[Scalar[s_type]](
                        a[global_row, i].cast[s_type]()
                    ) * rebind[Scalar[s_type]](b[i, global_col].cast[s_type]())

            comptime if elementwise_lambda_fn:
                comptime elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda[c_type, 1](
                    Index(global_row, global_col),
                    accum.cast[c_type](),
                )
            else:
                c[global_row, global_col] = accum.cast[c_type]()


fn _wmma_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """WMMA-based GEMM kernel for RDNA 3+ GPUs.

    Uses 16x16x16 WMMA instructions with shared memory tiling.
    Each workgroup computes a 64x64 output tile using 4 warps in a 2x2 layout.
    Each warp computes a 32x32 sub-tile using 2x2 = 4 WMMA operations per K step.
    """
    # Block coordinates
    var block_n = Int(block_idx.x)
    var block_m = Int(block_idx.y)

    var block_m_offset = block_m * BLOCK_M
    var block_n_offset = block_n * BLOCK_N

    # Thread identification
    var tid = Int(thread_idx.x)
    var wid = Int(warp_id())
    var lid = Int(lane_id())

    # Warp position in the 2x2 layout
    var warp_m = wid // WARPS_N  # 0 or 1
    var warp_n = wid % WARPS_N  # 0 or 1

    # Effective lane for RDNA WMMA (lanes 0-15 and 16-31 hold same data)
    var effective_lane = lid % 16

    # Shared memory for A and B tiles
    var a_shared = stack_allocation[
        BLOCK_M * BLOCK_K,
        a_type,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        BLOCK_N * BLOCK_K,
        b_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Initialize C accumulators to zero (4 tiles x 8 f32 elements each)
    var c_accum = InlineArray[SIMD[s_type, CD_FRAG_SIZE], NUM_C_TILES](
        fill=SIMD[s_type, CD_FRAG_SIZE](0)
    )

    # Main K-dimension loop
    for k_block in range(0, k, BLOCK_K):
        # --- Cooperative load of A tile to shared memory ---
        # 128 threads load BLOCK_M * BLOCK_K = 64 * 16 = 1024 elements
        # Each thread loads 1024 / 128 = 8 elements
        comptime for i in range(8):
            var elem_idx = tid * 8 + i
            var row = elem_idx // BLOCK_K
            var col = elem_idx % BLOCK_K
            var global_row = block_m_offset + row
            var global_col = k_block + col

            var val = Scalar[a_type](0)
            if global_row < m and global_col < k:
                val = rebind[Scalar[a_type]](a[global_row, global_col])
            a_shared[row * BLOCK_K + col] = val

        # --- Cooperative load of B tile to shared memory ---
        comptime for i in range(8):
            var elem_idx = tid * 8 + i
            var row = elem_idx // BLOCK_K
            var col = elem_idx % BLOCK_K
            var global_row = block_n_offset + row
            var global_col = k_block + col

            var val = Scalar[b_type](0)

            comptime if transpose_b:
                # B is stored as (N, K) in memory
                if global_row < n and global_col < k:
                    val = rebind[Scalar[b_type]](b[global_row, global_col])
            else:
                # B is stored as (K, N) - load transposed into shared memory
                if global_col < k and global_row < n:
                    val = rebind[Scalar[b_type]](b[global_col, global_row])
            b_shared[row * BLOCK_K + col] = val

        barrier()

        # --- Load fragments from shared memory and compute WMMA ---
        var a_frag = InlineArray[SIMD[a_type, AB_FRAG_SIZE], NUM_M_MMAS](
            fill=SIMD[a_type, AB_FRAG_SIZE](0)
        )
        var b_frag = InlineArray[SIMD[b_type, AB_FRAG_SIZE], NUM_N_MMAS](
            fill=SIMD[b_type, AB_FRAG_SIZE](0)
        )

        # Load A fragments: each lane loads one row of 16 elements
        comptime for m_mma in range(NUM_M_MMAS):
            var a_row = warp_m * WARP_M + m_mma * MMA_M + effective_lane

            comptime for ki in range(MMA_K):
                a_frag[m_mma][ki] = a_shared[a_row * BLOCK_K + ki]

        # Load B fragments: each lane loads one row of 16 elements
        comptime for n_mma in range(NUM_N_MMAS):
            var b_row = warp_n * WARP_N + n_mma * MMA_N + effective_lane

            comptime for ki in range(MMA_K):
                b_frag[n_mma][ki] = b_shared[b_row * BLOCK_K + ki]

        # Issue 4 WMMA operations (2x2 MMA tiles)
        comptime for m_mma in range(NUM_M_MMAS):
            comptime for n_mma in range(NUM_N_MMAS):
                var c_idx = m_mma * NUM_N_MMAS + n_mma
                _mma_intrinsic(
                    c_accum[c_idx],
                    a_frag[m_mma],
                    b_frag[n_mma],
                    c_accum[c_idx],
                )

        barrier()

    # --- Store C results to global memory ---
    # WMMA output mapping: lane l, element v -> C[row=v*2+l//16, col=l%16]
    var lane_col = lid % 16
    var lane_row_offset = lid // 16  # 0 for lanes 0-15, 1 for lanes 16-31

    comptime for m_mma in range(NUM_M_MMAS):
        comptime for n_mma in range(NUM_N_MMAS):
            var c_idx = m_mma * NUM_N_MMAS + n_mma

            comptime for v in range(CD_FRAG_SIZE):
                var global_row = (
                    block_m_offset
                    + warp_m * WARP_M
                    + m_mma * MMA_M
                    + v * 2
                    + lane_row_offset
                )
                var global_col = (
                    block_n_offset + warp_n * WARP_N + n_mma * MMA_N + lane_col
                )

                if global_row < m and global_col < n:
                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = (
                            elementwise_lambda_fn.value()
                        )
                        elementwise_lambda[c_type, 1](
                            Index(global_row, global_col),
                            c_accum[c_idx][v].cast[c_type](),
                        )
                    else:
                        c[global_row, global_col] = c_accum[c_idx][v].cast[
                            c_type
                        ]()
