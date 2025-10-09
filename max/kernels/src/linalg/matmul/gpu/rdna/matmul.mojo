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
"""RDNA-optimized GEMM kernel using WMMA instructions.

This module implements matrix multiplication kernels optimized for AMD RDNA GPUs
(Radeon RX 7000 series, etc.) using Wave Matrix Multiply Accumulate (WMMA) instructions.

RDNA uses 16x16x16 WMMA operations with Wave32 mode for optimal performance.
"""

from collections import OptionalReg
from sys import llvm_intrinsic, simd_width_of
from sys.info import _is_amd_rdna, _is_amd_rdna2

from gpu import block_idx, thread_idx, WARP_SIZE, barrier
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.swizzle import Swizzle
from memory import stack_allocation

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig

# RDNA WMMA operates on 16x16x16 tiles in Wave32 mode
alias WMMA_M = 16
alias WMMA_N = 16
alias WMMA_K = 16


@always_inline
fn wmma_rdna[
    accum_type: DType,
    a_type: DType,
    b_type: DType
](
    a_frag: SIMD[a_type, 8],
    b_frag: SIMD[b_type, 8],
    c_frag: SIMD[accum_type, 8]
) -> SIMD[accum_type, 8]:
    """RDNA WMMA intrinsic for 16x16x16 matrix multiply-accumulate.

    Performs C += A * B using RDNA WMMA instructions.
    Selects the correct intrinsic based on input types.
    """
    constrained[
        accum_type is DType.float32,
        "RDNA WMMA accumulator must be float32",
    ]()

    @parameter
    if a_type is DType.float16 and b_type is DType.float16:
        # F32 = F16 * F16 + F32
        return llvm_intrinsic[
            "llvm.amdgcn.wmma.f32.16x16x16.f16",
            SIMD[accum_type, 8],
            has_side_effect=False,
        ](a_frag, b_frag, c_frag)
    elif a_type is DType.bfloat16 and b_type is DType.bfloat16:
        # F32 = BF16 * BF16 + F32
        return llvm_intrinsic[
            "llvm.amdgcn.wmma.f32.16x16x16.bf16",
            SIMD[accum_type, 8],
            has_side_effect=False,
        ](a_frag, b_frag, c_frag)
    else:
        constrained[False, "Unsupported RDNA WMMA input types"]()
        return c_frag


fn gemm_kernel_rdna_naive[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    c_layout_int_type: DType,
    a_layout_int_type: DType,
    b_layout_int_type: DType,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        layout_int_type=c_layout_int_type,
        linear_idx_type=c_linear_idx_type,
    ],
    a: LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        layout_int_type=a_layout_int_type,
        linear_idx_type=a_linear_idx_type,
    ],
    b: LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
        layout_int_type=b_layout_int_type,
        linear_idx_type=b_linear_idx_type,
    ],
):
    """Simple RDNA GEMM kernel using WMMA instructions.

    This is a basic implementation to demonstrate RDNA support.
    A production kernel would include:
    - Shared memory tiling
    - Double buffering
    - Optimized memory access patterns
    - Better thread/warp scheduling
    """

    # Guard against RDNA2 and earlier
    constrained[
        not _is_amd_rdna2(),
        "RDNA2 and earlier require fallback paths that are not yet implemented"
    ]()

    # Get matrix dimensions
    var M = c.shape[0]()
    var N = c.shape[1]()
    var K = a.shape[1]()

    # Calculate thread position
    var tid = thread_idx.x
    var bid = block_idx.x

    # Each warp handles a 16x16 tile of C
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE

    # Calculate tile coordinates
    var tile_m = (bid // (N // WMMA_N)) * WMMA_M
    var tile_n = (bid % (N // WMMA_N)) * WMMA_N

    # Check bounds
    if tile_m >= M or tile_n >= N:
        return

    # Initialize accumulator
    var c_frag = SIMD[get_accum_type[c_type](), 8](0)

    # Loop over K dimension in WMMA_K tiles
    for k_tile in range(0, K, WMMA_K):
        # Load A tile (16x16) - simplified, would use shared memory in production
        # Each thread in Wave32 gets 8 elements for RDNA WMMA
        var a_frag = SIMD[a_type, 8](0)
        if tile_m < M and k_tile < K:
            # Each thread loads part of the tile
            @parameter
            for i in range(8):
                if lane_id == i // 2:
                    var row = tile_m + (i % WMMA_M)
                    var col = k_tile + (i % WMMA_K)
                    if row < M and col < K:
                        a_frag[i] = a[row, col][0]

        # Load B tile (16x16)
        var b_frag = SIMD[b_type, 8](0)
        if k_tile < K and tile_n < N:
            @parameter
            for i in range(8):
                if lane_id == i // 2:
                    if transpose_b:
                        var row = tile_n + (i % WMMA_N)
                        var col = k_tile + (i % WMMA_K)
                        if row < N and col < K:
                            b_frag[i] = b[row, col][0]
                    else:
                        var row = k_tile + (i % WMMA_K)
                        var col = tile_n + (i % WMMA_N)
                        if row < K and col < N:
                            b_frag[i] = b[row, col][0]

        # Perform WMMA operation
        c_frag = wmma_rdna[get_accum_type[c_type](), a_type, b_type](a_frag, b_frag, c_frag)

    # Store result back to C
    @parameter
    for i in range(8):
        if lane_id == i:
            var row = tile_m + (i // (WMMA_N // 2))
            var col = tile_n + (i % (WMMA_N // 2)) * 2
            if row < M and col < N:
                if elementwise_lambda_fn:
                    alias epilogue_fn = elementwise_lambda_fn.value()
                    var coords = IndexList[2](row, col)
                    var val = SIMD[c_type, 1](c_frag[i].cast[c_type]())
                    epilogue_fn(coords, val)
                else:
                    c[row, col] = c_frag[i].cast[c_type]()


fn gemm_kernel_rdna[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    c_layout_int_type: DType,
    a_layout_int_type: DType,
    b_layout_int_type: DType,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        layout_int_type=c_layout_int_type,
        linear_idx_type=c_linear_idx_type,
    ],
    a: LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        layout_int_type=a_layout_int_type,
        linear_idx_type=a_linear_idx_type,
    ],
    b: LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
        layout_int_type=b_layout_int_type,
        linear_idx_type=b_linear_idx_type,
    ],
):
    """RDNA-optimized GEMM kernel entry point.

    Routes to appropriate implementation based on data types and configuration.
    Currently uses naive implementation as a starting point.
    """

    # For now, use the naive implementation
    # TODO: Add optimized implementations for different configurations
    gemm_kernel_rdna_naive[
        c_type,
        c_layout,
        a_type,
        a_layout,
        b_type,
        b_layout,
        transpose_b,
        c_layout_int_type,
        a_layout_int_type,
        b_layout_int_type,
        c_linear_idx_type,
        a_linear_idx_type,
        b_linear_idx_type,
        config,
        elementwise_lambda_fn,
    ](c, a, b)