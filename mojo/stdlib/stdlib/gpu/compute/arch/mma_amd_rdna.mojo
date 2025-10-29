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
"""AMD RDNA3/4 WMMA implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for AMD RDNA3 and RDNA4 consumer GPUs
using the WMMA (Wave Matrix Multiply Accumulate) instructions.

Reference: https://gpuopen.com/learn/wmma_on_rdna3/
"""

from sys import llvm_intrinsic
from memory import bitcast
from gpu import lane_id

# Import helper functions from parent module
from ..mma import _has_type, _has_shape, _unsupported_mma_op


# ===----------------------------------------------------------------------=== #
# RDNA Matrix Loaders
# ===----------------------------------------------------------------------=== #


@always_inline
fn _load_matrix_a_amd_rdna[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 16]:
    """RDNA-specific implementation: loads 16 FP16 elements per thread."""
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_x = lane & 15
    var a = SIMD[DType.float16, 16]()

    @parameter
    for i in range(16):
        var a_idx = ldm * (tile_row + Int(thread_x)) + tile_col + i
        a[i] = a_ptr[a_idx]

    return a


@always_inline
fn _load_matrix_a_amd_rdna[
    m: Int, n: Int, k: Int
](
    a_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 16]:
    """RDNA-specific implementation: loads 16 BF16 elements per thread."""
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_x = lane & 15
    var a = SIMD[DType.bfloat16, 16]()

    @parameter
    for i in range(16):
        var a_idx = ldm * (tile_row + Int(thread_x)) + tile_col + i
        a[i] = a_ptr[a_idx]

    return a


@always_inline
fn _load_matrix_b_amd_rdna[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 16]:
    """RDNA-specific implementation: loads 16 FP16 elements per thread."""
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_y = lane & 15
    var b = SIMD[DType.float16, 16]()

    @parameter
    for i in range(16):
        var b_idx = ldm * (tile_row + i) + tile_col + Int(thread_y)
        b[i] = b_ptr[b_idx]

    return b


@always_inline
fn _load_matrix_b_amd_rdna[
    m: Int, n: Int, k: Int
](
    b_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 16]:
    """RDNA-specific implementation: loads 16 BF16 elements per thread."""
    constrained[m == 16 and n == 16 and k == 16]()
    var lane = lane_id()
    var thread_y = lane & 15
    var b = SIMD[DType.bfloat16, 16]()

    @parameter
    for i in range(16):
        var b_idx = ldm * (tile_row + i) + tile_col + Int(thread_y)
        b[i] = b_ptr[b_idx]

    return b


@always_inline
fn load_matrix_a_amd_rdna16x16x16(
    a_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 16]:
    """Loads 16×16×16 matrix A tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous FP16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 16 FP16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use the generic
        load_matrix_a_amd() which returns SIMD[4].
    """
    return _load_matrix_a_amd_rdna[16, 16, 16](a_ptr, tile_row, tile_col, ldm)


@always_inline
fn load_matrix_a_amd_rdna16x16x16(
    a_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 16]:
    """Loads 16×16×16 matrix A tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous BF16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 16 BF16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use the generic
        load_matrix_a_amd() which returns SIMD[4].
    """
    return _load_matrix_a_amd_rdna[16, 16, 16](a_ptr, tile_row, tile_col, ldm)


@always_inline
fn load_matrix_b_amd_rdna16x16x16(
    b_ptr: UnsafePointer[Float16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.float16, 16]:
    """Loads 16×16×16 matrix B tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous FP16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 16 FP16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use the generic
        load_matrix_b_amd() which returns SIMD[4].
    """
    return _load_matrix_b_amd_rdna[16, 16, 16](b_ptr, tile_row, tile_col, ldm)


@always_inline
fn load_matrix_b_amd_rdna16x16x16(
    b_ptr: UnsafePointer[BFloat16],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[DType.bfloat16, 16]:
    """Loads 16×16×16 matrix B tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous BF16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 16 BF16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use the generic
        load_matrix_b_amd() which returns SIMD[4].
    """
    return _load_matrix_b_amd_rdna[16, 16, 16](b_ptr, tile_row, tile_col, ldm)


# ===----------------------------------------------------------------------=== #
# RDNA WMMA Intrinsics
# ===----------------------------------------------------------------------=== #


@always_inline
fn _mma_wmma_rdna(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs AMD RDNA3+ WMMA (Wave Matrix Multiply-Accumulate) operations.

    This function implements matrix multiply-accumulate operations for AMD RDNA3+
    consumer GPUs using WMMA instructions. WMMA was introduced in RDNA3 and is not
    available on RDNA1/2 hardware.

    Supported operations by RDNA generation:

    RDNA3+ (all operations):
        - F32 = F16 * F16 + F32 (16x16x16 shape)
        - F32 = BF16 * BF16 + F32 (16x16x16 shape)

    RDNA4 additional operations:
        - F32 = FP8 * FP8 + F32 (16x16x32 shape, native hardware support)

    Args:
        d: Output accumulator SIMD vector (modified in-place).
        a: First input matrix as SIMD vector.
        b: Second input matrix as SIMD vector.
        c: Accumulator matrix as SIMD vector.

    RDNA WMMA Fragment Requirements:
        RDNA WMMA is a wave-cooperative operation where each lane holds a fragment
        of the full matrix. For the 16×16×16 WMMA operation (M×N×K dimensions):

        Matrix Dimensions:
            - Matrix A: 16×16 (M×K) = 256 fp16/bf16 elements total
            - Matrix B: 16×16 (K×N) = 256 fp16/bf16 elements total
            - Matrix C/D: 16×16 (M×N) = 256 fp32 elements total

        Per-Lane Fragment Sizes (wave32 mode):
            - A fragment: 16 fp16/bf16 elements (full K=16 dimension per lane)
            - B fragment: 16 fp16/bf16 elements (full K=16 dimension per lane)
            - C/D fragment: 8 fp32 elements (M×N=256 distributed: 256/32 lanes = 8)

        This means the SIMD sizes passed to mma() for wave32 must be:
            - a.size = 16, b.size = 16, c.size = 8, d.size = 8

        LLVM Intrinsic Signatures:
            - FP16: llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half>, <16 x half>, <8 x float>)
            - BF16: llvm.amdgcn.wmma.f32.16x16x16.bf16(<16 x i16>, <16 x i16>, <8 x float>)

            Note: BF16 fragments must be bitcast to <16 x i16> (packed BF16 as int16)
            before calling the intrinsic, not passed as <16 x bfloat>.

    References:
        - RDNA3 WMMA: https://gpuopen.com/learn/wmma_on_rdna3/
        - RDNA3 ISA: AMD RDNA3 Shader Instruction Set Architecture
        - RDNA4 ISA: AMD RDNA4 Instruction Set Architecture
    """

    @parameter
    fn get_intrinsic_name() -> String:
        # ===------------------------------------------------------------------===#
        # F32 = F16 * F16 + F32 (16x16x16)
        # Or
        # F32 = BF16 * BF16 + F32 (16x16x16)
        # ===------------------------------------------------------------------===#
        @parameter
        if _has_type[
            (DType.float16, DType.float16, DType.float32, DType.float32)
        ](a.dtype, b.dtype, c.dtype, d.dtype) or _has_type[
            (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
        ](
            a.dtype, b.dtype, c.dtype, d.dtype
        ):

            @parameter
            if _has_shape[(16, 16, 8, 8)](a.size, b.size, c.size, d.size):
                alias type_name = "f16" if a.dtype is DType.float16 else "bf16"
                return "llvm.amdgcn.wmma.f32.16x16x16." + type_name
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        elif a.dtype in [
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
        ] or b.dtype in [
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
        ]:
            # FP8 placeholder for RDNA4
            _unsupported_mma_op(d, a, b, c)
            return ""
        else:
            _unsupported_mma_op(d, a, b, c)
            return ""

    @parameter
    if a.size == 16 and b.size == 16 and c.size == 8 and d.size == 8:
        alias intrinsic_name = get_intrinsic_name()

        @parameter
        if a.dtype is DType.bfloat16:
            var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                bitcast[DType.int16, 16](a), bitcast[DType.int16, 16](b), c
            )
            d = rebind[type_of(d)](r)
        else:
            var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](a, b, c)
            d = rebind[type_of(d)](r)
    else:
        var r = llvm_intrinsic[get_intrinsic_name(), SIMD[c.dtype, c.size]](
            a, b, c
        )
        d = rebind[type_of(d)](r)
