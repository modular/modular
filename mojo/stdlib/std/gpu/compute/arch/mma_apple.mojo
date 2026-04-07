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
"""Apple Silicon MMA implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for Apple M5 GPUs using the
simdgroup_matrix hardware instructions (Metal 4.0 / AIR 2.8.0).

Supported operations:
- Float multiply-accumulate: {F16, BF16, F32} inputs, F32 accumulator
- Integer widening multiply-accumulate: {I8, U8} inputs, I32/U32 accumulator
"""

from std.sys import llvm_intrinsic

from std.gpu import lane_id

# Import helper functions from parent module
from ..mma import _has_shape, _unsupported_mma_op


@always_inline
def _apple_frag_layout(tid: Int) -> Tuple[Int, Int]:
    """Returns (row_lo, col_base) for the given simdgroup thread."""
    return (
        ((tid & 7) // 2) + ((tid & 16) >> 2),
        ((tid & 1) << 2) + (tid & 8),
    )


@always_inline
def apple_mma_load[
    dtype: DType,
](
    ptr: UnsafePointer[Scalar[dtype], ...],
    row_stride: Int,
    col_stride: Int = 1,
) -> SIMD[dtype, 8]:
    """Loads a 16x16 matrix fragment for the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 16x16 tile.
        row_stride: Distance between consecutive rows in the buffer.
        col_stride: Distance between consecutive columns within a row.

    Returns:
        SIMD vector of 8 elements for this thread's fragment.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    # If row-elems are contiguous, vectorize load
    if col_stride == 1:
        var lo = (ptr + row_lo * row_stride + col_base).load[width=4]()
        var hi = (ptr + (row_lo + 8) * row_stride + col_base).load[width=4]()
        return lo.join(hi)
    else:
        var frag = SIMD[dtype, 8]()
        for el in range(8):
            var row = row_lo + (Int(el > 3) * 8)
            var col = col_base + (el & 3)
            frag[el] = ptr[row * row_stride + col * col_stride]
        return frag


@always_inline
def apple_mma_store[
    dtype: DType,
](
    ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    row_stride: Int,
    frag: SIMD[dtype, 8],
    col_stride: Int = 1,
):
    """Stores a 16x16 matrix fragment from the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 16x16 tile.
        row_stride: Distance between consecutive rows in the buffer.
        frag: SIMD vector of 8 elements to store.
        col_stride: Distance between consecutive columns within a row.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    # If row-elems are contiguous, vectorize store
    if col_stride == 1:
        (ptr + row_lo * row_stride + col_base).store(frag.slice[4, offset=0]())
        (ptr + (row_lo + 8) * row_stride + col_base).store(
            frag.slice[4, offset=4]()
        )
    else:
        for el in range(8):
            var row = row_lo + (Int(el > 3) * 8)
            var col = col_base + (el & 3)
            ptr[row * row_stride + col * col_stride] = frag[el]


@always_inline
def _mma_apple(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    _mma_apple_transposable(d, a, b, c, False, False)


@always_inline
def _mma_apple_transposable(
    mut d: SIMD,
    a: SIMD,
    b: SIMD,
    c: SIMD,
    transpose_a: Bool,
    transpose_b: Bool,
):
    """Variant of `_mma_apple` with runtime transpose flags."""
    comptime assert _has_shape[8](
        a.size, b.size, c.size, d.size
    ), "Apple MMA requires 8-element fragments (16x16 / 32 threads)"

    comptime assert d.dtype in (
        DType.float32,
        DType.int32,
        DType.uint32,
    ) and c.dtype in (
        DType.float32,
        DType.int32,
        DType.uint32,
    ), "Apple MMA accumulator (C and D) must be 32-bit"

    comptime assert c.dtype == d.dtype, "Apple MMA C and D types must match"

    comptime _valid_float_input = (
        a.dtype in (DType.float16, DType.bfloat16, DType.float32)
        and b.dtype in (DType.float16, DType.bfloat16, DType.float32)
    )

    comptime _valid_int_input = (
        a.dtype in (DType.int8, DType.uint8)
        and b.dtype in (DType.int8, DType.uint8)
    )

    comptime if _valid_float_input and d.dtype == DType.float32:
        d = rebind[type_of(d)](
            llvm_intrinsic[
                "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
                SIMD[DType.float32, 8],
            ](a, transpose_a, b, transpose_b, c)
        )

    elif _valid_int_input and d.dtype in (DType.int32, DType.uint32):
        d = rebind[type_of(d)](
            llvm_intrinsic[
                "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
                SIMD[d.dtype, 8],
            ](a, transpose_a, b, transpose_b, c)
        )

    else:
        _unsupported_mma_op(d, a, b, c)
