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
"""MXFP4 and MXFP8 weight dequantization (E8M0 block-32 scales).

Both OCP microscaling formats share the scale structure: one `float8_e8m0fnu`
scale (a pure power of two) per 32 consecutive elements along `K`, and no
tensor-wide scale. They differ only in the element encoding, which is selected
by the dtype of `input`:

- MXFP4: `uint8`, two E2M1 values per byte (low nibble first), `[N, K // 2]`.
- MXFP8: `float8_e4m3fn`, one value per element, `[N, K]`.

The output is bfloat16 or float32:

    out[r, c] = element[r, c] * 2^(scale_code[r, c // 32] - 127)

Because the scale is a power of two the multiply is exact in float32 (barring
overflow), so float32 output is bit-identical to a host reference. bfloat16
output rounds once, at the final cast.

Two block-scale layouts are accepted, selected by the rank of `scales`:

- Rank 2, `[N, K // 32]`: the checkpoint layout.
- Rank 5, `[ceildiv(N, 128), ceildiv(K, 128), 32, 4, 4]`: the SM100/SM120
  TCGEN05 interleaved layout. With 32-element blocks one atom spans 128
  columns, so the second dimension differs from NVFP4's `ceildiv(K, 64)`.

An E8M0 NaN code (0xFF) propagates as NaN.

The kernel is an `elementwise` over `[N, K]` with the scale-block width as the
SIMD width, so one thread dequantizes one 32-element scale block, occupancy
does not depend on `K`, and consecutive threads touch consecutive bytes of
every tensor. This is separate from the older H100-only `mxfp4_dequant`
kernel, which stays the path for bf16/fp8 output with rank-2 scales on GPU.

GPU only: the graph compiler does not support FP8 or E8M0 tensors on a CPU
device, so there is no way to reach a CPU path from a graph.
"""

from std.math import ceildiv
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from layout import TileTensor
from layout.coord import Coord
from .fp4_utils import (
    MXFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_K_GROUP_SIZE,
    decode_e2m1_to_f32,
    get_scale_factor,
)

# MXFP4 and MXFP8 share the 32-element scale block.
comptime _MX_BLOCK = MXFP4_SF_VECTOR_SIZE


@inline(.always)
def _load_block_scale[
    scales_dtype: DType
](scales: TileTensor[scales_dtype, ...], row: Int, block_col: Int) -> Scalar[
    scales_dtype
]:
    """Loads the scale of the 32-element block `(row, block_col)`.

    Parameters:
        scales_dtype: Element type of the scales tensor.

    Args:
        scales: Rank-2 `[N, K // 32]` or rank-5 interleaved block scales.
        row: Row of the weight.
        block_col: Index of the 32-element block along `K`.

    Returns:
        The raw (not yet widened) E8M0 block scale.
    """
    comptime if scales.flat_rank >= 5:
        return get_scale_factor[SF_VECTOR_SIZE=_MX_BLOCK](
            scales, row, block_col * _MX_BLOCK
        )
    else:
        return rebind[Scalar[scales_dtype]](scales.load((row, block_col)))


@inline(.always)
def _dequant_mx_block[
    out_dtype: DType, in_dtype: DType, scales_dtype: DType
](
    output: TileTensor[mut=True, out_dtype, ...],
    input: TileTensor[in_dtype, ...],
    scales: TileTensor[scales_dtype, ...],
    row: Int,
    block_col: Int,
):
    """Dequantizes one 32-element scale block. Runs inside the GPU kernel.

    Parameters:
        out_dtype: Output element type (bfloat16 or float32).
        in_dtype: Input element type: uint8 (MXFP4) or float8_e4m3fn (MXFP8).
        scales_dtype: Scale element type (float8_e8m0fnu).

    Args:
        output: Output weight `[N, K]`.
        input: Packed MXFP4 `[N, K // 2]` or MXFP8 `[N, K]` weight.
        scales: Rank-2 or rank-5 block scales.
        row: Row of the weight.
        block_col: Index of the 32-element block along `K`.
    """
    var col = block_col * _MX_BLOCK

    var values: SIMD[.float32, _MX_BLOCK]
    comptime if in_dtype == .uint8:
        # Low nibble is the even element, high nibble the odd one, so
        # interleaving the two nibble vectors restores element order.
        var packed = input.load[_MX_BLOCK // 2]((row, col // 2))
        var lo = (packed & 0x0F).cast[.uint16]()
        var hi = (packed >> 4).cast[.uint16]()
        values = decode_e2m1_to_f32(lo.interleave(hi))
    else:
        comptime assert (
            in_dtype == .float8_e4m3fn
        ), "MXFP8 input must be float8_e4m3fn"
        values = input.load[_MX_BLOCK]((row, col)).cast[.float32]()

    var scale = _load_block_scale(scales, row, block_col).cast[.float32]()

    # The scale is a power of two, so this multiply is exact in float32.
    output.store((row, col), (values * scale).cast[out_dtype]())


@inline(.always)
def dequant_mx[
    target: StaticString
](
    ctx: DeviceContext,
    output: TileTensor[mut=True, ...],
    input: TileTensor,
    scales: TileTensor,
    num_rows: Int,
    num_cols: Int,
) raises:
    """Dequantize MXFP4 or MXFP8 weights to bfloat16 or float32.

    The format follows the dtype of `input`: `uint8` is packed MXFP4,
    `float8_e4m3fn` is MXFP8.

    Parameters:
        target: The compilation target. Must be a GPU.

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor `[num_rows, num_cols]` of bfloat16 or float32.
        input: Packed MXFP4 weights `[num_rows, num_cols // 2]` of uint8, or
            MXFP8 weights `[num_rows, num_cols]` of float8_e4m3fn.
        scales: Block scales of `float8_e8m0fnu`: rank 2
            `[num_rows, num_cols // 32]`, or the rank-5 interleaved layout
            `[ceildiv(num_rows, 128), ceildiv(num_cols, 128), 32, 4, 4]`.
        num_rows: Number of rows (N dimension for weights).
        num_cols: Number of columns (K dimension, unpacked). Must be a
            multiple of 32.

    Raises:
        If `num_cols` is not a multiple of 32, or if the shapes of `output`,
        `input` and `scales` do not agree with `num_rows` and `num_cols`. The
        kernels read scales on trust, so a short scale tensor would otherwise
        be an out-of-bounds read.
    """
    comptime assert is_gpu[target](), "MX dequant only supports GPUs"

    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype in (
        DType.bfloat16,
        DType.float32,
    ), "output must be bfloat16 or float32"
    comptime assert (
        scales_dtype == .float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype in (
        DType.uint8,
        DType.float8_e4m3fn,
    ), "input must be uint8 (packed MXFP4) or float8_e4m3fn (MXFP8)"
    comptime assert (
        scales.flat_rank == 2 or scales.flat_rank == 5
    ), "scales must be rank 2 or the rank-5 interleaved layout"

    comptime packing = 2 if in_dtype == DType.uint8 else 1

    if num_rows == 0 or num_cols == 0:
        return

    if num_cols % _MX_BLOCK != 0:
        raise Error(
            "MX dequant needs a column count that is a multiple of 32, got "
            + String(num_cols)
        )
    if Int(output.dim[0]()) != num_rows or Int(output.dim[1]()) != num_cols:
        raise Error("MX dequant output shape does not match [rows, cols]")
    if (
        Int(input.dim[0]()) != num_rows
        or Int(input.dim[1]()) * packing != num_cols
    ):
        raise Error(
            "MX dequant input must be [rows, cols // 2] (MXFP4) or [rows,"
            " cols] (MXFP8)"
        )
    comptime if scales.flat_rank == 5:
        if Int(scales.dim[0]()) != ceildiv(num_rows, SF_MN_GROUP_SIZE) or Int(
            scales.dim[1]()
        ) != ceildiv(num_cols, SF_K_GROUP_SIZE[_MX_BLOCK]):
            raise Error(
                "MX dequant rank-5 scales must be [ceildiv(rows, 128),"
                " ceildiv(cols, 128), 32, 4, 4]"
            )
    else:
        if (
            Int(scales.dim[0]()) != num_rows
            or Int(scales.dim[1]()) * _MX_BLOCK != num_cols
        ):
            raise Error("MX dequant rank-2 scales must be [rows, cols // 32]")

    # The tensors are captured by the closure below, which runs on the GPU.
    var out_tt = output.as_unsafe_any_origin()
    var in_tt = input.as_unsafe_any_origin()
    var scales_tt = scales.as_unsafe_any_origin()

    # One SIMD vector is one scale block, so each thread dequantizes a whole
    # block. `elementwise` flattens the `[rows, cols]` space, so consecutive
    # threads touch consecutive bytes of every tensor whatever `cols` is.
    # `cols` is a multiple of 32 (checked above), so there is no scalar
    # tail; the `width` argument is the block size whenever the body runs.
    @inline(.always)
    def dequant[width: Int, alignment: Int = 1](idx: Coord) {var}:
        _dequant_mx_block(
            out_tt,
            in_tt,
            scales_tt,
            Int(idx[0]),
            Int(idx[1]) // _MX_BLOCK,
        )

    elementwise[_MX_BLOCK, target=target, _trace_description="dequant_mx"](
        dequant, (num_rows, num_cols), ctx
    )
