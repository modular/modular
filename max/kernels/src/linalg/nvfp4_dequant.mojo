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
"""NVFP4 weight dequantization.

Converts packed NVFP4 weights (uint8, two E2M1 values per byte, low nibble
first) with `float8_e4m3fn` block scales and a tensor-wide float32 scale into
bfloat16 or float32:

    out[r, c] = (e2m1(nibble) * |block_scale[r, c // 16]|) * tensor_sf

`tensor_sf` is the *decode* scale, i.e. the checkpoint's `weight_scale_2`, the
same value the NVFP4 matmul applies to its output. It is not the inverted
factor the quantize op takes. The multiplication order matches the host-side
reference (`max.pipelines.weights.fp4_quantization.dequantize_nvfp4`): the
E2M1 value times the block scale is exact in float32, so only the final
multiply rounds, and float32 output is bit-identical to the reference.

Two block-scale layouts are accepted, selected by the rank of `scales`:

- Rank 2, `[N, K // 16]`: the checkpoint layout.
- Rank 5, `[ceildiv(N, 128), ceildiv(K, 64), 32, 4, 4]`: the SM100 TCGEN05
  interleaved layout that `quantize_dynamic_block_scaled` emits on SM100/SM120.

E4M3 NaN scale codes are not special-cased: they propagate as NaN.

The kernel is an `elementwise` over `[N, K]` with the scale-block width as the
SIMD width, so one thread dequantizes one 16-element scale block: one 8-byte
packed load, one scale load, and one contiguous store of 16 outputs. Occupancy
does not depend on `K`, and consecutive threads touch consecutive bytes of
every tensor.

GPU only: the graph compiler does not support FP8 tensors on a CPU device, so
there is no way to reach a CPU path from a graph.
"""

from std.math import ceildiv
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from layout import TileTensor
from layout.coord import Coord
from .fp4_utils import (
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_K_GROUP_SIZE,
    decode_e2m1_to_f32,
    get_scale_factor,
)


@inline(.always)
def _load_block_scale[
    scales_dtype: DType
](scales: TileTensor[scales_dtype, ...], row: Int, block_col: Int) -> Scalar[
    scales_dtype
]:
    """Loads the scale of the 16-element block `(row, block_col)`.

    Parameters:
        scales_dtype: Element type of the scales tensor.

    Args:
        scales: Rank-2 `[N, K // 16]` or rank-5 interleaved block scales.
        row: Row of the weight.
        block_col: Index of the 16-element block along `K`.

    Returns:
        The raw (not yet widened or abs'd) block scale.
    """
    comptime if scales.flat_rank >= 5:
        return get_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
            scales, row, block_col * NVFP4_SF_VECTOR_SIZE
        )
    else:
        return rebind[Scalar[scales_dtype]](scales.load((row, block_col)))


@inline(.always)
def _dequant_nvfp4_block[
    out_dtype: DType, in_dtype: DType, scales_dtype: DType
](
    output: TileTensor[mut=True, out_dtype, ...],
    input: TileTensor[in_dtype, ...],
    scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    row: Int,
    block_col: Int,
):
    """Dequantizes one 16-element scale block. Runs inside the GPU kernel.

    Parameters:
        out_dtype: Output element type (bfloat16 or float32).
        in_dtype: Packed input element type (uint8).
        scales_dtype: Scale element type (float8_e4m3fn).

    Args:
        output: Output weight `[N, K]`.
        input: Packed weight `[N, K // 2]`.
        scales: Rank-2 or rank-5 block scales.
        tensor_sf: Tensor-wide decode scale (`weight_scale_2`).
        row: Row of the weight.
        block_col: Index of the 16-element block along `K`.
    """
    comptime assert in_dtype == .uint8, "packed input must be uint8"
    comptime BYTES_PER_BLOCK = NVFP4_SF_VECTOR_SIZE // 2

    var col = block_col * NVFP4_SF_VECTOR_SIZE
    var packed = input.load[BYTES_PER_BLOCK]((row, col // 2))

    # Low nibble is the even element, high nibble the odd one, so interleaving
    # the two nibble vectors restores element order.
    var lo = (packed & 0x0F).cast[.uint16]()
    var hi = (packed >> 4).cast[.uint16]()
    var values = decode_e2m1_to_f32(lo.interleave(hi))

    var block_scale = abs(
        _load_block_scale(scales, row, block_col).cast[.float32]()
    )

    # (e2m1 * |scale|) is exact in float32; only the tensor scale rounds.
    var scaled = (values * block_scale) * tensor_sf
    output.store((row, col), scaled.cast[out_dtype]())


@inline(.always)
def dequant_nvfp4[
    target: StaticString
](
    ctx: DeviceContext,
    output: TileTensor[mut=True, ...],
    input: TileTensor,
    scales: TileTensor,
    tensor_sf: Float32,
    num_rows: Int,
    num_cols: Int,
) raises:
    """Dequantize NVFP4 packed weights to bfloat16 or float32.

    Parameters:
        target: The compilation target. Must be a GPU.

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor `[num_rows, num_cols]` of bfloat16 or float32.
        input: Packed weights `[num_rows, num_cols // 2]` of uint8.
        scales: Block scales of `float8_e4m3fn`: rank 2
            `[num_rows, num_cols // 16]`, or the rank-5 SM100 interleaved
            layout `[ceildiv(num_rows, 128), ceildiv(num_cols, 64), 32, 4, 4]`.
        tensor_sf: Tensor-wide decode scale (the checkpoint's
            `weight_scale_2`).
        num_rows: Number of rows (N dimension for weights).
        num_cols: Number of columns (K dimension, unpacked). Must be a
            multiple of 16.

    Raises:
        If `num_cols` is not a multiple of 16, or if the shapes of `output`,
        `input` and `scales` do not agree with `num_rows` and `num_cols`. The
        kernels read scales on trust, so a short scale tensor would otherwise
        be an out-of-bounds read.
    """
    comptime assert is_gpu[target](), "NVFP4 dequant only supports GPUs"

    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype in (
        DType.bfloat16,
        DType.float32,
    ), "output must be bfloat16 or float32"
    comptime assert (
        scales_dtype == .float8_e4m3fn
    ), "scales must be float8_e4m3fn"
    comptime assert in_dtype == .uint8, "input must be uint8 (packed FP4)"
    comptime assert (
        scales.flat_rank == 2 or scales.flat_rank == 5
    ), "scales must be rank 2 or the rank-5 interleaved layout"

    if num_rows == 0 or num_cols == 0:
        return

    if num_cols % NVFP4_SF_VECTOR_SIZE != 0:
        raise Error(
            "NVFP4 dequant needs a column count that is a multiple of 16, got "
            + String(num_cols)
        )
    if Int(output.dim[0]()) != num_rows or Int(output.dim[1]()) != num_cols:
        raise Error("NVFP4 dequant output shape does not match [rows, cols]")
    if Int(input.dim[0]()) != num_rows or Int(input.dim[1]()) * 2 != num_cols:
        raise Error("NVFP4 dequant packed input must be [rows, cols // 2]")
    comptime if scales.flat_rank == 5:
        if Int(scales.dim[0]()) != ceildiv(num_rows, SF_MN_GROUP_SIZE) or Int(
            scales.dim[1]()
        ) != ceildiv(num_cols, SF_K_GROUP_SIZE[NVFP4_SF_VECTOR_SIZE]):
            raise Error(
                "NVFP4 dequant rank-5 scales must be [ceildiv(rows, 128),"
                " ceildiv(cols, 64), 32, 4, 4]"
            )
    else:
        if (
            Int(scales.dim[0]()) != num_rows
            or Int(scales.dim[1]()) * NVFP4_SF_VECTOR_SIZE != num_cols
        ):
            raise Error(
                "NVFP4 dequant rank-2 scales must be [rows, cols // 16]"
            )

    # The tensors are captured by the closure below, which runs on the GPU.
    var out_tt = output.as_unsafe_any_origin()
    var in_tt = input.as_unsafe_any_origin()
    var scales_tt = scales.as_unsafe_any_origin()

    # One SIMD vector is one scale block, so each thread dequantizes a whole
    # block. `elementwise` flattens the `[rows, cols]` space, so consecutive
    # threads touch consecutive bytes of every tensor whatever `cols` is.
    # `cols` is a multiple of 16 (checked above), so there is no scalar
    # tail; the `width` argument is the block size whenever the body runs.
    @inline(.always)
    def dequant[width: Int, alignment: Int = 1](idx: Coord) {var}:
        _dequant_nvfp4_block(
            out_tt,
            in_tt,
            scales_tt,
            tensor_sf,
            Int(idx[0]),
            Int(idx[1]) // NVFP4_SF_VECTOR_SIZE,
        )

    elementwise[
        NVFP4_SF_VECTOR_SIZE, target=target, _trace_description="dequant_nvfp4"
    ](dequant, (num_rows, num_cols), ctx)
