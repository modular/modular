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

import std.math as math

from std.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_rope_gpu
from std.testing import assert_almost_equal
from std.utils.numerics import get_accum_type

from std.utils.index import Index, IndexList


def compute_rms_norm_rope_ref[
    dtype: DType, cos_sin_dtype: DType
](
    input_h: UnsafePointer[Scalar[dtype], _],
    gamma_h: UnsafePointer[Scalar[dtype], _],
    cos_h: UnsafePointer[Scalar[cos_sin_dtype], _],
    sin_h: UnsafePointer[Scalar[cos_sin_dtype], _],
    output_ref: UnsafePointer[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
) raises:
    """CPU reference: RMS norm followed by RoPE rotation."""
    comptime accum_type = get_accum_type[dtype]()

    var half_cols = cols // 2

    for r in range(rows):
        # Compute RMS norm factor for this row.
        var sum_sq = Scalar[accum_type](0)
        for c in range(cols):
            var v = input_h[r * cols + c].cast[accum_type]()
            sum_sq += v * v
        var rms = math.sqrt(
            sum_sq / Scalar[accum_type](cols) + epsilon.cast[accum_type]()
        )
        var inv_rms = Scalar[accum_type](1) / rms

        # Apply norm with gamma (multiply_before_cast=True: all ops in accum_type).
        var normed = alloc[Scalar[dtype]](cols)
        for c in range(cols):
            var v = input_h[r * cols + c].cast[accum_type]()
            var g = (
                gamma_h[c].cast[accum_type]() + weight_offset.cast[accum_type]()
            )
            normed[c] = (v * inv_rms * g).cast[dtype]()

        # Apply RoPE: rotated[c] = -normed[c+half] if c < half else normed[c-half]
        for c in range(cols):
            var normed_val = normed[c].cast[accum_type]()
            var cos_val = cos_h[r * cols + c].cast[accum_type]()
            var sin_val = sin_h[r * cols + c].cast[accum_type]()
            var paired_c = c + half_cols if c < half_cols else c - half_cols
            var paired_val = normed[paired_c].cast[accum_type]()
            var rotated = -paired_val if c < half_cols else paired_val
            output_ref[r * cols + c] = (
                normed_val * cos_val + rotated * sin_val
            ).cast[dtype]()

        normed.free()


def run_rms_norm_rope_gpu[
    rank: Int, //, dtype: DType, cos_sin_dtype: DType = dtype
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_rms_norm_rope_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var cos_h = alloc[Scalar[cos_sin_dtype]](rows * cols)
    var sin_h = alloc[Scalar[cos_sin_dtype]](rows * cols)
    var result_gpu = alloc[Scalar[dtype]](rows * cols)
    var result_ref = alloc[Scalar[dtype]](rows * cols)

    # Initialize with diverse deterministic values.
    for i in range(rows * cols):
        data_h[i] = (Float64((i % 37) - 18) / 10.0).cast[dtype]()
        cos_h[i] = (Float64((i % 19) - 9) / 10.0).cast[cos_sin_dtype]()
        sin_h[i] = (Float64((i % 23) - 11) / 10.0).cast[cos_sin_dtype]()

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    # Compute CPU reference.
    compute_rms_norm_rope_ref[dtype, cos_sin_dtype](
        data_h,
        gamma_h,
        cos_h,
        sin_h,
        result_ref,
        rows,
        cols,
        epsilon,
        weight_offset,
    )

    # Run GPU kernel.
    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var cos_d = ctx.enqueue_create_buffer[cos_sin_dtype](rows * cols)
    var sin_d = ctx.enqueue_create_buffer[cos_sin_dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(cos_d, cos_h)
    ctx.enqueue_copy(sin_d, sin_h)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Idx(cols)))
    var cos_vals = TileTensor(cos_d, row_major(Coord(shape)))
    var sin_vals = TileTensor(sin_d, row_major(Coord(shape)))

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    def input_fn[
        width: Int, _rank: Int, alignment: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    @__copy_capture(cos_vals)
    @parameter
    def cos_fn[
        width: Int, _rank: Int, alignment: Int
    ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
        var idx = cos_vals.layout(Coord(coords))
        return cos_vals.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    @__copy_capture(sin_vals)
    @parameter
    def sin_fn[
        width: Int, _rank: Int, alignment: Int
    ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
        var idx = sin_vals.layout(Coord(coords))
        return sin_vals.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    def output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = output_buf.layout(Coord(coords))
        output_buf.raw_store[width=width, alignment=alignment](idx, val)

    rms_norm_rope_gpu[
        input_fn, cos_fn, sin_fn, output_fn, multiply_before_cast=True
    ](shape, gamma, epsilon, weight_offset, cos_vals, sin_vals, ctx)

    ctx.enqueue_copy(result_gpu, output_d)
    ctx.synchronize()

    # Compare GPU result with CPU reference.
    for i in range(rows * cols):
        assert_almost_equal(result_gpu[i], result_ref[i], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = cos_d
    _ = sin_d
    _ = output_d

    data_h.free()
    gamma_h.free()
    cos_h.free()
    sin_h.free()
    result_gpu.free()
    result_ref.free()


def main() raises:
    with DeviceContext() as ctx:
        # Basic shapes
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(2, 4))
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(3, 8))
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(5, 16))
        # Higher rank
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(2, 3, 8))
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(1, 5, 6, 16))
        # Larger cols
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(4, 128))
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(2, 256))
        run_rms_norm_rope_gpu[DType.float32](ctx, Index(2, 4096))
        # bfloat16
        # BFloat16 accumulates rounding from both RMSNorm and RoPE; use 5%.
        run_rms_norm_rope_gpu[DType.bfloat16](ctx, Index(3, 128), rtol=5e-2)
        run_rms_norm_rope_gpu[DType.bfloat16](ctx, Index(2, 4096), rtol=5e-2)
        # Mixed cos/sin dtype
        run_rms_norm_rope_gpu[DType.bfloat16, cos_sin_dtype=DType.float32](
            ctx, Index(2, 128), rtol=5e-2
        )
