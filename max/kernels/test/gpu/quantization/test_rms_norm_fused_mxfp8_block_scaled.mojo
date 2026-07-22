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
"""Correctness test for fused RMSNorm + MXFP8 block-scaled quantization.

Compares `rms_norm_fused_quantize_dynamic_block_scaled` (SM100/SM120) against a
host reference that models the unfused pipeline it replaces:

    rms_norm (multiply_before_cast, bf16 store) -> quantize_dynamic_block_scaled[MXFP8]

The reference rounds the normalized value to bf16 before the per-32-element
block max and quantization (matching the bf16 intermediate the unfused path
stores and the standalone quant re-reads), derives the E8M0 block scale as
`(group_max / 448).cast[float8_e8m0fnu]`, and quantizes to E4M3 exactly as
`quantize_dynamic_scaled_fp4fp8_kernel` does. Device scales are read back
through `get_scale_factor`, validating the interleaved rank-5 layout including
the `align_up(rows, 128)` padding-row / padding-column zeroing.

Equivalence caveat: the fused kernel computes the RMSNorm sum-of-squares with a
GPU block (tree) reduction while the host reference sums sequentially, so
`norm_factor` can differ by an f32 ULP. E8M0 block scales are powers of two and
robust to this (asserted bit-exact); a small fraction of E4M3 values sitting
exactly on a rounding boundary may flip, so the E4M3 output is asserted equal
within a small mismatch budget.

RUN STATUS: authored but DEFERRED. Requires a B200 (SM100). Do not run until the
shared GPUs are free.
"""

from std.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from layout._fillers import random
from std.math import align_up, ceildiv, recip, rsqrt
from std.memory import bitcast
from std.utils import IndexList
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)
from linalg.rms_norm_block_scaled import (
    rms_norm_fused_quantize_dynamic_block_scaled,
)

comptime E4M3_MAX = Float32(448.0)


def test_fused_rms_norm_mxfp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext,
    rows: Int,
    cols: Int,
    epsilon: Float32 = 1e-6,
    weight_offset: Float32 = 0.0,
    fp8_mismatch_budget: Float64 = 0.001,
) raises:
    if cols % SF_VECTOR_SIZE != 0:
        raise Error("cols must be a multiple of SF_VECTOR_SIZE (32)")

    comptime SF_GROUP_COLS = SF_VECTOR_SIZE * SF_ATOM_K  # 128 for MXFP8
    var num_col_blocks = cols // SF_VECTOR_SIZE
    var num_sf_col_blocks = align_up(cols, SF_GROUP_COLS) // SF_VECTOR_SIZE
    var num_row_padded = align_up(rows, SF_MN_GROUP_SIZE)
    var weight_offset_id = Scalar[in_dtype](weight_offset)
    var weight_offset_f32 = weight_offset_id.cast[DType.float32]()

    # --- Buffers / layouts (mirrors test_scaled_mxfp8_quantization.mojo). ---
    comptime in_static = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var in_shape = IndexList[2](rows, cols)
    var in_rt = RuntimeLayout[in_static].row_major(in_shape)
    var in_device = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var in_tensor = LayoutTensor[in_dtype, in_static](in_device, in_rt)

    comptime out_static = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var out_device = ctx.enqueue_create_buffer[out_dtype](rows * cols)
    var out_tensor = LayoutTensor[out_dtype, out_static](
        out_device, RuntimeLayout[out_static].row_major(in_shape)
    )

    comptime gamma_static = Layout.row_major(UNKNOWN_VALUE)
    var gamma_device = ctx.enqueue_create_buffer[in_dtype](cols)
    var gamma_tensor = LayoutTensor[in_dtype, gamma_static](
        gamma_device, RuntimeLayout[gamma_static].row_major(IndexList[1](cols))
    )

    var scales_shape = IndexList[5](
        ceildiv(rows, SF_MN_GROUP_SIZE),
        ceildiv(cols, SF_GROUP_COLS),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime scales_static = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K
    )
    var scales_rt = RuntimeLayout[scales_static].row_major(scales_shape)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_shape.flattened_length()
    )
    var scales_tensor = LayoutTensor[scales_dtype, scales_static](
        scales_device, scales_rt
    )

    # Poison scales so padding-zeroing is actually exercised (not left as 0s
    # from a fresh allocation).
    with scales_device.map_to_host() as scales_host:
        for i in range(len(scales_host)):
            scales_host[i] = Scalar[scales_dtype](0)
    with out_device.map_to_host() as out_host:
        for i in range(len(out_host)):
            out_host[i] = Scalar[out_dtype](0)

    # Deterministic gamma; keep host copy for the reference.
    var gamma_vals = List[Scalar[in_dtype]](length=cols, fill=0)
    for i in range(cols):
        gamma_vals[i] = Scalar[in_dtype](0.5 + Float64(i % 11) * 0.1)
    with gamma_device.map_to_host() as gamma_host:
        for i in range(cols):
            gamma_host[i] = gamma_vals[i]

    # Random input.
    with in_device.map_to_host() as in_host:
        var in_host_tensor = LayoutTensor[in_dtype, in_static](in_host, in_rt)
        random(in_host_tensor, min=-2.0, max=2.0)

    # --- Host reference: rms_norm (bf16 store) -> MXFP8 block quant. ---
    var ref_out = List[Scalar[out_dtype]](length=rows * cols, fill=0)
    var ref_scale = List[Scalar[scales_dtype]](
        length=rows * num_col_blocks, fill=0
    )
    var normed = List[Scalar[in_dtype]](length=cols, fill=0)
    with in_device.map_to_host() as in_host:
        var in_host_tensor = LayoutTensor[in_dtype, in_static](in_host, in_rt)
        for r in range(rows):
            var m2 = Float32(0)
            for c in range(cols):
                var v = rebind[Scalar[DType.float32]](
                    in_host_tensor[r, c].cast[DType.float32]()
                )
                m2 += v * v
            var norm_factor = rsqrt(m2 / Float32(cols) + epsilon)
            for c in range(cols):
                var v = rebind[Scalar[DType.float32]](
                    in_host_tensor[r, c].cast[DType.float32]()
                )
                var g = gamma_vals[c].cast[DType.float32]() + weight_offset_f32
                # multiply_before_cast: single bf16 rounding, matching the
                # kernel default and rms_norm_gpu(multiply_before_cast=True).
                normed[c] = (v * norm_factor * g).cast[in_dtype]()
            for blk in range(num_col_blocks):
                var group_max = Float32(0)
                for j in range(SF_VECTOR_SIZE):
                    var a = abs(
                        normed[blk * SF_VECTOR_SIZE + j].cast[DType.float32]()
                    )
                    if a > group_max:
                        group_max = a
                var sf = (group_max * recip(E4M3_MAX)).cast[scales_dtype]()
                ref_scale[r * num_col_blocks + blk] = sf
                var output_scale = Float32(0)
                if group_max != 0:
                    output_scale = recip(sf.cast[DType.float32]())
                for j in range(SF_VECTOR_SIZE):
                    var idx = blk * SF_VECTOR_SIZE + j
                    ref_out[r * cols + idx] = (
                        normed[idx].cast[DType.float32]() * output_scale
                    ).cast[out_dtype]()

    # --- Run the fused kernel. ---
    var in_ptr = in_device.unsafe_ptr()

    @__copy_capture(in_ptr)
    @always_inline
    @parameter
    def input_fn[width: Int](row: Int, col: Int) -> SIMD[in_dtype, width]:
        return in_ptr.load[width=width](row * cols + col)

    rms_norm_fused_quantize_dynamic_block_scaled[
        input_fn,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
    ](
        rows,
        cols,
        lt_to_tt(out_tensor).as_unsafe_any_origin(),
        lt_to_tt(scales_tensor).as_unsafe_any_origin(),
        lt_to_tt(gamma_tensor).as_unsafe_any_origin(),
        epsilon,
        weight_offset_id,
        ctx,
    )
    ctx.synchronize()

    # --- Compare E4M3 output (bit-exact within a small budget). ---
    var fp8_mismatch = 0
    with out_device.map_to_host() as out_host:
        var out_host_tensor = LayoutTensor[out_dtype, out_static](
            out_host, RuntimeLayout[out_static].row_major(in_shape)
        )
        for r in range(rows):
            for c in range(cols):
                var got = bitcast[DType.uint8](
                    rebind[Scalar[out_dtype]](out_host_tensor[r, c])
                )
                var want = bitcast[DType.uint8](ref_out[r * cols + c])
                if got != want:
                    fp8_mismatch += 1

    # --- Compare E8M0 scales (bit-exact, including padding), read through the
    # interleaved-layout accessor `get_scale_factor`. ---
    var scale_mismatch = 0
    with scales_device.map_to_host() as scales_host:
        var scales_host_tensor = LayoutTensor[scales_dtype, scales_static](
            scales_host, scales_rt
        )
        for r in range(num_row_padded):
            for blk in range(num_sf_col_blocks):
                var got = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    scales_host_tensor.as_unsafe_any_origin(),
                    r,
                    blk * SF_VECTOR_SIZE,
                )
                var want = Scalar[scales_dtype](0)
                if r < rows and blk < num_col_blocks:
                    want = ref_scale[r * num_col_blocks + blk]
                if bitcast[DType.uint8](got) != bitcast[DType.uint8](want):
                    scale_mismatch += 1

    var fp8_rate = Float64(fp8_mismatch) / Float64(rows * cols)
    print(
        "rows =",
        rows,
        "cols =",
        cols,
        "fp8 mismatch =",
        fp8_mismatch,
        "(",
        fp8_rate * 100.0,
        "% ) scale mismatch =",
        scale_mismatch,
    )
    if scale_mismatch != 0:
        raise Error("E8M0 block scales must match bit-exactly")
    if fp8_rate > fp8_mismatch_budget:
        raise Error("E4M3 output mismatch rate exceeds budget")


def main() raises:
    with DeviceContext() as ctx:
        # (rows, cols): exercise row padding, column padding, and larger shapes.
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 4, 128)
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 128, 256)
        # cols not a multiple of 128 -> scale column padding.
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 13, 160)
        # nonzero weight_offset.
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 32, 512, weight_offset=0.25)
        # production-ish shapes.
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 999, 4096)
        test_fused_rms_norm_mxfp8[
            DType.bfloat16,
            DType.float8_e4m3fn,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
        ](ctx, 2, 6144)
        print("All fused RMSNorm + MXFP8 block-scaled tests passed!")
