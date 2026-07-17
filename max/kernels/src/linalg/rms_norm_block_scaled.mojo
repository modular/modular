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
"""Fused RMSNorm + MXFP8 block-scaled ("quantize-on-write") quantization.

SM100 (B200) / SM120. Computes RMSNorm over each row and, in the same kernel,
quantizes the normalized activation to MXFP8 (E4M3 values + E8M0 per-32-element
block scales) in the exact interleaved rank-5 scale layout the SM100
block-scaled GEMM consumes. Fusing the quant into the norm epilogue removes the
HBM round-trip of the bf16 normalized activation that a standalone
`quantize_dynamic_block_scaled` would otherwise re-read.

The result is numerically equivalent to the unfused pipeline
`rms_norm (bf16 store) -> quantize_dynamic_block_scaled[MXFP8]`:

- The normalized value is rounded to bf16 (`in_dtype`) BEFORE the per-block max
  and quantization, exactly as the unfused path stores a bf16 intermediate and
  the standalone quant re-reads it.
- The per-block E8M0 scale (`group_max / 448` cast to `float8_e8m0fnu`), the
  reciprocal-scale multiply, and the E4M3 rounding reproduce
  `quantize_dynamic_scaled_fp4fp8_kernel` bit-for-bit
  (`linalg/fp4_quantization.mojo`).
- The rank-5 interleaved scale write and the `align_up(rows, 128)` padding-row
  zeroing reuse `set_scale_factor` / `SF_ATOM_M` / `SF_ATOM_K` /
  `SF_MN_GROUP_SIZE` from `linalg/fp4_utils.mojo`, so the downstream GEMM is a
  drop-in consumer.
"""

from std.math import align_up, recip, rsqrt
from std.sys import align_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.host.info import _is_sm10x_gpu, _is_sm12x_gpu
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from std.gpu.primitives.warp import shuffle_xor
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.runtime.tracing import Trace, TraceLevel, trace_arg
from std.utils import StaticTuple
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type
from layout import Layout, LayoutTensor, TileTensor
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)

# E4M3 (float8_e4m3fn) max finite magnitude; the per-block scale maps the block
# max to this value. Matches `quantize_dynamic_scaled_fp4fp8_kernel`.
comptime _E4M3_MAX = Float32(448.0)


@always_inline
def rms_norm_fused_quantize_dynamic_block_scaled[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    //,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    *,
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
    multiply_before_cast: Bool = True,
    target: StaticString = "gpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    compile_only: Bool = False,
](
    num_rows: Int,
    num_cols: Int,
    output_device: TileTensor[
        mut=True, out_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_device: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    gamma_device: TileTensor[in_dtype, address_space=AddressSpace.GENERIC, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
) raises:
    """Fuses RMSNorm and MXFP8 block-scaled quantization in a single pass.

    Parameters:
        in_dtype: Input/gamma data type (must be `bfloat16`).
        out_dtype: Quantized output dtype (must be `float8_e4m3fn`).
        scales_dtype: Block-scale dtype (must be `float8_e8m0fnu`).
        input_fn: Loads `width` normalization inputs at `(row, col)`.
        SF_VECTOR_SIZE: Block size for the scale factor (32 for MXFP8).
        multiply_before_cast: If True, `x * norm * gamma` is computed in the
            accumulator dtype and cast to `in_dtype` once (matches
            `rms_norm_gpu` with `multiply_before_cast=True`). If False, the
            normalized value is cast to `in_dtype` before the gamma multiply.
        target: Target device ("gpu").
        pdl_level: Programmatic Dependent Launch level. Defaults to ON so the
            kernel overlaps with its stream neighbors; the kernel body waits on
            the producer grid via `PDL` before reading input.
        compile_only: If True, only compile the kernel (no launch).

    Args:
        num_rows: Number of rows (tokens).
        num_cols: Row length (hidden size); must be a multiple of
            `SF_VECTOR_SIZE`.
        output_device: Rank-2 `[num_rows, num_cols]` E4M3 output.
        scales_device: Rank-5 interleaved E8M0 block scales (same layout as
            `quantize_dynamic_block_scaled`).
        gamma_device: Rank-1 `[num_cols]` RMSNorm weight.
        epsilon: RMSNorm epsilon.
        weight_offset: Added to gamma after loading (e.g. Gemma-style `+1`).
        ctx: Device context.
    """
    comptime assert (
        in_dtype == DType.bfloat16
    ), "input/gamma dtype must be bfloat16"
    comptime assert (
        out_dtype == DType.float8_e4m3fn
    ), "output dtype must be float8_e4m3fn (MXFP8 values)"
    comptime assert (
        scales_dtype == MXFP8_SF_DTYPE
    ), "scales dtype must be float8_e8m0fnu (MXFP8 E8M0 block scales)"
    comptime assert (
        SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be MXFP8_SF_VECTOR_SIZE (32)"
    comptime assert output_device.rank == 2, "output must be rank 2"
    comptime assert (
        scales_device.rank == 5
    ), "scales must be rank 5 (SM100 interleaved layout)"
    comptime assert gamma_device.flat_rank == 1, "gamma must be rank 1"
    comptime assert target == "gpu", "only the GPU target is supported"
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "fused RMSNorm + MXFP8 block-scaled quant requires SM100 or SM120"

    if num_cols % SF_VECTOR_SIZE != 0:
        raise Error("num_cols must be a multiple of SF_VECTOR_SIZE (32)")

    @always_inline
    @parameter
    def description_fn() -> String:
        return (
            trace_arg("input", IndexList[2](num_rows, num_cols), in_dtype)
            + " -> "
            + trace_arg("output", IndexList[2](num_rows, num_cols), out_dtype)
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_quantize_dynamic_block_scaled",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        if num_rows == 0 or num_cols == 0:
            return

        var gamma = gamma_device.as_unsafe_any_origin().to_layout_tensor()
        var output = output_device.as_unsafe_any_origin().to_layout_tensor()
        var scales = scales_device.as_unsafe_any_origin().to_layout_tensor()

        comptime gamma_layout = gamma.layout
        comptime output_layout = output.layout
        comptime scales_layout = scales.layout

        # One block per row, including the scale-tensor padding rows in
        # [num_rows, align_up(num_rows, 128)) which only zero their scales.
        # A fixed comptime block size lets `block.sum` size its shared memory.
        comptime threads_per_block = (
            ctx.default_device_info.max_thread_block_size
        )
        var grid_dim = align_up(num_rows, SF_MN_GROUP_SIZE)

        comptime kernel = _rms_norm_fused_block_scaled_kernel[
            in_dtype,
            out_dtype,
            scales_dtype,
            gamma_layout,
            output_layout,
            scales_layout,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            threads_per_block=threads_per_block,
            multiply_before_cast=multiply_before_cast,
            input_fn=input_fn,
        ]

        comptime if compile_only:
            _ = ctx.compile_function[kernel]()
        else:
            ctx.enqueue_function[kernel](
                gamma,
                output,
                scales,
                epsilon,
                weight_offset,
                num_rows,
                num_cols,
                grid_dim=grid_dim,
                block_dim=threads_per_block,
                attributes=pdl_launch_attributes(pdl_level),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(t"rms_norm_fused_block_scaled_{in_dtype}_{out_dtype}")
def _rms_norm_fused_block_scaled_kernel[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    gamma_layout: Layout,
    output_layout: Layout,
    scales_layout: Layout,
    *,
    SF_VECTOR_SIZE: Int,
    threads_per_block: Int,
    multiply_before_cast: Bool,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
](
    gamma: LayoutTensor[in_dtype, gamma_layout, ImmutAnyOrigin],
    output: LayoutTensor[out_dtype, output_layout, MutAnyOrigin],
    scales: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    num_rows: Int,
    num_cols: Int,
):
    """One block per row: RMSNorm reduction, then per-32-block MXFP8 quant."""
    comptime accum_type = get_accum_type[in_dtype]()
    comptime ELEMENTS_PER_THREAD = 8
    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
    comptime assert NUM_THREADS_PER_SF in (
        2,
        4,
    ), "SF_VECTOR_SIZE must be 16 or 32 with 8 elements per thread"
    comptime gamma_align = align_of[SIMD[in_dtype, ELEMENTS_PER_THREAD]]()

    var row = block_idx.x
    var tid = thread_idx.x
    var eps_a = epsilon.cast[accum_type]()
    var weight_offset_a = weight_offset.cast[accum_type]()

    var num_col_threads = num_cols // ELEMENTS_PER_THREAD
    # Scale-tensor columns padded up to a 128-wide (SF_VECTOR_SIZE * SF_ATOM_K)
    # group; padded blocks get a zero scale so the tensor-core scale load is 0.
    var num_sf_cols = align_up(num_cols, SF_VECTOR_SIZE * SF_ATOM_K)
    var num_sf_threads = num_sf_cols // ELEMENTS_PER_THREAD

    var is_padded_row = row >= num_rows

    with PDL():
        if is_padded_row:
            # This row exists only in the scale tensor's last row-group; zero
            # its block scales (no corresponding output row).
            for col_idx in range(tid, num_sf_threads, block_dim.x):
                var global_col = col_idx * ELEMENTS_PER_THREAD
                if global_col % SF_VECTOR_SIZE == 0:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        scales, row, global_col, Scalar[scales_dtype](0)
                    )
        else:
            # --- Phase 1: sum of squares over the row -> norm factor. ---
            var thread_m2 = Scalar[accum_type](0)
            for col_idx in range(tid, num_col_threads, block_dim.x):
                var global_col = col_idx * ELEMENTS_PER_THREAD
                var x = input_fn[ELEMENTS_PER_THREAD](row, global_col).cast[
                    accum_type
                ]()
                thread_m2 += (x * x).reduce_add()

            var row_m2 = block.sum[
                block_size=threads_per_block, broadcast=True
            ](thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](num_cols)) + eps_a
            )

            # --- Phase 2: normalize -> bf16 -> MXFP8 block-scaled quant. ---
            for col_idx in range(tid, num_sf_threads, block_dim.x):
                var global_col = col_idx * ELEMENTS_PER_THREAD

                if col_idx >= num_col_threads:
                    # Scale-tensor column padding; only the block start writes.
                    if global_col % SF_VECTOR_SIZE == 0:
                        set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                            scales, row, global_col, Scalar[scales_dtype](0)
                        )
                    continue

                var x = input_fn[ELEMENTS_PER_THREAD](row, global_col).cast[
                    accum_type
                ]()
                var gamma_vec = gamma.load[
                    width=ELEMENTS_PER_THREAD, load_alignment=gamma_align
                ](IndexList[1](global_col))

                # Round the normalized value to bf16 exactly as the unfused
                # `rms_norm` stores its bf16 intermediate, so the block max and
                # quantization below operate on the same bits the standalone
                # quant would re-read.
                var normed: SIMD[in_dtype, ELEMENTS_PER_THREAD]
                comptime if multiply_before_cast:
                    var gamma_a = gamma_vec.cast[accum_type]() + weight_offset_a
                    normed = (x * norm_factor * gamma_a).cast[in_dtype]()
                else:
                    normed = (x * norm_factor).cast[in_dtype]() * (
                        gamma_vec + weight_offset
                    )
                var normed_f32 = normed.cast[DType.float32]()

                # Per-32-element block max: 8 lanes locally, then a butterfly
                # reduce across the NUM_THREADS_PER_SF lane-contiguous threads.
                var thread_max = abs(normed).reduce_max()
                thread_max = max(shuffle_xor(thread_max, 1), thread_max)
                comptime if NUM_THREADS_PER_SF == 4:
                    thread_max = max(shuffle_xor(thread_max, 2), thread_max)
                var group_max = thread_max.cast[DType.float32]()

                var scale_factor = group_max * recip(_E4M3_MAX)
                var fp8_scale_factor = scale_factor.cast[scales_dtype]()

                var output_scale = Float32(0.0)
                if group_max != 0:
                    output_scale = recip(fp8_scale_factor.cast[DType.float32]())

                if global_col % SF_VECTOR_SIZE == 0:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        scales, row, global_col, fp8_scale_factor
                    )

                var out_vec = (normed_f32 * output_scale).cast[out_dtype]()
                output.store(row, global_col, out_vec)
