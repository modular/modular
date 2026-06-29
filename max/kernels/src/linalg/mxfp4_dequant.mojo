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
"""FP4 (MXFP4 / NVFP4) dequantization kernels for GPUs without native FP4
matmul support (SM90 and older NVIDIA architectures).

Converts packed FP4 weights (uint8, 2 E2M1 values per byte) with block scales
into float8_e4m3fn or bfloat16:

- MXFP4: E8M0 scales, one per SF_VECTOR_SIZE=32 consecutive elements.
- NVFP4: E4M3 (or pre-multiplied float32) scales, one per SF_VECTOR_SIZE=16
  consecutive elements. A per-tensor scale (modelopt ``weight_scale_2``) is
  expected to be pre-multiplied into the block scales by the caller.

Scales are in 2D layout [N, K/SF_VECTOR_SIZE]. The E2M1 decode uses a software
LUT, so the kernels run on any GPU (no SM100+ instructions).
"""

from std.math import ceildiv
from std.gpu import block_idx, thread_idx, grid_dim, block_dim
from std.gpu.host import DeviceContext
from std.gpu.host.info import GPUInfo
from std.sys.info import _accelerator_arch
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.utils import StaticTuple
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA
from layout import TileTensor
from layout.coord import Coord, Idx
from layout.tile_layout import TensorLayout
from .fp4_utils import (
    cast_uint_to_fp4e2m1,
    MXFP4_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
)
from std.algorithm.functional import elementwise
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import Index, IndexList
from std.sys.info import simd_width_of


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(512))
)
@__name(t"dequant_fp4_{out_dtype}_{scales_dtype}_{in_dtype}_{SF_VECTOR_SIZE}")
def _dequant_fp4_kernel[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    *,
    SF_VECTOR_SIZE: Int = 32,
    ELEMENTS_PER_THREAD: Int = 8,
](
    output: TileTensor[out_dtype, output_layout, MutAnyOrigin],
    input: TileTensor[in_dtype, input_layout, MutAnyOrigin],
    scales: TileTensor[scales_dtype, scales_layout, MutAnyOrigin],
    num_rows: Int,
    num_cols: Int,
):
    """Kernel that dequantizes packed FP4 uint8 to out_dtype (FP8 or BF16).

    Scales are 2D [num_rows, num_cols // SF_VECTOR_SIZE], one scale per block
    of SF_VECTOR_SIZE elements (32 for MXFP4/E8M0, 16 for NVFP4/E4M3-or-f32).
    """
    comptime assert output.flat_rank >= 2
    comptime assert input.flat_rank >= 2
    comptime assert scales.flat_rank >= 2
    comptime BYTES_PER_THREAD = ELEMENTS_PER_THREAD // 2

    with PDL():
        for global_row_idx in range(block_idx.x, num_rows, grid_dim.x):
            for col_thread_idx in range(
                thread_idx.x,
                ceildiv(num_cols, ELEMENTS_PER_THREAD),
                block_dim.x,
            ):
                var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD

                if global_col_idx >= num_cols:
                    continue

                # Load packed uint8 bytes
                var packed_byte_col = global_col_idx // 2
                var packed_bytes = input.load[BYTES_PER_THREAD](
                    Coord(global_row_idx, packed_byte_col)
                )

                # Unpack to float32 via E2M1 lookup table.
                # NOTE: We use a software decode (cast_uint_to_fp4e2m1) rather
                # than the SM100+ PTX instruction (cast_f4e2m1x2_to_fp16x2)
                # because this kernel targets SM90 (H100).
                var fp32_values = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=ELEMENTS_PER_THREAD
                ](packed_bytes)

                # Load the block scale from 2D layout
                var scale_col = global_col_idx // SF_VECTOR_SIZE
                var scale_raw = rebind[Scalar[scales_dtype]](
                    scales.load(Coord(global_row_idx, scale_col))
                )

                # Convert the scale to float32 using stdlib SIMD cast. For
                # E8M0 on SM100+ this uses PTX cvt.rn.bf16x2.ue8m0x2; on
                # older architectures it falls back to the bitcast approach
                # with correct special-case handling for 0x00 and 0xFF.
                # E4M3 and float32 scales cast natively everywhere.
                var scale_f32 = scale_raw.cast[DType.float32]()

                # Apply scale and cast to output dtype
                var scaled_values = fp32_values * scale_f32
                var out_values = scaled_values.cast[out_dtype]()

                # Store output
                output.store[width=ELEMENTS_PER_THREAD](
                    Coord(global_row_idx, global_col_idx),
                    out_values,
                )


@always_inline
def _dequant_fp4[
    *, SF_VECTOR_SIZE: Int
](
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    scales: TileTensor,
    num_rows: Int,
    num_cols: Int,
    pdl_level: PDLLevel,
) raises:
    """Shared launch logic for the FP4 dequant kernel (format-agnostic)."""
    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.bfloat16,
    ), "output must be float8_e4m3fn or bfloat16"
    comptime assert in_dtype == DType.uint8, "input must be uint8 (packed FP4)"
    comptime ELEMENTS_PER_THREAD = 8
    comptime assert (
        SF_VECTOR_SIZE % ELEMENTS_PER_THREAD == 0
    ), "SF_VECTOR_SIZE must be a multiple of ELEMENTS_PER_THREAD"

    if num_rows == 0 or num_cols == 0:
        return

    debug_assert(
        num_cols % ELEMENTS_PER_THREAD == 0,
        "num_cols must be a multiple of ELEMENTS_PER_THREAD (8)",
    )
    comptime num_max_threads = 512
    comptime _gpu = GPUInfo.from_name[_accelerator_arch()]()
    comptime num_SMs = _gpu.sm_count

    var num_col_threads = ceildiv(num_cols, ELEMENTS_PER_THREAD)

    var block_dim_val = (min(num_col_threads, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, _gpu.threads_per_multiprocessor // block_dim_val[0]
    )
    var grid_dim_val = (
        min(num_rows, num_SMs * num_blocks_per_SM),
        1,
        1,
    )

    # Rebind immutable origins to MutAnyOrigin for the GPU kernel.
    var input_tt = rebind[
        TileTensor[in_dtype, type_of(input).LayoutType, MutAnyOrigin]
    ](input)
    var scales_tt = rebind[
        TileTensor[
            scales_dtype,
            type_of(scales).LayoutType,
            MutAnyOrigin,
        ]
    ](scales)

    comptime kernel = _dequant_fp4_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        type_of(output).LayoutType,
        type_of(scales_tt).LayoutType,
        type_of(input_tt).LayoutType,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
    ]

    ctx.enqueue_function[kernel](
        output,
        input_tt,
        scales_tt,
        num_rows,
        num_cols,
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
        attributes=pdl_launch_attributes(pdl_level),
    )


@always_inline
def dequant_mxfp4[
    *, SF_VECTOR_SIZE: Int = 32
](
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    scales: TileTensor,
    num_rows: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel(),
) raises:
    """Dequantize MXFP4 packed weights to FP8 or BF16.

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor [num_rows, num_cols] of float8_e4m3fn or bfloat16.
        input: Input tensor [num_rows, num_cols // 2] of uint8 (packed FP4).
        scales: Scale tensor [num_rows, num_cols // SF_VECTOR_SIZE] of float8_e8m0fnu.
        num_rows: Number of rows (N dimension for weights).
        num_cols: Number of columns (K dimension, unpacked).
        pdl_level: PDL optimization level for kernel launch.
    """
    comptime assert (
        scales.dtype == DType.float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert (
        SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be 32 for MXFP4"

    _dequant_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx, output, input, scales, num_rows, num_cols, pdl_level
    )


@always_inline
def dequant_nvfp4[
    *, SF_VECTOR_SIZE: Int = 16
](
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    scales: TileTensor,
    num_rows: Int,
    num_cols: Int,
    pdl_level: PDLLevel = PDLLevel(),
) raises:
    """Dequantize NVFP4 packed weights to FP8 or BF16.

    Runs on any GPU (software E2M1 LUT) — used as the fallback path on
    architectures without native FP4 matmul support (pre-Blackwell NVIDIA).

    The modelopt per-tensor ``weight_scale_2`` is NOT applied here: callers
    pre-multiply it into ``scales`` (yielding float32 scales) or pass the raw
    E4M3 block scales when no per-tensor scale exists.

    Args:
        ctx: Device context for kernel launch.
        output: Output tensor [num_rows, num_cols] of float8_e4m3fn or bfloat16.
        input: Input tensor [num_rows, num_cols // 2] of uint8 (packed FP4).
        scales: Scale tensor [num_rows, num_cols // SF_VECTOR_SIZE] of
            float8_e4m3fn or float32 (pre-multiplied).
        num_rows: Number of rows (N dimension for weights).
        num_cols: Number of columns (K dimension, unpacked).
        pdl_level: PDL optimization level for kernel launch.
    """
    comptime assert scales.dtype in (
        DType.float8_e4m3fn,
        DType.float32,
    ), "scales must be float8_e4m3fn or float32 (pre-multiplied)"
    comptime assert (
        SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be 16 for NVFP4"

    _dequant_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx, output, input, scales, num_rows, num_cols, pdl_level
    )


def _cast_bf16_to_fp8(
    ctx: DeviceContext,
    output: TileTensor,
    input: TileTensor,
    num_rows: Int,
    num_cols: Int,
) raises:
    """Cast BF16 tensor to FP8 using elementwise kernel."""
    var out_tt = output.as_unsafe_any_origin()
    var in_tt = input.as_unsafe_any_origin()
    comptime assert out_tt.flat_rank == 2, "output must be rank 2"
    comptime assert in_tt.flat_rank == 2, "input must be rank 2"
    comptime assert out_tt.mut, "output must be mutable"

    @always_inline
    def cast_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 2, "cast_fn only supports rank-2 tensors"
        out_tt.store[width=width](
            idx,
            in_tt.load[width=width](idx).cast[out_tt.dtype](),
        )

    elementwise[
        simd_width_of[input.dtype](),
        target="gpu",
        _trace_description="mxfp4_dequant_cast",
    ](cast_fn, (num_rows, num_cols), ctx)
