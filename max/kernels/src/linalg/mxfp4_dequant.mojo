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
"""MXFP4 dequantization kernel for H100 (SM90).

Converts packed MXFP4 weights (uint8, 2 FP4 values per byte) with E8M0 block
scales into float8_e4m3fn or bfloat16.

Scales are in 2D layout [N, K/SF_VECTOR_SIZE] where each scale covers
SF_VECTOR_SIZE (32) consecutive elements.
"""

from std.math import ceildiv
from std.gpu import (
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
    grid_dim_uint as grid_dim,
    block_dim_uint as block_dim,
)
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
from .fp4_utils import cast_uint_to_fp4e2m1, MXFP4_SF_VECTOR_SIZE


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(512))
)
def _dequant_mxfp4_to_fp8_kernel[
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
    """Kernel that dequantizes MXFP4 packed uint8 to out_dtype (FP8 or BF16).

    Scales are 2D [num_rows, num_cols // SF_VECTOR_SIZE], one scale per block
    of SF_VECTOR_SIZE elements.
    """
    comptime assert output.flat_rank >= 2
    comptime assert input.flat_rank >= 2
    comptime assert scales.flat_rank >= 2
    comptime BYTES_PER_THREAD = ELEMENTS_PER_THREAD // 2

    with PDL():
        for global_row_idx in range(
            Int(block_idx.x), num_rows, Int(grid_dim.x)
        ):
            for col_thread_idx in range(
                Int(thread_idx.x),
                ceildiv(num_cols, ELEMENTS_PER_THREAD),
                Int(block_dim.x),
            ):
                var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD

                if global_col_idx >= num_cols:
                    continue

                # Load packed uint8 bytes
                var packed_byte_col = global_col_idx // 2
                var packed_bytes = input.load[BYTES_PER_THREAD](
                    Coord(Idx(global_row_idx), Idx(packed_byte_col))
                )

                # Unpack to float32 via E2M1 lookup table.
                # NOTE: We use a software LUT (cast_uint_to_fp4e2m1) rather
                # than the SM100+ PTX instruction (cast_f4e2m1x2_to_fp16x2)
                # because this kernel targets SM90 (H100).
                var fp32_values = cast_uint_to_fp4e2m1[
                    out_dtype=DType.float32, out_width=ELEMENTS_PER_THREAD
                ](packed_bytes)

                # Load the E8M0 scale from 2D layout
                var scale_col = global_col_idx // SF_VECTOR_SIZE
                var scale_e8m0 = rebind[Scalar[scales_dtype]](
                    scales.load(Coord(Idx(global_row_idx), Idx(scale_col)))
                )

                # Convert E8M0 to float32 using stdlib SIMD cast.
                # On SM100+ this uses PTX cvt.rn.bf16x2.ue8m0x2; on SM90
                # it falls back to the bitcast approach with correct
                # special-case handling for 0x00 and 0xFF.
                var scale_f32 = scale_e8m0.cast[DType.float32]()

                # Apply scale and cast to output dtype
                var scaled_values = fp32_values * scale_f32
                var out_values = scaled_values.cast[out_dtype]()

                # Store output
                output.store[width=ELEMENTS_PER_THREAD](
                    Coord(Idx(global_row_idx), Idx(global_col_idx)),
                    out_values,
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
    comptime out_dtype = output.dtype
    comptime in_dtype = input.dtype
    comptime scales_dtype = scales.dtype

    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.bfloat16,
    ), "output must be float8_e4m3fn or bfloat16"
    comptime assert (
        scales_dtype == DType.float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype == DType.uint8, "input must be uint8 (packed FP4)"
    comptime assert (
        SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be 32 for MXFP4"
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

    comptime kernel = _dequant_mxfp4_to_fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        type_of(output).LayoutType,
        type_of(scales_tt).LayoutType,
        type_of(input_tt).LayoutType,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
    ]

    ctx.enqueue_function[kernel, kernel](
        output,
        input_tt,
        scales_tt,
        num_rows,
        num_cols,
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
        attributes=pdl_launch_attributes(pdl_level),
    )
