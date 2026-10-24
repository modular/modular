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

from std.math import ceildiv
from std.testing import assert_raises
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    swiglu_quantize_dynamic_block_scaled,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from linalg.swiglu_activation import (
    apply_swiglu_activation,
    get_swiglu_activation,
    swiglu_activation_type,
)


def _gated_activation[
    activation: swiglu_activation_type, N: Int
](
    h: MutPointer[BFloat16, MutAnyOrigin],
    gate_up: MutPointer[BFloat16, MutAnyOrigin],
    rows: Int32,
):
    var i = Int(global_idx.x)
    if i >= Int(rows) * (N // 8):
        return
    var row, c = divmod(i, N // 8)
    var src = row * 2 * N + c * 8
    h.unsafe_store(
        row * N + c * 8,
        apply_swiglu_activation[activation](
            gate_up.unsafe_load[width=8](src),
            gate_up.unsafe_load[width=8](src + N),
        ),
    )


def test_swiglu_quantize_matches_unfused[
    out_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
    activation: swiglu_activation_type,
](ctx: DeviceContext, m: Int) raises:
    """Checks the fused gated quantize writes the bytes of the activation
    followed by the plain quantize.

    Rows span 2^-8..2^8, some values sit far past the clamp, and every fifth
    row has an all-zero block. Both arms write into buffers pre-filled with a
    sentinel, so a byte either skips, padded scale rows included, differs.
    """
    comptime out_N = N // 2 if out_dtype == DType.uint8 else N
    comptime num_max_threads = 512 if out_dtype == DType.uint8 else 1024

    var gate_up_layout = row_major(m, Idx[2 * N])
    var h_layout = row_major(m, Idx[N])
    var out_layout = row_major(m, Idx[out_N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var gate_up = ctx.enqueue_create_buffer[DType.bfloat16](
        gate_up_layout.size()
    )
    var h = ctx.enqueue_create_buffer[DType.bfloat16](h_layout.size())
    var out_ref = ctx.enqueue_create_buffer[DType.uint8](out_layout.size())
    var out_fused = ctx.enqueue_create_buffer[DType.uint8](out_layout.size())
    var sf_ref = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    var sf_fused = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    out_ref.enqueue_fill(0xAB)
    out_fused.enqueue_fill(0xAB)
    sf_ref.enqueue_fill(0xAB)
    sf_fused.enqueue_fill(0xAB)

    with gate_up.map_to_host() as host:
        var t = TileTensor(host, gate_up_layout)
        random(t, min=-1.0, max=1.0)
        for row in range(m):
            var row_scale = Float32(2.0) ** Float32(row % 17 - 8)
            for col in range(2 * N):
                var v = t[row, col].cast[DType.float32]() * row_scale
                if row % 5 == 0 and col % N < SF_VECTOR_SIZE:
                    v = 0
                elif col % 37 == row % 37:
                    v *= 1000
                t[row, col] = v.cast[DType.bfloat16]()

    ctx.enqueue_function[_gated_activation[activation, N]](
        h,
        gate_up,
        Int32(m),
        grid_dim=ceildiv(m * (N // 8), 256),
        block_dim=256,
    )
    quantize_dynamic_scaled_fp4fp8[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, num_max_threads=num_max_threads
    ](
        ctx,
        TileTensor(
            out_ref.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            out_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_ref.unsafe_ptr().unsafe_bitcast[Scalar[scales_dtype]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        TileTensor(h, h_layout).as_unsafe_any_origin(),
        num_cols=N,
        num_cols_padded=N,
    )
    swiglu_quantize_dynamic_block_scaled[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, activation=activation
    ](
        TileTensor(
            out_fused.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            out_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_fused.unsafe_ptr().unsafe_bitcast[Scalar[scales_dtype]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        TileTensor(gate_up, gate_up_layout).as_unsafe_any_origin(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    with out_ref.map_to_host() as a, out_fused.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                print("M =", m, "N =", N, "output byte", i, a[i], "vs", b[i])
                raise Error("fused output differs from the unfused path")
    with sf_ref.map_to_host() as a, sf_fused.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                print("M =", m, "N =", N, "scale byte", i, a[i], "vs", b[i])
                raise Error("fused scales differ from the unfused path")
    print("fused matches unfused:", out_dtype, "M =", m, "N =", N)


def test_swiglu_quantize_rejects_unpacked_fp4_output(
    ctx: DeviceContext,
) raises:
    """Checks the wrapper refuses an FP4 output one byte per value wide,
    which the kernel would otherwise write past."""
    comptime N = 128
    comptime m = 3
    var gate_up = ctx.enqueue_create_buffer[DType.bfloat16](m * 2 * N)
    var out = ctx.enqueue_create_buffer[DType.uint8](m * N)
    var scales_layout = row_major(
        Idx[1],
        Idx[ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var sf = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    with assert_raises(contains="output must be"):
        swiglu_quantize_dynamic_block_scaled[
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            activation=get_swiglu_activation["silu"],
        ](
            TileTensor(out, row_major(Idx[m], Idx[N])).as_unsafe_any_origin(),
            TileTensor(
                sf.unsafe_ptr().unsafe_bitcast[Scalar[NVFP4_SF_DTYPE]](),
                scales_layout,
            ).as_unsafe_any_origin(),
            TileTensor(
                gate_up, row_major(Idx[m], Idx[2 * N])
            ).as_unsafe_any_origin(),
            Float32(1.0),
            ctx,
        )


def main() raises:
    comptime clamped = get_swiglu_activation["silu", limit=10.0]
    comptime plain = get_swiglu_activation["silu"]
    comptime oai = get_swiglu_activation["swigluoai", 1.702, 7.0]
    with DeviceContext() as ctx:
        for m in [1, 96, 126, 129, 1000]:
            test_swiglu_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                2304,
                clamped,
            ](ctx, m)
            test_swiglu_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                160,
                plain,
            ](ctx, m)
            test_swiglu_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                128,
                oai,
            ](ctx, m)
            test_swiglu_quantize_matches_unfused[
                DType.uint8,
                NVFP4_SF_DTYPE,
                NVFP4_SF_VECTOR_SIZE,
                2304,
                clamped,
            ](ctx, m)
        test_swiglu_quantize_rejects_unpacked_fp4_output(ctx)
