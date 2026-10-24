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

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4_async,
    quantize_dynamic_scaled_fp4fp8,
)
from std.math import ceildiv
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    get_scale_factor,
)
from std.math import isnan
from std.memory import bitcast


def test_dynamic_mxfp8_quant[
    in_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    N: Int,
](ctx: DeviceContext, m: Int, n: Int, tensor_scale: Float32 = 1.0) raises:
    if N % (SF_VECTOR_SIZE) != 0:
        raise Error(
            "n must be a multiple of (SF_VECTOR_SIZE // 2) due to kernel"
            " constraints"
        )

    comptime out_dtype = DType.float8_e4m3fn

    var input_layout = row_major(m, Idx[N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(n, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](input_layout.size())
    var out_device = ctx.enqueue_create_buffer[out_dtype](input_layout.size())
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_layout.size()
    )
    var input_tensor = TileTensor(in_device, input_layout)
    var output_tensor = TileTensor(out_device, input_layout)
    var scales_tensor = TileTensor(scales_device, scales_layout)

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = TileTensor(in_host, input_layout)
        random(in_host_tensor, min=-1.0, max=1.0)

        for idx0 in range(m):
            for idx1 in range(N):
                in_host_tensor[idx0, idx1] = (
                    in_host_tensor[idx0, idx1] * tensor_scale.cast[in_dtype]()
                )

    with out_device.map_to_host() as out_host:
        for i in range(len(out_host)):
            out_host[i] = 0

    # Run the quantization kernel
    quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor.as_unsafe_any_origin(),
        scales_tensor.as_unsafe_any_origin(),
        input_tensor.as_unsafe_any_origin(),
        num_cols=n,
        num_cols_padded=n,
    )

    ctx.synchronize()

    # Verify results by reading back from device
    var atol = Float32(1)
    var rtol = Float32(0)
    var mismatch_count = 0
    with in_device.map_to_host() as in_host:
        with out_device.map_to_host() as out_host:
            with scales_device.map_to_host() as scales_host:
                var input_tensor_host = TileTensor(in_host, input_layout)
                var output_tensor_host = TileTensor(out_host, input_layout)
                var scales_tensor_host = TileTensor(scales_host, scales_layout)

                for idx0 in range(m):
                    for idx1 in range(N):
                        var ref_output = input_tensor_host[idx0, idx1].cast[
                            DType.float32
                        ]()
                        var output = output_tensor_host[idx0, idx1].cast[
                            DType.float32
                        ]()

                        if isnan(ref_output):
                            raise Error("NaN value found in reference output!")
                        if isnan(output):
                            raise Error("NaN value found in quantized output!")

                        var fp8_sf = get_scale_factor[
                            SF_VECTOR_SIZE=SF_VECTOR_SIZE
                        ](scales_tensor_host.as_unsafe_any_origin(), idx0, idx1)

                        var output_dequantized = (
                            output * fp8_sf.cast[.float32]()
                        )

                        var left = abs(output_dequantized - ref_output)
                        var right = atol + rtol * abs(ref_output)

                        if left > right:
                            mismatch_count += 1

                var mismatch_rate = Float64(mismatch_count) / Float64(m * n)
                if (1 - mismatch_rate) < 0.999:
                    raise Error("Too many mismatches!")
                print(
                    "M = ",
                    m,
                    "N = ",
                    n,
                    "SF_VECTOR_SIZE = ",
                    SF_VECTOR_SIZE,
                    "in_dtype = ",
                    in_dtype,
                    "scales_dtype = ",
                    scales_dtype,
                    "mismatch percentage = ",
                    mismatch_rate * 100.0,
                    "%",
                )


def test_dynamic_mxfp8_quant_degenerate_blocks(
    ctx: DeviceContext, m: Int
) raises:
    """Checks that all-zero and E8M0-floor blocks quantize without NaN.

    E8M0 has no zero, and its smallest scale (2^-127) is an fp32 subnormal.
    A block of values around 2^-119, such as a checkpoint block stored with
    E8M0 byte 0x00, gets that floor scale. The block must still dequantize to
    finite values: a NaN anywhere in a row poisons every dot product the
    block-scaled MMA takes over that row.
    """
    comptime N = 256
    comptime in_dtype = DType.bfloat16
    comptime out_dtype = DType.float8_e4m3fn
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime last_block = N // SF_VECTOR_SIZE - 1
    comptime tiny_block = 5
    # 2^-119: values in [-1, 1] scaled by this land at the E8M0 floor scale.
    comptime tiny = Float32(1.504632769052528e-36)

    var input_layout = row_major(m, Idx[N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var in_device = ctx.enqueue_create_buffer[in_dtype](input_layout.size())
    var out_device = ctx.enqueue_create_buffer[out_dtype](input_layout.size())
    var scales_device = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
        scales_layout.size()
    )
    var input_tensor = TileTensor(in_device, input_layout)
    var output_tensor = TileTensor(out_device, input_layout)
    var scales_tensor = TileTensor(scales_device, scales_layout)

    with in_device.map_to_host() as in_host:
        var in_host_tensor = TileTensor(in_host, input_layout)
        random(in_host_tensor, min=-1.0, max=1.0)
        for row in range(m):
            for col in range(N):
                var block = col // SF_VECTOR_SIZE
                if block == 2 or block == last_block:
                    in_host_tensor[row, col] = Scalar[in_dtype](0)
                elif block == tiny_block:
                    var value = in_host_tensor[row, col].cast[DType.float32]()
                    if col % SF_VECTOR_SIZE == 0:
                        value = 0
                    in_host_tensor[row, col] = (value * tiny).cast[in_dtype]()

    quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor.as_unsafe_any_origin(),
        scales_tensor.as_unsafe_any_origin(),
        input_tensor.as_unsafe_any_origin(),
        num_cols=N,
        num_cols_padded=N,
    )
    ctx.synchronize()

    with out_device.map_to_host() as out_host:
        with scales_device.map_to_host() as scales_host:
            var output_tensor_host = TileTensor(out_host, input_layout)
            var scales_tensor_host = TileTensor(scales_host, scales_layout)
            print(
                "M =",
                m,
                "e8m0 scale byte: zero block",
                bitcast[DType.uint8](
                    get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        scales_tensor_host.as_unsafe_any_origin(), 0, 64
                    )
                ),
                "tiny block",
                bitcast[DType.uint8](
                    get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        scales_tensor_host.as_unsafe_any_origin(),
                        0,
                        tiny_block * SF_VECTOR_SIZE,
                    )
                ),
            )
            for row in range(m):
                for col in range(N):
                    var fp8_sf = get_scale_factor[
                        SF_VECTOR_SIZE=SF_VECTOR_SIZE
                    ](scales_tensor_host.as_unsafe_any_origin(), row, col)
                    var dequantized = (
                        output_tensor_host[row, col].cast[DType.float32]()
                        * fp8_sf.cast[DType.float32]()
                    )
                    if isnan(dequantized):
                        print(
                            "row",
                            row,
                            "col",
                            col,
                            "scale byte",
                            bitcast[DType.uint8](fp8_sf),
                        )
                        raise Error("NaN dequantized value for an MXFP8 block")
                    var block = col // SF_VECTOR_SIZE
                    if (block == 2 or block == last_block) and dequantized != 0:
                        raise Error("all-zero MXFP8 block dequantized nonzero")
                    if block == tiny_block and abs(dequantized) > 2 * tiny:
                        raise Error("E8M0-floor MXFP8 block dequantized wrong")


def test_mxfp8_async_matches_sync[
    N: Int, num_max_threads: Int = 512
](ctx: DeviceContext, m: Int) raises:
    """Checks the TMA-async MXFP8 quantize writes the sync kernel's bytes.

    Rows span 2^-8..2^8 and include an all-zero block, an E8M0-floor block
    and large outliers. Both kernels write into buffers pre-filled with a
    sentinel, so a byte either skips, padded scale rows included, differs.
    """
    comptime in_dtype = DType.bfloat16
    comptime out_dtype = DType.float8_e4m3fn
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime blocks = N // SF_VECTOR_SIZE
    # 2^-119: values in [-1, 1] scaled by this land at the E8M0 floor scale.
    comptime tiny = Float32(1.504632769052528e-36)

    var input_layout = row_major(m, Idx[N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var in_device = ctx.enqueue_create_buffer[in_dtype](input_layout.size())
    var out_sync = ctx.enqueue_create_buffer[DType.uint8](input_layout.size())
    var out_async = ctx.enqueue_create_buffer[DType.uint8](input_layout.size())
    var sf_sync = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    var sf_async = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    out_sync.enqueue_fill(0xAB)
    out_async.enqueue_fill(0xAB)
    sf_sync.enqueue_fill(0xAB)
    sf_async.enqueue_fill(0xAB)

    with in_device.map_to_host() as in_host:
        var t = TileTensor(in_host, input_layout)
        random(t, min=-1.0, max=1.0)
        for row in range(m):
            var row_scale = Float32(2.0) ** Float32(row % 17 - 8)
            for col in range(N):
                var block = col // SF_VECTOR_SIZE
                var v = t[row, col].cast[DType.float32]() * row_scale
                if block == 1 % blocks:
                    v = 0
                elif blocks > 3 and block == 3:
                    v = t[row, col].cast[DType.float32]() * tiny
                elif col % 37 == row % 37:
                    v *= 1000
                t[row, col] = v.cast[in_dtype]()

    var input_tensor = TileTensor(in_device, input_layout)
    quantize_dynamic_scaled_fp4fp8[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, num_max_threads=num_max_threads
    ](
        ctx,
        TileTensor(
            out_sync.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            input_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_sync.unsafe_ptr().unsafe_bitcast[Scalar[MXFP8_SF_DTYPE]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        input_tensor.as_unsafe_any_origin(),
        num_cols=N,
        num_cols_padded=N,
    )
    quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        TileTensor(
            out_async.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            input_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_async.unsafe_ptr().unsafe_bitcast[Scalar[MXFP8_SF_DTYPE]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        input_tensor.as_unsafe_any_origin(),
    )
    ctx.synchronize()

    with out_sync.map_to_host() as a, out_async.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                print("M =", m, "N =", N, "output byte", i, a[i], "vs", b[i])
                raise Error("async MXFP8 output differs from the sync kernel")
    with sf_sync.map_to_host() as a, sf_async.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                print("M =", m, "N =", N, "scale byte", i, a[i], "vs", b[i])
                raise Error("async MXFP8 scales differ from the sync kernel")
    print("async matches sync: M =", m, "N =", N)


def main() raises:
    with DeviceContext() as ctx:
        test_dynamic_mxfp8_quant_degenerate_blocks(ctx, 1)
        test_dynamic_mxfp8_quant_degenerate_blocks(ctx, 130)
        test_dynamic_mxfp8_quant[
            DType.bfloat16,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
            N=128,
        ](ctx, 1, 128)
        test_dynamic_mxfp8_quant[
            DType.bfloat16,
            MXFP8_SF_DTYPE,
            MXFP8_SF_VECTOR_SIZE,
            N=128,
        ](ctx, 258, 128)

        comptime for N in range(576, 16384, 1024):
            test_dynamic_mxfp8_quant[
                DType.bfloat16,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                N=N,
            ](ctx, 999, N, tensor_scale=32.0)

        for m in [1, 96, 126, 129, 1000]:
            test_mxfp8_async_matches_sync[128](ctx, m)
            test_mxfp8_async_matches_sync[160](ctx, m)
            test_mxfp8_async_matches_sync[1280](ctx, m)
            test_mxfp8_async_matches_sync[2304](ctx, m)
            test_mxfp8_async_matches_sync[4096](ctx, m)
            test_mxfp8_async_matches_sync[5120](ctx, m)
            test_mxfp8_async_matches_sync[8192](ctx, m)
            test_mxfp8_async_matches_sync[2304, 1024](ctx, m)
            test_mxfp8_async_matches_sync[5120, 1024](ctx, m)
            test_mxfp8_async_matches_sync[8192, 1024](ctx, m)
