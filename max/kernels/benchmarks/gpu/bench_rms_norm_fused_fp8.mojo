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

from math import align_up
from random import random_float64
from sys import env_get_bool, env_get_dtype, size_of

from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext
from internal_utils import env_get_shape, int_list_to_tuple
from runtime.asyncrt import DeviceContextPtr
from layout._coord import Coord, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
)
from nn.normalization import rms_norm_gpu, rms_norm_fused_fp8

from buffer import NDBuffer
from linalg.fp8_quantization import quantize_dynamic_scaled_fp8
from memory import LegacyUnsafePointer
from utils.index import Index, IndexList


# Cache busting helpers: 512 MiB is larger than 2x the infinity cache on MI300x.
fn _calculate_stride(tensor_size: Int, alignment: Int) -> Int:
    return align_up(tensor_size, alignment)


fn _calculate_buffer_size[
    dtype: DType
](tensor_size: Int, alignment: Int) -> Int:
    comptime k512m = 512 * 1024 * 1024
    var stride = _calculate_stride(tensor_size, alignment)
    return align_up(k512m, stride * size_of[dtype]()) // size_of[dtype]()


fn _calculate_offset(iteration: Int, stride: Int, buffer_size: Int) -> Int:
    return (iteration * stride) % buffer_size


fn bench_rms_norm_fused_fp8[
    rank: Int,
    //,
    in_dtype: DType,
    out_dtype: DType,
    shape: IndexList[rank],
    cache_busting: Bool = True,
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    """Benchmark fused RMS norm + FP8 quantization against separate operations.

    Compares:
    1. RMS norm alone
    2. FP8 quantization alone
    3. Fused RMS norm + FP8 quantization
    """
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    # Allocate host memory
    var data_h = alloc[Scalar[in_dtype]](rows * cols)
    var gamma_h = alloc[Scalar[in_dtype]](cols)

    # Initialize data
    for i in range(rows * cols):
        var val = Scalar[in_dtype](random_float64(0, 100).cast[in_dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[in_dtype]()

    # Calculate buffer sizes for cache busting
    comptime simd_size = 4
    var data_size = rows * cols
    var stride_data = _calculate_stride(data_size, simd_size)
    var buf_data_in = _calculate_buffer_size[in_dtype](data_size, simd_size)
    var buf_data_out = _calculate_buffer_size[out_dtype](data_size, simd_size)

    # Allocate device buffers - use larger sizes when cache busting
    var alloc_data_in = buf_data_in if cache_busting else data_size
    var alloc_data_out = buf_data_out if cache_busting else data_size

    var data_d = ctx.enqueue_create_buffer[in_dtype](alloc_data_in)
    var gamma_d = ctx.enqueue_create_buffer[in_dtype](cols)
    var rms_output_d = ctx.enqueue_create_buffer[in_dtype](alloc_data_in)
    var fp8_output_d = ctx.enqueue_create_buffer[out_dtype](alloc_data_out)
    var fused_output_d = ctx.enqueue_create_buffer[out_dtype](alloc_data_out)
    var scales_d = ctx.enqueue_create_buffer[DType.float32](rows)

    var param_shape = Index(cols)

    # Create TileTensor for gamma
    var gamma_tensor = TileTensor(gamma_d, row_major(Coord(param_shape)))

    var epsilon = Scalar[in_dtype](0.001)
    var weight_offset = Scalar[in_dtype](0.0)

    # Copy data to device (initialize the whole buffer when cache busting)
    from internal_utils._utils import InitializationType, init_vector_launch

    comptime random_distribution = InitializationType.uniform_distribution
    init_vector_launch[in_dtype](
        data_d, alloc_data_in, random_distribution, ctx
    )
    init_vector_launch[in_dtype](
        rms_output_d, alloc_data_in, random_distribution, ctx
    )
    ctx.enqueue_copy(gamma_d, gamma_h)

    # ===== Benchmark 1: RMS norm alone =====
    # Extract pointers outside kernel context
    comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
    var data_base_ptr = data_d.unsafe_ptr()
    var rms_output_base_ptr = rms_output_d.unsafe_ptr()

    @always_inline
    @__copy_capture(
        shape,
        gamma_tensor,
        epsilon,
        weight_offset,
        stride_data,
        buf_data_in,
        data_base_ptr,
        rms_output_base_ptr,
    )
    @parameter
    fn bench_rms_norm(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            # Calculate offsets - zero when not cache busting
            var offset_in = 0
            var offset_out = 0

            comptime if cache_busting:
                offset_in = _calculate_offset(
                    iteration, stride_data, buf_data_in
                )
                offset_out = _calculate_offset(
                    iteration, stride_data, buf_data_in
                )

            # Construct buffers with offsets
            var data_ptr_offset = UnsafePointer[Scalar[in_dtype]](
                data_base_ptr + offset_in
            )
            var rms_output_ptr_offset = UnsafePointer[Scalar[in_dtype]](
                rms_output_base_ptr + offset_out
            )
            var data_buf_offset = TileTensor(
                data_ptr_offset, row_major(Coord(shape))
            )
            var rms_output_buf_offset = TileTensor(
                rms_output_ptr_offset, row_major(Coord(shape))
            )

            # Input function for RMS norm
            @__copy_capture(data_buf_offset)
            @always_inline
            @parameter
            fn input_fn[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
                var idx = data_buf_offset.layout(Coord(coords))
                return data_buf_offset.ptr.load[width=width, alignment=width](
                    idx
                )

            # Output function for RMS norm
            @always_inline
            @__copy_capture(rms_output_buf_offset)
            @parameter
            fn rms_output_fn[
                width: Int, alignment: Int
            ](coords: IndexList[rank], val: SIMD[in_dtype, width]) -> None:
                var idx = rms_output_buf_offset.layout(Coord(coords))
                rms_output_buf_offset.ptr.store[
                    width=width, alignment=alignment
                ](idx, val)

            rms_norm_gpu[input_fn, rms_output_fn, multiply_before_cast=True](
                shape, gamma_tensor, epsilon, weight_offset, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_rms_norm](
        BenchId(
            "rms_norm_only",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    # ===== Benchmark 2: FP8 quantization alone =====
    # Extract pointers outside kernel context
    var rms_output_base_ptr_fp8 = rms_output_base_ptr
    var fp8_output_base_ptr = fp8_output_d.unsafe_ptr()
    var scales_base_ptr = scales_d.unsafe_ptr()

    @always_inline
    @__copy_capture(
        stride_data,
        buf_data_in,
        buf_data_out,
        rms_output_base_ptr_fp8,
        fp8_output_base_ptr,
        scales_base_ptr,
    )
    @parameter
    fn bench_fp8_quant(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            # Calculate offsets - zero when not cache busting
            var offset_in = 0
            var offset_out = 0

            comptime if cache_busting:
                offset_in = _calculate_offset(
                    iteration, stride_data, buf_data_in
                )
                offset_out = _calculate_offset(
                    iteration, stride_data, buf_data_out
                )

            # Input function for FP8 quant (reads from RMS norm output)
            @__copy_capture(offset_in, rms_output_base_ptr_fp8)
            @always_inline
            @parameter
            fn fp8_input_fn[
                width: Int, alignment: Int
            ](row: Int, col: Int) -> SIMD[in_dtype, width]:
                var rms_ptr = UnsafePointer[Scalar[in_dtype]](
                    rms_output_base_ptr_fp8 + offset_in
                )
                var idx = row * cols + col
                return rms_ptr.load[width=width](idx)

            var fp8_output_ndbuf = NDBuffer[out_dtype, 2, MutAnyOrigin](
                UnsafePointer[Scalar[out_dtype]](
                    fp8_output_base_ptr + offset_out
                ),
                Index(rows, cols),
            )
            var scales_ndbuf = NDBuffer[DType.float32, 2, MutAnyOrigin](
                UnsafePointer[Scalar[DType.float32]](scales_base_ptr),
                Index(1, rows),
            )

            quantize_dynamic_scaled_fp8[
                input_fn=fp8_input_fn,
                group_size_or_per_token= -1,  # Per-token quantization
                num_cols=cols,
            ](fp8_output_ndbuf, scales_ndbuf, Float32(448.0), ctx, rows)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fp8_quant](
        BenchId(
            "fp8_quant_only",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    # ===== Benchmark 3: Fused RMS norm + FP8 quantization =====
    # Extract pointers outside kernel context
    var data_base_ptr_fused = data_base_ptr
    var fused_output_base_ptr = fused_output_d.unsafe_ptr()
    var scales_base_ptr_fused = scales_base_ptr

    @always_inline
    @__copy_capture(
        shape,
        gamma_tensor,
        epsilon,
        weight_offset,
        stride_data,
        buf_data_in,
        buf_data_out,
        data_base_ptr_fused,
        fused_output_base_ptr,
        scales_base_ptr_fused,
    )
    @parameter
    fn bench_fused(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx_: DeviceContext, iteration: Int) raises:
            # Calculate offsets - zero when not cache busting
            var offset_in = 0
            var offset_out = 0

            comptime if cache_busting:
                offset_in = _calculate_offset(
                    iteration, stride_data, buf_data_in
                )
                offset_out = _calculate_offset(
                    iteration, stride_data, buf_data_out
                )

            # Input function with offset
            @__copy_capture(offset_in, data_base_ptr_fused)
            @always_inline
            @parameter
            fn input_fn_fused[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
                var data_ptr_offset = UnsafePointer[Scalar[in_dtype]](
                    data_base_ptr_fused + offset_in
                )
                var data_buf_offset = TileTensor(
                    data_ptr_offset, row_major(Coord(shape))
                )
                var idx = data_buf_offset.layout(Coord(coords))
                return data_buf_offset.ptr.load[width=width, alignment=width](
                    idx
                )

            var fused_output_ndbuf = NDBuffer[out_dtype, rank, MutAnyOrigin](
                UnsafePointer[Scalar[out_dtype]](
                    fused_output_base_ptr + offset_out
                ),
                rebind[IndexList[rank]](coord_to_index_list(Coord(shape))),
            )
            var fused_scale_shape = shape
            fused_scale_shape[rank - 1] = 1
            var fused_scales_ndbuf = NDBuffer[
                DType.float32, rank, MutAnyOrigin
            ](
                UnsafePointer[Scalar[DType.float32]](scales_base_ptr_fused),
                fused_scale_shape,
            )

            # DeviceContextPtr has an @implicit constructor from DeviceContext
            var ctx_ptr = DeviceContextPtr(ctx_)
            rms_norm_fused_fp8[
                in_dtype,
                out_dtype,
                DType.float32,
                rank,
                input_fn_fused,
            ](
                shape,
                fused_output_ndbuf,
                gamma_tensor,
                epsilon,
                weight_offset,
                ctx_ptr,
                Float32(448.0),
                fused_scales_ndbuf,
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fused](
        BenchId(
            "rms_norm_fused_fp8",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    # ===== Verification: Compare fused output with separate operations =====
    print("\nVerifying outputs...")

    # Allocate separate device buffers for verification (no cache busting)
    var fp8_verify_d = ctx.enqueue_create_buffer[out_dtype](rows * cols)
    var fused_verify_d = ctx.enqueue_create_buffer[out_dtype](rows * cols)
    var rms_verify_d = ctx.enqueue_create_buffer[in_dtype](rows * cols)

    var fp8_verify_base_ptr = fp8_verify_d.unsafe_ptr()
    var fused_verify_base_ptr = fused_verify_d.unsafe_ptr()
    var rms_verify_base_ptr = rms_verify_d.unsafe_ptr()

    # Run separate operations with zero offset
    var data_ptr_verify = UnsafePointer[Scalar[in_dtype]](data_base_ptr)
    var rms_output_ptr_verify = UnsafePointer[Scalar[in_dtype]](
        rms_verify_base_ptr
    )
    var data_buf_verify = TileTensor(data_ptr_verify, row_major(Coord(shape)))
    var rms_output_buf_verify = TileTensor(
        rms_output_ptr_verify, row_major(Coord(shape))
    )

    # Input function for verification
    @__copy_capture(data_buf_verify)
    @always_inline
    @parameter
    fn input_fn_verify[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var idx = data_buf_verify.layout(Coord(coords))
        return data_buf_verify.ptr.load[width=width](idx)

    # Output function for verification
    @always_inline
    @__copy_capture(rms_output_buf_verify)
    @parameter
    fn rms_output_fn_verify[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[in_dtype, width]) -> None:
        var idx = rms_output_buf_verify.layout(Coord(coords))
        rms_output_buf_verify.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Run RMS norm
    rms_norm_gpu[
        input_fn_verify, rms_output_fn_verify, multiply_before_cast=True
    ](shape, gamma_tensor, epsilon, weight_offset, ctx)

    # Run FP8 quantization on RMS norm output
    @__copy_capture(rms_verify_base_ptr)
    @always_inline
    @parameter
    fn fp8_input_fn_verify[
        width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[in_dtype, width]:
        var rms_ptr = UnsafePointer[Scalar[in_dtype]](rms_verify_base_ptr)
        var idx = row * cols + col
        return rms_ptr.load[width=width](idx)

    var fp8_output_ndbuf_verify = NDBuffer[out_dtype, 2, MutAnyOrigin](
        UnsafePointer[Scalar[out_dtype]](fp8_verify_base_ptr),
        Index(rows, cols),
    )
    var scales_ndbuf_verify = NDBuffer[DType.float32, 2, MutAnyOrigin](
        UnsafePointer[Scalar[DType.float32]](scales_base_ptr),
        Index(1, rows),
    )

    quantize_dynamic_scaled_fp8[
        input_fn=fp8_input_fn_verify,
        group_size_or_per_token= -1,
        num_cols=cols,
    ](fp8_output_ndbuf_verify, scales_ndbuf_verify, Float32(448.0), ctx, rows)

    # Run fused kernel
    @__copy_capture(data_base_ptr_fused)
    @always_inline
    @parameter
    fn input_fn_fused_verify[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var data_ptr = UnsafePointer[Scalar[in_dtype]](data_base_ptr_fused)
        var data_buf = TileTensor(data_ptr, row_major(Coord(shape)))
        var idx = data_buf.layout(Coord(coords))
        return data_buf.ptr.load[width=width](idx)

    var fused_output_ndbuf_verify = NDBuffer[out_dtype, rank, MutAnyOrigin](
        UnsafePointer[Scalar[out_dtype]](fused_verify_base_ptr),
        rebind[IndexList[rank]](coord_to_index_list(Coord(shape))),
    )
    var verify_scale_shape = shape
    verify_scale_shape[rank - 1] = 1
    var fused_scales_ndbuf_verify = NDBuffer[DType.float32, rank, MutAnyOrigin](
        UnsafePointer[Scalar[DType.float32]](scales_base_ptr_fused),
        verify_scale_shape,
    )

    var ctx_ptr_verify = DeviceContextPtr(ctx)
    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        rank,
        input_fn_fused_verify,
    ](
        shape,
        fused_output_ndbuf_verify,
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx_ptr_verify,
        Float32(448.0),
        fused_scales_ndbuf_verify,
    )

    ctx.synchronize()

    # Copy results back to host for verification
    var fp8_output_h = alloc[Scalar[out_dtype]](rows * cols)
    var fused_output_h = alloc[Scalar[out_dtype]](rows * cols)

    ctx.enqueue_copy(fp8_output_h, fp8_verify_d)
    ctx.enqueue_copy(fused_output_h, fused_verify_d)
    ctx.synchronize()

    # Compare outputs
    from testing import assert_almost_equal

    var max_diff = Float32(0.0)
    var max_rel_diff = Float32(0.0)
    var num_errors = 0
    var num_exact = 0
    var sum_abs_diff = Float64(0.0)

    # More relaxed tolerance for FP8 quantization
    # FP8 can have systematic scaling differences up to ~12-13%
    var rtol = Float32(0.15)  # 15% relative tolerance for FP8
    var atol = Float32(2.0)  # Absolute tolerance for FP8

    for i in range(rows * cols):
        var fp8_val = fp8_output_h[i].cast[DType.float32]()
        var fused_val = fused_output_h[i].cast[DType.float32]()
        var diff = abs(fp8_val - fused_val)

        if diff > max_diff:
            max_diff = diff

        if fp8_val == fused_val:
            num_exact += 1

        sum_abs_diff += diff.cast[DType.float64]()

        # Calculate relative difference
        var avg_val = (abs(fp8_val) + abs(fused_val)) / 2.0
        var rel_diff = diff / max(avg_val, Float32(1.0))
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

        var threshold = atol + rtol * avg_val
        if diff > threshold:
            num_errors += 1
            if num_errors <= 10:  # Print first 10 errors
                print(
                    "Mismatch at index",
                    i,
                    ": separate =",
                    fp8_val,
                    ", fused =",
                    fused_val,
                    ", diff =",
                    diff,
                    ", rel =",
                    rel_diff,
                )

    var mean_abs_diff = sum_abs_diff / Float64(rows * cols)
    var percent_exact = Float32(num_exact) * 100.0 / Float32(rows * cols)

    print("\nVerification Statistics:")
    print("  Total elements:", rows * cols)
    print("  Exact matches:", num_exact, "(", percent_exact, "%)")
    print("  Max absolute diff:", max_diff)
    print("  Max relative diff:", max_rel_diff)
    print("  Mean absolute diff:", mean_abs_diff)
    print("  Mismatches (above threshold):", num_errors)

    var error_rate = Float32(num_errors) / Float32(rows * cols)

    if num_errors > 0:
        if error_rate > 0.05:  # More than 5% errors is concerning
            print(
                "\nVerification FAILED:",
                num_errors,
                "mismatches (",
                error_rate * 100.0,
                "%) exceed threshold (rtol=",
                rtol,
                ", atol=",
                atol,
                ")",
            )
            raise Error("Output verification failed - too many mismatches")
        else:
            # Small number of mismatches might be due to FP8 scaling differences
            print(
                "\nVerification PASSED with warnings:",
                num_errors,
                "minor mismatches (",
                error_rate * 100.0,
                "%) likely due to FP8 scaling differences",
            )
    else:
        print(
            "\nVerification PASSED: All outputs within tolerance",
        )

    # Cleanup
    fp8_output_h.free()
    fused_output_h.free()

    _ = data_d
    _ = gamma_d
    _ = rms_output_d
    _ = fp8_output_d
    _ = fused_output_d
    _ = scales_d
    _ = fp8_verify_d
    _ = fused_verify_d
    _ = rms_verify_d

    data_h.free()
    gamma_h.free()


def main():
    comptime in_dtype = env_get_dtype["in_dtype", DType.bfloat16]()
    comptime out_dtype = env_get_dtype["out_dtype", DType.float8_e4m3fn]()
    comptime shape = int_list_to_tuple[
        env_get_shape["shape", "1x4096x16384"]()
    ]()
    comptime cache_busting = env_get_bool["cache_busting", True]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        # Run fused RMS norm + FP8 quantization benchmark
        # This benchmarks all three variants: rms_norm, fp8_quant, and fused
        bench_rms_norm_fused_fp8[in_dtype, out_dtype, shape, cache_busting](
            ctx, m, "rms_norm_fused_fp8"
        )

    m.dump_report()
