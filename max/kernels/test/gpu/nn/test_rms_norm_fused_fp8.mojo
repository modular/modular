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

"""Tests for fused RMSNorm + FP8 quantization kernel."""

from buffer import NDBuffer
from gpu.host import DeviceContext, DeviceBuffer
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from layout.int_tuple import fill_like
from memory import LegacyUnsafePointer, bitcast
from runtime.asyncrt import DeviceContextPtr
from utils.index import Index, IndexList
from math import rsqrt
from utils.numerics import max_finite, min_finite

from nn.normalization import rms_norm_fused_fp8

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


fn initialize_test_data[
    dtype: DType
](data: UnsafePointer[Scalar[dtype]], size: Int):
    """Initialize test data with diverse positive values to avoid FP8 saturation.

    Creates a mix of small, medium, and large magnitudes (0.05 to 3.0) that will
    produce a good distribution after RMSNorm without hitting FP8 edge cases.
    """
    for i in range(size):
        var pattern = i % 20
        var val: Float32

        # Use a wider range [0.05, 3.0] with varied spacing - all positive
        if pattern < 5:
            # Small values: 0.05, 0.1, 0.15, 0.2, 0.25
            val = Float32(0.05 + Float64(pattern) * 0.05)
        elif pattern < 10:
            # Medium values: 0.3, 0.4, 0.5, 0.6, 0.7
            val = Float32(0.3 + Float64((pattern - 5)) * 0.1)
        elif pattern < 15:
            # Medium-large values: 0.8, 1.0, 1.2, 1.4, 1.6
            val = Float32(0.8 + Float64((pattern - 10)) * 0.2)
        else:
            # Large values: 1.8, 2.1, 2.4, 2.7, 3.0
            val = Float32(1.8 + Float64((pattern - 15)) * 0.3)

        data[i] = val.cast[dtype]()


fn compute_reference_dynamic_scaling[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
](
    input_data: UnsafePointer[Scalar[in_dtype]],
    gamma_data: UnsafePointer[Scalar[in_dtype]],
    output_data: UnsafePointer[Scalar[out_dtype]],
    scales_data: UnsafePointer[Scalar[scales_dtype]],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    weight_offset: Float32,
    scale_ub: Float32,
):
    """Compute reference RMSNorm + dynamic FP8 quantization on host."""
    var fp8_max = Float32(max_finite[out_dtype]())
    var fp8_min = Float32(min_finite[out_dtype]())

    # Allocate temporary storage for normalized values
    var temp_storage = UnsafePointer[Scalar[DType.float32]].alloc(cols)

    for row in range(rows):
        # Step 1: Compute mean square for RMSNorm
        var mean_square = Float32(0.0)
        for col in range(cols):
            var val = input_data[row * cols + col].cast[DType.float32]()
            mean_square += val * val
        mean_square = mean_square / Float32(cols)

        # Step 2: Compute normalization factor
        var norm_factor = rsqrt(mean_square + epsilon)

        # Step 3: Normalize and apply gamma (store in temp buffer)
        var row_max = Float32(0.0)
        for col in range(cols):
            var val = input_data[row * cols + col].cast[DType.float32]()
            var normalized = val * norm_factor
            var gamma_val = gamma_data[col].cast[DType.float32]()
            var scaled_val = normalized * (gamma_val + weight_offset)

            # Track max for dynamic scaling
            row_max = max(row_max, abs(scaled_val))

            # Store normalized value in temporary buffer
            temp_storage[col] = scaled_val

        # Step 4: Compute scale and quantize (clamping row_max to scale_ub)
        # Match kernel precision: scale is computed in scales_dtype
        var clamped_max_sd = min(
            row_max.cast[scales_dtype](),
            scale_ub.cast[scales_dtype](),
        )
        var scale_sd = (
            clamped_max_sd / max_finite[out_dtype]().cast[scales_dtype]()
        )
        scales_data[row] = scale_sd
        var scale_factor_recip = (
            Float32(0.0) if scale_sd
            == 0.0 else 1.0 / scale_sd.cast[DType.float32]()
        )

        for col in range(cols):
            var scaled_val = temp_storage[col]
            var quantized = scaled_val * scale_factor_recip
            quantized = max(fp8_min, min(fp8_max, quantized))
            output_data[row * cols + col] = quantized.cast[out_dtype]()

    temp_storage.free()


fn test_dynamic[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    weight_offset: Float32 = 0.0,
    scale_ub: Float32 = 448.0,
) raises:
    """Test arbitrary rank tensor with dynamic scaling."""
    var input_size = shape.flattened_length()
    var cols = shape[rank - 1]
    var rows = input_size // cols

    # Allocate and initialize host memory
    var in_host = UnsafePointer[Scalar[in_dtype]].alloc(input_size)
    var out_host = UnsafePointer[Scalar[out_dtype]].alloc(input_size)
    var gamma_host = UnsafePointer[Scalar[in_dtype]].alloc(cols)
    var scales_host = UnsafePointer[Scalar[scales_dtype]].alloc(rows)
    var expected_host = UnsafePointer[Scalar[out_dtype]].alloc(input_size)
    var expected_scales_host = UnsafePointer[Scalar[scales_dtype]].alloc(rows)

    # Initialize with diverse values to avoid FP8 saturation
    initialize_test_data(in_host, input_size)
    for i in range(cols):
        # Gamma values between 0.5 and 1.5 to create variety after normalization
        gamma_host[i] = Scalar[in_dtype](0.5 + Float64((i % 11)) * 0.1)

    # Cast epsilon and weight_offset to in_dtype (matching kernel signature),
    # then back to Float32 so reference and kernel see the same values.
    var epsilon_id = Scalar[in_dtype](1e-5)
    var weight_offset_id = Scalar[in_dtype](weight_offset)
    var epsilon_f32 = epsilon_id.cast[DType.float32]()
    var weight_offset_f32 = weight_offset_id.cast[DType.float32]()

    # Compute reference
    compute_reference_dynamic_scaling[in_dtype, out_dtype](
        in_host,
        gamma_host,
        expected_host,
        expected_scales_host,
        rows,
        cols,
        epsilon_f32,
        weight_offset_f32,
        scale_ub,
    )

    # Setup GPU
    var in_device = ctx.enqueue_create_buffer[in_dtype](input_size)
    var out_device = ctx.enqueue_create_buffer[out_dtype](input_size)
    var gamma_device = ctx.enqueue_create_buffer[in_dtype](cols)
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](rows)

    ctx.enqueue_copy(in_device, in_host)
    ctx.enqueue_copy(gamma_device, gamma_host)

    comptime layout_nd = Layout.row_major[rank]()
    comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)

    var in_tensor = LayoutTensor[in_dtype, layout_nd](
        in_device, RuntimeLayout[layout_nd].row_major(shape)
    )
    var out_tensor = LayoutTensor[out_dtype, layout_nd](
        out_device, RuntimeLayout[layout_nd].row_major(shape)
    )

    var param_shape = Index(cols)
    var gamma_tensor = TileTensor(gamma_device, row_major(Coord(param_shape)))

    var scales_tensor = LayoutTensor[scales_dtype, layout_1d](
        scales_device, RuntimeLayout[layout_1d].row_major(Index(rows))
    )

    var out_buffer = NDBuffer[mut=True, out_dtype, rank, MutAnyOrigin](
        out_tensor.ptr, shape
    )
    var scale_shape = shape
    scale_shape[rank - 1] = 1
    var scale_buffer = NDBuffer[mut=True, scales_dtype, rank, MutAnyOrigin](
        scales_tensor.ptr, scale_shape
    )

    @__copy_capture(in_tensor)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var idx_linear = in_tensor.runtime_layout(
            RuntimeTuple[fill_like(in_tensor.layout.shape, UNKNOWN_VALUE)](idx)
        )
        return in_tensor.ptr.load[width=width, alignment=width](idx_linear)

    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        scales_dtype,
        rank,
        input_fn,
        target="gpu",
    ](
        shape,
        out_buffer,
        gamma_tensor,
        epsilon_id,
        weight_offset_id,
        DeviceContextPtr(ctx),
        scale_ub,
        scale_buffer,
    )

    # Verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.enqueue_copy(scales_host, scales_device)
    ctx.synchronize()

    var num_mismatches = 0
    for i in range(input_size):
        if bitcast[DType.uint8](out_host[i]) != bitcast[DType.uint8](
            expected_host[i]
        ):
            num_mismatches += 1
            if num_mismatches <= 5:
                print("Mismatch at", i)

    if num_mismatches > 0:
        raise Error("Higher rank tensor test failed")

    print(
        "✓ Rank-",
        rank,
        " tensor test passed (dynamic, weight_offset=",
        weight_offset,
        ") ",
        shape,
    )

    # Cleanup
    in_host.free()
    out_host.free()
    gamma_host.free()
    scales_host.free()
    expected_host.free()
    expected_scales_host.free()


fn main() raises:
    print("Running fused RMSNorm + FP8 tests...")
    var ctx = DeviceContext()

    comptime for scales_dtype in [DType.float32, DType.bfloat16]:
        print("\nTesting scales dtype: ", scales_dtype)
        # Rank-2 tests: Small sizes (warp-tiling)
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(32, 256)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(8, 512)
        )

        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(64, 4096)
        )

        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 8192)
        )

        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 16384)
        )

        # Rank-2 tests: Large (block-tiling)
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 32768)
        )

        # Rank-2 tests: Non-power-of-2 dimensions
        print("\nTesting non-power-of-2 dimensions...")
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(13, 97)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(7, 333)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(17, 513)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(23, 1001)
        )

        # Rank-3 and Rank-4 tests
        print("\nTesting higher rank tensors...")
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(2, 16, 256)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(3, 5, 97)
        )  # Non-power-of-2

        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 4, 8, 128)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 3, 5, 64)
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 4](
            ctx, Index(2, 3, 7, 97)
        )  # Non-power-of-2

        # Tests with nonzero weight_offset
        print("\nTesting nonzero weight_offset...")
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128), weight_offset=0.5
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128), weight_offset=0.1
        )

        # Tests with low scale_ub to exercise the clamping path
        print("\nTesting low scale_ub...")
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(16, 128), scale_ub=0.5
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 2](
            ctx, Index(32, 256), scale_ub=0.1
        )
        test_dynamic[DType.bfloat16, DType.float8_e4m3fn, scales_dtype, 3](
            ctx, Index(4, 8, 128), scale_ub=0.25
        )

    print("\n✅ All tests passed!")
