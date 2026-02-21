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
"""Tests for RMSNorm with fused residual connection."""

from math import sqrt
from sys.info import CompilationTarget

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import alloc
from random import Random
from state_space.rms_norm_fused_residual import rms_norm_fused_residual_cpu
from testing import TestSuite, assert_almost_equal

from utils.index import Index, IndexList


fn compute_rms_ref[
    dtype: DType
](
    data_ptr: UnsafePointer[Scalar[dtype]], size: Int, eps: Scalar[dtype]
) -> Scalar[DType.float32]:
    """Compute reference RMS value."""
    var sum_of_squares = Float32()
    for i in range(size):
        var d = data_ptr[i].cast[DType.float32]()
        sum_of_squares += d * d
    return sqrt((sum_of_squares / Float32(size)) + eps.cast[DType.float32]())


fn run_rms_norm_fused_residual_cpu[
    dtype: DType, rank: Int
](
    shape: IndexList[rank],
    rtol: Float64 = 0.001,
    dropout_p: Float64 = 0.0,
    seed: UInt64 = 0,
) raises:
    """Test rms_norm_fused_residual CPU implementation."""
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate memory
    var input_ptr = alloc[Scalar[dtype]](rows * cols)
    var residual_ptr = alloc[Scalar[dtype]](rows * cols)
    var output_ptr = alloc[Scalar[dtype]](rows * cols)
    var residual_output_ptr = alloc[Scalar[dtype]](rows * cols)
    var gamma_ptr = alloc[Scalar[dtype]](cols)

    # Initialize input data
    for i in range(rows * cols):
        input_ptr[i] = Scalar[dtype](Float64(i % 10) * 0.1)
        residual_ptr[i] = Scalar[dtype](Float64((i + 3) % 7) * 0.15)

    # Initialize gamma (weight)
    for i in range(cols):
        gamma_ptr[i] = Scalar[dtype](Float64(i + cols) / Float64(cols))

    # Create tensors
    comptime layout_nd = Layout.row_major[rank]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var input_tensor = LayoutTensor[dtype, layout_nd, MutAnyOrigin](
        input_ptr,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var residual_tensor = LayoutTensor[dtype, layout_nd, MutAnyOrigin](
        residual_ptr,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var output_tensor = LayoutTensor[dtype, layout_nd, MutAnyOrigin](
        output_ptr,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var residual_output_tensor = LayoutTensor[dtype, layout_nd, MutAnyOrigin](
        residual_output_ptr,
        RuntimeLayout[layout_nd].row_major(shape),
    )
    var gamma_tensor = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_ptr,
        RuntimeLayout[layout_1d].row_major(IndexList[1](cols)),
    )

    var epsilon = Scalar[dtype](1e-5)
    var weight_offset = Scalar[dtype](0.0)

    # Define input functions
    @__copy_capture(input_tensor)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return input_tensor.load[width=width](rebind[IndexList[rank]](coords))

    @__copy_capture(residual_tensor)
    @always_inline
    @parameter
    fn residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return residual_tensor.load[width=width](
            rebind[IndexList[rank]](coords)
        )

    # Define output functions
    @__copy_capture(output_tensor)
    @always_inline
    @parameter
    fn output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_tensor.store[width=width](coords, val)

    @__copy_capture(residual_output_tensor)
    @always_inline
    @parameter
    fn residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        residual_output_tensor.store[width=width](coords, val)

    # Read from the residual output buffer written by the first pass.
    # This correctly handles both the dropout and non-dropout paths:
    # the first pass writes dropout(input) + residual into residual_output_tensor,
    # and the normalization pass must read those same values.
    @__copy_capture(residual_output_tensor)
    @always_inline
    @parameter
    fn residual_read_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return residual_output_tensor.load[width=width](
            rebind[IndexList[rank]](coords)
        )

    var dropout_p_scalar = Scalar[dtype](dropout_p)

    # Run the kernel
    rms_norm_fused_residual_cpu[
        input_fn,
        residual_input_fn,
        output_fn,
        residual_output_fn,
        residual_read_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma_tensor,
        epsilon,
        weight_offset,
        dropout_p=dropout_p_scalar,
        seed=seed,
    )

    # Compute dropout scale
    var dropout_scale = Scalar[dtype](1.0)
    var zero_scalar = Scalar[dtype](0.0)
    if dropout_p_scalar > zero_scalar:
        var one_scalar = Scalar[dtype](1.0)
        dropout_scale = one_scalar / (one_scalar - dropout_p_scalar)

    # Verify results
    for r in range(rows):
        # Compute expected residual output: dropout(input) + residual
        var sum_ptr = alloc[Scalar[dtype]](cols)
        for c in range(cols):
            var idx = r * cols + c
            var input_val = input_ptr[idx]

            # Apply the same deterministic dropout as the kernel
            if dropout_p_scalar > zero_scalar:
                var element_offset = r * cols + c
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p_scalar:
                    input_val = input_val * dropout_scale
                else:
                    input_val = zero_scalar

            sum_ptr[c] = input_val + residual_ptr[idx]

        # Verify residual output
        for c in range(cols):
            var idx = r * cols + c
            assert_almost_equal(sum_ptr[c], residual_output_ptr[idx], rtol=rtol)

        # Compute RMS of the sum
        var rms_val = compute_rms_ref(sum_ptr, cols, epsilon)

        # Verify normalized output
        for c in range(cols):
            var idx = r * cols + c
            var sum_val = sum_ptr[c].cast[DType.float32]()
            var expected_norm = (sum_val / rms_val).cast[dtype]() * (
                gamma_ptr[c] + weight_offset
            )
            assert_almost_equal(expected_norm, output_ptr[idx], rtol=rtol)

        sum_ptr.free()

    # Cleanup
    input_ptr.free()
    residual_ptr.free()
    output_ptr.free()
    residual_output_ptr.free()
    gamma_ptr.free()


fn test_rms_norm_fused_residual_float32_2d() raises:
    """Test rms_norm_fused_residual with float32 and 2D shape."""
    run_rms_norm_fused_residual_cpu[DType.float32](Index(4, 16), rtol=1e-3)


fn test_rms_norm_fused_residual_float32_small() raises:
    """Test rms_norm_fused_residual with small dimensions."""
    run_rms_norm_fused_residual_cpu[DType.float32](Index(2, 8), rtol=1e-3)


fn test_rms_norm_fused_residual_float32_large_cols() raises:
    """Test rms_norm_fused_residual with larger column count."""
    run_rms_norm_fused_residual_cpu[DType.float32](Index(2, 128), rtol=1e-3)


fn test_rms_norm_fused_residual_float32_3d() raises:
    """Test rms_norm_fused_residual with 3D shape."""
    run_rms_norm_fused_residual_cpu[DType.float32](Index(2, 3, 16), rtol=1e-3)


fn test_rms_norm_fused_residual_bfloat16() raises:
    """Test rms_norm_fused_residual with bfloat16."""

    comptime if not CompilationTarget.has_neon():
        run_rms_norm_fused_residual_cpu[DType.bfloat16](Index(4, 16), rtol=1e-2)


# =============================================================================
# Dropout tests (dropout_p > 0)
# =============================================================================


fn test_rms_norm_fused_residual_dropout_float32_2d() raises:
    """Test rms_norm_fused_residual CPU with dropout enabled (float32, 2D)."""
    run_rms_norm_fused_residual_cpu[DType.float32](
        Index(4, 16), rtol=1e-3, dropout_p=0.3, seed=42
    )


fn test_rms_norm_fused_residual_dropout_float32_3d() raises:
    """Test rms_norm_fused_residual CPU with dropout enabled (float32, 3D)."""
    run_rms_norm_fused_residual_cpu[DType.float32](
        Index(2, 3, 16), rtol=1e-3, dropout_p=0.5, seed=123
    )


fn test_rms_norm_fused_residual_dropout_float32_large_cols() raises:
    """Test CPU dropout path with larger column count."""
    run_rms_norm_fused_residual_cpu[DType.float32](
        Index(2, 128), rtol=1e-3, dropout_p=0.1, seed=7
    )


fn test_rms_norm_fused_residual_dropout_bfloat16() raises:
    """Test rms_norm_fused_residual CPU with dropout enabled (bfloat16)."""

    comptime if not CompilationTarget.has_neon():
        run_rms_norm_fused_residual_cpu[DType.bfloat16](
            Index(4, 16), rtol=1e-2, dropout_p=0.2, seed=99
        )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
