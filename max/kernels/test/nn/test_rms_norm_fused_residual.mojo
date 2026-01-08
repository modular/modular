# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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

from math import sqrt
from sys.info import simd_width_of

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import random
from layout.int_tuple import fill_like
from memory import alloc
from nn.normalization import rms_norm_cpu, rms_norm_fused_residual_cpu
from std.random import Random
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_rms_norm_fused_residual[
    rank: Int,
    //,
    dtype: DType,
](shape: IndexList[rank], rtol: Float64 = 0.01, dropout_p: Float64 = 0.0) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    comptime layout = Layout.row_major[rank]()
    var input_heap = alloc[Scalar[dtype]](rows * cols)
    var input_h = LayoutTensor[dtype, layout](
        input_heap, RuntimeLayout[layout].row_major(shape)
    )
    var residual_heap = alloc[Scalar[dtype]](rows * cols)
    var residual_h = LayoutTensor[dtype, layout](
        residual_heap, RuntimeLayout[layout].row_major(shape)
    )
    var unfused_intermediate_heap = alloc[Scalar[dtype]](rows * cols)
    var unfused_intermediate_h = LayoutTensor[dtype, layout](
        unfused_intermediate_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var result_unfused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_unfused_h = LayoutTensor[dtype, layout](
        result_unfused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var result_fused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_fused_h = LayoutTensor[dtype, layout](
        result_fused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var residual_fused_output_heap = alloc[Scalar[dtype]](rows * cols)
    var residual_fused_output_h = LayoutTensor[dtype, layout](
        residual_fused_output_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var gamma_heap = alloc[Scalar[dtype]](cols)
    var gamma_h = LayoutTensor[dtype, layout_1d](
        gamma_heap, RuntimeLayout[layout_1d].row_major(Index(cols))
    )

    # Initialize input data
    random(input_h)
    random(residual_h)
    random(gamma_h)

    var input_buf = input_h
    var residual_buf = residual_h
    var gamma = gamma_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var unfused_intermediate_buf = unfused_intermediate_h
    var residual_fused_output_buf = residual_fused_output_h
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)
    var dropout_p_scalar = Scalar[dtype](dropout_p)
    var zero_scalar = Scalar[dtype](0.0)
    var seed = UInt64(42)

    # Test fused operation
    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = input_buf.runtime_layout(
            RuntimeTuple[fill_like(input_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return input_buf.ptr.load[width=width](idx)

    @__copy_capture(residual_buf)
    @parameter
    @always_inline
    fn residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = residual_buf.runtime_layout(
            RuntimeTuple[fill_like(residual_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return residual_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(result_fused_buf)
    @parameter
    fn fused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_fused_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_fused_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_fused_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(residual_fused_output_buf)
    @parameter
    fn fused_residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = residual_fused_output_buf.runtime_layout(
            RuntimeTuple[
                fill_like(residual_fused_output_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        residual_fused_output_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Call fused kernel
    rms_norm_fused_residual_cpu[
        input_fn,
        residual_input_fn,
        fused_output_fn,
        fused_residual_output_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma,
        epsilon,
        weight_offset,
        dropout_p_scalar,
        seed,
    )

    # Test unfused operations for comparison
    # Step 1: Apply dropout to input if enabled, then add residual
    # We need to match the kernel's row-by-row processing exactly
    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    comptime simd_width = simd_width_of[dtype]()
    
    # Calculate dropout scale if needed
    var one_scalar = Scalar[dtype](1.0)
    var dropout_scale = one_scalar
    if dropout_p_scalar > zero_scalar:
        dropout_scale = one_scalar / (one_scalar - dropout_p_scalar)
    
    # Process row by row to match kernel exactly
    for row in range(prod_all_but_last_dim):
        for col in range(0, last_dim, simd_width):
            var indices = _get_start_indices_of_nth_subvolume(row, shape)
            var input_vals = SIMD[dtype, simd_width](0)
            var residual_vals = SIMD[dtype, simd_width](0)
            
            for i in range(simd_width):
                if col + i < last_dim:
                    indices[rank - 1] = col + i
                    var input_val = input_fn[1](indices.canonicalize())[0]
                    
                    # Apply dropout if enabled (matching kernel exactly)
                    if dropout_p_scalar > zero_scalar:
                        var element_offset = row * last_dim + col + i
                        var generator = Random(seed=seed, offset=UInt64(element_offset))
                        var rng = generator.step_uniform()
                        var rng_val = rng[0].cast[dtype]()
                        if rng_val >= dropout_p_scalar:
                            input_val = input_val * dropout_scale
                        else:
                            input_val = zero_scalar
                    
                    input_vals[i] = input_val
                    residual_vals[i] = residual_input_fn[1](indices.canonicalize())[0]
            
            # Add residual and store in intermediate buffer
            var sum_vals = input_vals + residual_vals
            
            for i in range(simd_width):
                if col + i < last_dim:
                    indices[rank - 1] = col + i
                    var intermediate_idx = unfused_intermediate_buf.runtime_layout(
                        RuntimeTuple[
                            fill_like(
                                unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE
                            )
                        ](indices)
                    )
                    unfused_intermediate_buf.ptr.store[width=1, alignment=1](
                        intermediate_idx, sum_vals[i]
                    )

    # Step 2: Add residual
    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf, residual_buf)
    fn sum_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](coords: IndexList[rank_]):
        var intermediate_idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(
                    unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE
                )
            ](coords)
        )
        var intermediate_val = unfused_intermediate_buf.ptr.load[width=width](
            intermediate_idx
        )
        var residual_idx = residual_buf.runtime_layout(
            RuntimeTuple[fill_like(residual_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        var residual_val = residual_buf.ptr.load[width=width](residual_idx)

        var residual_add_val = intermediate_val + residual_val
        unfused_intermediate_buf.ptr.store[width=width](
            intermediate_idx, residual_add_val
        )

    elementwise[sum_fn, simd_width_of[dtype](), target="cpu"](
        unfused_intermediate_buf.runtime_layout.shape.value.canonicalize(),
    )

    # Step 3: Apply RMSNorm
    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    fn unfused_input_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        return unfused_intermediate_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(result_unfused_buf)
    @parameter
    fn unfused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_unfused_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_unfused_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_unfused_buf.ptr.store[width=width, alignment=alignment](idx, val)

    rms_norm_cpu[
        unfused_input_fn,
        unfused_output_fn,
        multiply_before_cast=True,
    ](shape, gamma, epsilon, weight_offset)

    # Compare results
    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.ptr[i],
            result_unfused_h.ptr[i],
            rtol=rtol,
        )
        assert_almost_equal(
            residual_fused_output_h.ptr[i],
            unfused_intermediate_h.ptr[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_heap.free()
    residual_heap.free()
    unfused_intermediate_heap.free()
    result_unfused_heap.free()
    result_fused_heap.free()
    residual_fused_output_heap.free()
    gamma_heap.free()


def main():
    # Test various shapes without dropout
    run_rms_norm_fused_residual[DType.float32](Index(5), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(3, 4, 10, 20, 8), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(1, 5, 6, 10, 128), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 5), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 55), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(7, 557), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 8191), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 8192), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 16384), dropout_p=0.0)
    run_rms_norm_fused_residual[DType.float32](Index(2, 16385), dropout_p=0.0)

    # Test with dropout
    run_rms_norm_fused_residual[DType.float32](Index(2, 128), dropout_p=0.1)
    run_rms_norm_fused_residual[DType.float32](Index(4, 256), dropout_p=0.2)
    run_rms_norm_fused_residual[DType.float32](Index(2, 512), dropout_p=0.5)

