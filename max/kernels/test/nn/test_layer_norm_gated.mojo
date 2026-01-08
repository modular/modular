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

from math import exp, rsqrt, sqrt
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
from nn.normalization import layer_norm_gated_cpu
from testing import assert_almost_equal

from utils.index import Index, IndexList


@always_inline
fn silu_ref[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Reference SiLU implementation: x * sigmoid(x) = x / (1 + exp(-x))."""
    var x_f32 = x.cast[DType.float32]()
    var neg_x = -x_f32
    var exp_neg_x = exp(neg_x)
    var one = SIMD[DType.float32, simd_width](1.0)
    var sigmoid_x = one / (one + exp_neg_x)
    return (x_f32 * sigmoid_x).cast[dtype]()


fn run_layer_norm_gated[
    rank: Int,
    //,
    dtype: DType,
    has_z: Bool = True,
    has_bias: Bool = True,
    is_rms_norm: Bool = False,
    norm_before_gate: Bool = True,
](
    shape: IndexList[rank],
    rtol: Float64 = 0.01,
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    comptime layout = Layout.row_major[rank]()
    var input_heap = alloc[Scalar[dtype]](rows * cols)
    var input_h = LayoutTensor[dtype, layout](
        input_heap, RuntimeLayout[layout].row_major(shape)
    )
    var z_heap = alloc[Scalar[dtype]](rows * cols)
    var z_h = LayoutTensor[dtype, layout](
        z_heap, RuntimeLayout[layout].row_major(shape)
    )
    var result_fused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_fused_h = LayoutTensor[dtype, layout](
        result_fused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var result_unfused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_unfused_h = LayoutTensor[dtype, layout](
        result_unfused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var gamma_heap = alloc[Scalar[dtype]](cols)
    var gamma_h = LayoutTensor[dtype, layout_1d](
        gamma_heap, RuntimeLayout[layout_1d].row_major(Index(cols))
    )
    var beta_heap = alloc[Scalar[dtype]](cols)
    var beta_h = LayoutTensor[dtype, layout_1d](
        beta_heap, RuntimeLayout[layout_1d].row_major(Index(cols))
    )

    # Initialize input data
    random(input_h)
    random(z_h)
    random(gamma_h)
    random(beta_h)

    var input_buf = input_h
    var z_buf = z_h
    var gamma = gamma_h
    var beta = beta_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var epsilon = Scalar[dtype](0.001)

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

    @__copy_capture(z_buf)
    @always_inline
    @parameter
    fn z_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = z_buf.runtime_layout(
            RuntimeTuple[fill_like(z_buf.layout.shape, UNKNOWN_VALUE)](coords)
        )
        return z_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var col = coords[_rank - 1]
        var idx = gamma.runtime_layout(
            RuntimeTuple[fill_like(gamma.layout.shape, UNKNOWN_VALUE)](
                IndexList[1](col)
            )
        )
        return gamma.ptr.load[width=width](idx)

    @__copy_capture(beta)
    @always_inline
    @parameter
    fn beta_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var col = coords[_rank - 1]
        var idx = beta.runtime_layout(
            RuntimeTuple[fill_like(beta.layout.shape, UNKNOWN_VALUE)](
                IndexList[1](col)
            )
        )
        return beta.ptr.load[width=width](idx)

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

    # Call fused kernel
    layer_norm_gated_cpu[
        input_fn,
        z_fn,
        gamma_fn,
        beta_fn,
        fused_output_fn,
        has_z=has_z,
        has_bias=has_bias,
        is_rms_norm=is_rms_norm,
        norm_before_gate=norm_before_gate,
    ](shape, epsilon)

    # Test unfused operations for comparison
    comptime simd_width = simd_width_of[dtype]()
    comptime intermediate_type = DType.float32

    # Unfused reference implementation
    for row in range(rows):
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        
        # First pass: compute statistics
        var sum_val = Scalar[intermediate_type](0.0)
        var sum_sq_val = Scalar[intermediate_type](0.0)
        
        for col in range(0, cols, simd_width):
            var actual_width = min(simd_width, cols - col)
            indices[rank - 1] = col
            
            var x_idx = input_buf.runtime_layout(
                RuntimeTuple[fill_like(input_buf.layout.shape, UNKNOWN_VALUE)](
                    indices.canonicalize()
                )
            )
            var x = input_buf.ptr.load[width=simd_width](x_idx).cast[intermediate_type]()
            
            # Apply gating if z is provided and norm_before_gate is False
            if has_z and not norm_before_gate:
                var z_idx = z_buf.runtime_layout(
                    RuntimeTuple[fill_like(z_buf.layout.shape, UNKNOWN_VALUE)](
                        indices.canonicalize()
                    )
                )
                var z = z_buf.ptr.load[width=simd_width](z_idx).cast[intermediate_type]()
                var z_silu = silu_ref[intermediate_type, simd_width](z)
                x = x * z_silu
            
            # Accumulate for mean/variance computation
            for i in range(actual_width):
                sum_val += x[i]
                sum_sq_val += x[i] * x[i]
        
        var mean_val = sum_val / Scalar[intermediate_type](cols)
        var variance_val: Scalar[intermediate_type]
        
        if is_rms_norm:
            variance_val = sum_sq_val / Scalar[intermediate_type](cols)
        else:
            variance_val = (sum_sq_val / Scalar[intermediate_type](cols)) - mean_val * mean_val
        
        var rstd = rsqrt(variance_val + epsilon.cast[intermediate_type]())
        
        # Second pass: normalize and apply transformation
        for col in range(0, cols, simd_width):
            var actual_width = min(simd_width, cols - col)
            indices[rank - 1] = col
            
            var x_idx = input_buf.runtime_layout(
                RuntimeTuple[fill_like(input_buf.layout.shape, UNKNOWN_VALUE)](
                    indices.canonicalize()
                )
            )
            var x = input_buf.ptr.load[width=simd_width](x_idx).cast[intermediate_type]()
            
            # Apply gating if z is provided and norm_before_gate is False
            if has_z and not norm_before_gate:
                var z_idx = z_buf.runtime_layout(
                    RuntimeTuple[fill_like(z_buf.layout.shape, UNKNOWN_VALUE)](
                        indices.canonicalize()
                    )
                )
                var z = z_buf.ptr.load[width=simd_width](z_idx).cast[intermediate_type]()
                var z_silu = silu_ref[intermediate_type, simd_width](z)
                x = x * z_silu
            
            # Normalize
            var x_normalized: SIMD[intermediate_type, simd_width]
            if is_rms_norm:
                x_normalized = x * rstd
            else:
                x_normalized = (x - mean_val) * rstd
            
            # Apply gamma and beta
            var gamma_val = SIMD[intermediate_type, simd_width]()
            var beta_val = SIMD[intermediate_type, simd_width]()
            
            for i in range(actual_width):
                if col + i < cols:
                    var gamma_col_idx = gamma.runtime_layout(
                        RuntimeTuple[fill_like(gamma.layout.shape, UNKNOWN_VALUE)](
                            IndexList[1](col + i)
                        )
                    )
                    gamma_val[i] = gamma.ptr.load[width=1](gamma_col_idx)[0].cast[intermediate_type]()
                    
                    if has_bias:
                        var beta_col_idx = beta.runtime_layout(
                            RuntimeTuple[fill_like(beta.layout.shape, UNKNOWN_VALUE)](
                                IndexList[1](col + i)
                            )
                        )
                        beta_val[i] = beta.ptr.load[width=1](beta_col_idx)[0].cast[intermediate_type]()
                    else:
                        beta_val[i] = Scalar[intermediate_type](0.0)
            
            var output_val = x_normalized * gamma_val
            if has_bias:
                output_val = output_val + beta_val
            
            # Apply gating if z is provided and norm_before_gate is True
            if has_z and norm_before_gate:
                var z_idx = z_buf.runtime_layout(
                    RuntimeTuple[fill_like(z_buf.layout.shape, UNKNOWN_VALUE)](
                        indices.canonicalize()
                    )
                )
                var z = z_buf.ptr.load[width=simd_width](z_idx).cast[intermediate_type]()
                var z_silu = silu_ref[intermediate_type, simd_width](z)
                output_val = output_val * z_silu
            
            # Write output
            var output_final = output_val.cast[dtype]()
            var output_idx = result_unfused_buf.runtime_layout(
                RuntimeTuple[fill_like(result_unfused_buf.layout.shape, UNKNOWN_VALUE)](
                    indices.canonicalize()
                )
            )
            result_unfused_buf.ptr.store[width=simd_width, alignment=1](
                output_idx, output_final
            )

    # Compare results
    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.ptr[i],
            result_unfused_h.ptr[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_heap.free()
    z_heap.free()
    result_fused_heap.free()
    result_unfused_heap.free()
    gamma_heap.free()
    beta_heap.free()


def main():
    # Test CPU LayerNorm with gating (norm_before_gate=True)
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(5))
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(3, 4, 10, 20, 8))
    
    # Test CPU LayerNorm with gating (norm_before_gate=False)
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=False](Index(5))
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=False](Index(2, 128))
    
    # Test CPU RMSNorm with gating
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=False, is_rms_norm=True, norm_before_gate=True](Index(5))
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=False, is_rms_norm=True, norm_before_gate=True](Index(2, 128))
    
    # Test CPU without gating
    run_layer_norm_gated[DType.float32, has_z=False, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(5))
    run_layer_norm_gated[DType.float32, has_z=False, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(2, 128))
    
    # Test CPU without bias
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=False, is_rms_norm=False, norm_before_gate=True](Index(5))
    run_layer_norm_gated[DType.float32, has_z=True, has_bias=False, is_rms_norm=False, norm_before_gate=True](Index(2, 128))
    
    # Note: GPU tests should be added in a separate GPU test file (e.g., in max/kernels/test/gpu/nn/)
    # since DeviceContext requires GPU architecture at compile time

