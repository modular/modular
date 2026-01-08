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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]
from math import exp, rsqrt
from sys.info import simd_width_of

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from gpu.host import DeviceContext, get_gpu_target
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from random import rand
from layout.int_tuple import fill_like
from nn.normalization import layer_norm_gated_cpu, layer_norm_gated_gpu
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


fn run_layer_norm_gated_gpu[
    rank: Int,
    //,
    dtype: DType,
    has_z: Bool = True,
    has_bias: Bool = True,
    is_rms_norm: Bool = False,
    norm_before_gate: Bool = True,
](
    shape: IndexList[rank],
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var input_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var z_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var result_gpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var result_cpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    # Initialize input data
    rand[dtype](input_h, rows * cols)
    rand[dtype](z_h, rows * cols)
    rand[dtype](gamma_h, cols)
    rand[dtype](beta_h, cols)

    # Allocate device memory
    var input_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var z_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var result_gpu_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    # Copy to device
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(z_d, z_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)
    ctx.synchronize()

    # Create LayoutTensors with MutAnyOrigin for GPU (use DeviceBuffer directly)
    comptime layout = Layout.row_major[rank]()
    comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var input_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        input_d, RuntimeLayout[layout].row_major(shape)
    )
    var z_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        z_d, RuntimeLayout[layout].row_major(shape)
    )
    var result_gpu_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        result_gpu_d, RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_d, RuntimeLayout[layout_1d].row_major(Index(cols))
    )
    var beta = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        beta_d, RuntimeLayout[layout_1d].row_major(Index(cols))
    )
    var epsilon = Scalar[dtype](0.001)

    # Test CPU operation for comparison (using host memory)
    comptime layout_host = Layout.row_major[rank]()
    comptime layout_1d_host = Layout.row_major(UNKNOWN_VALUE)
    var input_host = LayoutTensor[dtype, layout_host](
        input_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var z_host = LayoutTensor[dtype, layout_host](
        z_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var result_cpu_host = LayoutTensor[dtype, layout_host](
        result_cpu_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var gamma_host = LayoutTensor[dtype, layout_1d_host](
        gamma_h, RuntimeLayout[layout_1d_host].row_major(Index(cols))
    )
    var beta_host = LayoutTensor[dtype, layout_1d_host](
        beta_h, RuntimeLayout[layout_1d_host].row_major(Index(cols))
    )

    @__copy_capture(input_host)
    @always_inline
    @parameter
    fn cpu_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = input_host.runtime_layout(
            RuntimeTuple[fill_like(input_host.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return input_host.ptr.load[width=width](idx)

    @__copy_capture(z_host)
    @always_inline
    @parameter
    fn cpu_z_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = z_host.runtime_layout(
            RuntimeTuple[fill_like(z_host.layout.shape, UNKNOWN_VALUE)](coords)
        )
        return z_host.ptr.load[width=width](idx)

    @__copy_capture(gamma_host)
    @always_inline
    @parameter
    fn cpu_gamma_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var col = coords[_rank - 1]
        var idx = gamma_host.runtime_layout(
            RuntimeTuple[fill_like(gamma_host.layout.shape, UNKNOWN_VALUE)](
                IndexList[1](col)
            )
        )
        return gamma_host.ptr.load[width=width](idx)

    @__copy_capture(beta_host)
    @always_inline
    @parameter
    fn cpu_beta_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var col = coords[_rank - 1]
        var idx = beta_host.runtime_layout(
            RuntimeTuple[fill_like(beta_host.layout.shape, UNKNOWN_VALUE)](
                IndexList[1](col)
            )
        )
        return beta_host.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(result_cpu_host)
    @parameter
    fn cpu_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_cpu_host.runtime_layout(
            RuntimeTuple[
                fill_like(result_cpu_host.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_cpu_host.ptr.store[width=width, alignment=alignment](idx, val)

    # Call CPU kernel
    layer_norm_gated_cpu[
        cpu_input_fn,
        cpu_z_fn,
        cpu_gamma_fn,
        cpu_beta_fn,
        cpu_output_fn,
        has_z=has_z,
        has_bias=has_bias,
        is_rms_norm=is_rms_norm,
        norm_before_gate=norm_before_gate,
    ](shape, epsilon)

    # Test GPU operation
    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn gpu_input_fn[
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
    fn gpu_z_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = z_buf.runtime_layout(
            RuntimeTuple[fill_like(z_buf.layout.shape, UNKNOWN_VALUE)](coords)
        )
        return z_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gpu_gamma_fn[
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
    fn gpu_beta_fn[
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
    @__copy_capture(result_gpu_buf)
    @parameter
    fn gpu_output_fn[
        width: Int, rank: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_gpu_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_gpu_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_gpu_buf.ptr.store[width=width, alignment=alignment](idx, val)

    # Call GPU kernel
    layer_norm_gated_gpu[
        gpu_input_fn,
        gpu_z_fn,
        gpu_gamma_fn,
        gpu_beta_fn,
        gpu_output_fn,
        has_z=has_z,
        has_bias=has_bias,
        is_rms_norm=is_rms_norm,
        norm_before_gate=norm_before_gate,
    ](
        shape,
        beta,
        epsilon,
        ctx=ctx,
    )

    # Copy GPU result back to host
    ctx.enqueue_copy(result_gpu_h, result_gpu_d)
    ctx.synchronize()

    # Compare results
    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h[i],
            result_cpu_h[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_h.free()
    z_h.free()
    result_gpu_h.free()
    result_cpu_h.free()
    gamma_h.free()
    beta_h.free()


def main():
    with DeviceContext() as ctx:
        # Test GPU LayerNorm with gating (norm_before_gate=True)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(2, 128), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(4, 256), ctx)
        print("✓ LayerNorm with gating (norm_before_gate=True) passed")
        
        # Test GPU LayerNorm with gating (norm_before_gate=False)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=False](Index(2, 128), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=True, is_rms_norm=False, norm_before_gate=False](Index(4, 256), ctx)
        print("✓ LayerNorm with gating (norm_before_gate=False) passed")
        
        # Test GPU RMSNorm with gating
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=False, is_rms_norm=True, norm_before_gate=True](Index(2, 64), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=False, is_rms_norm=True, norm_before_gate=True](Index(2, 128), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=False, is_rms_norm=True, norm_before_gate=True](Index(4, 256), ctx)
        print("✓ RMSNorm with gating passed")
        
        # Test GPU without gating
        run_layer_norm_gated_gpu[DType.float32, has_z=False, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(2, 128), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=False, has_bias=True, is_rms_norm=False, norm_before_gate=True](Index(4, 256), ctx)
        print("✓ LayerNorm without gating passed")
        
        # Test GPU LayerNorm without bias
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=False, is_rms_norm=False, norm_before_gate=True](Index(2, 128), ctx)
        run_layer_norm_gated_gpu[DType.float32, has_z=True, has_bias=False, is_rms_norm=False, norm_before_gate=True](Index(4, 256), ctx)
        print("✓ LayerNorm without bias passed")

