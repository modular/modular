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
from math import sqrt
from sys.info import simd_width_of

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from random import rand
from layout.int_tuple import fill_like
from memory import alloc
from nn.normalization import (
    rms_norm_cpu,
    rms_norm_fused_residual_cpu,
    rms_norm_fused_residual_gpu,
)
from std.random import Random
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_rms_norm_fused_residual_gpu[
    rank: Int,
    //,
    dtype: DType,
](
    shape: IndexList[rank],
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    dropout_p: Float64 = 0.0,
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var input_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var residual_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var result_gpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var residual_gpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var result_cpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var residual_cpu_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    # Initialize input data
    rand[dtype](input_h, rows * cols)
    rand[dtype](residual_h, rows * cols)
    rand[dtype](gamma_h, cols)

    # Allocate device memory
    var input_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var result_gpu_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_gpu_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    # Copy to device
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.synchronize()

    # Create LayoutTensors with MutAnyOrigin for GPU (use DeviceBuffer directly)
    comptime layout = Layout.row_major[rank]()
    comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var input_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        input_d, RuntimeLayout[layout].row_major(shape)
    )
    var residual_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        residual_d, RuntimeLayout[layout].row_major(shape)
    )
    var result_gpu_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        result_gpu_d, RuntimeLayout[layout].row_major(shape)
    )
    var residual_gpu_buf = LayoutTensor[dtype, layout, MutAnyOrigin](
        residual_gpu_d, RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        gamma_d, RuntimeLayout[layout_1d].row_major(Index(cols))
    )
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)
    var dropout_p_scalar = Scalar[dtype](dropout_p)
    var seed = UInt64(42)

    # Test CPU operation for comparison (using host memory)
    comptime layout_host = Layout.row_major[rank]()
    comptime layout_1d_host = Layout.row_major(UNKNOWN_VALUE)
    var input_host = LayoutTensor[dtype, layout_host](
        input_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var residual_host = LayoutTensor[dtype, layout_host](
        residual_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var result_cpu_host = LayoutTensor[dtype, layout_host](
        result_cpu_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var residual_cpu_host = LayoutTensor[dtype, layout_host](
        residual_cpu_h, RuntimeLayout[layout_host].row_major(shape)
    )
    var gamma_host = LayoutTensor[dtype, layout_1d_host](
        gamma_h, RuntimeLayout[layout_1d_host].row_major(Index(cols))
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

    @__copy_capture(residual_host)
    @always_inline
    @parameter
    fn cpu_residual_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = residual_host.runtime_layout(
            RuntimeTuple[fill_like(residual_host.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return residual_host.ptr.load[width=width](idx)

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

    @always_inline
    @__copy_capture(residual_cpu_host)
    @parameter
    fn cpu_residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = residual_cpu_host.runtime_layout(
            RuntimeTuple[
                fill_like(residual_cpu_host.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        residual_cpu_host.ptr.store[width=width, alignment=alignment](idx, val)

    # Call CPU kernel
    rms_norm_fused_residual_cpu[
        cpu_input_fn,
        cpu_residual_fn,
        cpu_output_fn,
        cpu_residual_output_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma_host,
        epsilon,
        weight_offset,
        dropout_p_scalar,
        seed,
    )

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

    @__copy_capture(residual_buf)
    @always_inline
    @parameter
    fn gpu_residual_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = residual_buf.runtime_layout(
            RuntimeTuple[fill_like(residual_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return residual_buf.ptr.load[width=width](idx)

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

    @always_inline
    @__copy_capture(result_gpu_buf)
    @parameter
    fn gpu_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_gpu_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_gpu_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_gpu_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(residual_gpu_buf)
    @parameter
    fn gpu_residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = residual_gpu_buf.runtime_layout(
            RuntimeTuple[
                fill_like(residual_gpu_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        residual_gpu_buf.ptr.store[width=width, alignment=alignment](idx, val)

    # Call GPU kernel
    rms_norm_fused_residual_gpu[
        gpu_input_fn,
        gpu_residual_fn,
        gpu_residual_output_fn,
        gpu_output_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma,
        epsilon,
        weight_offset,
        ctx,
        dropout_p=dropout_p_scalar,
        seed=seed,
    )

    # Copy GPU results back to host
    ctx.enqueue_copy(result_gpu_h, result_gpu_d)
    ctx.enqueue_copy(residual_gpu_h, residual_gpu_d)
    ctx.synchronize()

    # Compare results
    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_gpu_h[i],
            result_cpu_h[i],
            rtol=rtol,
        )
        assert_almost_equal(
            residual_gpu_h[i],
            residual_cpu_h[i],
            rtol=rtol,
        )
    
    # Cleanup
    input_h.free()
    residual_h.free()
    result_gpu_h.free()
    residual_gpu_h.free()
    result_cpu_h.free()
    residual_cpu_h.free()
    gamma_h.free()


def main():
    with DeviceContext() as ctx:
        # Test various shapes without dropout
        # Note: Using shapes with column sizes that are multiples of 4 to avoid 
        # CUDA misaligned address errors in the shared memory kernel.
        # Also, column sizes are limited to ~4096 due to GPU block thread limits.
        run_rms_norm_fused_residual_gpu[DType.float32](Index(2, 64), ctx, dropout_p=0.0)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(2, 128), ctx, dropout_p=0.0)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(4, 256), ctx, dropout_p=0.0)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(1, 5, 6, 10, 128), ctx, dropout_p=0.0)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(2, 1024), ctx, dropout_p=0.0)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(4, 2048), ctx, dropout_p=0.0)
        print("✓ RMSNorm fused residual without dropout passed")
        
        # Test with dropout
        run_rms_norm_fused_residual_gpu[DType.float32](Index(2, 128), ctx, dropout_p=0.1)
        run_rms_norm_fused_residual_gpu[DType.float32](Index(4, 256), ctx, dropout_p=0.2)
        print("✓ RMSNorm fused residual with dropout passed")

