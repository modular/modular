# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from algorithm.functional import elementwise
from gpu.host import DeviceContext
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._fillers import random
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_rms_norm_fused_residual_add_gpu[
    rank: Int, //,
    dtype: DType,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    comptime layout = Layout.row_major[rank]()
    comptime layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var runtime_layout = RuntimeLayout[layout].row_major(shape)
    var param_shape = Index(cols)
    var param_runtime_layout = RuntimeLayout[layout_1d].row_major(param_shape)

    # Allocate device buffers
    var data_device = ctx.enqueue_create_buffer[dtype](shape.flattened_length())
    var unfused_intermediate_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var result_unfused_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var result_fused_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var residual_fused_output_device = ctx.enqueue_create_buffer[dtype](
        shape.flattened_length()
    )
    var gamma1_device = ctx.enqueue_create_buffer[dtype](cols)
    var gamma2_device = ctx.enqueue_create_buffer[dtype](cols)

    # Initialize input data on host
    with data_device.map_to_host() as data_host:
        var data_host_tensor = LayoutTensor[dtype, layout](
            data_host, runtime_layout
        )
        random(data_host_tensor)

    with gamma1_device.map_to_host() as gamma1_host:
        var gamma1_host_tensor = LayoutTensor[dtype, layout_1d](
            gamma1_host, param_runtime_layout
        )
        random(gamma1_host_tensor)

    with gamma2_device.map_to_host() as gamma2_host:
        var gamma2_host_tensor = LayoutTensor[dtype, layout_1d](
            gamma2_host, param_runtime_layout
        )
        random(gamma2_host_tensor)

    # Initialize output buffers with zeros
    with unfused_intermediate_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with result_unfused_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with result_fused_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    with residual_fused_output_device.map_to_host() as host:
        for i in range(len(host)):
            host[i] = 0

    # Create device layout tensors
    var data_buf = LayoutTensor[dtype, layout](data_device, runtime_layout)
    var gamma1 = LayoutTensor[dtype, layout_1d](
        gamma1_device, param_runtime_layout
    )
    var gamma2 = LayoutTensor[dtype, layout_1d](
        gamma2_device, param_runtime_layout
    )
    var result_fused_buf = LayoutTensor[dtype, layout](
        result_fused_device, runtime_layout
    )
    var result_unfused_buf = LayoutTensor[dtype, layout](
        result_unfused_device, runtime_layout
    )
    var unfused_intermediate_buf = LayoutTensor[dtype, layout](
        unfused_intermediate_device, runtime_layout
    )
    var residual_fused_output_buf = LayoutTensor[dtype, layout](
        residual_fused_output_device, runtime_layout
    )

    var epsilon1 = Scalar[dtype](0.001)
    var epsilon2 = Scalar[dtype](0.002)
    var weight_offset1 = Scalar[dtype](0.0)
    var weight_offset2 = Scalar[dtype](0.0)

    # Test fused operation
    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return data_buf.ptr.load[width=width](idx)

    @parameter
    @always_inline
    @__copy_capture(data_buf)
    fn residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return data_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(result_fused_buf)
    @parameter
    fn fused_output_fn[
        width: Int, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[dtype, width]) -> None:
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
        width: Int, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[dtype, width]) -> None:
        var idx = residual_fused_output_buf.runtime_layout(
            RuntimeTuple[
                fill_like(residual_fused_output_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        residual_fused_output_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Call fused kernel
    rms_norm_fused_residual_add[
        input_fn,
        residual_input_fn,
        fused_output_fn,
        fused_residual_output_fn,
        target="gpu",
        multiply_before_cast=True,
    ](
        shape,
        gamma1,
        epsilon1,
        weight_offset1,
        gamma2,
        epsilon2,
        weight_offset2,
        ctx,
    )

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    @parameter
    fn unfused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        unfused_intermediate_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Step 1: First RMS norm
    rms_norm_gpu[input_fn, unfused_output_fn, multiply_before_cast=True](
        shape, gamma1, epsilon1, weight_offset1, ctx
    )

    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf, data_buf)
    fn sum_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](coords: IndexList[rank_]):
        var data_idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        var residual_val = data_buf.ptr.load[width=width](data_idx)
        var unfused_idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        var result_val = unfused_intermediate_buf.ptr.load[width=width](
            unfused_idx
        )
        unfused_intermediate_buf.ptr.store[width=width](
            unfused_idx, residual_val + result_val
        )

    elementwise[sum_fn, simd_width_of[dtype](), target="gpu"](
        unfused_intermediate_buf.runtime_layout.shape.value.canonicalize(),
        ctx,
    )

    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    fn unfused_input2_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        return unfused_intermediate_buf.ptr.load[width=width](idx)

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(result_unfused_buf)
    @parameter
    fn unfused_output2_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_unfused_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_unfused_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_unfused_buf.ptr.store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[
        unfused_input2_fn,
        unfused_output2_fn,
        multiply_before_cast=True,
    ](shape, gamma2, epsilon2, weight_offset2, ctx)

    ctx.synchronize()

    # Verify results
    var flattened_size = rows * cols
    with result_fused_device.map_to_host() as result_fused_host:
        with result_unfused_device.map_to_host() as result_unfused_host:
            with residual_fused_output_device.map_to_host() as residual_fused_host:
                with unfused_intermediate_device.map_to_host() as unfused_intermediate_host:
                    for i in range(flattened_size):
                        assert_almost_equal(
                            result_fused_host[i],
                            result_unfused_host[i],
                            rtol=rtol,
                        )
                        assert_almost_equal(
                            residual_fused_host[i],
                            unfused_intermediate_host[i],
                            rtol=rtol,
                        )


def main():
    with DeviceContext() as ctx:
        # Test various shapes similar to test_rms_norm.mojo
        run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(5))
        run_rms_norm_fused_residual_add_gpu[DType.float32](
            ctx, Index(3, 4, 10, 20, 8)
        )
        run_rms_norm_fused_residual_add_gpu[DType.bfloat16](
            ctx, Index(1, 5, 6, 10, 128)
        )
        run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 5))
        run_rms_norm_fused_residual_add_gpu[DType.bfloat16](ctx, Index(2, 55))
        run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(7, 557))
        run_rms_norm_fused_residual_add_gpu[DType.bfloat16](ctx, Index(2, 8191))
        run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 8192))
        run_rms_norm_fused_residual_add_gpu[DType.bfloat16](
            ctx, Index(2, 16384)
        )
        run_rms_norm_fused_residual_add_gpu[DType.bfloat16](
            ctx, Index(2, 16385)
        )

        # TODO(KERN-1951): the following fails with CUDA_ERROR_INVALID_VALUE, not sure why
        # run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 16384))
        # run_rms_norm_fused_residual_add_gpu[DType.float32](ctx, Index(2, 16385))
