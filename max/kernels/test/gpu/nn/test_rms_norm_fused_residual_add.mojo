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

from buffer import NDBuffer
from gpu.host import DeviceContext
from nn.normalization import *
from testing import assert_almost_equal
from internal_utils import HostNDBuffer, DeviceNDBuffer, random
from algorithm.functional import elementwise

from utils.index import Index, IndexList


fn run_rms_norm_fused_residual_add_gpu[
    rank: Int, //,
    dtype: DType,
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var data_h = HostNDBuffer[dtype, rank](shape)
    var unfused_intermediate_h = HostNDBuffer[dtype, rank](shape)
    var result_unfused_h = HostNDBuffer[dtype, rank](shape)
    var result_fused_h = HostNDBuffer[dtype, rank](shape)
    var residual_fused_output_h = HostNDBuffer[dtype, rank](shape)
    var gamma1_h = HostNDBuffer[dtype, 1](Index(cols))
    var gamma2_h = HostNDBuffer[dtype, 1](Index(cols))

    # Initialize input data
    random(data_h.tensor)
    random(gamma1_h.tensor)
    random(gamma2_h.tensor)

    # Allocate device memory
    var data_d = data_h.copy_to_device(ctx)
    var gamma1_d = gamma1_h.copy_to_device(ctx)
    var gamma2_d = gamma2_h.copy_to_device(ctx)
    var unfused_intermediate_d = unfused_intermediate_h.copy_to_device(ctx)
    var result_fused_d = result_fused_h.copy_to_device(ctx)
    var result_unfused_d = result_unfused_h.copy_to_device(ctx)
    var residual_fused_output_d = residual_fused_output_h.copy_to_device(ctx)

    var param_shape = Index(cols)

    var data_buf = NDBuffer[dtype, rank](data_d.buffer.unsafe_ptr(), shape)
    var gamma1 = NDBuffer[dtype, 1](gamma1_d.buffer.unsafe_ptr(), param_shape)
    var gamma2 = NDBuffer[dtype, 1](gamma2_d.buffer.unsafe_ptr(), param_shape)
    var result_fused_buf = NDBuffer[dtype, rank](
        result_fused_d.buffer.unsafe_ptr(), shape
    )
    var result_unfused_buf = NDBuffer[dtype, rank](
        result_unfused_d.buffer.unsafe_ptr(), shape
    )
    var unfused_intermediate_buf = NDBuffer[dtype, rank](
        unfused_intermediate_d.buffer.unsafe_ptr(), shape
    )
    var residual_fused_output_buf = NDBuffer[dtype, rank](
        residual_fused_output_d.buffer.unsafe_ptr(), shape
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
    ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @parameter
    @always_inline
    @__copy_capture(data_buf)
    fn residual_input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
        return data_buf.load[width=width](rebind[IndexList[rank]](idx))

    @always_inline
    @__copy_capture(result_fused_buf)
    @parameter
    fn fused_output_fn[
        width: Int, rank_: Int, alignment: Int
    ](idx: IndexList[rank_], val: SIMD[dtype, width]) -> None:
        result_fused_buf.store[width=width, alignment=alignment](
            rebind[IndexList[rank]](idx), val
        )

    @always_inline
    @__copy_capture(residual_fused_output_buf)
    @parameter
    fn fused_residual_output_fn[
        width: Int, rank_: Int, alignment: Int
    ](idx: IndexList[rank_], val: SIMD[dtype, width]) -> None:
        residual_fused_output_buf.store[width=width, alignment=alignment](
            rebind[IndexList[rank]](idx), val
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
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        unfused_intermediate_buf.store[width=width, alignment=alignment](
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
    ](idx: IndexList[rank_]):
        var residual_val = data_buf.load[width=width](
            rebind[IndexList[rank]](idx)
        )
        var result_val = unfused_intermediate_buf.load[width=width](
            rebind[IndexList[rank]](idx)
        )
        unfused_intermediate_buf.store[width=width](
            rebind[IndexList[rank]](idx), residual_val + result_val
        )

    elementwise[sum_fn, simd_width_of[dtype](), target="gpu"](
        unfused_intermediate_buf.dynamic_shape,
        ctx,
    )

    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    fn unfused_input2_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        return unfused_intermediate_buf.load[width=width](
            rebind[IndexList[rank]](idx)
        )

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(result_unfused_buf)
    @parameter
    fn unfused_output2_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        result_unfused_buf.store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[
        unfused_input2_fn,
        unfused_output2_fn,
        multiply_before_cast=True,
    ](shape, gamma2, epsilon2, weight_offset2, ctx)

    ctx.enqueue_copy(result_fused_h.tensor.data, result_fused_d.buffer)
    ctx.enqueue_copy(result_unfused_h.tensor.data, result_unfused_d.buffer)
    ctx.enqueue_copy(
        residual_fused_output_h.tensor.data, residual_fused_output_d.buffer
    )
    ctx.enqueue_copy(
        unfused_intermediate_h.tensor.data, unfused_intermediate_d.buffer
    )
    ctx.synchronize()

    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.tensor.data[i],
            result_unfused_h.tensor.data[i],
            rtol=rtol,
        )
        assert_almost_equal(
            residual_fused_output_h.tensor.data[i],
            unfused_intermediate_h.tensor.data[i],
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
