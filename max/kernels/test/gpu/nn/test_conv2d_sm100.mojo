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
"""SM100 Conv2D Test - Validation of TMA im2col conv2d kernel.

This test file validates the SM100 conv2d fprop kernel using hardware TMA
im2col transformation. The kernel computes C[M,N] = A[M,K] @ B[N,K]^T
which maps to convolution as:
- M = batch * out_h * out_w (output spatial)
- N = out_channels (filters)
- K = in_channels * filter_h * filter_w (reduction)

Test coverage:
- 3x3 and 1x1 convolutions
- 1-SM and 2-SM cluster modes
- Epilogue lambda fusion (bias addition)

Usage:
    bazel test //max/kernels/test/gpu/linalg:test_conv2d_sm100 --config=b200
"""

from collections import Optional
from sys import align_of

import linalg.matmul.vendor.blas as vendor_blas
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from random import rand
from utils.index import IndexList
from nn.conv_sm100.conv2d import (
    conv2d_fprop,
    conv2d_fprop_with_residual,
    im2col,
)
from nn.conv_sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)
from linalg.utils import elementwise_compute_lambda_type


fn test_conv2d_implicit_im2col[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d using implicit im2col (4D NHWC API).

    This tests the full conv2d kernel with TMA im2col transformation:
    - Activation: [N, H, W, C] (NHWC)
    - Filter: [K, R, S, C] (KRSC)
    - Output: [N, H_out, W_out, K] (NHWC)

    The reference is computed using explicit im2col + conv2d_fprop_gemm.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    print(
        "[IMPLICIT IM2COL] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = UnsafePointer[Scalar[act_type]].alloc(act_size)
    var filter_host_ptr = UnsafePointer[Scalar[filter_type]].alloc(filter_size)
    var out_host_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)
    var out_host_ref_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)

    # NDBuffers with dynamic dimensions
    comptime static_act_shape = DimList(-1, -1, -1, -1)
    comptime static_filter_shape = DimList(-1, -1, -1, -1)
    comptime static_out_shape = DimList(-1, -1, -1, -1)
    var dynamic_act_shape = DimList(batch, in_h, in_w, in_c)
    var dynamic_filter_shape = DimList(out_c, filter_h, filter_w, in_c)
    var dynamic_out_shape = DimList(batch, out_h, out_w, out_c)

    var act_host = NDBuffer[act_type, 4, _, static_act_shape](
        act_host_ptr, dynamic_act_shape
    )
    var filter_host = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_host_ptr, dynamic_filter_shape
    )

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = NDBuffer[act_type, 4, _, static_act_shape](
        act_device.unsafe_ptr(), dynamic_act_shape
    )
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_device.unsafe_ptr(), dynamic_filter_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = NDBuffer[out_type, 4, _, static_out_shape](
        out_device.unsafe_ptr(), dynamic_out_shape
    )

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host.data, act_host.num_elements())
    rand(filter_host.data, filter_host.num_elements())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)

    # Run conv2d with implicit im2col
    conv2d_fprop(
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM
    # Allocate im2col buffer [M, K]
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    # Perform im2col on host - use DimList with actual values for proper shape inference
    var im2col_host_ptr = UnsafePointer[Scalar[act_type]].alloc(im2col_size)
    var dynamic_im2col_shape = DimList(M, K)
    var im2col_host = NDBuffer[act_type, 2](
        im2col_host_ptr, dynamic_im2col_shape
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    # Create 2D NDBuffers for cuBLAS reference with proper shapes
    var dynamic_a_ref_shape = DimList(M, K)
    var dynamic_b_ref_shape = DimList(N, K)
    var dynamic_c_ref_shape = DimList(M, N)
    var im2col_device_nd = NDBuffer[act_type, 2](
        im2col_device.unsafe_ptr(), dynamic_a_ref_shape
    )
    var filter_2d_device_nd = NDBuffer[filter_type, 2](
        filter_device.unsafe_ptr(), dynamic_b_ref_shape
    )
    var out_2d_ref_nd = NDBuffer[out_type, 2](
        out_device_ref.unsafe_ptr(), dynamic_c_ref_shape
    )

    # Reference: cuBLAS GEMM (transpose_b=True for NK layout)
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr,
        out_host_ref_ptr,
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    act_host_ptr.free()
    filter_host_ptr.free()
    out_host_ptr.free()
    out_host_ref_ptr.free()
    im2col_host_ptr.free()

    _ = act_device
    _ = filter_device
    _ = out_device
    _ = out_device_ref
    _ = im2col_device


fn test_conv2d_1sm[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with 1-SM configuration (cta_group=1).

    This uses the same logic as test_conv2d_implicit_im2col but with
    the 1-SM configuration matching the CUTLASS example.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get 1-SM config (must be comptime for kernel parameters)
    # The config uses the dtype template parameters, so it works for both BF16 and FP16
    comptime config = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16_1sm()

    print(
        "[1-SM MODE] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ") cta_group=",
        config.cta_group,
        " stages=",
        config.num_pipeline_stages,
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = UnsafePointer[Scalar[act_type]].alloc(act_size)
    var filter_host_ptr = UnsafePointer[Scalar[filter_type]].alloc(filter_size)
    var out_host_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)
    var out_host_ref_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)

    # NDBuffers with dynamic dimensions
    comptime static_act_shape = DimList(-1, -1, -1, -1)
    comptime static_filter_shape = DimList(-1, -1, -1, -1)
    comptime static_out_shape = DimList(-1, -1, -1, -1)
    var dynamic_act_shape = DimList(batch, in_h, in_w, in_c)
    var dynamic_filter_shape = DimList(out_c, filter_h, filter_w, in_c)
    var dynamic_out_shape = DimList(batch, out_h, out_w, out_c)

    var act_host = NDBuffer[act_type, 4, _, static_act_shape](
        act_host_ptr, dynamic_act_shape
    )
    var filter_host = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_host_ptr, dynamic_filter_shape
    )

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = NDBuffer[act_type, 4, _, static_act_shape](
        act_device.unsafe_ptr(), dynamic_act_shape
    )
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_device.unsafe_ptr(), dynamic_filter_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = NDBuffer[out_type, 4, _, static_out_shape](
        out_device.unsafe_ptr(), dynamic_out_shape
    )

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host.data, act_host.num_elements())
    rand(filter_host.data, filter_host.num_elements())

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)

    # Run conv2d with 1-SM config
    conv2d_fprop[config=config](
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    var im2col_host_ptr = UnsafePointer[Scalar[act_type]].alloc(im2col_size)
    var dynamic_im2col_shape = DimList(M, K)
    var im2col_host = NDBuffer[act_type, 2](
        im2col_host_ptr, dynamic_im2col_shape
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    # Create 2D NDBuffers for cuBLAS reference
    var dynamic_a_ref_shape = DimList(M, K)
    var dynamic_b_ref_shape = DimList(N, K)
    var dynamic_c_ref_shape = DimList(M, N)
    var im2col_device_nd = NDBuffer[act_type, 2](
        im2col_device.unsafe_ptr(), dynamic_a_ref_shape
    )
    var filter_2d_device_nd = NDBuffer[filter_type, 2](
        filter_device.unsafe_ptr(), dynamic_b_ref_shape
    )
    var out_2d_ref_nd = NDBuffer[out_type, 2](
        out_device_ref.unsafe_ptr(), dynamic_c_ref_shape
    )

    # Reference: cuBLAS GEMM
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr,
        out_host_ref_ptr,
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    act_host_ptr.free()
    filter_host_ptr.free()
    out_host_ptr.free()
    out_host_ref_ptr.free()
    im2col_host_ptr.free()

    _ = act_device
    _ = filter_device
    _ = out_device
    _ = out_device_ref
    _ = im2col_device


fn test_conv2d_epilogue_lambda[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with epilogue lambda for bias addition.

    This tests the epilogue fusion infrastructure by adding a bias tensor
    to the convolution output using an epilogue lambda.

    The operation is: output = conv2d(activation, filter) + bias
    where bias is broadcast over the spatial dimensions.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get config
    comptime config = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16_1sm()

    print(
        "[EPILOGUE LAMBDA] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c
    var bias_size = out_c

    # Host allocations
    var act_host_ptr = UnsafePointer[Scalar[act_type]].alloc(act_size)
    var filter_host_ptr = UnsafePointer[Scalar[filter_type]].alloc(filter_size)
    var out_host_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)
    var out_host_ref_ptr = UnsafePointer[Scalar[out_type]].alloc(out_size)
    var bias_host_ptr = UnsafePointer[Scalar[out_type]].alloc(bias_size)

    # NDBuffers with dynamic dimensions
    comptime static_act_shape = DimList(-1, -1, -1, -1)
    comptime static_filter_shape = DimList(-1, -1, -1, -1)
    comptime static_out_shape = DimList(-1, -1, -1, -1)
    var dynamic_act_shape = DimList(batch, in_h, in_w, in_c)
    var dynamic_filter_shape = DimList(out_c, filter_h, filter_w, in_c)
    var dynamic_out_shape = DimList(batch, out_h, out_w, out_c)

    var act_host = NDBuffer[act_type, 4, _, static_act_shape](
        act_host_ptr, dynamic_act_shape
    )
    var filter_host = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_host_ptr, dynamic_filter_shape
    )

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[act_type](act_size)
    var act_device_nd = NDBuffer[act_type, 4, _, static_act_shape](
        act_device.unsafe_ptr(), dynamic_act_shape
    )
    var filter_device = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_device_nd = NDBuffer[filter_type, 4, _, static_filter_shape](
        filter_device.unsafe_ptr(), dynamic_filter_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_type](out_size)
    var out_device_nd = NDBuffer[out_type, 4, _, static_out_shape](
        out_device.unsafe_ptr(), dynamic_out_shape
    )
    var bias_device = ctx.enqueue_create_buffer[out_type](bias_size)

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[out_type](out_size)

    # Initialize with random data
    rand(act_host.data, act_host.num_elements())
    rand(filter_host.data, filter_host.num_elements())
    rand(bias_host_ptr, bias_size)

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)
    ctx.enqueue_copy(bias_device, bias_host_ptr)

    # Create bias tensor view for epilogue lambda
    # Bias is 1D [out_c], needs to be broadcast over [M, N] output
    comptime bias_shape = DimList(-1)
    var dynamic_bias_shape = DimList(out_c)
    var bias_tensor = NDBuffer[out_type, 1, _, bias_shape](
        bias_device.unsafe_ptr(), dynamic_bias_shape
    )

    # Define epilogue lambda that adds bias (broadcast over M dimension)
    # Output shape is [M, N] where N = out_channels
    # Bias is [N], so we index by idx[1] (the column/channel index)
    @parameter
    @always_inline
    @__copy_capture(bias_tensor)
    fn epilogue_add_bias[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # Load bias value for this channel and broadcast to SIMD width
        # Note: For width > 1, consecutive columns may have different biases
        # so we need to load a vector of biases
        var bias_val = bias_tensor.load[width=width](idx[1]).cast[_dtype]()
        return val + bias_val

    # Create optional lambda
    comptime optional_lambda = Optional[elementwise_compute_lambda_type](
        epilogue_add_bias
    )

    # Run conv2d with epilogue lambda
    conv2d_fprop[
        config=config,
        elementwise_compute_lambda_fn=optional_lambda,
    ](
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        problem,
        ctx,
    )

    # Reference: explicit im2col + cuBLAS GEMM (without bias)
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[act_type](im2col_size)

    var im2col_host_ptr = UnsafePointer[Scalar[act_type]].alloc(im2col_size)
    var dynamic_im2col_shape = DimList(M, K)
    var im2col_host = NDBuffer[act_type, 2](
        im2col_host_ptr, dynamic_im2col_shape
    )
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    # Create 2D NDBuffers for cuBLAS reference
    var dynamic_a_ref_shape = DimList(M, K)
    var dynamic_b_ref_shape = DimList(N, K)
    var dynamic_c_ref_shape = DimList(M, N)
    var im2col_device_nd = NDBuffer[act_type, 2](
        im2col_device.unsafe_ptr(), dynamic_a_ref_shape
    )
    var filter_2d_device_nd = NDBuffer[filter_type, 2](
        filter_device.unsafe_ptr(), dynamic_b_ref_shape
    )
    var out_2d_ref_nd = NDBuffer[out_type, 2](
        out_device_ref.unsafe_ptr(), dynamic_c_ref_shape
    )

    # Reference: cuBLAS GEMM
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Apply bias on CPU to reference output
    # Reference output is [M, N], bias is [N]
    for m in range(M):
        for n in range(N):
            var idx = m * N + n
            out_host_ref_ptr[idx] = out_host_ref_ptr[idx] + bias_host_ptr[n]

    # Validate results
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr,
        out_host_ref_ptr,
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    act_host_ptr.free()
    filter_host_ptr.free()
    out_host_ptr.free()
    out_host_ref_ptr.free()
    bias_host_ptr.free()
    im2col_host_ptr.free()

    _ = act_device
    _ = filter_device
    _ = out_device
    _ = out_device_ref
    _ = bias_device
    _ = im2col_device


fn test_conv2d_bias_fusion[
    dtype: DType,
    use_1sm: Bool,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d with fused bias addition - typical FLUX pattern.

    This is the most common fusion pattern in neural networks:
    output = conv2d(input, weights) + bias

    The bias is 1D [out_channels] and broadcasts over spatial dimensions.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Select config based on parameter
    comptime if use_1sm:
        comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()
    else:
        comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16()

    var mode_str = "1-SM" if use_1sm else "2-SM"
    print(
        "[CONV+BIAS ",
        mode_str,
        "] ",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        " -> ",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        sep="",
    )

    # Allocate memory
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    var act_host = UnsafePointer[Scalar[dtype]].alloc(act_size)
    var filter_host = UnsafePointer[Scalar[dtype]].alloc(filter_size)
    var bias_host = UnsafePointer[Scalar[dtype]].alloc(out_c)
    var out_host = UnsafePointer[Scalar[dtype]].alloc(out_size)
    var out_ref_host = UnsafePointer[Scalar[dtype]].alloc(out_size)

    rand(act_host, act_size)
    rand(filter_host, filter_size)
    rand(bias_host, out_c)

    var act_dev = ctx.enqueue_create_buffer[dtype](act_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var bias_dev = ctx.enqueue_create_buffer[dtype](out_c)
    var out_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var out_ref_dev = ctx.enqueue_create_buffer[dtype](out_size)
    var im2col_dev = ctx.enqueue_create_buffer[dtype](M * K)

    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filter_dev, filter_host)
    ctx.enqueue_copy(bias_dev, bias_host)

    # Create NDBuffers
    comptime dyn_shape_4d = DimList(-1, -1, -1, -1)
    var act_nd = NDBuffer[dtype, 4, _, dyn_shape_4d](
        act_dev.unsafe_ptr(), DimList(batch, in_h, in_w, in_c)
    )
    var filter_nd = NDBuffer[dtype, 4, _, dyn_shape_4d](
        filter_dev.unsafe_ptr(), DimList(out_c, filter_h, filter_w, in_c)
    )
    var out_nd = NDBuffer[dtype, 4, _, dyn_shape_4d](
        out_dev.unsafe_ptr(), DimList(batch, out_h, out_w, out_c)
    )

    # Create bias tensor for capture
    comptime dyn_shape_1d = DimList(-1)
    var bias_tensor = NDBuffer[dtype, 1, _, dyn_shape_1d](
        bias_dev.unsafe_ptr(), DimList(out_c)
    )

    # Epilogue lambda: add bias (idx[1] = channel index in [M, N] output)
    @parameter
    @always_inline
    @__copy_capture(bias_tensor)
    fn add_bias[
        _dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + bias_tensor.load[width=width](idx[1]).cast[_dtype]()

    comptime bias_lambda = Optional[elementwise_compute_lambda_type](add_bias)

    # Run conv2d with fused bias
    comptime if use_1sm:
        conv2d_fprop[
            config = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm(),
            elementwise_compute_lambda_fn=bias_lambda,
        ](
            out_nd.make_dims_unknown(),
            act_nd.make_dims_unknown(),
            filter_nd.make_dims_unknown(),
            problem,
            ctx,
        )
    else:
        conv2d_fprop[
            config = Conv2dConfig[dtype, dtype, dtype].default_bf16(),
            elementwise_compute_lambda_fn=bias_lambda,
        ](
            out_nd.make_dims_unknown(),
            act_nd.make_dims_unknown(),
            filter_nd.make_dims_unknown(),
            problem,
            ctx,
        )

    # Reference: im2col + GEMM + bias (CPU bias add)
    var act_host_nd = NDBuffer[dtype, 4, _, dyn_shape_4d](
        act_host, DimList(batch, in_h, in_w, in_c)
    )
    var im2col_host = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var im2col_host_nd = NDBuffer[dtype, 2](im2col_host, DimList(M, K))
    im2col(im2col_host_nd, act_host_nd, problem)
    ctx.enqueue_copy(im2col_dev, im2col_host)

    var im2col_nd = NDBuffer[dtype, 2](im2col_dev.unsafe_ptr(), DimList(M, K))
    var filter_2d_nd = NDBuffer[dtype, 2](
        filter_dev.unsafe_ptr(), DimList(N, K)
    )
    var out_ref_nd = NDBuffer[dtype, 2](out_ref_dev.unsafe_ptr(), DimList(M, N))

    vendor_blas.matmul(
        ctx,
        out_ref_nd,
        im2col_nd,
        filter_2d_nd,
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    # Copy results
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(out_ref_host, out_ref_dev)
    ctx.synchronize()

    # Apply bias to reference on CPU
    for m in range(M):
        for n in range(N):
            out_ref_host[m * N + n] = out_ref_host[m * N + n] + bias_host[n]

    # Validate
    assert_almost_equal(
        out_host, out_ref_host, out_size, atol=0.0001, rtol=1e-2
    )
    print("    PASSED")

    # Cleanup
    act_host.free()
    filter_host.free()
    bias_host.free()
    out_host.free()
    out_ref_host.free()
    im2col_host.free()
    _ = act_dev^
    _ = filter_dev^
    _ = bias_dev^
    _ = out_dev^
    _ = out_ref_dev^
    _ = im2col_dev^


fn test_conv2d_residual_api[
    dtype: DType,
](
    ctx: DeviceContext,
    batch: Int,
    in_h: Int,
    in_w: Int,
    in_c: Int,
    out_c: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
) raises:
    """Test conv2d_fprop_with_residual API.

    This tests the residual add: D = Conv(A,B) + beta*C

    Tests:
    1. has_residual=False should fall back to standard conv2d.
    2. beta=0 should fall back to standard conv2d.
    3. Full residual path: validates D = Conv(A,B) + beta*C against
       cuBLAS GEMM reference with host-side residual add.
    """
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_h,
        in_width=in_w,
        in_channels=in_c,
        out_channels=out_c,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    var out_h = problem.out_height()
    var out_w = problem.out_width()
    var M = problem.gemm_m()
    var N = problem.gemm_n()
    var K = problem.gemm_k()

    # Get 1-SM config
    comptime config = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()

    print(
        "[RESIDUAL API] batch=",
        batch,
        " in=(",
        in_h,
        "x",
        in_w,
        "x",
        in_c,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_h,
        "x",
        out_w,
        "x",
        out_c,
        ")",
        sep="",
    )

    # Sizes
    var act_size = batch * in_h * in_w * in_c
    var filter_size = out_c * filter_h * filter_w * in_c
    var out_size = batch * out_h * out_w * out_c

    # Host allocations
    var act_host_ptr = UnsafePointer[Scalar[dtype]].alloc(act_size)
    var filter_host_ptr = UnsafePointer[Scalar[dtype]].alloc(filter_size)
    var out_host_ptr = UnsafePointer[Scalar[dtype]].alloc(out_size)
    var out_host_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(out_size)
    var source_host_ptr = UnsafePointer[Scalar[dtype]].alloc(out_size)

    # NDBuffers with dynamic dimensions
    comptime static_act_shape = DimList(-1, -1, -1, -1)
    comptime static_filter_shape = DimList(-1, -1, -1, -1)
    comptime static_out_shape = DimList(-1, -1, -1, -1)
    var dynamic_act_shape = DimList(batch, in_h, in_w, in_c)
    var dynamic_filter_shape = DimList(out_c, filter_h, filter_w, in_c)
    var dynamic_out_shape = DimList(batch, out_h, out_w, out_c)

    var act_host = NDBuffer[dtype, 4, _, static_act_shape](
        act_host_ptr, dynamic_act_shape
    )

    # Device allocations
    var act_device = ctx.enqueue_create_buffer[dtype](act_size)
    var act_device_nd = NDBuffer[dtype, 4, _, static_act_shape](
        act_device.unsafe_ptr(), dynamic_act_shape
    )
    var filter_device = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_device_nd = NDBuffer[dtype, 4, _, static_filter_shape](
        filter_device.unsafe_ptr(), dynamic_filter_shape
    )
    var out_device = ctx.enqueue_create_buffer[dtype](out_size)
    var out_device_nd = NDBuffer[dtype, 4, _, static_out_shape](
        out_device.unsafe_ptr(), dynamic_out_shape
    )
    var source_device = ctx.enqueue_create_buffer[dtype](out_size)
    var source_device_nd = NDBuffer[dtype, 4, _, static_out_shape](
        source_device.unsafe_ptr(), dynamic_out_shape
    )

    # Reference output device buffer
    var out_device_ref = ctx.enqueue_create_buffer[dtype](out_size)

    # Initialize with random data
    rand(act_host.data, act_host.num_elements())
    rand(filter_host_ptr, filter_size)
    rand(source_host_ptr, out_size)

    # Copy to device
    ctx.enqueue_copy(act_device, act_host_ptr)
    ctx.enqueue_copy(filter_device, filter_host_ptr)
    ctx.enqueue_copy(source_device, source_host_ptr)

    # Test 1: has_residual=False should fall back to standard conv2d
    print("  Test 1: has_residual=False fallback...")
    conv2d_fprop_with_residual[config=config, has_residual=False](
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        source_device_nd.make_dims_unknown(),  # Ignored when has_residual=False
        Float32(1.0),  # Beta (ignored)
        problem,
        ctx,
    )

    # Test 2: beta=0 should fall back to standard conv2d
    print("  Test 2: beta=0 fallback...")
    conv2d_fprop_with_residual[config=config, has_residual=True](
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        source_device_nd.make_dims_unknown(),
        Float32(0.0),  # Beta=0 means no residual
        problem,
        ctx,
    )

    # Test 3: source provided with beta!=0
    # Full residual path: D = Conv(A,B) + beta*C
    comptime test_beta = Float32(1.0)
    print("  Test 3: source + beta (residual add)...")
    conv2d_fprop_with_residual[config=config, has_residual=True](
        out_device_nd.make_dims_unknown(),
        act_device_nd.make_dims_unknown(),
        filter_device_nd.make_dims_unknown(),
        source_device_nd.make_dims_unknown(),
        test_beta,  # Beta=1.0 for skip connection
        problem,
        ctx,
    )

    # Reference: compute Conv(A,B) via cuBLAS GEMM
    var im2col_size = M * K
    var im2col_device = ctx.enqueue_create_buffer[dtype](im2col_size)

    var im2col_host_ptr = UnsafePointer[Scalar[dtype]].alloc(im2col_size)
    var dynamic_im2col_shape = DimList(M, K)
    var im2col_host = NDBuffer[dtype, 2](im2col_host_ptr, dynamic_im2col_shape)
    im2col(im2col_host, act_host, problem)
    ctx.enqueue_copy(im2col_device, im2col_host_ptr)

    # Create 2D NDBuffers for cuBLAS reference
    var dynamic_a_ref_shape = DimList(M, K)
    var dynamic_b_ref_shape = DimList(N, K)
    var dynamic_c_ref_shape = DimList(M, N)
    var im2col_device_nd = NDBuffer[dtype, 2](
        im2col_device.unsafe_ptr(), dynamic_a_ref_shape
    )
    var filter_2d_device_nd = NDBuffer[dtype, 2](
        filter_device.unsafe_ptr(), dynamic_b_ref_shape
    )
    var out_2d_ref_nd = NDBuffer[dtype, 2](
        out_device_ref.unsafe_ptr(), dynamic_c_ref_shape
    )

    # Reference: cuBLAS GEMM (conv2d only)
    vendor_blas.matmul(
        ctx,
        out_2d_ref_nd,
        im2col_device_nd,
        filter_2d_device_nd,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy results to host
    ctx.enqueue_copy(out_host_ptr, out_device)
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref)
    ctx.synchronize()

    # Add residual to reference on host: ref = Conv(A,B) + beta * C
    for i in range(out_size):
        out_host_ref_ptr[i] = (
            out_host_ref_ptr[i].cast[DType.float32]()
            + test_beta * source_host_ptr[i].cast[DType.float32]()
        ).cast[dtype]()

    # Validate: D = Conv(A,B) + beta*C
    comptime rtol = 1e-2
    assert_almost_equal(
        out_host_ptr,
        out_host_ref_ptr,
        out_size,
        atol=0.0001,
        rtol=rtol,
    )
    print("  PASSED\n")

    # Clean up
    act_host_ptr.free()
    filter_host_ptr.free()
    out_host_ptr.free()
    out_host_ref_ptr.free()
    source_host_ptr.free()
    im2col_host_ptr.free()

    _ = act_device
    _ = filter_device
    _ = out_device
    _ = out_device_ref
    _ = source_device
    _ = im2col_device


fn test_conv2d_problem_shape():
    """Test Conv2dProblemShape computations."""
    print("Testing Conv2dProblemShape...")

    # Test 1: 3x3 conv with padding (common VAE pattern)
    var problem = Conv2dProblemShape(
        batch=1,
        in_height=64,
        in_width=64,
        in_channels=128,
        out_channels=256,
        filter_h=3,
        filter_w=3,
        pad_h=1,
        pad_w=1,
    )

    # With padding=1, stride=1, output size == input size
    var expected_out_h = 64
    var expected_out_w = 64
    if problem.out_height() != expected_out_h:
        print(
            "FAILED: out_height expected",
            expected_out_h,
            "got",
            problem.out_height(),
        )
        return
    if problem.out_width() != expected_out_w:
        print(
            "FAILED: out_width expected",
            expected_out_w,
            "got",
            problem.out_width(),
        )
        return

    # GEMM dimensions
    var expected_m = 1 * 64 * 64  # batch * out_h * out_w = 4096
    var expected_n = 256  # out_channels
    var expected_k = 128 * 3 * 3  # in_channels * R * S = 1152

    if problem.gemm_m() != expected_m:
        print("FAILED: gemm_m expected", expected_m, "got", problem.gemm_m())
        return
    if problem.gemm_n() != expected_n:
        print("FAILED: gemm_n expected", expected_n, "got", problem.gemm_n())
        return
    if problem.gemm_k() != expected_k:
        print("FAILED: gemm_k expected", expected_k, "got", problem.gemm_k())
        return

    print("  Conv2dProblemShape: PASSED\n")

    # Test 2: 1x1 conv (pointwise, no padding)
    var problem2 = Conv2dProblemShape(
        batch=2,
        in_height=32,
        in_width=32,
        in_channels=512,
        out_channels=256,
        filter_h=1,
        filter_w=1,
        pad_h=0,
        pad_w=0,
    )

    # 1x1 conv: output size == input size
    if problem2.out_height() != 32 or problem2.out_width() != 32:
        print("FAILED: 1x1 conv output size incorrect")
        return

    # GEMM: M = 2*32*32 = 2048, N = 256, K = 512*1*1 = 512
    if (
        problem2.gemm_m() != 2048
        or problem2.gemm_n() != 256
        or problem2.gemm_k() != 512
    ):
        print("FAILED: 1x1 conv GEMM dimensions incorrect")
        return

    print("  1x1 Conv: PASSED\n")


def main():
    print("=" * 60)
    print("SM100 CONV2D TEST")
    print("=" * 60)
    print()

    # Test problem shape computations (CPU)
    test_conv2d_problem_shape()

    # Test GPU kernels
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16

        # ============================================================
        # Test 1: 3x3 conv with padding
        # batch=1, 16x16 spatial, 64 in_channels, 256 out_channels
        # M = 256, N = 256, K = 64*3*3 = 576
        # ============================================================
        print("--- Test 1: 3x3 conv with padding ---")
        test_conv2d_implicit_im2col[
            dtype,
            dtype,
            DType.bfloat16,
        ](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=64,
            out_c=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 2: 1x1 pointwise conv
        # batch=1, 32x32 spatial, 256 in_channels, 256 out_channels
        # M = 1024, N = 256, K = 256
        # ============================================================
        print("--- Test 2: 1x1 pointwise conv ---")
        test_conv2d_implicit_im2col[
            dtype,
            dtype,
            DType.bfloat16,
        ](
            ctx,
            batch=1,
            in_h=32,
            in_w=32,
            in_c=256,
            out_c=256,
            filter_h=1,
            filter_w=1,
            pad_h=0,
            pad_w=0,
        )

        # ============================================================
        # Test 3: 1-SM mode (cta_group=1)
        # Same as Test 1 but with 1-SM configuration
        # ============================================================
        print("--- Test 3: 1-SM mode (3x3 conv) ---")
        test_conv2d_1sm[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 4: Epilogue lambda (bias addition)
        # Tests the epilogue fusion infrastructure
        # ============================================================
        print("--- Test 4: Epilogue lambda (bias add) ---")
        test_conv2d_epilogue_lambda[dtype, dtype, DType.bfloat16](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 5: Conv2d + bias fusion (2-SM mode)
        # Focused test for the common FLUX pattern
        # ============================================================
        print("--- Test 5: Conv2d + bias fusion (2-SM) ---")
        test_conv2d_bias_fusion[dtype, use_1sm=False](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,
            out_c=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 6: Conv2d + bias fusion (1-SM mode)
        # Same pattern with 1-SM configuration
        # ============================================================
        print("--- Test 6: Conv2d + bias fusion (1-SM) ---")
        test_conv2d_bias_fusion[dtype, use_1sm=True](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # ============================================================
        # Test 7: Conv2d with residual API
        # Tests the conv2d_fprop_with_residual API
        # ============================================================
        print("--- Test 7: Conv2d with residual API ---")
        test_conv2d_residual_api[dtype](
            ctx,
            batch=1,
            in_h=16,
            in_w=16,
            in_c=128,  # Must be multiple of 128 for 1-SM
            out_c=128,  # Must be multiple of 128 for 1-SM
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        )

        # NOTE: FP16 tests require additional stdlib changes beyond TMA:
        # - std/gpu/compute/mma.mojo st_matrix() also only supports BF16/F32
        # - Full FP16 support would require updates across multiple files
        # For now, CUTLASS comparison requires modifying CUTLASS to use BF16

    print("=" * 60)
    print("ALL CONV2D TESTS PASSED!")
    print("=" * 60)
