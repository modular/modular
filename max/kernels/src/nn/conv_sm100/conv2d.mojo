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

"""Public API for SM100 Conv2D forward propagation.

This module provides the high-level API for launching Conv2D fprop kernels
on NVIDIA Blackwell (SM100) GPUs. It handles:

- TMA descriptor setup for activation (with im2col), filter, and output
- Kernel configuration selection
- Kernel launch with proper grid/block dimensions

Usage (4D NHWC API with implicit im2col):
    from nn.conv_sm100 import conv2d_fprop

    var problem = Conv2dProblemShape(
        batch=1,
        in_height=256, in_width=256, in_channels=64,
        out_channels=128,
        filter_h=3, filter_w=3,
        pad_h=1, pad_w=1,
    )
    conv2d_fprop(output, input, filter, problem, ctx)

Note: This implementation currently supports:
- stride=1, dilation=1
- NHWC layout for activation and output
- KRSC layout for filter
- BF16/FP16 data types
"""

from collections import Optional
from math import align_up, ceildiv

from sys import size_of

from buffer.buffer import NDBuffer
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import create_tensor_tile_im2col

from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    create_tma_tile,
)
from layout import Layout as LegacyLayout, LayoutTensor, RuntimeLayout
from linalg.utils import elementwise_compute_lambda_type
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from .conv_config import Conv2dConfig, Conv2dProblemShape
from .conv_smem import Conv2dSmem
from .conv2d_fprop_kernel import Conv2dFpropKernel

# =============================================================================
# conv2d_fprop - 4D NHWC API with implicit im2col via TMA
# =============================================================================


fn conv2d_fprop[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    *,
    config: Conv2dConfig[act_type, filter_type, out_type] = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16(),
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
](
    output: NDBuffer[out_type, 4],  # NHWC
    activation: NDBuffer[act_type, 4],  # NHWC
    filter: NDBuffer[filter_type, 4],  # KRSC (out_ch, R, S, in_ch)
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
) raises:
    """Launch Conv2D forward propagation with 4D NHWC API and implicit im2col.

    This function provides a 4D tensor API for conv2d forward propagation
    using hardware TMA im2col transformation. The TMA descriptor encodes the
    convolution geometry (padding, stride, dilation) and performs coordinate
    transformation on-the-fly, eliminating the need for explicit im2col buffers.

    The convolution is implemented as implicit GEMM:
    - Activation matrix A[M, K] where M = batch*H_out*W_out, K = C*R*S
    - Filter matrix B[K, N] where N = out_channels (transposed)
    - Output matrix C[M, N]

    The TMA im2col descriptor handles the linear K iteration by decomposing
    k_coord into (channel, filter_r, filter_s) using the corner parameters:
    - lower_corner defines the starting filter offset (negative for padding)
    - upper_corner defines the ending filter offset
    - channels_per_pixel is the number of input channels (C)
    - pixels_per_column is the output spatial tile size (BM)

    Parameters:
        act_type: Data type of the input activation tensor.
        filter_type: Data type of the filter weights tensor.
        out_type: Data type of the output tensor.
        config: Kernel configuration (tile sizes, pipeline stages, etc.).
        elementwise_compute_lambda_fn: Optional element-wise lambda function
            for epilogue fusion (bias add, activation, residual connection).
            Signature: `fn(coords: IndexList[2], val: SIMD) -> SIMD`.
        register_based_epilogue: If True, apply lambda in registers (faster).
            If False, apply lambda after SMEM write (more flexible).

    Args:
        output: Output tensor [N, H_out, W_out, C_out] in NHWC layout.
        activation: Input activation [N, H, W, C] in NHWC layout.
        filter: Filter weights [K, R, S, C] in KRSC layout.
        problem: Convolution problem shape specification.
        ctx: Device context for kernel launch.

    Raises:
        Error if kernel launch fails or constraints are violated.
    """
    # Validate problem constraints
    if problem.stride_h != 1 or problem.stride_w != 1:
        raise Error("Only stride=1 is currently supported")
    if problem.dilation_h != 1 or problem.dilation_w != 1:
        raise Error("Only dilation=1 is currently supported")
    if problem.groups != 1:
        raise Error("Only groups=1 is currently supported")

    # Tile dimensions from config
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime cluster_shape = config.cluster_shape

    # Compute GEMM dimensions
    var M = problem.gemm_m()  # batch * H_out * W_out
    var N = problem.gemm_n()  # out_channels
    var K = problem.gemm_k()  # in_channels * filter_h * filter_w

    # Im2col corner offsets for stride=1, dilation=1
    # CUTLASS formula from detail.hpp (compute_upper_corner_whd):
    #   lower_corner = -lower_padding  (negative for padding region)
    #   upper_corner = upper_padding - (filter - 1) * dilation
    # For 3x3 with pad=1, dilation=1: lower=-1, upper=1-(3-1)*1=-1
    # Both corners are -1, meaning the filter window extends 1 pixel
    # into the padding region on each side.
    var lower_corner_h = -problem.pad_h
    var lower_corner_w = -problem.pad_w
    var upper_corner_h = problem.pad_h - (problem.filter_h - 1)
    var upper_corner_w = problem.pad_w - (problem.filter_w - 1)

    # TMA im2col bounds validation (following CUTLASS sm90_implicit_gemm pattern)
    # Corners are stored as signed 8-bit in the TMA descriptor: [-128, 127]
    # Filter offsets are stored as unsigned 8-bit: [0, 255]
    comptime corner_limit = 128  # signed 8-bit range
    comptime offset_limit = 255  # unsigned 8-bit max

    debug_assert(
        lower_corner_h >= -corner_limit and lower_corner_h < corner_limit,
        "lower_corner_h out of TMA im2col range [-128, 127]",
    )
    debug_assert(
        lower_corner_w >= -corner_limit and lower_corner_w < corner_limit,
        "lower_corner_w out of TMA im2col range [-128, 127]",
    )
    debug_assert(
        upper_corner_h >= -corner_limit and upper_corner_h < corner_limit,
        "upper_corner_h out of TMA im2col range [-128, 127]",
    )
    debug_assert(
        upper_corner_w >= -corner_limit and upper_corner_w < corner_limit,
        "upper_corner_w out of TMA im2col range [-128, 127]",
    )

    # Filter offsets range from 0 to (filter_size - 1), multiplied by dilation
    # For now we assume dilation=1
    debug_assert(
        problem.filter_h - 1 <= offset_limit,
        "filter_h offset exceeds TMA im2col limit [0, 255]",
    )
    debug_assert(
        problem.filter_w - 1 <= offset_limit,
        "filter_w offset exceeds TMA im2col limit [0, 255]",
    )

    # Create activation LayoutTensor view (4D NHWC)
    comptime act_4d_layout = LegacyLayout.row_major(1, 1, 1, 1)  # Dynamic
    var act_tensor = LayoutTensor[act_type, act_4d_layout](
        activation.data,
        RuntimeLayout[act_4d_layout](
            Index(
                activation.dim[0](),
                activation.dim[1](),
                activation.dim[2](),
                activation.dim[3](),
            ),
            Index(
                activation.stride[0](),
                activation.stride[1](),
                activation.stride[2](),
                activation.stride[3](),
            ),
        ),
    )

    # Shared memory size
    comptime SmemType = Conv2dSmem[
        act_type, filter_type, out_type, config=config
    ]
    comptime smem_size = size_of[SmemType]()
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Instantiate kernel first -- TMA layouts computed from config
    comptime conv_kernel = Conv2dFpropKernel[
        act_type,
        filter_type,
        out_type,
        config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
    ]
    comptime KernelType = type_of(conv_kernel)

    # Create TMA descriptors using kernel-derived layout types
    act_tma_op = create_tensor_tile_im2col[
        act_type,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
        __tile_layout = KernelType.ActTmaOp.layout,
        __desc_layout = KernelType.ActTmaOp.desc_layout,
    ](
        ctx,
        act_tensor,
        lower_corner_h,
        lower_corner_w,
        upper_corner_h,
        upper_corner_w,
        problem.out_height(),
        problem.out_width(),
        problem.filter_h,
        problem.filter_w,
    )

    # Create filter 2D view: [N, K] transposed (K-major)
    comptime filter_2d_layout = LegacyLayout.row_major(1, 1)  # Dynamic
    var filter_tensor = LayoutTensor[filter_type, filter_2d_layout](
        filter.data,
        RuntimeLayout[filter_2d_layout](
            Index(N, K),
            Index(K, 1),  # K-major (transposed)
        ),
    )

    filter_tma_op = create_tma_tile[
        KernelType.FilterTmaTile.tile_layout,
        KernelType.FilterTmaTile.desc_layout,
        Index(BN // (cluster_shape[0] // config.cta_group), BK),
        swizzle_mode = config.b_swizzle,
    ](ctx, filter_tensor)

    # Create output 2D view: [M, N] row-major
    comptime out_2d_layout = LegacyLayout.row_major(1, 1)  # Dynamic
    var out_tensor = LayoutTensor[out_type, out_2d_layout](
        output.data,
        RuntimeLayout[out_2d_layout](
            Index(M, N),
            Index(N, 1),  # Row-major
        ),
    )

    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    out_tma_op = create_tma_tile[
        KernelType.OutTmaTile.tile_layout,
        KernelType.OutTmaTile.desc_layout,
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, out_tensor)

    comptime kernel = conv_kernel.run

    # Grid dimensions
    var grid_dim = (
        align_up(ceildiv(M, BM), cluster_shape[0]),
        align_up(ceildiv(N, MMA_N), cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # GEMM dimensions
    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    # Launch kernel with im2col TMA
    ctx.enqueue_function[kernel, kernel](
        act_tma_op,
        filter_tma_op,
        out_tma_op,
        cluster_dim,
        mnk,
        grid_dim=grid_dim,
        block_dim=(conv_kernel.NUM_THREADS),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )


# =============================================================================
# conv2d_fprop_with_residual - Conv2D with optional residual add
# =============================================================================


fn conv2d_fprop_with_residual[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    *,
    config: Conv2dConfig[act_type, filter_type, out_type] = Conv2dConfig[
        act_type, filter_type, out_type
    ].default_bf16(),
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    has_residual: Bool = False,
](
    output: NDBuffer[out_type, 4],  # NHWC - D = Conv(A,B) + beta*C
    activation: NDBuffer[act_type, 4],  # NHWC - A
    filter: NDBuffer[filter_type, 4],  # KRSC - B
    source: NDBuffer[out_type, 4],  # NHWC - C (residual input)
    beta: Float32,  # Residual scale factor
    problem: Conv2dProblemShape,
    ctx: DeviceContext,
) raises:
    """Launch Conv2D fprop with residual add.

    Computes D = Conv(A,B) + beta*C. This function extends conv2d_fprop with
    residual add support. The epilogue load warp pre-fetches source tensor C
    via TMA, overlapping with MMA computation for better performance.

    The residual add is applied after the optional epilogue lambda:
        D = lambda(Conv(A,B)) + beta * C

    This supports common patterns like:
    - Skip connections: D = Conv(A,B) + C (beta=1.0)
    - Residual scaling: D = Conv(A,B) + 0.5*C (beta=0.5)
    - Fused residual+activation: D = ReLU(Conv(A,B)) + C

    Parameters:
        act_type: Data type of the input activation tensor.
        filter_type: Data type of the filter weights tensor.
        out_type: Data type of the output tensor.
        config: Kernel configuration (tile sizes, pipeline stages, etc.).
        elementwise_compute_lambda_fn: Optional element-wise lambda function
            for epilogue fusion (bias add, activation). Applied before residual.
        register_based_epilogue: If True, apply lambda in registers (faster).
        has_residual: If True, apply residual add. If False, source is ignored.

    Args:
        output: Output tensor [N, H_out, W_out, C_out] in NHWC layout (D).
        activation: Input activation [N, H, W, C] in NHWC layout (A).
        filter: Filter weights [K, R, S, C] in KRSC layout (B).
        source: Source tensor [N, H_out, W_out, C_out] for residual (C).
        beta: Residual scale factor. If 0.0, no residual is applied.
        problem: Convolution problem shape specification.
        ctx: Device context for kernel launch.

    Raises:
        Error if kernel launch fails, constraints are violated, or source
        tensor shape doesn't match output shape.

    Note:
        The epilogue load warp (warp ID 7) handles C loading when residual is
        enabled. When has_residual is False or beta is 0, this warp exits early
        and the kernel behaves identically to conv2d_fprop.
    """

    # If no residual requested or beta is 0, fall back to standard conv2d
    comptime if not has_residual:
        conv2d_fprop[
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
        ](output, activation, filter, problem, ctx)
        return

    if beta == 0.0:
        conv2d_fprop[
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
        ](output, activation, filter, problem, ctx)
        return

    # Validate source tensor shape matches output
    if source.dim[0]() != output.dim[0]():
        raise Error("Source batch size must match output batch size")
    if source.dim[1]() != output.dim[1]():
        raise Error("Source height must match output height")
    if source.dim[2]() != output.dim[2]():
        raise Error("Source width must match output width")
    if source.dim[3]() != output.dim[3]():
        raise Error("Source channels must match output channels")

    # ========== Compute GEMM dimensions ==========
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime cluster_shape = config.cluster_shape

    var M = problem.gemm_m()  # batch * H_out * W_out
    var N = problem.gemm_n()  # out_channels
    var K = problem.gemm_k()  # in_channels * filter_h * filter_w

    # Im2col corner offsets (same as conv2d_fprop)
    var lower_corner_h = -problem.pad_h
    var lower_corner_w = -problem.pad_w
    var upper_corner_h = problem.pad_h - (problem.filter_h - 1)
    var upper_corner_w = problem.pad_w - (problem.filter_w - 1)

    # ========== Create TMA descriptors ==========
    # Activation TMA with im2col (4D NHWC)
    comptime act_4d_layout = LegacyLayout.row_major(1, 1, 1, 1)
    var act_tensor = LayoutTensor[act_type, act_4d_layout](
        activation.data,
        RuntimeLayout[act_4d_layout](
            Index(
                activation.dim[0](),
                activation.dim[1](),
                activation.dim[2](),
                activation.dim[3](),
            ),
            Index(
                activation.stride[0](),
                activation.stride[1](),
                activation.stride[2](),
                activation.stride[3](),
            ),
        ),
    )

    # ========== Instantiate kernel ==========
    comptime SmemType = Conv2dSmem[
        act_type, filter_type, out_type, config=config
    ]
    comptime smem_size = size_of[SmemType]()
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Instantiate kernel first -- TMA layouts computed from config
    comptime conv_kernel = Conv2dFpropKernel[
        act_type,
        filter_type,
        out_type,
        config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
    ]
    comptime KernelType = type_of(conv_kernel)

    # Create TMA descriptors using kernel-derived layout types
    act_tma_op = create_tensor_tile_im2col[
        act_type,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
        __tile_layout = KernelType.ActTmaOp.layout,
        __desc_layout = KernelType.ActTmaOp.desc_layout,
    ](
        ctx,
        act_tensor,
        lower_corner_h,
        lower_corner_w,
        upper_corner_h,
        upper_corner_w,
        problem.out_height(),
        problem.out_width(),
        problem.filter_h,
        problem.filter_w,
    )

    # Filter TMA (2D K-major)
    comptime filter_2d_layout = LegacyLayout.row_major(1, 1)
    var filter_tensor = LayoutTensor[filter_type, filter_2d_layout](
        filter.data,
        RuntimeLayout[filter_2d_layout](
            Index(N, K),
            Index(K, 1),
        ),
    )
    filter_tma_op = create_tma_tile[
        KernelType.FilterTmaTile.tile_layout,
        KernelType.FilterTmaTile.desc_layout,
        Index(BN // (cluster_shape[0] // config.cta_group), BK),
        swizzle_mode = config.b_swizzle,
    ](ctx, filter_tensor)

    # Output TMA (D) - 2D row-major
    comptime out_2d_layout = LegacyLayout.row_major(1, 1)
    var out_tensor = LayoutTensor[out_type, out_2d_layout](
        output.data,
        RuntimeLayout[out_2d_layout](
            Index(M, N),
            Index(N, 1),
        ),
    )
    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    out_tma_op = create_tma_tile[
        KernelType.OutTmaTile.tile_layout,
        KernelType.OutTmaTile.desc_layout,
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, out_tensor)

    # Source TMA (C) - same shape and layout as output
    var src_tensor = LayoutTensor[out_type, out_2d_layout](
        source.data,
        RuntimeLayout[out_2d_layout](
            Index(M, N),
            Index(N, 1),
        ),
    )
    src_tma_op = create_tma_tile[
        KernelType.SrcTmaTile.tile_layout,
        KernelType.SrcTmaTile.desc_layout,
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, src_tensor)

    comptime kernel = conv_kernel.run_with_residual

    # Grid dimensions
    var grid_dim = (
        align_up(ceildiv(M, BM), cluster_shape[0]),
        align_up(ceildiv(N, MMA_M), cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    # Launch kernel with residual
    ctx.enqueue_function[kernel, kernel](
        act_tma_op,
        filter_tma_op,
        out_tma_op,
        src_tma_op,
        cluster_dim,
        mnk,
        beta,
        grid_dim=grid_dim,
        block_dim=(conv_kernel.NUM_THREADS),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )


# =============================================================================
# Helper: explicit im2col transformation
# =============================================================================


fn im2col[
    dtype: DType,
](
    output: NDBuffer[mut=True, dtype, 2, ...],  # [M, K] output
    activation: NDBuffer[dtype, 4, ...],  # [N, H, W, C] input
    problem: Conv2dProblemShape,
):
    """Explicit im2col transformation for convolution.

    Transforms a 4D activation tensor [N, H, W, C] into a 2D matrix [M, K]
    for GEMM-based convolution.

    M = batch * out_h * out_w
    K = in_channels * filter_h * filter_w

    Args:
        output: Output 2D buffer [M, K].
        activation: Input 4D buffer [N, H, W, C].
        problem: Convolution problem shape.

    Note: This is a CPU reference implementation. For production use,
    the implicit im2col in the kernel is preferred.
    """
    var out_h = problem.out_height()
    var out_w = problem.out_width()

    # For each output position (batch, oh, ow)
    for b in range(problem.batch):
        for oh in range(out_h):
            for ow in range(out_w):
                var m_idx = b * out_h * out_w + oh * out_w + ow

                # For each filter position (r, s, c) - filter-first, channel-last
                # K = r * S * C + s * C + c (matches TMA im2col hardware)
                for fh in range(problem.filter_h):
                    for fw in range(problem.filter_w):
                        for c in range(problem.in_channels):
                            var k_idx = (
                                fh * problem.filter_w * problem.in_channels
                                + fw * problem.in_channels
                                + c
                            )

                            # Compute input coordinates
                            var ih = (
                                oh * problem.stride_h
                                + fh * problem.dilation_h
                                - problem.pad_h
                            )
                            var iw = (
                                ow * problem.stride_w
                                + fw * problem.dilation_w
                                - problem.pad_w
                            )

                            # Check bounds (padding)
                            if (
                                ih >= 0
                                and ih < problem.in_height
                                and iw >= 0
                                and iw < problem.in_width
                            ):
                                var in_idx = (
                                    b
                                    * problem.in_height
                                    * problem.in_width
                                    * problem.in_channels
                                    + ih
                                    * problem.in_width
                                    * problem.in_channels
                                    + iw * problem.in_channels
                                    + c
                                )
                                output[m_idx, k_idx] = activation.data[in_idx]
                            else:
                                # Zero padding
                                output[m_idx, k_idx] = Scalar[dtype](0)
