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
"""Mamba-specific kernel registrations.

This package contains only the custom operations required for Mamba models:
- causal_conv1d: Causal 1D convolution with optional SiLU activation
- selective_scan_fwd: Selective scan forward pass for Mamba SSM
- rms_norm: Root Mean Square normalization for Mamba blocks

These operations are separated from MOGGKernelAPI to avoid conflicts with
default kernel registrations.
"""

from math import ceildiv

import compiler_internal as compiler
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from runtime.asyncrt import DeviceContextPtr

from nn.causal_conv1d import (
    causal_conv1d_channel_first_fwd_cpu,
    causal_conv1d_channel_first_fwd_gpu,
    causal_conv1d_update_cpu,
    causal_conv1d_update_gpu,
)
from nn.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_fwd_cpu_minimal,
    selective_scan_fwd_gpu_minimal,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
)
from nn.normalization import (
    rms_norm,
    rms_norm_fused_residual,
    rms_norm_fused_residual_add,
)

from utils.index import IndexList
from tensor import InputTensor, OutputTensor
from tensor.managed_tensor_slice import (
    _FusedInputTensor as FusedInputTensor,
    _FusedOutputTensor as FusedOutputTensor,
)


# ============================================================================
# Causal Conv1D Registration
# ============================================================================

@compiler.register("causal_conv1d")
struct CausalConv1D[activation: StaticString]:
    """Causal 1D convolution operation with bias.
    
    Performs causal (autoregressive) 1D convolution where each output position
    depends only on current and past input positions. Supports optional SiLU
    activation with SIMD-vectorized implementations for widths 1, 2, 3, 4.
    
    Parameters:
        activation: Activation function to apply after convolution.
            - "none": No activation (identity).
            - "silu": SiLU/Swish activation (x * sigmoid(x)).
    
    Tensor Shapes:
        - input: (batch, channels, seqlen) - Input sequence tensor.
        - weight: (channels, width) - Convolution weights per channel.
        - bias: (channels,) - Per-channel bias to add.
        - output: (batch, channels, seqlen) - Output tensor (same shape as input).
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],  # Changed from FusedInputTensor for GPU compatibility
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Validate ranks using compile-time parameter (runtime checks are optimized away)
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")

        var X = input.to_layout_tensor()
        var W = weight.to_layout_tensor()
        var O = output.to_layout_tensor()
        var B = bias.to_layout_tensor()

        # Get dimensions and strides from original tensors before conversion
        # This ensures we have valid values even if layout conversion has issues
        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)
        
        var x_batch_stride: UInt32 = UInt32(input.strides()[0])
        var x_c_stride: UInt32 = UInt32(input.strides()[1])
        var x_l_stride: UInt32 = UInt32(input.strides()[2])
        
        var weight_c_stride: UInt32 = UInt32(weight.strides()[0])
        var weight_width_stride: UInt32 = UInt32(weight.strides()[1])
        
        var out_batch_stride: UInt32 = UInt32(output.strides()[0])
        var out_c_stride: UInt32 = UInt32(output.strides()[1])
        var out_l_stride: UInt32 = UInt32(output.strides()[2])
        
        var bias_stride: UInt32 = UInt32(bias.strides()[0])
        
        var silu_activation = Self.activation == "silu"
        
        @parameter
        if is_cpu[target]():
            causal_conv1d_channel_first_fwd_cpu[
                X.dtype,
                X.layout,
                W.dtype,
                W.layout,
                O.dtype,
                O.layout,
                B.dtype,
                B.layout,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                X,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                silu_activation,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()
            comptime kNThreads = 128
            comptime kNElts = 4
            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function_checked[
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                    causal_conv1d_channel_first_fwd_gpu[
                        X.dtype,
                        X.layout,
                        W.dtype,
                        W.layout,
                        O.dtype,
                        O.layout,
                        kNThreads,
                        kWidth,
                        kNElts,
                        B.dtype,
                        B.layout,
                    ],
                ]()
                var silu_activation_int8 = Int8(silu_activation)
                gpu_ctx.enqueue_function_checked(
                    compiled_func,
                    batch_size,
                    dim,
                    seqlen,
                    width,
                    X,
                    W,
                    O,
                    B,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation_int8,
                    grid_dim=(ceildiv(X.dim(2), kNThreads * kNElts), X.dim(1), X.dim(0)),
                    block_dim=(kNThreads),
                )
            else:
                raise Error("Unsupported kernel width: only widths 1, 2, 3, 4 are supported")
        else:
            raise Error("Unsupported target device")
    
    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# Selective Scan Forward Operation
# ===----------------------------------------------------------------------=== #

@compiler.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.
    
    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.
    
    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.
    
    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        x: OutputTensor[dtype=dtype, rank=4],
        out_z: OutputTensor[dtype=dtype, rank=3],
        u: InputTensor[dtype=dtype, rank=3],  # Changed from FusedInputTensor for GPU compatibility
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=3],
        delta_bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Rank validation - output is always rank 3 for selective_scan_fwd
        # Other tensors have fixed ranks that are validated by the compiler
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")
        
        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups
        
        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var out_z_lt = out_z.to_layout_tensor()
        var u_lt = u.to_layout_tensor()
        var delta_lt = delta.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var delta_bias_lt = delta_bias.to_layout_tensor()
        
        var output_strides = output.strides()
        var x_strides = x.strides()
        var out_z_strides = out_z.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var delta_bias_strides = delta_bias.strides()
        
        comptime delta_softplus_int8: Int8 = Int8(1) if Self.delta_softplus else Int8(0)
        
        @parameter
        if is_cpu[target]():
            selective_scan_fwd_cpu[
                dtype,
                output_lt.layout,
                x_lt.layout,
                out_z_lt.layout,
                u_lt.layout,
                delta_lt.layout,
                A_lt.layout,
                B_lt.layout,
                C_lt.layout,
                D_lt.layout,
                z_lt.layout,
                delta_bias_lt.layout,
            ](
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                out_z_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                delta_bias_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(out_z_strides[0]),
                UInt32(out_z_strides[1]),
                UInt32(out_z_strides[2]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(z_strides[2]),
                UInt32(delta_bias_strides[0]),
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
            
            var compiled_kernel = gpu_ctx.compile_function_checked[
                selective_scan_fwd_gpu[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    delta_bias_lt.layout,
                ],
                selective_scan_fwd_gpu[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    out_z_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    delta_bias_lt.layout,
                ]
            ]()
            
            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                total_batch_dim,
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                out_z_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                delta_bias_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(out_z_strides[0]),
                UInt32(out_z_strides[1]),
                UInt32(out_z_strides[2]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(z_strides[2]),
                UInt32(delta_bias_strides[0]),
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            raise Error("Unsupported target: " + target)
    
    @staticmethod
    fn shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=3],
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=3],
        delta_bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[3]:
        return u.shape()


# ===----------------------------------------------------------------------=== #
# Minimal Selective Scan Forward Operation (no D, z, delta_bias)
# ===----------------------------------------------------------------------=== #

@compiler.register("selective_scan_fwd_minimal")
struct SelectiveScanFwdMinimal[delta_softplus: Bool = False]:
    """Minimal selective scan forward pass - no optional D, z, or delta_bias.
    
    This variant avoids passing empty tensors that could have null pointers.
    Use when D, z, and delta_bias are not provided.
    
    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.
    
    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
    """
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        x: OutputTensor[dtype=dtype, rank=4],
        u: InputTensor[dtype=dtype, rank=3],  # Changed from FusedInputTensor
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")
        
        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups
        
        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var u_lt = u.to_layout_tensor()
        var delta_lt = delta.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        
        var output_strides = output.strides()
        var x_strides = x.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        
        comptime delta_softplus_int8: Int8 = Int8(1) if Self.delta_softplus else Int8(0)
        
        @parameter
        if is_cpu[target]():
            selective_scan_fwd_cpu_minimal[
                dtype,
                output_lt.layout,
                x_lt.layout,
                u_lt.layout,
                delta_lt.layout,
                A_lt.layout,
                B_lt.layout,
                C_lt.layout,
            ](
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
            
            var compiled_kernel = gpu_ctx.compile_function_checked[
                selective_scan_fwd_gpu_minimal[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                ],
                selective_scan_fwd_gpu_minimal[
                    dtype,
                    output_lt.layout,
                    x_lt.layout,
                    u_lt.layout,
                    delta_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                ]
            ]()
            
            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                total_batch_dim,
                batch,
                dim,
                seqlen,
                dstate,
                group_size,
                delta_softplus_int8,
                output_lt,
                x_lt,
                u_lt,
                delta_lt,
                A_lt,
                B_lt,
                C_lt,
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(output_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(x_strides[2]),
                UInt32(x_strides[3]),
                UInt32(u_strides[0]),
                UInt32(u_strides[1]),
                UInt32(u_strides[2]),
                UInt32(delta_strides[0]),
                UInt32(delta_strides[1]),
                UInt32(delta_strides[2]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(B_strides[3]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(C_strides[3]),
                grid_dim=(num_blocks),
                block_dim=(BLOCK_SIZE),
            )
        else:
            raise Error("Unsupported target device")
    
    @staticmethod
    fn shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=3],
        delta: InputTensor[dtype=dtype, rank=3],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=4],
        C: InputTensor[dtype=dtype, rank=4],
    ) -> IndexList[3]:
        return u.shape()


# ===----------------------------------------------------------------------=== #
# Causal Conv1D Update Operation (Autoregressive)
# ===----------------------------------------------------------------------=== #

@compiler.register("causal_conv1d_update")
struct CausalConv1DUpdate[activation: StaticString]:
    """Incremental causal conv1d update for autoregressive decoding.
    
    This operation is designed for token-by-token generation where:
        1. A sliding window of recent inputs is maintained in conv_state.
        2. Each new input updates the state and produces an output.
        3. The conv_state is modified in-place for efficiency.
    
    Use this for:
        - Language model inference with autoregressive generation.
        - Real-time streaming applications.
        - Efficient incremental convolution without full sequence recomputation.
    
    Parameters:
        activation: "none" or "silu" - activation function to apply.
    
    Tensor Shapes:
        - input: (batch, channels, seqlen) - New input tokens (typically seqlen=1).
        - weight: (channels, width) - Convolution weights.
        - bias: (channels,) - Per-channel bias.
        - conv_state: (batch, channels, state_len) - Sliding window state (modified in-place).
        - output: (batch, channels, seqlen) - Convolution output for new tokens.
    
    State Management:
        The conv_state maintains the last (width-1) inputs for each channel.
        After update, the oldest values are shifted out and new inputs are appended.
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        conv_state: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],  # Changed from FusedInputTensor for GPU compatibility
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Validate ranks
        if rank != 3:
            raise Error("Input tensor must be rank 3 (batch, channels, seqlen)")
        if output.shape() != input.shape():
            raise Error("Output shape must match input shape")
        if conv_state.dim_size(0) != input.dim_size(0) or conv_state.dim_size(1) != input.dim_size(1):
            raise Error("conv_state batch and channel dimensions must match input")

        var X = input.to_layout_tensor()
        var CS = conv_state.to_layout_tensor()
        var W = weight.to_layout_tensor()
        var O = output.to_layout_tensor()
        var B = bias.to_layout_tensor()

        var batch_size: Int = input.dim_size(0)
        var dim: Int = input.dim_size(1)
        var seqlen: Int = input.dim_size(2)
        var width: Int = weight.dim_size(1)
        var state_len: Int = conv_state.dim_size(2)
        
        var x_batch_stride: UInt32 = UInt32(input.strides()[0])
        var x_c_stride: UInt32 = UInt32(input.strides()[1])
        var x_l_stride: UInt32 = UInt32(input.strides()[2])
        
        var conv_state_batch_stride: UInt32 = UInt32(conv_state.strides()[0])
        var conv_state_c_stride: UInt32 = UInt32(conv_state.strides()[1])
        var conv_state_l_stride: UInt32 = UInt32(conv_state.strides()[2])
        
        var weight_c_stride: UInt32 = UInt32(weight.strides()[0])
        var weight_width_stride: UInt32 = UInt32(weight.strides()[1])
        
        var out_batch_stride: UInt32 = UInt32(output.strides()[0])
        var out_c_stride: UInt32 = UInt32(output.strides()[1])
        var out_l_stride: UInt32 = UInt32(output.strides()[2])
        
        var bias_stride: UInt32 = UInt32(bias.strides()[0])
        
        var silu_activation = Self.activation == "silu"
        
        @parameter
        if is_cpu[target]():
            causal_conv1d_update_cpu[
                X.dtype,
                X.layout,
                CS.dtype,
                CS.layout,
                W.dtype,
                W.layout,
                O.dtype,
                O.layout,
                B.dtype,
                B.layout,
            ](
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X,
                CS,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                silu_activation,
            )
        elif is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()
            comptime kNThreads = 128
            var compiled_func = gpu_ctx.compile_function_checked[
                causal_conv1d_update_gpu[
                    X.dtype,
                    X.layout,
                    CS.dtype,
                    CS.layout,
                    W.dtype,
                    W.layout,
                    O.dtype,
                    O.layout,
                    B.dtype,
                    B.layout,
                    kNThreads,
                ],
                causal_conv1d_update_gpu[
                    X.dtype,
                    X.layout,
                    CS.dtype,
                    CS.layout,
                    W.dtype,
                    W.layout,
                    O.dtype,
                    O.layout,
                    B.dtype,
                    B.layout,
                    kNThreads,
                ],
            ]()
            var silu_activation_int8 = Int8(silu_activation)
            gpu_ctx.enqueue_function_checked(
                compiled_func,
                batch_size,
                dim,
                seqlen,
                width,
                state_len,
                X,
                CS,
                W,
                O,
                B,
                x_batch_stride,
                x_c_stride,
                x_l_stride,
                conv_state_batch_stride,
                conv_state_c_stride,
                conv_state_l_stride,
                weight_c_stride,
                weight_width_stride,
                out_batch_stride,
                out_c_stride,
                out_l_stride,
                bias_stride,
                silu_activation_int8,
                grid_dim=(batch_size, ceildiv(dim, kNThreads)),
                block_dim=(kNThreads),
            )
        else:
            raise Error("Unsupported target device")
    
    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        weight: InputTensor[dtype=dtype, rank=2],
        bias: InputTensor[dtype=dtype, rank=1],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# Selective Scan Update Operation (Autoregressive)
# ===----------------------------------------------------------------------=== #

@compiler.register("selective_scan_update")
struct SelectiveScanUpdate[delta_softplus: Bool = False]:
    """Selective scan update operation for autoregressive inference.
    
    Performs a single step of the SSM recurrence for incremental token generation.
    
    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.
    
    Tensor Shapes:
        - state_out: (batch, dim, dstate) - Updated state output
        - output: (batch, dim) - Output tensor
        - state_in: (batch, dim, dstate) - Input state
        - x: (batch, dim) - Input tensor
        - dt: (batch, dim) - Time delta tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate) - Input matrix
        - C: (batch, n_groups, dstate) - Output matrix
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (dim,) - Time delta bias (optional, can be empty)
    """
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        state_out: OutputTensor[dtype=dtype, rank=3],
        output: OutputTensor[dtype=dtype, rank=2],
        state_in: InputTensor[dtype=dtype, rank=3],  # Changed from FusedInputTensor for GPU compatibility
        x: InputTensor[dtype=dtype, rank=2],
        dt: InputTensor[dtype=dtype, rank=2],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=3],
        C: InputTensor[dtype=dtype, rank=3],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=2],
        dt_bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Tensor ranks are enforced at compile time by InputTensor/OutputTensor type definitions
        var batch = state_out.dim_size(0)
        var dim = state_out.dim_size(1)
        var dstate = state_out.dim_size(2)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups
        
        var state_out_lt = state_out.to_layout_tensor()
        var output_lt = output.to_layout_tensor()
        var state_in_lt = state_in.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var dt_lt = dt.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var dt_bias_lt = dt_bias.to_layout_tensor()
        
        var state_out_strides = state_out.strides()
        var output_strides = output.strides()
        var state_in_strides = state_in.strides()
        var x_strides = x.strides()
        var dt_strides = dt.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var dt_bias_strides = dt_bias.strides()
        
        comptime delta_softplus_int8: Int8 = Int8(1) if Self.delta_softplus else Int8(0)
        
        @parameter
        if is_cpu[target]():
            selective_scan_update_cpu[
                dtype,
                state_out_lt.layout,
                output_lt.layout,
                state_in_lt.layout,
                x_lt.layout,
                dt_lt.layout,
                A_lt.layout,
                B_lt.layout,
                C_lt.layout,
                D_lt.layout,
                z_lt.layout,
                dt_bias_lt.layout,
            ](
                batch,
                dim,
                dstate,
                group_size,
                delta_softplus_int8,
                state_out_lt,
                output_lt,
                state_in_lt,
                x_lt,
                dt_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                dt_bias_lt,
                UInt32(state_out_strides[0]),
                UInt32(state_out_strides[1]),
                UInt32(state_out_strides[2]),
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(state_in_strides[0]),
                UInt32(state_in_strides[1]),
                UInt32(state_in_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(dt_strides[0]),
                UInt32(dt_strides[1]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(dt_bias_strides[0]),
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx.get_device_context()
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)
            
            var compiled_kernel = gpu_ctx.compile_function_checked[
                selective_scan_update_gpu[
                    dtype,
                    state_out_lt.layout,
                    output_lt.layout,
                    state_in_lt.layout,
                    x_lt.layout,
                    dt_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    dt_bias_lt.layout,
                ],
                selective_scan_update_gpu[
                    dtype,
                    state_out_lt.layout,
                    output_lt.layout,
                    state_in_lt.layout,
                    x_lt.layout,
                    dt_lt.layout,
                    A_lt.layout,
                    B_lt.layout,
                    C_lt.layout,
                    D_lt.layout,
                    z_lt.layout,
                    dt_bias_lt.layout,
                ]
            ]()
            
            gpu_ctx.enqueue_function_checked(
                compiled_kernel,
                total_batch_dim,
                batch,
                dim,
                dstate,
                group_size,
                delta_softplus_int8,
                state_out_lt,
                output_lt,
                state_in_lt,
                x_lt,
                dt_lt,
                A_lt,
                B_lt,
                C_lt,
                D_lt,
                z_lt,
                dt_bias_lt,
                UInt32(state_out_strides[0]),
                UInt32(state_out_strides[1]),
                UInt32(state_out_strides[2]),
                UInt32(output_strides[0]),
                UInt32(output_strides[1]),
                UInt32(state_in_strides[0]),
                UInt32(state_in_strides[1]),
                UInt32(state_in_strides[2]),
                UInt32(x_strides[0]),
                UInt32(x_strides[1]),
                UInt32(dt_strides[0]),
                UInt32(dt_strides[1]),
                UInt32(A_strides[0]),
                UInt32(A_strides[1]),
                UInt32(B_strides[0]),
                UInt32(B_strides[1]),
                UInt32(B_strides[2]),
                UInt32(C_strides[0]),
                UInt32(C_strides[1]),
                UInt32(C_strides[2]),
                UInt32(D_strides[0]),
                UInt32(z_strides[0]),
                UInt32(z_strides[1]),
                UInt32(dt_bias_strides[0]),
                grid_dim=(num_blocks,),
                block_dim=(BLOCK_SIZE,),
            )
        else:
            raise Error("Unsupported target: " + target)
    
    @staticmethod
    fn shape[
        dtype: DType,
    ](
        state_in: InputTensor[dtype=dtype, rank=3],
        x: InputTensor[dtype=dtype, rank=2],
        dt: InputTensor[dtype=dtype, rank=2],
        A: InputTensor[dtype=dtype, rank=2],
        B: InputTensor[dtype=dtype, rank=3],
        C: InputTensor[dtype=dtype, rank=3],
        D: InputTensor[dtype=dtype, rank=1],
        z: InputTensor[dtype=dtype, rank=2],
        dt_bias: InputTensor[dtype=dtype, rank=1],
    ) -> Tuple[IndexList[3], IndexList[2]]:
        return (state_in.shape(), x.shape())


# ===----------------------------------------------------------------------=== #
# RMSNorm Operation
# ===----------------------------------------------------------------------=== #

@compiler.register("rms_norm")
struct RMSNorm:
    """Root Mean Square normalization operation for Mamba blocks.
    
    Performs RMS normalization on the input tensor, normalizing along the last
    dimension. This matches the RMSNorm implementation used in Mamba models.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py
    
    Tensor Shapes:
        - output: (..., hidden_size) - Output tensor (same shape as input).
        - input: (..., hidden_size) - Input tensor to normalize.
        - weight: (hidden_size,) - Weight tensor (gamma) for normalization.
        - eps: Scalar - Epsilon value for numerical stability (default: 1e-6).
        - weight_offset: Scalar - Offset added to weight before normalization (default: 0.0).
    
    Compile-time Options:
        - multiply_before_cast: If True, multiplies by weight before casting to output dtype.
          If False, casts to output dtype before multiplying by weight.
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")
        
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            # Use _fused_load for FusedInputTensor
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        rms_norm[
            dtype,
            rank,
            input_fn,
            output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma.to_layout_tensor(),
            epsilon,
            weight_offset,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# RMSNorm Fused Residual Operation
# ===----------------------------------------------------------------------=== #

@compiler.register("rms_norm_fused_residual")
struct RMSNormFusedResidual:
    """RMS normalization with fused residual connection for Mamba blocks.
    
    Performs RMS normalization on (input + residual), returning both the
    normalized output and the pre-normalized input (residual output).
    This matches the fused residual + norm pattern used in Mamba models.
    
    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py
    
    Tensor Shapes:
        - output: (..., hidden_size) - Normalized output tensor (same shape as input).
        - residual_output: (..., hidden_size) - Pre-normalized input (input + residual).
        - input: (..., hidden_size) - Input tensor to normalize.
        - residual_input: (..., hidden_size) - Residual tensor to add before normalization.
        - weight: (hidden_size,) - Weight tensor (gamma) for normalization.
        - eps: Scalar - Epsilon value for numerical stability (default: 1e-6).
        - weight_offset: Scalar - Offset added to weight before normalization (default: 0.0).
        - dropout_p: Scalar - Dropout probability (default: 0.0).
        - seed: Scalar[uint64] - Random seed for dropout (default: 0).
    
    Compile-time Options:
        - multiply_before_cast: If True, multiplies by weight before casting to output dtype.
          If False, casts to output dtype before multiplying by weight.
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        residual_output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        dropout_p: Scalar[dtype=dtype],
        seed: Scalar[dtype=DType.uint64],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Validate shapes
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")
        
        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")
        
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn residual_input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return residual_input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @parameter
        @always_inline
        fn residual_output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[residual_output.rank]](coords),
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        rms_norm_fused_residual[
            input_fn,
            residual_input_fn,
            output_fn,
            residual_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma.to_layout_tensor(),
            epsilon,
            weight_offset,
            ctx,
            dropout_p,
            UInt64(seed),
        )
    
    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        dropout_p: Scalar[dtype=dtype],
        seed: Scalar[dtype=DType.uint64],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# RMSNorm Fused Residual Add Operation
# ===----------------------------------------------------------------------=== #

@compiler.register("rms_norm_fused_residual_add")
struct RMSNormFusedResidualAdd:
    """RMS normalization with fused residual add for Mamba blocks.
    
    Performs two RMSNorm operations with a residual add in between:
    1. norm1 = RMSNorm(input, gamma1)
    2. residual_output = norm1 + residual_input
    3. output = RMSNorm(residual_output, gamma2)
    
    Compile-time Options:
        - multiply_before_cast: If True, multiplies by weight before casting to output dtype.
          If False, casts to output dtype before multiplying by weight.
    """
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        residual_output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma1: InputTensor[dtype=dtype, rank=1],
        gamma2: InputTensor[dtype=dtype, rank=1],
        epsilon1: Scalar[dtype=dtype],
        epsilon2: Scalar[dtype=dtype],
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn residual_input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return residual_input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @parameter
        @always_inline
        fn residual_output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[residual_output.rank]](coords),
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        rms_norm_fused_residual_add[
            input_fn,
            residual_input_fn,
            output_fn,
            residual_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma1.to_layout_tensor(),
            epsilon1,
            weight_offset1,
            gamma2.to_layout_tensor(),
            epsilon2,
            weight_offset2,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma1: InputTensor[dtype=dtype, rank=1],
        gamma2: InputTensor[dtype=dtype, rank=1],
        epsilon1: Scalar[dtype=dtype],
        epsilon2: Scalar[dtype=dtype],
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()
