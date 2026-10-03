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
"""Core Causal Conv1D Kernel Implementations.

This module provides CPU and GPU kernel implementations for causal 1D convolution,
supporting both channel-first and channel-last memory layouts.

Causal Convolution:
    In causal convolution, the output at position `i` depends only on inputs at
    positions `[i - width + 1, i]`. This ensures no information leakage from
    future positions, making it suitable for autoregressive sequence modeling.

    Mathematical form for width=4:
        out[i] = sum(x[i-3:i+1] * w[0:4]) + bias  (with boundary handling)

Kernel Categories:

    1. Forward Kernels (CPU & GPU):
        - `causal_conv1d_channel_first_fwd_cpu[_no_bias]`
        - `causal_conv1d_channel_last_fwd_cpu[_with_seq_idx][_no_bias]`
        - `causal_conv1d_channel_first_fwd_gpu[_with_seq_idx][_no_bias]`
        - `causal_conv1d_channel_last_fwd_gpu[_with_seq_idx][_no_bias]`

        SIMD-vectorized implementations with compile-time width specialization.
        Supported widths: 1, 2, 3, 4.

    2. Update Kernels (for autoregressive decode):
        - `causal_conv1d_update_cpu[_no_bias]`
        - `causal_conv1d_update_gpu[_no_bias]`

        Incremental update operations that maintain conv state for efficient
        autoregressive token generation.

Memory Layouts:
    - Channel-first (B, C, L): Standard layout, contiguous channels per position.
    - Channel-last (B, L, C): Contiguous positions per channel, used by some frameworks.

GPU Optimization Parameters:
    - kNThreads=128: Threads per block for sequence processing
    - kNElts=4: Elements processed per thread for better ILP
    - SIMD width 4: Vectorized weight loading for width=4 kernels

Activation Support:
    - None: Direct convolution output
    - SiLU: Sigmoid Linear Unit activation (x * sigmoid(x))
"""

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from layout import TensorLayout, TensorEngine, TileTensor
from nn.activations import silu


# ===----------------------------------------------------------------------=== #
# CPU Implementations
# ===----------------------------------------------------------------------=== #


def causal_conv1d_channel_first_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, C, L)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    silu_activation: Bool,
    ctx: Optional[DeviceContext] = None,
):
    """CPU implementation of causal conv1d for channel-first layout with bias.

    Optimizations:
    1. Parallelization across batch*channel dimensions using sync_parallelize.
    2. Pre-loaded weights in registers to reduce memory access.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width.
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        silu_activation: Whether to apply SiLU activation.
        ctx: The context to execute the work on.
    """
    var width_minus_1: Int = width - 1
    var total_bc = batch * dim

    # Parallelize across batch*channel combinations
    def process_bc(bc_idx: Int) {imm}:
        var b, c = divmod(bc_idx, dim)

        # Bounds checking
        if b >= batch or c >= dim:
            return

        # Validate bias tensor has valid dimensions (use debug_assert since we can't raise in @__parameter fn)
        assert (
            Int(bias.dim[0]()) > 0
        ), "Bias tensor must have at least one element"
        assert c < Int(
            bias.dim[0]()
        ), "Channel index out of bounds for bias tensor"

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.load[width=1]((c,))
        )

        # Pre-load weights for this channel to reduce memory access
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0
        if width >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c, 0)))
        if width >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c, 1)))
        if width >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c, 2)))
        if width >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c, 3)))

        # Process all sequence positions
        for l in range(seqlen):
            var conv_sum: Scalar[output_dtype] = cur_bias

            for w in range(width):
                var input_l: Int = l - (width_minus_1 - w)
                if input_l >= 0:
                    var input_val: Scalar[x_dtype] = x.load[width=1](
                        (b, c, input_l)
                    )
                    # Select weight based on position
                    var weight_val: Scalar[weight_dtype] = w0 if w == 0 else (
                        w1 if w == 1 else (w2 if w == 2 else w3)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_activation:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            output.store[width=1]((b, c, l), out_val)

    sync_parallelize(process_bc, total_bc, ctx)


def causal_conv1d_channel_first_fwd_cpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, C, L)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    silu_activation: Bool,
    ctx: Optional[DeviceContext] = None,
):
    """CPU implementation of causal conv1d for channel-first layout without bias.

    Optimizations:
    1. Parallelization across batch*channel dimensions using sync_parallelize.
    2. Pre-loaded weights in registers to reduce memory access.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width.
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        silu_activation: Whether to apply SiLU activation.
        ctx: The context to execute the work on.
    """
    var width_minus_1: Int = width - 1
    var total_bc = batch * dim

    # Parallelize across batch*channel combinations
    def process_bc(bc_idx: Int) {imm}:
        var b, c = divmod(bc_idx, dim)

        # Pre-load weights for this channel to reduce memory access
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0
        if width >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c, 0)))
        if width >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c, 1)))
        if width >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c, 2)))
        if width >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c, 3)))

        # Process all sequence positions
        for l in range(seqlen):
            var conv_sum: Scalar[output_dtype] = 0.0

            for w in range(width):
                var input_l: Int = l - (width_minus_1 - w)
                if input_l >= 0:
                    var input_val: Scalar[x_dtype] = x.load[width=1](
                        (b, c, input_l)
                    )
                    # Select weight based on position
                    var weight_val: Scalar[weight_dtype] = w0 if w == 0 else (
                        w1 if w == 1 else (w2 if w == 2 else w3)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_activation:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            output.store[width=1]((b, c, l), out_val)

    sync_parallelize(process_bc, total_bc, ctx)


def causal_conv1d_channel_last_fwd_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    silu_activation: Bool,
):
    """
    Optimized CPU implementation of causal conv1d for channel-last layout with bias.

    Structured for potential SIMD optimizations. Currently similar to naive but
    organized for future vectorization improvements.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = Scalar[output_dtype](
                    bias.load[width=1]((c,))
                )

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    if input_l >= 0:
                        var input_val: Scalar[x.dtype] = x.load[width=1](
                            (b, input_l, c)
                        )
                        var weight_val: Scalar[weight.dtype] = weight.load[
                            width=1
                        ]((c, w))
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x.dtype](weight_val)
                        )

                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, l, c), out_val)


def causal_conv1d_channel_last_fwd_cpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    silu_activation: Bool,
):
    """
    Optimized CPU implementation of causal conv1d for channel-last layout without bias.

    Structured for potential SIMD optimizations. Currently similar to naive but
    organized for future vectorization improvements.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    if input_l >= 0:
                        var input_val: Scalar[x.dtype] = x.load[width=1](
                            (b, input_l, c)
                        )
                        var weight_val: Scalar[weight.dtype] = weight.load[
                            width=1
                        ]((c, w))
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x.dtype](weight_val)
                        )

                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, l, c), out_val)


def causal_conv1d_channel_last_fwd_cpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    seq_idx: TileTensor[mut=False, seq_idx_dtype, ...],  # Shape (B, L)
    silu_activation: Bool,
):
    """Optimized implementation of causal conv1d for channel last data layout with seq_idx.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            var cur_seq_idx_val = seq_idx.load[width=1]((b, l))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = Scalar[output_dtype](
                    bias.load[width=1]((c,))
                )

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    var valid_seq: Bool = True
                    if input_l >= 0:
                        var input_seq_idx_val = seq_idx.load[width=1](
                            (b, input_l)
                        )
                        var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                        if input_seq_idx != cur_seq_idx:
                            valid_seq = False

                    if valid_seq and input_l >= 0:
                        var input_val: Scalar[x_dtype] = x.load[width=1](
                            (b, input_l, c)
                        )
                        var weight_val: Scalar[weight_dtype] = weight.load[
                            width=1
                        ]((c, w))
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x_dtype](weight_val)
                        )

                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, l, c), out_val)


def causal_conv1d_channel_last_fwd_cpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (B, L, C)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, L, C)
    seq_idx: TileTensor[mut=False, seq_idx_dtype, ...],  # Shape (B, L)
    silu_activation: Bool,
):
    """Optimized implementation of causal conv1d for channel last data layout without bias but with seq_idx.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for l in range(seqlen):
            var cur_seq_idx_val = seq_idx.load[width=1]((b, l))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            for c in range(dim):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var input_l: Int = l - (width_minus_1 - w)
                    var valid_seq: Bool = True
                    if input_l >= 0:
                        var input_seq_idx_val = seq_idx.load[width=1](
                            (b, input_l)
                        )
                        var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                        if input_seq_idx != cur_seq_idx:
                            valid_seq = False

                    if valid_seq and input_l >= 0:
                        var input_val: Scalar[x_dtype] = x.load[width=1](
                            (b, input_l, c)
                        )
                        var weight_val: Scalar[weight_dtype] = weight.load[
                            width=1
                        ]((c, w))
                        conv_sum = conv_sum + Scalar[output_dtype](
                            input_val * Scalar[x_dtype](weight_val)
                        )

                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, l, c), out_val)


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


def causal_conv1d_channel_first_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, C, L)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],  # Shape (C,)
    silu_activation: Int8,
):
    """Optimized GPU implementation of causal conv1d for channel-first layout with bias.

    Key optimizations:
    1. SIMD vectorization for input/output operations (kNElts elements per thread).
    2. Efficient memory access patterns with coalesced loads.
    3. Vectorized weight loading and computation for width=2 and width=4.
    4. Optimized activation function with SIMD operations.
    5. Better thread utilization and memory bandwidth usage.

    Grid: (ceildiv(seqlen, kNThreads * kNElts), dim, batch)
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        bias_dtype: Element type of the bias tensor `bias`.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        x_engine: Engine of the input tensor `x`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.
        bias_engine: Engine of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match kWidth compile-time parameter).
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = Int(x.dim[0]())
    var nChannels: Int = Int(x.dim[1]())
    var nSeqLen: Int = Int(x.dim[2]())

    if batch_id >= nBatches or channel_id >= nChannels or kWidth != _width:
        return

    # Safety check for bias dimension - if bias is empty or channel_id is out of bounds, use zero bias
    var bias_dim = Int(bias.dim[0]())
    var cur_bias: Scalar[x_dtype] = 0
    if bias_dim > 0 and channel_id < bias_dim:
        cur_bias = Scalar[x_dtype](bias.load[width=1]((channel_id,)))

    var out_vals: SIMD[output_dtype, kNElts] = 0
    # For _width 3, we need to use scalars instead of SIMD (SIMD requires power-of-2 widths)
    # Declare variables for both cases - only one will be used based on kWidth
    var W_2: SIMD[x_dtype, 2] = 0
    var W_4: SIMD[x_dtype, 4] = 0
    var w0: Scalar[x_dtype] = 0
    var w1: Scalar[x_dtype] = 0
    var w2: Scalar[x_dtype] = 0
    var w_single: Scalar[x_dtype] = 0  # For _width 1

    if kWidth == 1:
        w_single = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
    elif kWidth == 2:
        var w0_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        var w1_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        W_2 = SIMD[x_dtype, 2](w0_val, w1_val)
    elif kWidth == 4:
        var w0_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        var w1_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        var w2_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 2)))
        var w3_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 3)))
        W_4 = SIMD[x_dtype, 4](w0_val, w1_val, w2_val, w3_val)
    else:
        w0 = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        w1 = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        w2 = Scalar[x_dtype](weight.load[width=1]((channel_id, 2)))

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return
    var silu_active = Bool(Int(silu_activation) != 0)

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break

        # Build input window by loading directly from memory
        # This avoids SIMD slice issues while maintaining correctness
        var conv_result: Scalar[x_dtype] = 0

        comptime if kWidth == 1:
            if seq_idx >= 0 and seq_idx < nSeqLen:
                var x_val = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, seq_idx))
                )
                conv_result = w_single * x_val
        elif kWidth == 2:
            var input_window: SIMD[x_dtype, 2] = 0

            comptime for w in range(2):
                var input_l: Int = seq_idx - (1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, channel_id, input_l))
                    )
            var tmp: SIMD[x_dtype, 2] = W_2 * input_window
            conv_result = tmp.reduce_add[1]()
        elif kWidth == 4:
            var input_window: SIMD[x_dtype, 4] = 0

            comptime for w in range(4):
                var input_l: Int = seq_idx - (3 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, channel_id, input_l))
                    )
            var tmp: SIMD[x_dtype, 4] = W_4 * input_window
            conv_result = tmp.reduce_add[1]()
        else:
            # kWidth == 3 case
            var x0: Scalar[x_dtype] = 0
            var x1: Scalar[x_dtype] = 0
            var x2: Scalar[x_dtype] = 0
            var input_l0: Int = seq_idx - 2
            var input_l1: Int = seq_idx - 1
            var input_l2: Int = seq_idx
            if input_l0 >= 0 and input_l0 < nSeqLen:
                x0 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l0))
                )
            if input_l1 >= 0 and input_l1 < nSeqLen:
                x1 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l1))
                )
            if input_l2 >= 0 and input_l2 < nSeqLen:
                x2 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l2))
                )
            conv_result = w0 * x0 + w1 * x1 + w2 * x2

        var out_val: Scalar[output_dtype] = Scalar[output_dtype](
            cur_bias
        ) + Scalar[output_dtype](conv_result)
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        out_vals[i] = out_val

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break
        output.store[width=1](
            (batch_id, channel_id, seq_idx), Scalar[output_dtype](out_vals[i])
        )


# Optimized GPU version without bias
def causal_conv1d_channel_first_fwd_gpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, C, L)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel first data layout using SIMD operations (no bias).

    Key optimizations:
    1. SIMD vectorization for input/output operations
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage

    Grid: (ceildiv(seqlen, kNThreads * kNElts), dim, batch)
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        x_engine: Engine of the input tensor `x`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match kWidth compile-time parameter).
        x: Input tensor of shape (B, C, L).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = Int(x.dim[0]())
    var nChannels: Int = Int(x.dim[1]())
    var nSeqLen: Int = Int(x.dim[2]())

    if batch_id >= nBatches or channel_id >= nChannels:
        return

    var out_vals: SIMD[x_dtype, kNElts] = 0
    var W_2: SIMD[x_dtype, 2] = 0
    var W_4: SIMD[x_dtype, 4] = 0
    var w0: Scalar[x_dtype] = 0
    var w1: Scalar[x_dtype] = 0
    var w2: Scalar[x_dtype] = 0
    var w_single: Scalar[x_dtype] = 0

    if kWidth == 1:
        w_single = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
    elif kWidth == 2:
        var w0_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        var w1_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        W_2 = SIMD[x_dtype, 2](w0_val, w1_val)
    elif kWidth == 4:
        var w0_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        var w1_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        var w2_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 2)))
        var w3_val = Scalar[x_dtype](weight.load[width=1]((channel_id, 3)))
        W_4 = SIMD[x_dtype, 4](w0_val, w1_val, w2_val, w3_val)
    else:
        w0 = Scalar[x_dtype](weight.load[width=1]((channel_id, 0)))
        w1 = Scalar[x_dtype](weight.load[width=1]((channel_id, 1)))
        w2 = Scalar[x_dtype](weight.load[width=1]((channel_id, 2)))

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return
    var silu_active = Bool(Int(silu_activation) != 0)

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break

        # Build input window by loading directly from memory
        var conv_result: Scalar[x_dtype] = 0

        comptime if kWidth == 1:
            if seq_idx >= 0 and seq_idx < nSeqLen:
                var x_val = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, seq_idx))
                )
                conv_result = w_single * x_val
        elif kWidth == 2:
            var input_window: SIMD[x_dtype, 2] = 0

            comptime for w in range(2):
                var input_l: Int = seq_idx - (1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, channel_id, input_l))
                    )
            var tmp: SIMD[x_dtype, 2] = W_2 * input_window
            conv_result = tmp.reduce_add[1]()
        elif kWidth == 4:
            var input_window: SIMD[x_dtype, 4] = 0

            comptime for w in range(4):
                var input_l: Int = seq_idx - (3 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, channel_id, input_l))
                    )
            var tmp: SIMD[x_dtype, 4] = W_4 * input_window
            conv_result = tmp.reduce_add[1]()
        else:
            # kWidth == 3 case
            var x0: Scalar[x_dtype] = 0
            var x1: Scalar[x_dtype] = 0
            var x2: Scalar[x_dtype] = 0
            var input_l0: Int = seq_idx - 2
            var input_l1: Int = seq_idx - 1
            var input_l2: Int = seq_idx
            if input_l0 >= 0 and input_l0 < nSeqLen:
                x0 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l0))
                )
            if input_l1 >= 0 and input_l1 < nSeqLen:
                x1 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l1))
                )
            if input_l2 >= 0 and input_l2 < nSeqLen:
                x2 = Scalar[x_dtype](
                    x.load[width=1]((batch_id, channel_id, input_l2))
                )
            conv_result = w0 * x0 + w1 * x1 + w2 * x2

        var out_val: Scalar[x_dtype] = conv_result
        if silu_active:
            comptime if x_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[x_dtype]()
        out_vals[i] = out_val

    comptime for i in range(kNElts):
        var seq_idx: Int = seq_start + i
        if seq_idx >= seq_end:
            break
        output.store[width=1](
            (batch_id, channel_id, seq_idx), Scalar[output_dtype](out_vals[i])
        )


def causal_conv1d_channel_last_fwd_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    bias_dtype: DType,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, L, C)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],  # Shape (C,)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads using vectorized tensor views
    3. Vectorized weight loading and computation
    4. Chunked processing of multiple sequence positions per thread
    5. Optimized activation function with SIMD operations
    6. Better thread utilization and memory bandwidth usage

    For channel-last layout (B, L, C), we reshape to (B*L, C) to enable vectorized
    operations along channels, and process multiple sequence positions per thread.
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.load[width=1]((c_idx,))
        )
        var W = SIMD[weight_dtype, kWidth](0)

        comptime for tap in range(kWidth):
            W[tap] = weight.load[width=1]((c_idx, tap))[0]
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Build input window by loading directly from memory
            var input_window: SIMD[x_dtype, kWidth] = 0

            comptime for w in range(kWidth):
                var input_l: Int = seq_idx - (kWidth - 1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, input_l, c_idx))
                    )

            var tmp = rebind[SIMD[output_dtype, kWidth]](
                input_window * rebind[type_of(input_window)](W)
            )
            conv_sum = conv_sum + tmp.reduce_add[1]()

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break
            output.store[width=1](
                (batch_id, seq_idx, c_idx), out_vals_channel[i]
            )


# Optimized GPU version without bias for channel last
def causal_conv1d_channel_last_fwd_gpu_no_bias[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, L, C)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations (no bias).

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        var W = SIMD[weight_dtype, kWidth](0)

        comptime for tap in range(kWidth):
            W[tap] = weight.load[width=1]((c_idx, tap))[0]
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break

            var conv_sum: Scalar[output_dtype] = 0.0

            # Build input window by loading directly from memory
            var input_window: SIMD[x_dtype, kWidth] = 0

            comptime for w in range(kWidth):
                var input_l: Int = seq_idx - (kWidth - 1 - w)
                if input_l >= 0 and input_l < nSeqLen:
                    input_window[w] = Scalar[x_dtype](
                        x.load[width=1]((batch_id, input_l, c_idx))
                    )

            var tmp = rebind[SIMD[output_dtype, kWidth]](
                input_window * rebind[type_of(input_window)](W)
            )
            conv_sum = conv_sum + tmp.reduce_add[1]()

            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_idx: Int = seq_start + i
            if seq_idx >= seq_end:
                break
            output.store[width=1](
                (batch_id, seq_idx, c_idx), out_vals_channel[i]
            )


# ============================================================================
# Optimized GPU Implementations with seq_idx as TileTensor
# ============================================================================


# Optimized GPU implementation for channel-last with bias and seq_idx as TileTensor
def causal_conv1d_channel_last_fwd_gpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
    seq_idx_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, L, C)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],  # Shape (C,)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin, Engine=seq_idx_engine
    ],  # Shape (B, L)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations with seq_idx support.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads using vectorized tensor views
    3. Vectorized weight loading and computation
    4. Chunked processing of multiple sequence positions per thread
    5. Optimized activation function with SIMD operations
    6. Better thread utilization and memory bandwidth usage
    7. seq_idx support for conditional processing

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.
        seq_idx_dtype: Element type of the `seq_idx` tensor.
        kNThreads: Number of threads per block used to process the sequence
            dimension.
        kWidth: Compile-time convolution kernel width; must match the runtime
            `width` argument.
        kNElts: Number of sequence elements each thread processes, used for
            SIMD vectorization and ILP.
        x_LT: TensorLayout of the input tensor `x`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        seq_idx_LT: TensorLayout of the `seq_idx` tensor.
        x_engine: Engine of the input tensor `x`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.
        bias_engine: Engine of the bias tensor `bias`.
        seq_idx_engine: Engine of the `seq_idx` tensor.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length.
        width: Kernel width (must match `kWidth` compile-time parameter).
        x: Input tensor of shape (B, L, C).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, L, C).
        bias: Bias tensor of shape (C,).
        seq_idx: Per-position sequence id tensor of shape (B, L); a
            convolution tap at position `input_l` only contributes when its
            sequence id matches the id at the output position.
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    # Helper function to load SIMD vector from 3D tensor at (_batch, seq, channel_start)
    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.load[width=1]((c_idx,))
        )

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 0)))

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 1)))

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 2)))

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 3)))

        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_val = seq_idx.load[width=1]((batch_id, seq_pos))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l)
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var x_val = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l, c_idx))
                        )
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l2, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l2, c_idx))
                        )
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l3)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x3 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l3, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            output.store[width=1](
                (batch_id, seq_pos, c_idx), out_vals_channel[i]
            )


# Optimized GPU implementation for channel-last without bias but with seq_idx as TileTensor
def causal_conv1d_channel_last_fwd_gpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    seq_idx_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, L, C)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, L, C)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin, Engine=seq_idx_engine
    ],  # Shape (B, L)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel last data layout using SIMD operations (no bias) with seq_idx support.

    Key optimizations:
    1. SIMD vectorization for input/output operations across channels
    2. Efficient memory access patterns with coalesced loads
    3. Vectorized weight loading and computation
    4. Optimized activation function with SIMD operations
    5. Better thread utilization and memory bandwidth usage
    6. seq_idx support for conditional processing
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 0)))

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 1)))

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 2)))

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 3)))

        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_val = seq_idx.load[width=1]((batch_id, seq_pos))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = 0.0

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l)
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var x_val = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l, c_idx))
                        )
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l2, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l0, c_idx))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l1, c_idx))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l2, c_idx))
                        )
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l3)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x3 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, input_l3, c_idx))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            output.store[width=1](
                (batch_id, seq_pos, c_idx), out_vals_channel[i]
            )


# ============================================================================
# Channel-First GPU Implementations with seq_idx as TileTensor
# ============================================================================


# Optimized GPU implementation for channel-first with bias and seq_idx as TileTensor
def causal_conv1d_channel_first_fwd_gpu_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
    seq_idx_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, C, L)
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],  # Shape (C,)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin, Engine=seq_idx_engine
    ],  # Shape (B, L)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel-first data layout using SIMD operations with seq_idx support.

    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    # Safety check for bias tensor dimensions
    var bias_dim = Int(bias.dim[0]())
    if bias_dim == 0:
        return

    # For channel-first (B, C, L), we process each channel separately
    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Safety check for bias dimension
        if c_idx >= bias_dim:
            break

        var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
            bias.load[width=1]((c_idx,))
        )
        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 0)))

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 1)))

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 2)))

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 3)))
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_val = seq_idx.load[width=1]((batch_id, seq_pos))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = cur_bias

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l)
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var x_val = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l))
                        )
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l2))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l2))
                        )
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l3)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x3 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l3))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            output.store[width=1](
                (batch_id, c_idx, seq_pos), out_vals_channel[i]
            )


# Optimized GPU implementation for channel-first without bias but with seq_idx as TileTensor
def causal_conv1d_channel_first_fwd_gpu_no_bias_with_seq_idx[
    x_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    seq_idx_dtype: DType,
    kNThreads: Int,
    kWidth: Int,
    kNElts: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    seq_idx_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    seq_idx_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (B, C, L)
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],  # Shape (C, W)
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],  # Shape (B, C, L)
    seq_idx: TileTensor[
        seq_idx_dtype, seq_idx_LT, MutUntrackedOrigin, Engine=seq_idx_engine
    ],  # Shape (B, L)
    silu_activation: Int8,
):
    """
    Optimized causal conv1d implementation for channel-first data layout using SIMD operations (no bias) with seq_idx support.

    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_chunk_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x
    var kChunkSize: Int = block_dim.x

    var nBatches: Int = _batch
    var nSeqLen: Int = _seqlen
    var nChannels: Int = _dim

    if batch_id >= nBatches or kWidth != _width:
        return

    var seq_start: Int = chunk_id * kChunkSize * kNElts + tidx * kNElts
    var seq_end: Int = min(seq_start + kNElts, nSeqLen)

    if seq_start >= nSeqLen:
        return

    var channel_start: Int = channel_chunk_id * kNElts

    if channel_start >= nChannels:
        return

    for c_offset in range(kNElts):
        var c_idx: Int = channel_start + c_offset
        if c_idx >= nChannels:
            break

        # Load weights directly from memory to avoid vectorize issues
        # For kWidth == 3, use scalar operations to avoid SIMD issues
        var w0: Scalar[weight_dtype] = 0
        var w1: Scalar[weight_dtype] = 0
        var w2: Scalar[weight_dtype] = 0
        var w3: Scalar[weight_dtype] = 0

        comptime if kWidth >= 1:
            w0 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 0)))

        comptime if kWidth >= 2:
            w1 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 1)))

        comptime if kWidth >= 3:
            w2 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 2)))

        comptime if kWidth >= 4:
            w3 = Scalar[weight_dtype](weight.load[width=1]((c_idx, 3)))
        var out_vals_channel: SIMD[output_dtype, kNElts] = 0
        var silu_active = Bool(Int(silu_activation) != 0)

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break

            # Get current seq_idx value
            var cur_seq_idx_val = seq_idx.load[width=1]((batch_id, seq_pos))
            var cur_seq_idx: Int32 = Int32(cur_seq_idx_val)

            var conv_sum: Scalar[output_dtype] = 0.0

            # Use scalar operations for all kWidth values to avoid SIMD issues with non-power-of-2 sizes
            comptime if kWidth == 1:
                var input_l: Int = seq_pos
                if input_l >= 0 and input_l < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l)
                    )
                    var input_seq_idx: Int32 = Int32(input_seq_idx_val)
                    if input_seq_idx == cur_seq_idx:
                        var x_val = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l))
                        )
                        conv_sum += Scalar[output_dtype](
                            Scalar[output_dtype](x_val)
                            * Scalar[output_dtype](w0)
                        )
            elif kWidth == 2:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 1
                var input_l1: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                )
            elif kWidth == 3:
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 2
                var input_l1: Int = seq_pos - 1
                var input_l2: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l2))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                )
            else:  # kWidth == 4
                var x0: Scalar[x_dtype] = 0
                var x1: Scalar[x_dtype] = 0
                var x2: Scalar[x_dtype] = 0
                var x3: Scalar[x_dtype] = 0
                var input_l0: Int = seq_pos - 3
                var input_l1: Int = seq_pos - 2
                var input_l2: Int = seq_pos - 1
                var input_l3: Int = seq_pos
                if input_l0 >= 0 and input_l0 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l0)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x0 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l0))
                        )
                if input_l1 >= 0 and input_l1 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l1)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x1 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l1))
                        )
                if input_l2 >= 0 and input_l2 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l2)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x2 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l2))
                        )
                if input_l3 >= 0 and input_l3 < nSeqLen:
                    var input_seq_idx_val = seq_idx.load[width=1](
                        (batch_id, input_l3)
                    )
                    if Int32(input_seq_idx_val) == cur_seq_idx:
                        x3 = Scalar[x_dtype](
                            x.load[width=1]((batch_id, c_idx, input_l3))
                        )
                conv_sum += Scalar[output_dtype](
                    Scalar[output_dtype](w0) * Scalar[output_dtype](x0)
                    + Scalar[output_dtype](w1) * Scalar[output_dtype](x1)
                    + Scalar[output_dtype](w2) * Scalar[output_dtype](x2)
                    + Scalar[output_dtype](w3) * Scalar[output_dtype](x3)
                )
            var out_val: Scalar[output_dtype] = conv_sum
            if silu_active:
                comptime if output_dtype.is_floating_point():
                    out_val = silu(out_val)
                else:
                    out_val = silu(out_val.cast[.float32]()).cast[
                        output_dtype
                    ]()
            out_vals_channel[i] = out_val

        comptime for i in range(kNElts):
            var seq_pos: Int = seq_start + i
            if seq_pos >= seq_end:
                break
            output.store[width=1](
                (batch_id, c_idx, seq_pos), out_vals_channel[i]
            )


# ============================================================================
# Causal Conv1D Update Kernels
# ============================================================================
# These kernels implement incremental (step-by-step) convolution for inference,
# maintaining a conv_state buffer that gets updated with each step.


def causal_conv1d_update_cpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,  # seqlen of x (typically 1 for autoregressive inference)
    width: Int,
    state_len: Int,  # state_len of conv_state (>= width - 1)
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # Shape (B, C, L) or (B, C) when L=1
    conv_state: TileTensor[mut=True, conv_state_dtype, ...],  # Shape (B, C, S)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (C, W)
    output: TileTensor[mut=True, output_dtype, ...],  # Shape (B, C, L)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (C,)
    silu_activation: Bool,
):
    """
    CPU implementation of causal conv1d update for incremental inference.

    This kernel:
    1. Concatenates conv_state with x to form a sliding window
    2. Computes convolution output for the new positions
    3. Updates conv_state with the new values from x

    Simple mode (no circular buffer):
    - conv_state holds the last (state_len) values
    - New x values are appended, old values are shifted out

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of input x (typically 1).
        width: Kernel width.
        state_len: Length of conv_state (>= width - 1).
        x: Input tensor.
        conv_state: Convolution state buffer (modified in-place).
        weight: Convolution weights.
        output: Output tensor.
        bias: Bias tensor.
        silu_activation: Whether to apply SiLU activation.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
                bias.load[width=1]((c,))
            )
            # Process each position in the input sequence
            for l in range(seqlen):
                # Compute convolution sum using conv_state and x
                var conv_sum: Scalar[output_dtype] = cur_bias

                for w in range(width):
                    # Position in the virtual concatenated sequence [conv_state, x]
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[x_dtype] = 0.0

                    if src_pos >= state_len:
                        # Read from x
                        var x_l_pos = src_pos - state_len
                        input_val = x.load[width=1]((b, c, x_l_pos))
                    elif src_pos >= 0:
                        # Read from conv_state
                        input_val = Scalar[x_dtype](
                            conv_state.load[width=1]((b, c, src_pos))
                        )
                    # else: src_pos < 0, treat as 0 (zero padding)

                    var weight_val: Scalar[weight_dtype] = weight.load[width=1](
                        (c, w)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

                # Write output
                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, c, l), out_val)

            # Update conv_state: shift old values and add new x values
            if seqlen >= state_len:
                # x is longer than state, just copy last state_len values from x
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_val = x.load[width=1]((b, c, x_l_pos))
                    conv_state.store[width=1](
                        (b, c, s), Scalar[conv_state_dtype](x_val)
                    )
            else:
                # Shift conv_state left by seqlen positions, then append x
                for s in range(state_len - seqlen):
                    var val = conv_state.load[width=1]((b, c, (s + seqlen)))
                    conv_state.store[width=1]((b, c, s), val)

                # Copy x values to the end
                for l in range(seqlen):
                    var x_val = x.load[width=1]((b, c, l))
                    conv_state.store[width=1](
                        (b, c, (state_len - seqlen + l)),
                        Scalar[conv_state_dtype](x_val),
                    )


def causal_conv1d_update_cpu_no_bias[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: TileTensor[mut=False, x_dtype, ...],
    conv_state: TileTensor[mut=True, conv_state_dtype, ...],
    weight: TileTensor[mut=False, weight_dtype, ...],
    output: TileTensor[mut=True, output_dtype, ...],
    silu_activation: Bool,
):
    """CPU implementation of causal conv1d update without bias.

    Performs incremental convolution for autoregressive decode by treating
    `conv_state` followed by `x` as a virtual sliding window, computing the
    output for the new positions, then shifting the newest `state_len`
    values back into `conv_state` in place.

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.

    Args:
        batch: Number of sequences processed in parallel.
        dim: Number of channels per sequence position.
        seqlen: Number of new input positions in `x` (1 for autoregressive
            decode).
        width: Convolution kernel width in positions.
        state_len: Length of the rolling buffer stored in `conv_state`;
            must be at least `width - 1`.
        x: Input tensor of shape (B, C, L) holding the new positions to
            convolve.
        conv_state: Rolling convolution state of shape (B, C, S) that
            holds the last `state_len` values; updated in place.
        weight: Convolution weights of shape (C, W).
        output: Output tensor of shape (B, C, L) receiving the convolved
            values for the new positions.
        silu_activation: Whether to apply the SiLU activation to the
            output values before storing.
    """
    var width_minus_1: Int = width - 1

    for b in range(batch):
        for c in range(dim):
            for l in range(seqlen):
                var conv_sum: Scalar[output_dtype] = 0.0

                for w in range(width):
                    var src_pos = state_len + l - (width_minus_1 - w)
                    var input_val: Scalar[x_dtype] = 0.0

                    if src_pos >= state_len:
                        var x_l_pos = src_pos - state_len
                        input_val = x.load[width=1]((b, c, x_l_pos))
                    elif src_pos >= 0:
                        input_val = Scalar[x_dtype](
                            conv_state.load[width=1]((b, c, src_pos))
                        )
                    var weight_val: Scalar[weight_dtype] = weight.load[width=1](
                        (c, w)
                    )
                    conv_sum = conv_sum + Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weight_val)
                    )

                var out_val: Scalar[output_dtype] = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[.float32]()).cast[
                            output_dtype
                        ]()
                output.store[width=1]((b, c, l), out_val)

            # Update conv_state
            if seqlen >= state_len:
                for s in range(state_len):
                    var x_l_pos = seqlen - state_len + s
                    var x_val = x.load[width=1]((b, c, x_l_pos))
                    conv_state.store[width=1](
                        (b, c, s), Scalar[conv_state_dtype](x_val)
                    )
            else:
                for s in range(state_len - seqlen):
                    var val = conv_state.load[width=1]((b, c, (s + seqlen)))
                    conv_state.store[width=1]((b, c, s), val)

                for l in range(seqlen):
                    var x_val = x.load[width=1]((b, c, l))
                    conv_state.store[width=1](
                        (b, c, (state_len - seqlen + l)),
                        Scalar[conv_state_dtype](x_val),
                    )


def causal_conv1d_update_gpu[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    bias_dtype: DType,
    kNThreads: Int,
    x_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    bias_LT: TensorLayout,
    x_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
    bias_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    silu_activation: Int8,
):
    """GPU kernel for causal conv1d update operation (for autoregressive decode).

    This kernel performs incremental updates to maintain convolution state for efficient
    autoregressive token generation. It processes a new input sequence and updates both
    the output and the internal convolution state.

    Grid: (batch, ceildiv(dim, kNThreads))
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        bias_dtype: Element type of the bias tensor `bias`.
        kNThreads: Number of threads per block used to process the channel
            dimension.
        x_LT: TensorLayout of the input tensor `x`.
        conv_state_LT: TensorLayout of the convolution state tensor
            `conv_state`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        bias_LT: TensorLayout of the bias tensor `bias`.
        x_engine: Engine of the input tensor `x`.
        conv_state_engine: Engine of the convolution state tensor
            `conv_state`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.
        bias_engine: Engine of the bias tensor `bias`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of the new input.
        width: Kernel width.
        state_len: Length of the convolution state buffer.
        x: Input tensor of shape (B, C, L).
        conv_state: Convolution state tensor of shape (B, C, state_len).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        bias: Bias tensor of shape (C,).
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)
    var _state_len = Int(state_len)
    var b = block_idx.x
    var c_base = block_idx.y * kNThreads
    var c = c_base + thread_idx.x

    if b >= _batch or c >= _dim:
        return

    var width_minus_1: Int = _width - 1
    var cur_bias: Scalar[output_dtype] = Scalar[output_dtype](
        bias.load[width=1]((c,))
    )
    var silu_active = Bool(silu_activation != 0)

    for l in range(_seqlen):
        var conv_sum: Scalar[output_dtype] = cur_bias

        for w in range(_width):
            var src_pos = _state_len + l - (width_minus_1 - w)
            var input_val: Scalar[x_dtype] = 0.0

            if src_pos >= _state_len:
                var x_l_pos = src_pos - _state_len
                input_val = x.load[width=1]((b, c, x_l_pos))
            elif src_pos >= 0:
                input_val = Scalar[x_dtype](
                    conv_state.load[width=1]((b, c, src_pos))
                )
            var weight_val: Scalar[weight_dtype] = weight.load[width=1]((c, w))
            conv_sum = conv_sum + Scalar[output_dtype](
                input_val * Scalar[x_dtype](weight_val)
            )
        var out_val: Scalar[output_dtype] = conv_sum
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        output.store[width=1]((b, c, l), out_val)

    # Update conv_state
    if _seqlen >= _state_len:
        for s in range(_state_len):
            var x_l_pos = _seqlen - _state_len + s
            var x_val = x.load[width=1]((b, c, x_l_pos))
            conv_state.store[width=1](
                (b, c, s), Scalar[conv_state_dtype](x_val)
            )
    else:
        for s in range(_state_len - _seqlen):
            var val = conv_state.load[width=1]((b, c, (s + _seqlen)))
            conv_state.store[width=1]((b, c, s), val)

        for l in range(_seqlen):
            var x_val = x.load[width=1]((b, c, l))
            conv_state.store[width=1](
                (b, c, (_state_len - _seqlen + l)),
                Scalar[conv_state_dtype](x_val),
            )


def causal_conv1d_update_gpu_no_bias[
    x_dtype: DType,
    conv_state_dtype: DType,
    weight_dtype: DType,
    output_dtype: DType,
    kNThreads: Int,
    x_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    weight_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    weight_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    width: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
    silu_activation: Int8,
):
    """GPU kernel for causal conv1d update operation without bias (for autoregressive decode).

    This kernel performs incremental updates to maintain convolution state for efficient
    autoregressive token generation. It processes a new input sequence and updates both
    the output and the internal convolution state.

    Grid: (batch, ceildiv(dim, kNThreads))
    Block: kNThreads

    Parameters:
        x_dtype: Element type of the input tensor `x`.
        conv_state_dtype: Element type of the convolution state tensor
            `conv_state`.
        weight_dtype: Element type of the weight tensor `weight`.
        output_dtype: Element type of the output tensor `output`.
        kNThreads: Number of threads per block used to process the channel
            dimension.
        x_LT: TensorLayout of the input tensor `x`.
        conv_state_LT: TensorLayout of the convolution state tensor
            `conv_state`.
        weight_LT: TensorLayout of the weight tensor `weight`.
        output_LT: TensorLayout of the output tensor `output`.
        x_engine: Engine of the input tensor `x`.
        conv_state_engine: Engine of the convolution state tensor
            `conv_state`.
        weight_engine: Engine of the weight tensor `weight`.
        output_engine: Engine of the output tensor `output`.

    Args:
        batch: Batch size.
        dim: Number of channels.
        seqlen: Sequence length of the new input.
        width: Kernel width.
        state_len: Length of the convolution state buffer.
        x: Input tensor of shape (B, C, L).
        conv_state: Convolution state tensor of shape (B, C, state_len).
        weight: Weight tensor of shape (C, W).
        output: Output tensor of shape (B, C, L).
        silu_activation: Whether to apply SiLU activation (Int8: 0 or 1).
    """
    var _batch = Int(batch)
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _width = Int(width)
    var _state_len = Int(state_len)
    var b = block_idx.x
    var c_base = block_idx.y * kNThreads
    var c = c_base + thread_idx.x

    if b >= _batch or c >= _dim:
        return

    var width_minus_1: Int = _width - 1
    var silu_active = Bool(silu_activation != 0)

    for l in range(_seqlen):
        var conv_sum: Scalar[output_dtype] = 0.0

        for w in range(_width):
            var src_pos = _state_len + l - (width_minus_1 - w)
            var input_val: Scalar[x_dtype] = 0.0

            if src_pos >= _state_len:
                var x_l_pos = src_pos - _state_len
                input_val = x.load[width=1]((b, c, x_l_pos))
            elif src_pos >= 0:
                input_val = Scalar[x_dtype](
                    conv_state.load[width=1]((b, c, src_pos))
                )
            var weight_val: Scalar[weight_dtype] = weight.load[width=1]((c, w))
            conv_sum = conv_sum + Scalar[output_dtype](
                input_val * Scalar[x_dtype](weight_val)
            )
        var out_val: Scalar[output_dtype] = conv_sum
        if silu_active:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[.float32]()).cast[output_dtype]()
        output.store[width=1]((b, c, l), out_val)

    if _seqlen >= _state_len:
        for s in range(_state_len):
            var x_l_pos = _seqlen - _state_len + s
            var x_val = x.load[width=1]((b, c, x_l_pos))
            conv_state.store[width=1](
                (b, c, s), Scalar[conv_state_dtype](x_val)
            )
    else:
        for s in range(_state_len - _seqlen):
            var val = conv_state.load[width=1]((b, c, (s + _seqlen)))
            conv_state.store[width=1]((b, c, s), val)

        for l in range(_seqlen):
            var x_val = x.load[width=1]((b, c, l))
            conv_state.store[width=1](
                (b, c, (_state_len - _seqlen + l)),
                Scalar[conv_state_dtype](x_val),
            )
