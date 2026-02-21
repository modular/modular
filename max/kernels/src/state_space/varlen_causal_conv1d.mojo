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
"""Causal Conv1D with variable length sequence support (vLLM interface).

This module implements causal 1D convolution operations that support variable
length sequences using cumulative sequence lengths (cu_seqlens), compatible
with the vLLM inference interface.

Key Functions:
    - causal_conv1d_varlen_fwd: Forward pass for varlen sequences
    - causal_conv1d_varlen_update: Update function for decode
    - causal_conv1d_varlen_states: Extract states from varlen sequences

vLLM Interface:
    - x: (dim, cu_seq_len) for varlen - sequences concatenated left to right
    - query_start_loc: (batch + 1) int32 - cumulative sequence lengths
    - cache_indices: (batch) int32 - indices into conv_states
    - has_initial_state: (batch) bool - whether to use initial state
    - conv_states: (..., dim, width - 1) - states updated in-place
    - activation: None or "silu" or "swish"
    - pad_slot_id: int - for identifying padded entries
"""

from collections import Optional
from math import ceildiv, exp

from algorithm import vectorize
from buffer.buffer import NDBuffer
from buffer.dimlist import Dim, DimList

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx

from memory import UnsafePointer, memcpy

from layout import Layout, LayoutTensor
from state_space.causal_conv1d import silu
from tensor import InputTensor, OutputTensor


# ============================================================================
# Constants
# ============================================================================

comptime PAD_SLOT_ID: Int32 = -1


# ============================================================================
# CPU Reference Implementations
# ============================================================================


fn causal_conv1d_varlen_states_cpu[
    x_dtype: DType,
    x_layout: Layout,
    cu_seqlens_dtype: DType,
    cu_seqlens_layout: Layout,
    states_dtype: DType,
    states_layout: Layout,
](
    total_tokens: Int,
    dim: Int,
    batch: Int,
    state_len: Int,
    x: LayoutTensor[
        x_dtype, x_layout, MutAnyOrigin
    ],  # Shape (total_tokens, dim)
    cu_seqlens: LayoutTensor[
        cu_seqlens_dtype, cu_seqlens_layout, MutAnyOrigin
    ],  # Shape (batch + 1,)
    states: LayoutTensor[
        states_dtype, states_layout, MutAnyOrigin
    ],  # Shape (batch, dim, state_len)
    x_seqlen_stride: UInt32,
    x_dim_stride: UInt32,
    states_batch_stride: UInt32,
    states_dim_stride: UInt32,
    states_seqlen_stride: UInt32,
):
    """Extract the last state_len elements from each variable length sequence.

    For each sequence in the batch, copies the last state_len tokens (or fewer
    if the sequence is shorter) to the states tensor. If a sequence is shorter
    than state_len, the earlier positions in states are zero-padded.

    This is the CPU reference implementation for causal_conv1d_varlen_states.

    Parameters:
        x_dtype: Data type of the input tensor.
        x_layout: Layout of the input tensor.
        cu_seqlens_dtype: Data type of the cumulative sequence lengths.
        cu_seqlens_layout: Layout of the cumulative sequence lengths.
        states_dtype: Data type of the output states tensor.
        states_layout: Layout of the output states tensor.

    Args:
        total_tokens: Total number of tokens across all sequences.
        dim: Number of channels/dimensions.
        batch: Number of sequences.
        state_len: Number of elements to extract per sequence (typically width - 1).
        x: Input tensor of shape (total_tokens, dim).
        cu_seqlens: Cumulative sequence lengths of shape (batch + 1,).
        states: Output states tensor of shape (batch, dim, state_len).
        x_seqlen_stride: Stride for sequence dimension in x.
        x_dim_stride: Stride for dimension in x.
        states_batch_stride: Stride for batch dimension in states.
        states_dim_stride: Stride for dimension in states.
        states_seqlen_stride: Stride for sequence dimension in states.
    """
    # Initialize states to zero
    for b in range(batch):
        for d in range(dim):
            for s in range(state_len):
                var states_offset = (
                    UInt32(b) * states_batch_stride
                    + UInt32(d) * states_dim_stride
                    + UInt32(s) * states_seqlen_stride
                )
                states.ptr[states_offset] = Scalar[states_dtype](0)

    # Extract states for each sequence
    for b in range(batch):
        var end_idx = Int(cu_seqlens.ptr[b + 1])
        var start_idx_seq = Int(cu_seqlens.ptr[b])
        var start_idx = max(start_idx_seq, end_idx - state_len)
        var num_elements = end_idx - start_idx

        # Copy elements from x to states
        # states[b, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
        for i in range(num_elements):
            var x_seq_idx = start_idx + i
            var states_seq_idx = state_len - num_elements + i

            for d in range(dim):
                var x_offset = (
                    UInt32(x_seq_idx) * x_seqlen_stride
                    + UInt32(d) * x_dim_stride
                )
                var states_offset = (
                    UInt32(b) * states_batch_stride
                    + UInt32(d) * states_dim_stride
                    + UInt32(states_seq_idx) * states_seqlen_stride
                )
                var val = x.ptr[x_offset]
                states.ptr[states_offset] = Scalar[states_dtype](val)


fn causal_conv1d_varlen_fwd_cpu[
    x_dtype: DType,
    x_layout: Layout,
    weight_dtype: DType,
    weight_layout: Layout,
    bias_dtype: DType,
    bias_layout: Layout,
    output_dtype: DType,
    output_layout: Layout,
    cu_seqlens_dtype: DType,
    cu_seqlens_layout: Layout,
    cache_indices_dtype: DType,
    cache_indices_layout: Layout,
    has_initial_state_dtype: DType,
    has_initial_state_layout: Layout,
    conv_states_dtype: DType,
    conv_states_layout: Layout,
](
    dim: Int,
    total_seqlen: Int,
    width: Int,
    batch: Int,
    x: LayoutTensor[
        x_dtype, x_layout, MutAnyOrigin
    ],  # Shape (dim, total_seqlen) for varlen
    weight: LayoutTensor[
        weight_dtype, weight_layout, MutAnyOrigin
    ],  # Shape (dim, width)
    bias: LayoutTensor[bias_dtype, bias_layout, MutAnyOrigin],  # Shape (dim,)
    query_start_loc: LayoutTensor[
        cu_seqlens_dtype, cu_seqlens_layout, MutAnyOrigin
    ],  # Shape (batch + 1,)
    cache_indices: LayoutTensor[
        cache_indices_dtype, cache_indices_layout, MutAnyOrigin
    ],  # Shape (batch,)
    has_initial_state: LayoutTensor[
        has_initial_state_dtype, has_initial_state_layout, MutAnyOrigin
    ],  # Shape (batch,)
    conv_states: LayoutTensor[
        conv_states_dtype, conv_states_layout, MutAnyOrigin
    ],  # Shape (..., dim, width - 1)
    output: LayoutTensor[
        output_dtype, output_layout, MutAnyOrigin
    ],  # Shape (dim, total_seqlen)
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    conv_states_batch_stride: UInt32,
    conv_states_dim_stride: UInt32,
    conv_states_width_stride: UInt32,
    silu_activation: Bool,
    pad_slot_id: Int32,
    has_cache_indices: Bool,
    has_initial_state_flag: Bool,
    has_conv_states: Bool,
    has_bias: Bool,
):
    """Forward pass for causal conv1d with variable length sequences.

    Performs causal 1D convolution on variable length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.
    """
    var width_minus_1 = width - 1

    # Process each sequence in the batch
    for b in range(batch):
        # Check if this is a padded entry
        if has_cache_indices:
            var cache_idx_val = Int32(cache_indices.ptr[b])
            if cache_idx_val == pad_slot_id:
                continue

        var seq_start = Int(query_start_loc.ptr[b])
        var seq_end = Int(query_start_loc.ptr[b + 1])
        var seqlen = seq_end - seq_start

        # Determine if we should use initial state
        var use_initial_state = False
        if has_initial_state_flag:
            use_initial_state = Bool(has_initial_state.ptr[b])

        # Get cache index for this batch
        var cache_idx: Int = b
        if has_cache_indices:
            cache_idx = Int(cache_indices.ptr[b])

        # Process each channel
        for d in range(dim):
            # Load bias
            var bias_val: Scalar[output_dtype] = 0
            if has_bias:
                bias_val = Scalar[output_dtype](bias.ptr[d])

            # Load weights for this channel
            var weights = List[Scalar[weight_dtype]]()
            for w_idx in range(width):
                var weight_offset = (
                    UInt32(d) * weight_dim_stride
                    + UInt32(w_idx) * weight_width_stride
                )
                weights.append(weight.ptr[weight_offset])

            # Process each position in the sequence
            for l in range(seqlen):
                var conv_sum = bias_val

                # Convolution sum
                for w_idx in range(width):
                    var input_l = l - (width_minus_1 - w_idx)
                    var input_val: Scalar[x_dtype] = 0

                    if input_l >= 0:
                        # Within current sequence
                        var x_offset = (
                            UInt32(d) * x_dim_stride
                            + UInt32((seq_start + input_l)) * x_seqlen_stride
                        )
                        input_val = x.ptr[x_offset]
                    elif use_initial_state and has_conv_states:
                        # Use initial state from conv_states
                        var state_idx = (
                            width_minus_1 + input_l
                        )  # Maps negative to state index
                        if state_idx >= 0:
                            var state_offset = (
                                UInt32(cache_idx) * conv_states_batch_stride
                                + UInt32(d) * conv_states_dim_stride
                                + UInt32(state_idx) * conv_states_width_stride
                            )
                            input_val = Scalar[x_dtype](
                                conv_states.ptr[state_offset]
                            )

                    conv_sum += Scalar[output_dtype](
                        input_val * Scalar[x_dtype](weights[w_idx])
                    )

                # Apply activation
                var out_val = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[DType.float32]()).cast[
                            output_dtype
                        ]()

                # Store output
                var out_offset = (
                    UInt32(d) * out_dim_stride
                    + UInt32((seq_start + l)) * out_seqlen_stride
                )
                output.ptr[out_offset] = out_val

            # Update conv_states with final state if provided
            if has_conv_states:
                # Copy last (width-1) elements to conv_states
                for s in range(width_minus_1):
                    var src_l = seqlen - width_minus_1 + s
                    var val: Scalar[conv_states_dtype] = 0

                    if src_l >= 0:
                        var x_offset = (
                            UInt32(d) * x_dim_stride
                            + UInt32((seq_start + src_l)) * x_seqlen_stride
                        )
                        val = Scalar[conv_states_dtype](x.ptr[x_offset])
                    elif use_initial_state:
                        # Carry over from initial state
                        var state_idx = width_minus_1 + src_l - (width_minus_1)
                        if state_idx >= 0 and state_idx < width_minus_1:
                            var state_offset = (
                                UInt32(cache_idx) * conv_states_batch_stride
                                + UInt32(d) * conv_states_dim_stride
                                + UInt32(state_idx) * conv_states_width_stride
                            )
                            val = conv_states.ptr[state_offset]

                    var state_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(s) * conv_states_width_stride
                    )
                    conv_states.ptr[state_offset] = val


fn causal_conv1d_varlen_update_cpu[
    x_dtype: DType,
    x_layout: Layout,
    weight_dtype: DType,
    weight_layout: Layout,
    bias_dtype: DType,
    bias_layout: Layout,
    output_dtype: DType,
    output_layout: Layout,
    conv_state_dtype: DType,
    conv_state_layout: Layout,
    cache_seqlens_dtype: DType,
    cache_seqlens_layout: Layout,
    conv_state_indices_dtype: DType,
    conv_state_indices_layout: Layout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: LayoutTensor[
        x_dtype, x_layout, MutAnyOrigin
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
    weight: LayoutTensor[
        weight_dtype, weight_layout, MutAnyOrigin
    ],  # Shape (dim, width)
    bias: LayoutTensor[bias_dtype, bias_layout, MutAnyOrigin],  # Shape (dim,)
    conv_state: LayoutTensor[
        conv_state_dtype, conv_state_layout, MutAnyOrigin
    ],  # Shape (batch, dim, state_len)
    cache_seqlens: LayoutTensor[
        cache_seqlens_dtype, cache_seqlens_layout, MutAnyOrigin
    ],  # Shape (batch,)
    conv_state_indices: LayoutTensor[
        conv_state_indices_dtype, conv_state_indices_layout, MutAnyOrigin
    ],  # Shape (batch,)
    output: LayoutTensor[
        output_dtype, output_layout, MutAnyOrigin
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
    x_batch_stride: UInt32,
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_dim_stride: UInt32,
    conv_state_seqlen_stride: UInt32,
    out_batch_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    silu_activation: Bool,
    pad_slot_id: Int32,
    has_conv_state_indices: Bool,
    has_cache_seqlens: Bool,
    has_bias: Bool,
):
    """Update function for causal conv1d decode.

    Updates the convolution state and computes output for decode steps.
    Supports circular buffer state management with cache_seqlens.
    """
    var width_minus_1 = width - 1

    for b in range(batch):
        # Check for padded entry
        if has_conv_state_indices:
            var state_idx_val = Int32(conv_state_indices.ptr[b])
            if state_idx_val == pad_slot_id:
                continue

        # Determine actual batch index for conv_state
        var state_batch_idx = b
        if has_conv_state_indices:
            state_batch_idx = Int(conv_state_indices.ptr[b])

        for d in range(dim):
            # Load bias
            var bias_val: Scalar[output_dtype] = 0
            if has_bias:
                bias_val = Scalar[output_dtype](bias.ptr[d])

            # Load weights
            var weights = List[Scalar[weight_dtype]]()
            for w_idx in range(width):
                var weight_offset = (
                    UInt32(d) * weight_dim_stride
                    + UInt32(w_idx) * weight_width_stride
                )
                weights.append(weight.ptr[weight_offset])

            for l in range(seqlen):
                # Get cache position for circular buffer
                var cache_offset = 0
                if has_cache_seqlens:
                    var cache_seqlen = Int(cache_seqlens.ptr[b])
                    cache_offset = (cache_seqlen + l) % state_len

                # Gather input values from state and current x
                var input_vals = List[Scalar[x_dtype]]()
                for w_idx in range(width):
                    var input_val: Scalar[x_dtype] = 0
                    var rel_pos = (
                        w_idx - width_minus_1
                    )  # Ranges from -(width-1) to 0

                    if rel_pos + l < 0:
                        # Read from state
                        var state_pos: Int
                        if has_cache_seqlens:
                            # Circular buffer
                            var cache_seqlen = Int(cache_seqlens.ptr[b])
                            state_pos = (
                                cache_seqlen + rel_pos + l + state_len
                            ) % state_len
                        else:
                            # Linear buffer: position in state
                            state_pos = width_minus_1 + rel_pos + l

                        if state_pos >= 0 and state_pos < state_len:
                            var state_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(state_pos) * conv_state_seqlen_stride
                            )
                            input_val = Scalar[x_dtype](
                                conv_state.ptr[state_offset]
                            )
                    else:
                        # Read from current x
                        var x_l = rel_pos + l
                        if x_l >= 0 and x_l < seqlen:
                            var x_offset = (
                                UInt32(b) * x_batch_stride
                                + UInt32(d) * x_dim_stride
                                + UInt32(x_l) * x_seqlen_stride
                            )
                            input_val = x.ptr[x_offset]

                    input_vals.append(input_val)

                # Compute convolution
                var conv_sum = bias_val
                for w_idx in range(width):
                    conv_sum += Scalar[output_dtype](
                        input_vals[w_idx] * Scalar[x_dtype](weights[w_idx])
                    )

                # Apply activation
                var out_val = conv_sum
                if silu_activation:
                    comptime if output_dtype.is_floating_point():
                        out_val = silu(out_val)
                    else:
                        out_val = silu(out_val.cast[DType.float32]()).cast[
                            output_dtype
                        ]()

                # Store output
                var out_offset = (
                    UInt32(b) * out_batch_stride
                    + UInt32(d) * out_dim_stride
                    + UInt32(l) * out_seqlen_stride
                )
                output.ptr[out_offset] = out_val

            # Update state with new x values
            for l in range(seqlen):
                var x_offset = (
                    UInt32(b) * x_batch_stride
                    + UInt32(d) * x_dim_stride
                    + UInt32(l) * x_seqlen_stride
                )
                var x_val = x.ptr[x_offset]

                var state_pos: Int
                if has_cache_seqlens:
                    # Circular buffer
                    var cache_seqlen = Int(cache_seqlens.ptr[b])
                    state_pos = (cache_seqlen + l) % state_len
                else:
                    # Shift state left and add new value at end
                    if l == 0:
                        # Shift existing values
                        for s in range(state_len - seqlen):
                            var src_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32((s + seqlen))
                                * conv_state_seqlen_stride
                            )
                            var dst_offset = (
                                UInt32(state_batch_idx)
                                * conv_state_batch_stride
                                + UInt32(d) * conv_state_dim_stride
                                + UInt32(s) * conv_state_seqlen_stride
                            )
                            var val = conv_state.ptr[src_offset]
                            conv_state.ptr[dst_offset] = val
                    state_pos = state_len - seqlen + l

                var state_offset = (
                    UInt32(state_batch_idx) * conv_state_batch_stride
                    + UInt32(d) * conv_state_dim_stride
                    + UInt32(state_pos) * conv_state_seqlen_stride
                )
                conv_state.ptr[state_offset] = Scalar[conv_state_dtype](x_val)


# ============================================================================
# GPU Kernel Implementations
# ============================================================================


fn causal_conv1d_varlen_states_gpu[
    x_dtype: DType,
    x_layout: Layout,
    cu_seqlens_dtype: DType,
    cu_seqlens_layout: Layout,
    states_dtype: DType,
    states_layout: Layout,
    BLOCK_M: Int,
    BLOCK_N: Int,
](
    total_tokens: Int,
    dim: Int,
    batch: Int,
    state_len: Int,
    x: LayoutTensor[
        x_dtype, x_layout, MutAnyOrigin
    ],  # Shape (total_tokens, dim)
    cu_seqlens: LayoutTensor[
        cu_seqlens_dtype, cu_seqlens_layout, MutAnyOrigin
    ],  # Shape (batch + 1,)
    states: LayoutTensor[
        states_dtype, states_layout, MutAnyOrigin
    ],  # Shape (batch, dim, state_len)
    x_seqlen_stride: UInt32,
    x_dim_stride: UInt32,
    states_batch_stride: UInt32,
    states_dim_stride: UInt32,
    states_seqlen_stride: UInt32,
):
    """GPU kernel for extracting states from variable length sequences.

    Each thread block processes a tile of (BLOCK_M x BLOCK_N) elements.
    Grid dimensions: (ceildiv(dim, BLOCK_N), ceildiv(state_len, BLOCK_M), batch)

    Parameters:
        x_dtype: Data type of input.
        x_layout: Layout of input.
        cu_seqlens_dtype: Data type of cumulative sequence lengths.
        cu_seqlens_layout: Layout of cumulative sequence lengths.
        states_dtype: Data type of output states.
        states_layout: Layout of output states.
        BLOCK_M: Tile size for sequence dimension.
        BLOCK_N: Tile size for channel dimension.

    Args:
        total_tokens: Total number of tokens.
        dim: Number of channels.
        batch: Number of sequences.
        state_len: State length to extract.
        x: Input tensor.
        cu_seqlens: Cumulative sequence lengths.
        states: Output states tensor.
        x_seqlen_stride: Stride for sequence in x.
        x_dim_stride: Stride for dimension in x.
        states_batch_stride: Stride for batch in states.
        states_dim_stride: Stride for dimension in states.
        states_seqlen_stride: Stride for sequence in states.
    """
    var batch_idx = Int(block_idx.z)
    var block_row = Int(block_idx.y)
    var block_col = Int(block_idx.x)
    var tid_row = Int(thread_idx.y)
    var tid_col = Int(thread_idx.x)

    # Load sequence boundaries
    var end_idx = Int(cu_seqlens.ptr[batch_idx + 1])
    var start_idx_seq = Int(cu_seqlens.ptr[batch_idx])
    var start_idx = max(start_idx_seq, end_idx - state_len)

    # Calculate row indices (processing from end backwards)
    var row = end_idx - (block_row * BLOCK_M + tid_row + 1)
    var col = block_col * BLOCK_N + tid_col

    # Load value from x if in valid range
    var val: Scalar[states_dtype] = 0
    if row >= start_idx and col < dim:
        var x_offset = (
            UInt32(row) * x_seqlen_stride + UInt32(col) * x_dim_stride
        )
        val = Scalar[states_dtype](x.ptr[x_offset])

    # Calculate state row index
    var states_row = state_len - (block_row * BLOCK_M + tid_row + 1)

    # Store to states if in valid range
    if states_row >= 0 and col < dim:
        var states_offset = (
            UInt32(batch_idx) * states_batch_stride
            + UInt32(col) * states_dim_stride
            + UInt32(states_row) * states_seqlen_stride
        )
        states.ptr[states_offset] = val


fn causal_conv1d_varlen_fwd_gpu[
    x_dtype: DType,
    x_layout: Layout,
    weight_dtype: DType,
    weight_layout: Layout,
    bias_dtype: DType,
    bias_layout: Layout,
    output_dtype: DType,
    output_layout: Layout,
    cu_seqlens_dtype: DType,
    cu_seqlens_layout: Layout,
    cache_indices_dtype: DType,
    cache_indices_layout: Layout,
    has_initial_state_dtype: DType,
    has_initial_state_layout: Layout,
    conv_states_dtype: DType,
    conv_states_layout: Layout,
    WIDTH: Int,
    BLOCK_DIM: Int,
    BLOCK_SEQ: Int,
](
    dim: Int,
    total_seqlen: Int,
    batch: Int,
    x: LayoutTensor[x_dtype, x_layout, MutAnyOrigin],
    weight: LayoutTensor[weight_dtype, weight_layout, MutAnyOrigin],
    bias: LayoutTensor[bias_dtype, bias_layout, MutAnyOrigin],
    query_start_loc: LayoutTensor[
        cu_seqlens_dtype, cu_seqlens_layout, MutAnyOrigin
    ],
    cache_indices: LayoutTensor[
        cache_indices_dtype, cache_indices_layout, MutAnyOrigin
    ],
    has_initial_state: LayoutTensor[
        has_initial_state_dtype, has_initial_state_layout, MutAnyOrigin
    ],
    conv_states: LayoutTensor[
        conv_states_dtype, conv_states_layout, MutAnyOrigin
    ],
    output: LayoutTensor[output_dtype, output_layout, MutAnyOrigin],
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    conv_states_batch_stride: UInt32,
    conv_states_dim_stride: UInt32,
    conv_states_width_stride: UInt32,
    silu_activation: Int8,
    pad_slot_id: Int32,
    has_cache_indices: Int8,
    has_initial_state_flag: Int8,
    has_conv_states: Int8,
    has_bias: Int8,
):
    """GPU kernel for causal conv1d forward with variable length sequences.

    Grid: (batch, ceildiv(dim, BLOCK_DIM))
    Block: (BLOCK_DIM, BLOCK_SEQ)

    Each block processes BLOCK_DIM channels for one sequence.

    Note: silu_activation and flag parameters are Int8 (0 or 1) instead of Bool
    for DevicePassable compatibility on GPU.
    """
    var batch_idx = Int(block_idx.x)
    var dim_block_idx = Int(block_idx.y)
    var tid = Int(thread_idx.x)

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_cache_indices != 0:
        var cache_idx_val = Int32(cache_indices.ptr[batch_idx])
        if cache_idx_val == pad_slot_id:
            return

    # Get sequence bounds
    var seq_start = Int(query_start_loc.ptr[batch_idx])
    var seq_end = Int(query_start_loc.ptr[batch_idx + 1])
    var seqlen = seq_end - seq_start

    if d >= dim:
        return

    # Check for initial state
    var use_initial_state = False
    if has_initial_state_flag != 0:
        use_initial_state = Bool(has_initial_state.ptr[batch_idx])

    # Get cache index
    var cache_idx: Int = batch_idx
    if has_cache_indices != 0:
        cache_idx = Int(cache_indices.ptr[batch_idx])

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](bias.ptr[d])

    # Load weights into registers
    var weights = SIMD[weight_dtype, 8](0)  # Initialize with zeros
    for w_idx in range(WIDTH):
        var weight_offset = (
            UInt32(d) * weight_dim_stride + UInt32(w_idx) * weight_width_stride
        )
        weights[w_idx] = weight.ptr[weight_offset]

    comptime WIDTH_MINUS_1 = WIDTH - 1

    # Process sequence
    for l in range(seqlen):
        var conv_sum = bias_val

        # Gather inputs and compute convolution
        comptime for w_idx in range(WIDTH):
            var input_l = l - (WIDTH_MINUS_1 - w_idx)
            var input_val: Scalar[x_dtype] = 0

            if input_l >= 0:
                var x_offset = (
                    UInt32(d) * x_dim_stride
                    + UInt32((seq_start + input_l)) * x_seqlen_stride
                )
                input_val = x.ptr[x_offset]
            elif use_initial_state and has_conv_states != 0:
                var state_idx = WIDTH_MINUS_1 + input_l
                if state_idx >= 0:
                    var state_offset = (
                        UInt32(cache_idx) * conv_states_batch_stride
                        + UInt32(d) * conv_states_dim_stride
                        + UInt32(state_idx) * conv_states_width_stride
                    )
                    input_val = Scalar[x_dtype](conv_states.ptr[state_offset])

            conv_sum += Scalar[output_dtype](
                input_val * Scalar[x_dtype](weights[w_idx])
            )

        # Apply activation
        var out_val = conv_sum
        if silu_activation != 0:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[DType.float32]()).cast[
                    output_dtype
                ]()

        # Store output
        var out_offset = (
            UInt32(d) * out_dim_stride
            + UInt32((seq_start + l)) * out_seqlen_stride
        )
        output.ptr[out_offset] = out_val

    # Update conv_states
    if has_conv_states != 0:
        comptime for s in range(WIDTH_MINUS_1):
            var src_l = seqlen - WIDTH_MINUS_1 + s
            var val: Scalar[conv_states_dtype] = 0

            if src_l >= 0:
                var x_offset = (
                    UInt32(d) * x_dim_stride
                    + UInt32((seq_start + src_l)) * x_seqlen_stride
                )
                val = Scalar[conv_states_dtype](x.ptr[x_offset])

            var state_offset = (
                UInt32(cache_idx) * conv_states_batch_stride
                + UInt32(d) * conv_states_dim_stride
                + UInt32(s) * conv_states_width_stride
            )
            conv_states.ptr[state_offset] = val


fn causal_conv1d_varlen_update_gpu[
    x_dtype: DType,
    x_layout: Layout,
    weight_dtype: DType,
    weight_layout: Layout,
    bias_dtype: DType,
    bias_layout: Layout,
    output_dtype: DType,
    output_layout: Layout,
    conv_state_dtype: DType,
    conv_state_layout: Layout,
    cache_seqlens_dtype: DType,
    cache_seqlens_layout: Layout,
    conv_state_indices_dtype: DType,
    conv_state_indices_layout: Layout,
    WIDTH: Int,
    BLOCK_DIM: Int,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    state_len: Int,
    x: LayoutTensor[x_dtype, x_layout, MutAnyOrigin],
    weight: LayoutTensor[weight_dtype, weight_layout, MutAnyOrigin],
    bias: LayoutTensor[bias_dtype, bias_layout, MutAnyOrigin],
    conv_state: LayoutTensor[conv_state_dtype, conv_state_layout, MutAnyOrigin],
    cache_seqlens: LayoutTensor[
        cache_seqlens_dtype, cache_seqlens_layout, MutAnyOrigin
    ],
    conv_state_indices: LayoutTensor[
        conv_state_indices_dtype, conv_state_indices_layout, MutAnyOrigin
    ],
    output: LayoutTensor[output_dtype, output_layout, MutAnyOrigin],
    x_batch_stride: UInt32,
    x_dim_stride: UInt32,
    x_seqlen_stride: UInt32,
    weight_dim_stride: UInt32,
    weight_width_stride: UInt32,
    conv_state_batch_stride: UInt32,
    conv_state_dim_stride: UInt32,
    conv_state_seqlen_stride: UInt32,
    out_batch_stride: UInt32,
    out_dim_stride: UInt32,
    out_seqlen_stride: UInt32,
    silu_activation: Int8,
    pad_slot_id: Int32,
    has_conv_state_indices: Int8,
    has_cache_seqlens: Int8,
    has_bias: Int8,
):
    """GPU kernel for causal conv1d update (decode step).

    Grid: (batch, ceildiv(dim, BLOCK_DIM))
    Block: (BLOCK_DIM,)

    Note: silu_activation and flag parameters are Int8 (0 or 1) instead of Bool
    for DevicePassable compatibility on GPU.
    """
    var batch_idx = Int(block_idx.x)
    var dim_block_idx = Int(block_idx.y)
    var tid = Int(thread_idx.x)

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_conv_state_indices != 0:
        var state_idx_val = Int32(conv_state_indices.ptr[batch_idx])
        if state_idx_val == pad_slot_id:
            return

    if d >= dim:
        return

    # Get state batch index
    var state_batch_idx: Int = batch_idx
    if has_conv_state_indices != 0:
        state_batch_idx = Int(conv_state_indices.ptr[batch_idx])

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](bias.ptr[d])

    # Load weights
    var weights = SIMD[weight_dtype, 8](0)  # Initialize with zeros
    for w_idx in range(WIDTH):
        var weight_offset = (
            UInt32(d) * weight_dim_stride + UInt32(w_idx) * weight_width_stride
        )
        weights[w_idx] = weight.ptr[weight_offset]

    comptime WIDTH_MINUS_1 = WIDTH - 1

    for l in range(seqlen):
        # Get cache position
        var cache_offset = 0
        if has_cache_seqlens != 0:
            var cache_seqlen = Int(cache_seqlens.ptr[batch_idx])
            cache_offset = cache_seqlen

        # Gather inputs and compute
        var conv_sum = bias_val

        comptime for w_idx in range(WIDTH):
            var rel_pos = w_idx - WIDTH_MINUS_1
            var input_val: Scalar[x_dtype] = 0

            if rel_pos + l < 0:
                # From state
                var state_pos: Int
                if has_cache_seqlens != 0:
                    state_pos = (
                        cache_offset + rel_pos + l + state_len
                    ) % state_len
                else:
                    state_pos = WIDTH_MINUS_1 + rel_pos + l

                if state_pos >= 0 and state_pos < state_len:
                    var state_offset = (
                        UInt32(state_batch_idx) * conv_state_batch_stride
                        + UInt32(d) * conv_state_dim_stride
                        + UInt32(state_pos) * conv_state_seqlen_stride
                    )
                    input_val = Scalar[x_dtype](conv_state.ptr[state_offset])
            else:
                # From x
                var x_l = rel_pos + l
                if x_l >= 0 and x_l < seqlen:
                    var x_offset = (
                        UInt32(batch_idx) * x_batch_stride
                        + UInt32(d) * x_dim_stride
                        + UInt32(x_l) * x_seqlen_stride
                    )
                    input_val = x.ptr[x_offset]

            conv_sum += Scalar[output_dtype](
                input_val * Scalar[x_dtype](weights[w_idx])
            )
        # Apply activation
        var out_val = conv_sum
        if silu_activation != 0:
            comptime if output_dtype.is_floating_point():
                out_val = silu(out_val)
            else:
                out_val = silu(out_val.cast[DType.float32]()).cast[
                    output_dtype
                ]()

        # Store output
        var out_offset = (
            UInt32(batch_idx) * out_batch_stride
            + UInt32(d) * out_dim_stride
            + UInt32(l) * out_seqlen_stride
        )
        output.ptr[out_offset] = out_val

        # Update state
        var x_offset = (
            UInt32(batch_idx) * x_batch_stride
            + UInt32(d) * x_dim_stride
            + UInt32(l) * x_seqlen_stride
        )
        var x_val = x.ptr[x_offset]

        var state_pos: Int
        if has_cache_seqlens != 0:
            state_pos = (cache_offset + l) % state_len
        else:
            state_pos = state_len - seqlen + l

        var state_offset = (
            UInt32(state_batch_idx) * conv_state_batch_stride
            + UInt32(d) * conv_state_dim_stride
            + UInt32(state_pos) * conv_state_seqlen_stride
        )
        conv_state.ptr[state_offset] = Scalar[conv_state_dtype](x_val)
