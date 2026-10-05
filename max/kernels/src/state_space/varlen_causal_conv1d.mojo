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
    - cache_indices: (batch) uint32 - indices into conv_states
    - has_initial_state: (batch) bool - whether to use initial state
    - conv_states: (..., dim, width - 1) - states updated in-place
    - activation: None or "silu" or "swish"
    - pad_slot_id: int - for identifying padded entries
"""

from std.algorithm import vectorize
from std.bit import count_leading_zeros, next_power_of_two
from std.math import ceildiv, clamp, exp, recip
from std.sys import size_of
from std.sys.info import is_nvidia_gpu
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.host import DeviceContext


from layout import TensorLayout, TileTensor
from layout.tensor_engine import TensorEngine

from nn.activations import silu


# ============================================================================
# Constants
# ============================================================================

comptime PAD_SLOT_ID: Int32 = -1

# Lane count of a tap vector holding a WIDTH-tap filter. WIDTH dispatches in
# {1, 2, 3, 4} and 3 is not a valid SIMD width, so round up and zero the
# padding lane; a padded dot product still sums to the WIDTH-tap result.
comptime _TAP_LANES[WIDTH: Int] = Int(next_power_of_two(WIDTH))


# ============================================================================
# Shared helpers
# ============================================================================


@inline(.always)
def _apply_silu[
    output_dtype: DType, width: Int = 1
](out_val: SIMD[output_dtype, width], silu_activation: Bool) -> SIMD[
    output_dtype, width
]:
    """Optionally apply the SiLU activation, preserving the accumulator dtype.

    Floating-point outputs run SiLU in place; integral outputs promote to f32
    for the activation then cast back. Works on a scalar or a channel vector.

    Parameters:
        output_dtype: The output element type.
        width: The number of channels, 1 for a scalar.

    Args:
        out_val: The pre-activation convolution output.
        silu_activation: Whether to apply SiLU.

    Returns:
        `out_val`, with SiLU applied when `silu_activation` is True.
    """
    if silu_activation:
        comptime if output_dtype.is_floating_point():
            comptime if output_dtype == .float32 and is_nvidia_gpu():
                # IEEE division branches per lane to handle denormals, which
                # serializes the channel vector. An approximate reciprocal
                # with one Newton step gives the same f32 bits for |x| <= 80;
                # the clamp keeps `d * r` from becoming `inf * 0` when
                # `exp(-x)` overflows.
                var d = 1 + exp(-out_val)
                var r = recip(d)
                var cap = SIMD[output_dtype, width](8.5070592e37)
                return out_val * (r + r * (1 - min(d, cap) * r))
            else:
                return silu(out_val)
        else:
            return silu(out_val.cast[.float32]()).cast[output_dtype]()
    return out_val


@inline(.always)
def _channel_weights[
    weight_dtype: DType,
    WIDTH: Int,
    weight_LT: TensorLayout,
    weight_engine: TensorEngine,
](
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    d: Int,
) -> SIMD[weight_dtype, _TAP_LANES[WIDTH]]:
    """Load channel `d`'s `WIDTH` conv weights into a register vector.

    Parameters:
        weight_dtype: The weight element type.
        WIDTH: The convolution width (number of taps).
        weight_LT: Layout type of the weight tensor.
        weight_engine: Engine of the weight tensor.

    Args:
        weight: The `(dim, width)` weight tensor.
        d: The channel index.

    Returns:
        Channel `d`'s weights, tap `w_idx` in lane `w_idx`, with the padding
        lanes zeroed so a padded dot product still sums to the `WIDTH`-tap
        result.
    """
    var weights = SIMD[weight_dtype, _TAP_LANES[WIDTH]](0)
    comptime for w_idx in range(WIDTH):
        weights[w_idx] = weight.load[width=1, alignment=1]((d, w_idx))
    return weights


@inline(.always)
def _use_initial_state(
    has_initial_state: TileTensor[DType.bool, ...],
    conv_states: TileTensor[...],
    b: Int,
) -> Bool:
    """Whether sequence `b` continues from its stored conv state.

    An empty `has_initial_state` or `conv_states` means no sequence does.
    """
    return (
        Int(conv_states.dim[0]()) > 0
        and Int(has_initial_state.dim[0]()) > 0
        and Bool(has_initial_state.load[width=1]((b,)))
    )


@inline(.always)
def _cache_slot[
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
](slot_fn: SlotFn, b: Int) -> Int:
    """Returns sequence `b`'s conv-state pool slot.

    A slot equal to `PAD_SLOT_ID` marks a padded entry. The `uint32` slot is
    read back as `Int32`, so the pad value survives the round trip.
    """
    return Int(Int32(slot_fn[1, 1](b)))


# ============================================================================
# Forward-path DRAM I/O owner
# ============================================================================


@fieldwise_init
struct VarlenConvIO[
    weight_origin: Origin,
    bias_origin: Origin,
    out_origin: MutOrigin,
    dtype: DType,
    weight_layout: TensorLayout,
    bias_layout: TensorLayout,
    out_layout: TensorLayout,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    out_engine: TensorEngine,
    weight_addr: AddressSpace,
    bias_addr: AddressSpace,
    out_addr: AddressSpace,
    weight_idx: DType,
    bias_idx: DType,
    out_idx: DType,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[dtype, width],
    //,
    channels_last: Bool,
](ImplicitlyCopyable, Movable):
    """Owner of the forward-path DRAM reads/writes for varlen causal conv1d.

    Holds the input reader, the `(dim, width)` `weight`, `(dim,)` `bias`, and
    `output` TileTensor views and exposes one method per access verb.

    `x` and `output` are the caller's physical tensors: `(dim, seqlen)`, or
    `(seqlen, dim)` when `channels_last`. The methods take the logical
    `(d, s)` and order the coordinates by `channels_last`, so the axis order stays
    out of the kernels.

    Every view parameter is inferred from the constructor arguments, so the
    owner adapts to whatever storage the caller's views carry; the CPU
    reference and the GPU kernels pass differently-parameterized views. Only
    the store target `output` pins a mutable origin.

    Parameters:
        weight_origin: Inferred origin of the weight view.
        bias_origin: Inferred origin of the bias view.
        out_origin: Inferred mutable origin of the output view.
        dtype: Element type of `x`, `weight`, `bias` and `output`.
        weight_layout: Layout type of the `(dim, width)` weight view.
        bias_layout: Layout type of the `(dim,)` bias view.
        out_layout: Layout type of the output view.
        weight_engine: Inferred engine of the weight view.
        bias_engine: Inferred engine of the bias view.
        out_engine: Inferred engine of the output view.
        weight_addr: Inferred address space of the weight view.
        bias_addr: Inferred address space of the bias view.
        out_addr: Inferred address space of the output view.
        weight_idx: Inferred linear-index type of the weight view.
        bias_idx: Inferred linear-index type of the bias view.
        out_idx: Inferred linear-index type of the output view.
        XFn: Inferred reader of `x` at its physical `(row, column)` index.
        channels_last: Whether `x` and `output` are `(seqlen, dim)` rather than
            `(dim, seqlen)`.
    """

    var x_fn: Self.XFn
    var weight: TileTensor[
        Self.dtype,
        Self.weight_layout,
        Self.weight_origin,
        Engine=Self.weight_engine,
        address_space=Self.weight_addr,
        linear_idx_type=Self.weight_idx,
    ]
    var bias: TileTensor[
        Self.dtype,
        Self.bias_layout,
        Self.bias_origin,
        Engine=Self.bias_engine,
        address_space=Self.bias_addr,
        linear_idx_type=Self.bias_idx,
    ]
    var output: TileTensor[
        Self.dtype,
        Self.out_layout,
        Self.out_origin,
        Engine=Self.out_engine,
        address_space=Self.out_addr,
        linear_idx_type=Self.out_idx,
    ]

    @inline(.always)
    def dim(self) -> Int:
        """Returns the number of channels."""
        # `output` has the physical shape of `x`.
        comptime dim_axis = 1 if Self.channels_last else 0
        return Int(self.output.dim[dim_axis]())

    @inline(.always)
    def load_x[width: Int = 1](self, d: Int, s: Int) -> SIMD[Self.dtype, width]:
        """Load `x[d:d + width, s]`: windowed history read of the input.

        Parameters:
            width: The number of consecutive channels; above 1 needs
                `channels_last`, where they are contiguous.

        Args:
            d: The first channel index.
            s: The packed sequence position.
        """
        comptime if Self.channels_last:
            return self.x_fn[width, width](s, d)
        else:
            comptime assert width == 1, "channel vectors need channels_last"
            return self.x_fn[1, 1](d, s)

    @inline(.always)
    def load_weight(self, d: Int, w: Int) -> Scalar[Self.dtype]:
        """Load `weight[d, w]`: per-channel conv tap.

        Args:
            d: The channel index into the `(dim, width)` weight view.
            w: The convolution tap index in `[0, width)`.
        """
        return self.weight.load[width=1]((d, w))

    @inline(.always)
    def load_bias[width: Int = 1](self, d: Int) -> SIMD[Self.dtype, width]:
        """Load `bias[d:d + width]`, or zero when `bias` is empty.

        Parameters:
            width: The number of consecutive channels.

        Args:
            d: The first channel index into the `(dim,)` bias view.
        """
        if Int(self.bias.dim[0]()) == 0:
            return 0
        return self.bias.load[width=width]((d,))

    @inline(.always)
    def store_out[
        width: Int = 1
    ](self, d: Int, s: Int, val: SIMD[Self.dtype, width]):
        """Store `output[d:d + width, s] = val`: convolution output.

        Parameters:
            width: The number of consecutive channels, as in `load_x`.

        Args:
            d: The first channel index.
            s: The packed sequence position.
            val: The convolution result to store.
        """
        comptime if Self.channels_last:
            self.output.store((s, d), val)
        else:
            comptime assert width == 1, "channel vectors need channels_last"
            self.output.store((d, s), val)


# ============================================================================
# CPU Reference Implementations
# ============================================================================


def causal_conv1d_varlen_states_cpu[
    x_dtype: DType,
    cu_seqlens_dtype: DType,
    states_dtype: DType,
](
    total_tokens: Int,
    dim: Int,
    batch: Int,
    state_len: Int,
    x: TileTensor[mut=False, x_dtype, ...],  # Shape (total_tokens, dim)
    cu_seqlens: TileTensor[
        mut=False, cu_seqlens_dtype, ...
    ],  # Shape (batch + 1,)
    states: TileTensor[
        mut=True, states_dtype, ...
    ],  # Shape (batch, dim, state_len)
):
    """Extract the last state_len elements from each variable length sequence.

    For each sequence in the batch, copies the last state_len tokens (or fewer
    if the sequence is shorter) to the states tensor. If a sequence is shorter
    than state_len, the earlier positions in states are zero-padded.

    This is the CPU reference implementation for causal_conv1d_varlen_states.

    Parameters:
        x_dtype: Data type of the input tensor.
        cu_seqlens_dtype: Data type of the cumulative sequence lengths.
        states_dtype: Data type of the output states tensor.

    Args:
        total_tokens: Total number of tokens across all sequences.
        dim: Number of channels/dimensions.
        batch: Number of sequences.
        state_len: Number of elements to extract per sequence (typically width - 1).
        x: Input tensor of shape (total_tokens, dim).
        cu_seqlens: Cumulative sequence lengths of shape (batch + 1,).
        states: Output states tensor of shape (batch, dim, state_len).
    """
    # Initialize states to zero
    for b in range(batch):
        for d in range(dim):
            for s in range(state_len):
                states.store[width=1]((b, d, s), Scalar[states_dtype](0))

    # Extract states for each sequence
    for b in range(batch):
        var end_idx = Int(cu_seqlens.load[width=1]((b + 1,)))
        var start_idx_seq = Int(cu_seqlens.load[width=1]((b,)))
        var start_idx = max(start_idx_seq, end_idx - state_len)
        var num_elements = end_idx - start_idx

        # Copy elements from x to states
        # states[b, :, -(end_idx - start_idx):] = x[start_idx:end_idx].T
        for i in range(num_elements):
            var x_seq_idx = start_idx + i
            var states_seq_idx = state_len - num_elements + i

            for d in range(dim):
                var val = x.load[width=1]((x_seq_idx, d))
                states.store[width=1](
                    (b, d, states_seq_idx), Scalar[states_dtype](val)
                )


def causal_conv1d_varlen_fwd_cpu[
    dtype: DType,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[dtype, width],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    //,
    silu_activation: Bool,
    use_residual: Bool = False,
    channels_last: Bool = False,
](
    weight: TileTensor[mut=False, dtype, ...],
    bias: TileTensor[mut=False, dtype, ...],
    query_start_loc: TileTensor[mut=False, .int32, ...],
    has_initial_state: TileTensor[mut=False, .bool, ...],
    conv_states: TileTensor[mut=True, ...],
    output: TileTensor[mut=True, dtype, ...],
    x_fn: XFn,
    slot_fn: SlotFn,
):
    """Forward pass for causal conv1d with variable length sequences.

    Performs causal 1D convolution on variable length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.

    This is the CPU reference implementation for causal_conv1d_varlen_fwd.

    Parameters:
        dtype: Element type of `x`, `weight`, `bias` and `output`.
        XFn: Reads `x` at its physical `(row, column)` index, claiming
            `alignment` elements of alignment.
        SlotFn: Reads the conv-state slot of each sequence.
        silu_activation: Whether to apply the SiLU activation to the output.
        use_residual: If True, adds `x[d, s]` to the convolution sum at each
            output position, before the activation.
        channels_last: If True, `x` and `output` are `(total_seqlen, dim)`
            rather than `(dim, total_seqlen)`.

    Args:
        weight: Weight tensor of shape `(dim, width)`.
        bias: Bias tensor of shape `(dim,)`, or empty.
        query_start_loc: Cumulative sequence lengths of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flags of shape `(batch,)` selecting
            whether to continue the stored state, or empty.
        conv_states: Convolution states of shape `(max_slots, dim, width - 1)`,
            updated in place, or empty.
        output: Output tensor of shape `(dim, total_seqlen)`.
        x_fn: Reads the input `x`, of shape `(dim, total_seqlen)`.
        slot_fn: Reads each sequence's slot in `conv_states`. `PAD_SLOT_ID`
            skips the sequence.
    """
    comptime accum_dtype = get_accum_type[dtype]()
    comptime state_dtype = type_of(conv_states).dtype

    var width = Int(weight.dim[1]())
    var width_minus_1 = width - 1
    var batch = Int(query_start_loc.dim[0]()) - 1
    var has_conv_states = Int(conv_states.dim[0]()) > 0

    # `conv_states` is indexed, not addressed: the pool outgrows a 32-bit
    # offset, and TileTensor forms this one at 64 bits.
    var io = VarlenConvIO[channels_last](x_fn, weight, bias, output)

    for b in range(batch):
        var slot = _cache_slot(slot_fn, b)
        if slot == Int(PAD_SLOT_ID):
            continue

        var seq_start = Int(query_start_loc.load[width=1]((b,)))
        var seqlen = Int(query_start_loc.load[width=1]((b + 1,))) - seq_start
        var use_initial_state = _use_initial_state(
            has_initial_state, conv_states, b
        )

        for d in range(io.dim()):
            var bias_val = io.load_bias(d).cast[accum_dtype]()

            for l in range(seqlen):
                var conv_sum = bias_val
                for w_idx in range(width):
                    # Positions before the sequence start map to state index
                    # `width_minus_1 + input_l`.
                    var input_l = l - (width_minus_1 - w_idx)
                    var input_val = Scalar[dtype](0)
                    if input_l >= 0:
                        input_val = io.load_x(d, seq_start + input_l)
                    elif use_initial_state:
                        input_val = conv_states.load[width=1](
                            (slot, d, width_minus_1 + input_l)
                        ).cast[dtype]()
                    conv_sum += (
                        input_val.cast[accum_dtype]()
                        * io.load_weight(d, w_idx).cast[accum_dtype]()
                    )

                comptime if use_residual:
                    conv_sum += io.load_x(d, seq_start + l).cast[accum_dtype]()

                io.store_out(
                    d,
                    seq_start + l,
                    _apply_silu[accum_dtype](conv_sum, silu_activation).cast[
                        dtype
                    ](),
                )

            if has_conv_states:
                for s in range(width_minus_1):
                    var src_l = seqlen - width_minus_1 + s
                    var val = Scalar[state_dtype](0)
                    if src_l >= 0:
                        val = io.load_x(d, seq_start + src_l).cast[
                            state_dtype
                        ]()
                    elif use_initial_state:
                        val = conv_states.load[width=1](
                            (slot, d, width_minus_1 + src_l)
                        )
                    conv_states.store((slot, d, s), val)


def causal_conv1d_varlen_update_cpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    conv_state_dtype: DType,
    cache_seqlens_dtype: DType,
    conv_state_indices_dtype: DType,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
    x: TileTensor[
        mut=False, x_dtype, ...
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
    weight: TileTensor[mut=False, weight_dtype, ...],  # Shape (dim, width)
    bias: TileTensor[mut=False, bias_dtype, ...],  # Shape (dim,)
    conv_state: TileTensor[
        mut=True, conv_state_dtype, ...
    ],  # Shape (batch, dim, state_len)
    cache_seqlens: TileTensor[
        mut=False, cache_seqlens_dtype, ...
    ],  # Shape (batch,)
    conv_state_indices: TileTensor[
        mut=False, conv_state_indices_dtype, ...
    ],  # Shape (batch,)
    output: TileTensor[
        mut=True, output_dtype, ...
    ],  # Shape (batch, dim) or (batch, dim, seqlen)
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
            var state_idx_val = Int32(conv_state_indices.load[width=1]((b,)))
            if state_idx_val == pad_slot_id:
                continue

        # Determine actual batch index for conv_state
        var state_batch_idx = b
        if has_conv_state_indices:
            state_batch_idx = Int(conv_state_indices.load[width=1]((b,)))

        for d in range(dim):
            # Load bias
            var bias_val: Scalar[output_dtype] = 0
            if has_bias:
                bias_val = Scalar[output_dtype](bias.load[width=1]((d,)))

            # Load weights
            var weights = List[Scalar[weight_dtype]]()
            for w_idx in range(width):
                weights.append(weight.load[width=1]((d, w_idx)))

            for l in range(seqlen):
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
                            var cache_seqlen = Int(
                                cache_seqlens.load[width=1]((b,))
                            )
                            state_pos = (
                                cache_seqlen + rel_pos + l + state_len
                            ) % state_len
                        else:
                            # Linear buffer: position in state
                            state_pos = width_minus_1 + rel_pos + l

                        if state_pos >= 0 and state_pos < state_len:
                            input_val = Scalar[x_dtype](
                                conv_state.load[width=1](
                                    (state_batch_idx, d, state_pos)
                                )
                            )
                    else:
                        # Read from current x
                        var x_l = rel_pos + l
                        if x_l >= 0 and x_l < seqlen:
                            input_val = x.load[width=1]((b, d, x_l))

                    input_vals.append(input_val)

                # Compute convolution
                var conv_sum = bias_val
                for w_idx in range(width):
                    conv_sum += Scalar[output_dtype](
                        input_vals[w_idx] * Scalar[x_dtype](weights[w_idx])
                    )

                # Apply activation
                var out_val = _apply_silu[output_dtype](
                    conv_sum, silu_activation
                )

                # Store output
                output.store[width=1]((b, d, l), out_val)

            # Update state with new x values
            for l in range(seqlen):
                var x_val = x.load[width=1]((b, d, l))

                var state_pos: Int
                if has_cache_seqlens:
                    # Circular buffer
                    var cache_seqlen = Int(cache_seqlens.load[width=1]((b,)))
                    state_pos = (cache_seqlen + l) % state_len
                else:
                    # Shift state left and add new value at end
                    if l == 0:
                        # Shift existing values
                        for s in range(state_len - seqlen):
                            var val = conv_state.load[width=1](
                                (state_batch_idx, d, s + seqlen)
                            )
                            conv_state.store[width=1](
                                (state_batch_idx, d, s), val
                            )
                    state_pos = state_len - seqlen + l

                conv_state.store[width=1](
                    (state_batch_idx, d, state_pos),
                    Scalar[conv_state_dtype](x_val),
                )


# ============================================================================
# GPU Kernel Implementations
# ============================================================================


def causal_conv1d_varlen_states_gpu[
    x_dtype: DType,
    cu_seqlens_dtype: DType,
    states_dtype: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    x_LT: TensorLayout,
    cu_seqlens_LT: TensorLayout,
    states_LT: TensorLayout,
    x_engine: TensorEngine,
    cu_seqlens_engine: TensorEngine,
    states_engine: TensorEngine,
](
    total_tokens: Int32,
    dim: Int32,
    batch: Int32,
    state_len: Int32,
    x: TileTensor[
        x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine
    ],  # Shape (total_tokens, dim)
    cu_seqlens: TileTensor[
        cu_seqlens_dtype,
        cu_seqlens_LT,
        MutUntrackedOrigin,
        Engine=cu_seqlens_engine,
    ],  # Shape (batch + 1,)
    states: TileTensor[
        states_dtype, states_LT, MutUntrackedOrigin, Engine=states_engine
    ],  # Shape (batch, dim, state_len)
):
    """GPU kernel for extracting states from variable length sequences.

    Each thread block processes a tile of (BLOCK_M x BLOCK_N) elements.
    Grid dimensions: (ceildiv(dim, BLOCK_N), ceildiv(state_len, BLOCK_M), batch)

    Parameters:
        x_dtype: Data type of input.
        cu_seqlens_dtype: Data type of cumulative sequence lengths.
        states_dtype: Data type of output states.
        BLOCK_M: Tile size for sequence dimension.
        BLOCK_N: Tile size for channel dimension.
        x_LT: Layout type of input tensor.
        cu_seqlens_LT: Layout type of cumulative sequence lengths tensor.
        states_LT: Layout type of output states tensor.
        x_engine: Engine of input tensor.
        cu_seqlens_engine: Engine of cumulative sequence lengths tensor.
        states_engine: Engine of output states tensor.

    Args:
        total_tokens: Total number of tokens.
        dim: Number of channels.
        batch: Number of sequences.
        state_len: State length to extract.
        x: Input tensor.
        cu_seqlens: Cumulative sequence lengths.
        states: Output states tensor.
    """
    var _dim = Int(dim)
    var _state_len = Int(state_len)
    var batch_idx = block_idx.z
    var block_row = block_idx.y
    var block_col = block_idx.x
    var tid_row = thread_idx.y
    var tid_col = thread_idx.x

    # Load sequence boundaries
    var end_idx = Int(cu_seqlens.load[width=1]((batch_idx + 1,)))
    var start_idx_seq = Int(cu_seqlens.load[width=1]((batch_idx,)))
    var start_idx = max(start_idx_seq, end_idx - _state_len)

    # Calculate row indices (processing from end backwards)
    var row = end_idx - (block_row * BLOCK_M + tid_row + 1)
    var col = block_col * BLOCK_N + tid_col

    # Load value from x if in valid range
    var val: Scalar[states_dtype] = 0
    if row >= start_idx and col < _dim:
        val = Scalar[states_dtype](x.load[width=1]((row, col)))

    # Calculate state row index
    var states_row = _state_len - (block_row * BLOCK_M + tid_row + 1)

    # Store to states if in valid range
    if states_row >= 0 and col < _dim:
        states.store[width=1]((batch_idx, col, states_row), val)


# Tuned on B200, untuned elsewhere. Without the register cap the compiler keeps
# about 150 registers per thread and too few blocks stay resident to hide the
# load latency.
comptime _FWD_BLOCK_DIM = 64
comptime _FWD_TILE_SEQ = 64
comptime _FWD_UNROLL = 8
comptime _FWD_MIN_CTAS = 12


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_FWD_BLOCK_DIM))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(_FWD_MIN_CTAS))
def _causal_conv1d_varlen_fwd_kernel[
    dtype: DType,
    conv_states_dtype: DType,
    WIDTH: Int,
    VEC: Int,
    SEQ_PER_ROW: Bool,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    conv_states_LT: TensorLayout,
    output_LT: TensorLayout,
    Engine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[dtype, width],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    silu_activation: Bool,
    use_residual: Bool,
    channels_last: Bool,
](
    weight: TileTensor[dtype, weight_LT, MutUntrackedOrigin, Engine=Engine],
    bias: TileTensor[dtype, bias_LT, MutUntrackedOrigin, Engine=Engine],
    query_start_loc: TileTensor[
        .int32, query_start_loc_LT, MutUntrackedOrigin, Engine=Engine
    ],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    conv_states: TileTensor[
        conv_states_dtype, conv_states_LT, MutUntrackedOrigin, Engine=Engine
    ],
    output: TileTensor[dtype, output_LT, MutUntrackedOrigin, Engine=Engine],
    x_fn: XFn,
    slot_fn: SlotFn,
):
    """GPU kernel for causal conv1d forward with variable length sequences.

    Grid: `(ceildiv(dim, BLOCK_DIM * VEC), rows)`, block: `(BLOCK_DIM,)`.

    Each thread owns `VEC` consecutive channels and walks a `TILE_SEQ`-token
    tile of one sequence with a register window. A depthwise conv has no
    cross-position recurrence, so tiles are independent: the causal gather
    reads read-only `x` across tile boundaries, and `conv_states` is read and
    written only by each sequence's first tile.

    With `SEQ_PER_ROW`, row `b` walks all of sequence `b` and `rows` is the
    batch. Otherwise tile `k` of sequence `b` runs in row `b + seq_start(b) //
    TILE_SEQ + k`, which grows by at least one from tile to tile. A block finds
    its sequence by search. `rows` is `ceildiv(total_seqlen, TILE_SEQ) + batch`,
    enough to cover every tile without a host-side maximum over the ragged
    lengths, and still larger than the tile count. Rows that map to no tile
    return early, so the grid reduces empty rows rather than removing them.

    Parameters:
        dtype: Element type of `x`, `weight`, `bias` and `output`.
        conv_states_dtype: Element type of `conv_states`.
        WIDTH: Convolution width (number of taps).
        VEC: Channels per thread; above 1 needs `channels_last` and channel
            vectors aligned to their size in `x`, `weight`, `bias` and
            `output`.
        SEQ_PER_ROW: Whether a grid row walks a whole sequence instead of a
            tile.
        weight_LT: Layout type of `weight`.
        bias_LT: Layout type of `bias`.
        query_start_loc_LT: Layout type of `query_start_loc`.
        has_initial_state_LT: Layout type of `has_initial_state`.
        conv_states_LT: Layout type of `conv_states`.
        output_LT: Layout type of `output`.
        Engine: Engine shared by all tile operands.
        XFn: Reads `x` at its physical `(row, column)` index, claiming
            `alignment` elements of alignment.
        SlotFn: Reads the conv-state slot of each sequence.
        silu_activation: Whether to apply SiLU to the output.
        use_residual: If True, adds `x[d, s]` to the convolution sum before
            the activation.
        channels_last: If True, `x` and `output` are `(total_seqlen, dim)`
            rather than `(dim, total_seqlen)`.

    Args:
        weight: Weight tensor of shape `(dim, width)`.
        bias: Bias tensor of shape `(dim,)`, or empty.
        query_start_loc: Cumulative sequence lengths of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flags of shape `(batch,)`, or empty.
        conv_states: Convolution states of shape `(max_slots, dim, width - 1)`,
            updated in place, or empty.
        output: Output tensor, same shape as `x`.
        x_fn: Reads the input `x`.
        slot_fn: Reads each sequence's slot in `conv_states`. `PAD_SLOT_ID`
            skips the sequence.
    """
    comptime accum_dtype = get_accum_type[dtype]()
    comptime WIDTH_MINUS_1 = WIDTH - 1
    comptime TILE_SEQ = _FWD_TILE_SEQ
    comptime Vec = SIMD[dtype, VEC]
    comptime AccVec = SIMD[accum_dtype, VEC]

    var batch = Int(query_start_loc.dim[0]()) - 1
    var row = block_idx.y
    var b = row
    comptime if not SEQ_PER_ROW:
        # Each warp searches on its own, so no thread leaves before the
        # search: `lane_id()` probes `WARP_SIZE`-way and the sequence is the
        # last one whose grid row starts at or before this block's. A row is
        # at least the sequence index and at most `total_seqlen // TILE_SEQ`
        # above it, which leaves a couple of candidates at decode.
        comptime seq_axis = 0 if channels_last else 1
        b = Int(
            clamp(row - Int(output.dim[seq_axis]()) // TILE_SEQ, 0, batch - 1)
        )
        var hi = min(batch, row + 1)
        while hi - b > 1:
            var step = ceildiv(hi - b, WARP_SIZE)
            var p = b + lane_id() * step
            var le = p < hi and (
                Int(query_start_loc.load[width=1]((p,))) // TILE_SEQ + p <= row
            )
            # Monotonic in the lane, so the last hit is the highest ballot
            # bit. An empty ballot means that hit is before this window.
            comptime mask_type = DType.uint32 if WARP_SIZE <= 32 else DType.uint64
            var ballot = warp.vote[mask_type](le)
            var last = -1
            if ballot != 0:
                last = WARP_SIZE - 1 - Int(count_leading_zeros(ballot))
            b += last * step
            hi = min(hi, b + step)

    var d = (block_idx.x * _FWD_BLOCK_DIM + thread_idx.x) * VEC
    var io = VarlenConvIO[channels_last](x_fn, weight, bias, output)
    if d >= io.dim():
        return

    # These loads do not depend on each other, so they all go out before the
    # first branch on one of their results.
    var seq_start = Int(query_start_loc.load[width=1]((b,)))
    var seqlen = Int(query_start_loc.load[width=1]((b + 1,))) - seq_start
    var slot = _cache_slot(slot_fn, b)
    var use_initial_state = _use_initial_state(
        has_initial_state, conv_states, b
    )
    var bias_val = io.load_bias[VEC](d).cast[accum_dtype]()
    var tap_weights = Array[AccVec, WIDTH](fill=AccVec(0))
    comptime for w in range(WIDTH):
        comptime for c in range(VEC):
            tap_weights[w][c] = io.load_weight(d + c, w).cast[accum_dtype]()

    var tile_start = 0
    var tile_end = seqlen
    comptime if not SEQ_PER_ROW:
        # Tile 0 stays alive even for an empty sequence so it still reaches
        # the epilogue below and zeroes conv_states.
        tile_start = (row - seq_start // TILE_SEQ - b) * TILE_SEQ
        if tile_start >= max(seqlen, 1):
            return
        tile_end = min(tile_start + TILE_SEQ, seqlen)
    if slot == Int(PAD_SLOT_ID):
        return

    # Register sliding window over the WIDTH-1 inputs preceding the current
    # position, so the steady state costs one global load per output instead
    # of WIDTH. Negative positions fall back to the continued sequence's
    # stored state.
    var win = Array[AccVec, WIDTH_MINUS_1](fill=AccVec(0))
    comptime for i in range(WIDTH_MINUS_1):
        var pos = tile_start - WIDTH_MINUS_1 + i
        if pos >= 0:
            win[i] = io.load_x[VEC](d, seq_start + pos).cast[accum_dtype]()
        elif use_initial_state:
            comptime for c in range(VEC):
                win[i][c] = (
                    conv_states.load[width=1](
                        (slot, d + c, WIDTH_MINUS_1 + pos)
                    )
                    .cast[dtype]()
                    .cast[accum_dtype]()
                )

    # One U-output trip. The steady state calls this with U=UNROLL so the U
    # loads share no dependencies and can all be in flight; the tile remainder
    # reuses it with U=1.
    def _emit_chunk[U: Int](tile_off: Int) {mut win, imm}:
        var l0 = tile_start + tile_off
        var raw = Array[Vec, U](fill=Vec(0))
        comptime for u in range(U):
            raw[u] = io.load_x[VEC](d, seq_start + l0 + u)

        comptime for u in range(U):
            var cur = raw[u].cast[accum_dtype]()
            var conv_sum = bias_val
            comptime for w in range(WIDTH_MINUS_1):
                conv_sum += win[w] * tap_weights[w]
            conv_sum += cur * tap_weights[WIDTH_MINUS_1]

            comptime if use_residual:
                conv_sum += cur

            io.store_out[VEC](
                d,
                seq_start + l0 + u,
                _apply_silu[accum_dtype, VEC](conv_sum, silu_activation).cast[
                    dtype
                ](),
            )

            comptime if WIDTH_MINUS_1 > 0:
                comptime for w in range(WIDTH_MINUS_1 - 1):
                    win[w] = win[w + 1]
                win[WIDTH_MINUS_1 - 1] = cur

    vectorize[_FWD_UNROLL](tile_end - tile_start, _emit_chunk)

    # Tile 0 is the only block that reads the slot, so it also writes it;
    # the tail tile writing it would race with that read. When tile 0 is not
    # the tail, the sequence is longer than WIDTH-1 and its new state is
    # plain `x`; otherwise the window holds it, with the continued state
    # filling any entries a short chunk lacks.
    if Int(conv_states.dim[0]()) > 0 and tile_start == 0:
        comptime for i in range(WIDTH_MINUS_1):
            if tile_end != seqlen:
                win[i] = io.load_x[VEC](
                    d, seq_start + seqlen - WIDTH_MINUS_1 + i
                ).cast[accum_dtype]()
            comptime for c in range(VEC):
                conv_states.store(
                    (slot, d + c, i), win[i][c].cast[conv_states_dtype]()
                )


def causal_conv1d_varlen_fwd_gpu[
    dtype: DType,
    conv_states_dtype: DType,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    query_start_loc_LT: TensorLayout,
    has_initial_state_LT: TensorLayout,
    conv_states_LT: TensorLayout,
    output_LT: TensorLayout,
    Engine: TensorEngine,
    XFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int, Int) -> SIMD[dtype, width],
    SlotFn: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int](Int) -> SIMD[.uint32, width],
    //,
    WIDTH: Int,
    silu_activation: Bool = False,
    use_residual: Bool = False,
    channels_last: Bool = False,
](
    weight: TileTensor[dtype, weight_LT, _, Engine=Engine],
    bias: TileTensor[dtype, bias_LT, _, Engine=Engine],
    query_start_loc: TileTensor[.int32, query_start_loc_LT, _, Engine=Engine],
    has_initial_state: TileTensor[
        .bool, has_initial_state_LT, _, Engine=Engine
    ],
    conv_states: TileTensor[
        conv_states_dtype, conv_states_LT, _, Engine=Engine
    ],
    output: TileTensor[dtype, output_LT, _, Engine=Engine],
    x_addr: Int,
    x_row_stride: Int,
    x_fn: XFn,
    slot_fn: SlotFn,
    ctx: DeviceContext,
    x_vector_loads: Bool = True,
    x_col_stride: Int = 1,
) raises:
    """Launches the causal conv1d forward kernel with variable length sequences.

    Prefill moves 8 bytes of channels per thread when `channels_last` and the
    operands are aligned for it, and one channel per thread otherwise. Pure
    decode, where no sequence has more than a token on average, gives each
    thread one channel and each grid row a whole sequence.

    `x_addr`, `x_row_stride` and `x_col_stride` describe the view `x_fn` reads,
    copied out before the call so the tensor and the closure are not both live
    arguments. They are the address and strides of that view's first element.
    A fused input does not have them: the closure applies the slice offset
    inside the load, and the slice pointer is not that address. Pass
    `x_vector_loads=False` unless those three values are the ones the load
    uses, in which case `x_fn` may only issue a channel vector when they are
    8-byte aligned and the channel stride is 1.

    Parameters:
        dtype: Element type of `x`, `weight`, `bias` and `output`.
        conv_states_dtype: Element type of `conv_states`.
        weight_LT: Layout type of `weight`.
        bias_LT: Layout type of `bias`.
        query_start_loc_LT: Layout type of `query_start_loc`.
        has_initial_state_LT: Layout type of `has_initial_state`.
        conv_states_LT: Layout type of `conv_states`.
        output_LT: Layout type of `output`.
        Engine: Engine shared by all tile operands.
        XFn: Reads `x` at its physical `(row, column)` index, claiming
            `alignment` elements of alignment.
        SlotFn: Reads the conv-state slot of each sequence.
        WIDTH: Convolution width (number of taps).
        silu_activation: Whether to apply SiLU to the output.
        use_residual: If True, adds `x[d, s]` to the convolution sum before
            the activation.
        channels_last: If True, `x` and `output` are `(total_seqlen, dim)`
            rather than `(dim, total_seqlen)`.

    Args:
        weight: Weight tensor of shape `(dim, width)`.
        bias: Bias tensor of shape `(dim,)`, or empty.
        query_start_loc: Cumulative sequence lengths of shape `(batch + 1,)`.
        has_initial_state: Per-sequence flags of shape `(batch,)`, or empty.
        conv_states: Convolution states of shape `(max_slots, dim, width - 1)`,
            updated in place, or empty.
        output: Output tensor, same shape as `x`.
        x_addr: Address of the first element `x_fn` loads, when
            `x_vector_loads` is True.
        x_row_stride: Stride, in elements, of that view's first axis.
        x_fn: Reads `x`.
        slot_fn: Reads each sequence's slot in `conv_states`. `PAD_SLOT_ID`
            skips the sequence.
        ctx: The device context to launch on.
        x_vector_loads: Whether `x_fn` issues channel vectors. False when the
            address of those loads is not the one in `x_addr`.
        x_col_stride: Stride, in elements, of that view's channel axis. A
            channel vector needs this to be 1.
    """
    comptime seq_axis = 0 if channels_last else 1
    var dim = Int(output.dim[1 - seq_axis]())
    var total_seqlen = Int(output.dim[seq_axis]())
    var batch = Int(query_start_loc.dim[0]()) - 1

    @inline(.always)
    def launch[VEC: Int, SEQ_PER_ROW: Bool]() raises {imm}:
        comptime kernel = _causal_conv1d_varlen_fwd_kernel[
            dtype,
            conv_states_dtype,
            WIDTH,
            VEC,
            SEQ_PER_ROW,
            weight_LT,
            bias_LT,
            query_start_loc_LT,
            has_initial_state_LT,
            conv_states_LT,
            output_LT,
            Engine,
            XFn,
            SlotFn,
            silu_activation,
            use_residual,
            channels_last,
        ]
        ctx.enqueue_function[kernel](
            weight,
            bias,
            query_start_loc,
            has_initial_state,
            conv_states,
            output,
            host_arg=x_fn,
            host_arg2=slot_fn,
            grid_dim=(
                ceildiv(dim, _FWD_BLOCK_DIM * VEC),
                batch if SEQ_PER_ROW else ceildiv(total_seqlen, _FWD_TILE_SEQ)
                + batch,
            ),
            block_dim=_FWD_BLOCK_DIM,
        )

    if total_seqlen <= batch:
        launch[1, True]()
        return

    comptime if channels_last:
        # Channel vectors are contiguous 8-byte accesses. `x` is often a
        # column slice, so its own base and strides have to be the ones the
        # load uses; a fused input cannot prove that and passes
        # `x_vector_loads=False`. Bias and output are read and written along
        # their last axis, which has to be unit stride, not merely aligned.
        comptime V = 8 // size_of[dtype]()
        var x_ok = not x_vector_loads or (
            x_addr % 8 == 0 and x_row_stride % V == 0 and x_col_stride == 1
        )
        var bias_ok = Int(bias.dim[0]()) == 0 or (
            Int(bias.unsafe_ptr()) % 8 == 0
            and Int(bias.layout.stride[0]().value()) == 1
        )
        if (
            dim % V == 0
            and x_ok
            and bias_ok
            and Int(output.unsafe_ptr()) % 8 == 0
            and Int(weight.unsafe_ptr()) % 8 == 0
            and Int(output.layout.stride[0]().value()) % V == 0
            and Int(output.layout.stride[1]().value()) == 1
        ):
            launch[V, False]()
            return
    launch[1, False]()


def causal_conv1d_varlen_update_gpu[
    x_dtype: DType,
    weight_dtype: DType,
    bias_dtype: DType,
    output_dtype: DType,
    conv_state_dtype: DType,
    cache_seqlens_dtype: DType,
    conv_state_indices_dtype: DType,
    WIDTH: Int,
    BLOCK_DIM: Int,
    x_LT: TensorLayout,
    weight_LT: TensorLayout,
    bias_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    cache_seqlens_LT: TensorLayout,
    conv_state_indices_LT: TensorLayout,
    output_LT: TensorLayout,
    x_engine: TensorEngine,
    weight_engine: TensorEngine,
    bias_engine: TensorEngine,
    conv_state_engine: TensorEngine,
    cache_seqlens_engine: TensorEngine,
    conv_state_indices_engine: TensorEngine,
    output_engine: TensorEngine,
](
    batch: Int32,
    dim: Int32,
    seqlen: Int32,
    state_len: Int32,
    x: TileTensor[x_dtype, x_LT, MutUntrackedOrigin, Engine=x_engine],
    weight: TileTensor[
        weight_dtype, weight_LT, MutUntrackedOrigin, Engine=weight_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_LT, MutUntrackedOrigin, Engine=bias_engine
    ],
    conv_state: TileTensor[
        conv_state_dtype,
        conv_state_LT,
        MutUntrackedOrigin,
        Engine=conv_state_engine,
    ],
    cache_seqlens: TileTensor[
        cache_seqlens_dtype,
        cache_seqlens_LT,
        MutUntrackedOrigin,
        Engine=cache_seqlens_engine,
    ],
    conv_state_indices: TileTensor[
        conv_state_indices_dtype,
        conv_state_indices_LT,
        MutUntrackedOrigin,
        Engine=conv_state_indices_engine,
    ],
    output: TileTensor[
        output_dtype, output_LT, MutUntrackedOrigin, Engine=output_engine
    ],
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
    var _dim = Int(dim)
    var _seqlen = Int(seqlen)
    var _state_len = Int(state_len)
    var batch_idx = block_idx.x
    var dim_block_idx = block_idx.y
    var tid = thread_idx.x

    var d = dim_block_idx * BLOCK_DIM + tid

    # Check for padding
    if has_conv_state_indices != 0:
        var state_idx_val = Int32(
            conv_state_indices.load[width=1]((batch_idx,))
        )
        if state_idx_val == pad_slot_id:
            return

    if d >= _dim:
        return

    # Get state _batch index
    var state_batch_idx: Int = batch_idx
    if has_conv_state_indices != 0:
        state_batch_idx = Int(conv_state_indices.load[width=1]((batch_idx,)))

    # Load bias
    var bias_val: Scalar[output_dtype] = 0
    if has_bias != 0:
        bias_val = Scalar[output_dtype](bias.load[width=1]((d,)))

    # Load weights
    var weights = _channel_weights[weight_dtype, WIDTH](weight, d)

    comptime WIDTH_MINUS_1 = WIDTH - 1

    for l in range(_seqlen):
        # Get cache position
        var cache_offset = 0
        if has_cache_seqlens != 0:
            var cache_seqlen = Int(cache_seqlens.load[width=1]((batch_idx,)))
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
                        cache_offset + rel_pos + l + _state_len
                    ) % _state_len
                else:
                    state_pos = WIDTH_MINUS_1 + rel_pos + l

                if state_pos >= 0 and state_pos < _state_len:
                    input_val = Scalar[x_dtype](
                        conv_state.load[width=1](
                            (state_batch_idx, d, state_pos)
                        )
                    )
            else:
                # From x
                var x_l = rel_pos + l
                if x_l >= 0 and x_l < _seqlen:
                    input_val = x.load[width=1]((batch_idx, d, x_l))

            conv_sum += Scalar[output_dtype](
                input_val * Scalar[x_dtype](weights[w_idx])
            )
        # Apply activation
        var out_val = _apply_silu[output_dtype](conv_sum, silu_activation != 0)

        # Store output
        output.store[width=1]((batch_idx, d, l), out_val)

        # Update state
        var x_val = x.load[width=1]((batch_idx, d, l))

        var state_pos: Int
        if has_cache_seqlens != 0:
            state_pos = (cache_offset + l) % _state_len
        else:
            state_pos = _state_len - _seqlen + l

        conv_state.store[width=1](
            (state_batch_idx, d, state_pos),
            Scalar[conv_state_dtype](x_val),
        )
